//! A resource pack's `core/lightmap` shader (26.3's `Lightmap.render`, `RenderPipelines.LIGHTMAP`):
//! the 16x16 light texture drawn by the pack's GLSL from the frame's `LightmapInfo`, as a pack that
//! darkens caves or tints the night expects (wiki "Lightmaps", 1.21.11 and up). Without one the
//! texture is computed natively (`Renderer::update_lightmap`).

use crate::core_shader::{CoreProgram, Glue};

/// `core/screenquad`: no vertex input, three vertices from `gl_VertexIndex`. GL's framebuffer
/// rows run from the bottom and wgpu's from the top: the flip keeps row `n` at sky level `n`.
pub(crate) const GLUE: Glue = Glue {
    inputs: &[],
    prelude: "",
    setup: "",
    finish: "gl_Position.y = -gl_Position.y;",
    vertex_id: "int(gl_VertexIndex)",
};

/// The block `Lightmap.render` fills, beside the vanilla ones.
pub(crate) const INFO_BLOCK: &str = "LightmapInfo";

/// The pipeline's shader ids, which a pack's `shader_modifiers` target.
pub const STAGES: [&str; 2] = ["minecraft:core/screenquad", "minecraft:core/lightmap"];

/// Each vanilla uniform block's offset in the one buffer (`core_shader::BLOCKS` order), then
/// `LightmapInfo`'s.
const BLOCK_OFFSETS: [u64; 4] = [0, 256, 512, 768];
const BLOCK_SIZES: [u64; 4] = [48, 64, 160, 48];
const INFO_OFFSET: u64 = 1024;
const INFO_SIZE: u64 = 96;

/// `LightmapRenderState` as `Lightmap.render` writes it (std140: six floats, four vec3s).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LightmapInfo {
    pub sky_factor: f32,
    pub block_factor: f32,
    pub night_vision: f32,
    pub darkness: f32,
    pub boss_darkening: f32,
    pub brightness: f32,
    pub block_light_tint: [f32; 3],
    pub sky_light_color: [f32; 3],
    pub ambient_color: [f32; 3],
    pub night_vision_color: [f32; 3],
}

impl LightmapInfo {
    pub(crate) fn std140(&self) -> [u8; INFO_SIZE as usize] {
        let mut floats = [0f32; 24];
        floats[..6].copy_from_slice(&[
            self.sky_factor,
            self.block_factor,
            self.night_vision,
            self.darkness,
            self.boss_darkening,
            self.brightness,
        ]);
        // vec3s start on 16 bytes: floats 8, 12, 16 and 20.
        for (i, v) in [
            self.block_light_tint,
            self.sky_light_color,
            self.ambient_color,
            self.night_vision_color,
        ]
        .into_iter()
        .enumerate()
        {
            floats[8 + 4 * i..11 + 4 * i].copy_from_slice(&v);
        }
        let mut bytes = [0u8; INFO_SIZE as usize];
        bytes.copy_from_slice(bytemuck::cast_slice(&floats));
        bytes
    }
}

/// The light texture `lightmap.fsh` draws for `info`, computed here: RGBA rows of sky level by
/// block level, what the native path uploads.
pub(crate) fn native_lightmap(info: &LightmapInfo) -> [u8; 16 * 16 * 4] {
    use glam::Vec3;
    let brightness = |level: f32| level / (4.0 - 3.0 * level);
    let mut pixels = [0u8; 16 * 16 * 4];
    for s in 0..16 {
        for b in 0..16 {
            let block_level = b as f32 / 15.0;
            let sky_level = s as f32 / 15.0;
            let block_brightness = brightness(block_level) * info.block_factor;
            let sky_brightness = brightness(sky_level) * info.sky_factor;
            let mut color = Vec3::from(info.ambient_color)
                .max(Vec3::from(info.night_vision_color) * info.night_vision);
            color += Vec3::from(info.sky_light_color) * sky_brightness;
            let parabolic = (2.0 * block_level - 1.0).powi(2);
            let block_color = Vec3::from(info.block_light_tint).lerp(Vec3::ONE, 0.9 * parabolic);
            color += block_color * block_brightness;
            // A boss bar mixes the light toward 70/60/60%.
            color = color.lerp(color * Vec3::new(0.7, 0.6, 0.6), info.boss_darkening);
            color -= Vec3::splat(info.darkness);
            color = color.clamp(Vec3::ZERO, Vec3::ONE);
            let max = color.max_element().max(1e-6);
            let inverted = 1.0 - max;
            let scaled = 1.0 - inverted.powi(4);
            let not_gamma = color * (scaled / max);
            color = color.lerp(not_gamma, info.brightness);
            let i = (s * 16 + b) * 4;
            pixels[i] = (color.x * 255.0) as u8;
            pixels[i + 1] = (color.y * 255.0) as u8;
            pixels[i + 2] = (color.z * 255.0) as u8;
            pixels[i + 3] = 255;
        }
    }
    pixels
}

/// The packs' lightmap pipeline and its uniforms.
pub(super) struct CoreLightmap {
    pipeline: wgpu::RenderPipeline,
    group: wgpu::BindGroup,
    uniform: wgpu::Buffer,
    /// The packs' expression uniforms the shader reads (`shader_modifiers`), in their blocks'
    /// order after `LightmapInfo`'s.
    extra: Vec<String>,
}

/// The offset of the `i`th expression uniform's block.
fn extra_offset(i: usize) -> u64 {
    INFO_OFFSET + 256 * (i as u64 + 1)
}

impl CoreLightmap {
    /// The pipeline for `program`, drawing into an `Rgba8Unorm` texture; `None` when the pack's
    /// shader does not compile for the backend or reads a block it is not given.
    pub(super) fn new(device: &wgpu::Device, program: &CoreProgram) -> Option<Self> {
        let info_binding = program
            .other
            .iter()
            .position(|n| n == INFO_BLOCK)
            .map(crate::core_shader::other_binding);
        // The others are the expression uniforms the loader let through.
        let extra: Vec<(u32, String)> = program
            .other
            .iter()
            .enumerate()
            .filter(|(_, n)| *n != INFO_BLOCK)
            .map(|(i, n)| (crate::core_shader::other_binding(i), n.clone()))
            .collect();
        // A pack's shader is untrusted: the backend failing to compile it is an internal error.
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
        let module = |stage, module: &naga::Module| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("Resource pack lightmap ({stage})")),
                source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(module.clone())),
            })
        };
        let vertex = module("vertex", &program.vertex);
        let fragment = module("fragment", &program.fragment);
        let block = |binding: u32, size: u64| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(size),
            },
            count: None,
        };
        let mut entries: Vec<wgpu::BindGroupLayoutEntry> = (0..4)
            .map(|binding| block(binding, BLOCK_SIZES[binding as usize]))
            .collect();
        entries.extend(info_binding.map(|b| block(b, INFO_SIZE)));
        entries.extend(extra.iter().map(|(b, _)| block(*b, 16)));
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Resource pack lightmap"),
            entries: &entries,
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Resource pack lightmap"),
            size: extra_offset(extra.len()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let range = |binding: u32, offset: u64, size: u64| wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &uniform,
                offset,
                size: wgpu::BufferSize::new(size),
            }),
        };
        let mut resources: Vec<wgpu::BindGroupEntry> = (0..4)
            .map(|i| range(i as u32, BLOCK_OFFSETS[i], BLOCK_SIZES[i]))
            .collect();
        resources.extend(info_binding.map(|b| range(b, INFO_OFFSET, INFO_SIZE)));
        resources.extend(
            extra
                .iter()
                .enumerate()
                .map(|(i, (b, _))| range(*b, extra_offset(i), 16)),
        );
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Resource pack lightmap"),
            layout: &layout,
            entries: &resources,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Resource pack lightmap"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Resource pack lightmap"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vertex,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &fragment,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let internal = pollster::block_on(internal.pop());
        if let Some(e) = internal.or(pollster::block_on(validation.pop())) {
            log::warn!("the resource pack's lightmap shader is not used: {e}");
            return None;
        }
        Some(Self {
            pipeline,
            group,
            uniform,
            extra: extra.into_iter().map(|(_, n)| n).collect(),
        })
    }

    /// The buffer and the bytes the frame's blocks write to it: `globals` at 0, `info` at its
    /// own, and each expression uniform from `blocks` by its name (0 where a frame has none).
    pub(super) fn uniforms(
        &self,
        globals: [u8; 48],
        info: &LightmapInfo,
        blocks: Option<&std::collections::BTreeMap<String, Vec<u8>>>,
    ) -> (&wgpu::Buffer, Vec<u8>) {
        let mut data = vec![0u8; extra_offset(self.extra.len()) as usize];
        data[..48].copy_from_slice(&globals);
        let info_at = INFO_OFFSET as usize;
        data[info_at..info_at + INFO_SIZE as usize].copy_from_slice(&info.std140());
        for (i, name) in self.extra.iter().enumerate() {
            if let Some(bytes) = blocks.and_then(|b| b.get(name)) {
                let at = extra_offset(i) as usize;
                let n = bytes.len().min(16);
                data[at..at + n].copy_from_slice(&bytes[..n]);
            }
        }
        (&self.uniform, data)
    }

    /// `Lightmap.render`'s pass: the screen quad over the whole texture behind `view`.
    pub(super) fn draw(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Resource pack lightmap"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.group, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// The renderer's light texture: the pack's shader draws it, or it is computed here.
impl super::Renderer {
    /// The resource packs' `core/lightmap` (`CoreShaders::lightmap`), or the native one.
    pub fn set_lightmap_shader(&mut self, program: Option<&crate::core_shader::CoreProgram>) {
        self.core_lightmap = program.and_then(|p| CoreLightmap::new(&self.device, p));
        self.lightmap_uploaded = false;
    }
    pub(super) fn update_lightmap(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &super::Frame<'_>,
    ) {
        let (sky, darkness) = (&frame.sky, frame.darkness.1);
        let gamma = (self.gamma - frame.darkness.0).max(0.0);
        let (night_vision, flicker) = (frame.night_vision, frame.block_flicker);
        let (boss_darken, globals) = (frame.boss_darken, self.pack_globals(frame));
        let blocks = frame.post_effects.extras.map(|e| &e.blocks);
        // LightmapRenderStateExtractor.extract: what lightmap.fsh reads (NightVisionColor's
        // default is #999999).
        let info = LightmapInfo {
            sky_factor: sky.sky_light_factor,
            block_factor: 1.4 + flicker,
            night_vision,
            darkness,
            boss_darkening: boss_darken,
            brightness: gamma,
            block_light_tint: sky.block_light_tint,
            sky_light_color: sky.sky_light_color,
            ambient_color: sky.ambient_light_color,
            night_vision_color: [0.6; 3],
        };
        if let Some(core) = &self.core_lightmap
            && !self.fullbright
        {
            let (buffer, data) = core.uniforms(globals, &info, blocks);
            self.uploads.buffer(encoder, buffer, 0, &data);
            let view = self.lightmap.create_view(&Default::default());
            core.draw(encoder, &view);
            // Native pixels go on being made for F3's view of the lightmap.
            self.lightmap_uploaded = false;
        }
        let mut pixels = native_lightmap(&info);
        if self.fullbright {
            pixels.fill(255);
        }
        // The lightmap changes with the time of day, flicker and effects,
        // not every frame: upload only what differs from the texture.
        if self.core_lightmap.is_some() && !self.fullbright {
            self.lightmap_pixels = pixels;
            return;
        }
        if pixels == self.lightmap_pixels && self.lightmap_uploaded {
            return;
        }
        self.uploads.texture(
            encoder,
            wgpu::TexelCopyTextureInfo {
                texture: &self.lightmap,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            16,
            16,
            &pixels,
        );
        self.lightmap_pixels = pixels;
        self.lightmap_uploaded = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resources() -> std::path::PathBuf {
        std::env::var_os("HORIZON_RESOURCES")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../../.local/horizon-client")
                    .join(horizon_core::VANILLA_VERSION)
            })
    }

    /// Vanilla's own `screenquad` and `lightmap` through the pack path draw the texture the
    /// native path computes, row by sky level and column by block level, for a dim, tinted
    /// End-like state (green ambient, no sky light, night vision, a boss bar, some darkness).
    #[test]
    fn vanillas_lightmap_shader_draws_the_native_lightmap() {
        let root = resources();
        let shader = |name: &str| {
            std::fs::read_to_string(root.join(format!("assets/minecraft/shaders/core/{name}")))
        };
        let (Ok(vertex), Ok(fragment)) = (shader("screenquad.vsh"), shader("lightmap.fsh")) else {
            eprintln!("skipping: no prepared resources at {}", root.display());
            return;
        };
        let include = |id: &str| {
            let (namespace, path) = id.split_once(':')?;
            std::fs::read_to_string(root.join(format!("assets/{namespace}/{path}"))).ok()
        };
        let program = crate::core_shader::translate_core_with(
            &vertex,
            &fragment,
            &[],
            &include,
            &GLUE,
            &|b| b == INFO_BLOCK,
        )
        .expect("vanilla's lightmap shaders translate");
        let instance = crate::gfx::test_gpu::instance();
        let Some((device, queue)) =
            pollster::block_on(instance.request_adapter(&Default::default()))
                .ok()
                .and_then(|adapter| {
                    pollster::block_on(adapter.request_device(&Default::default())).ok()
                })
        else {
            eprintln!("no GPU: skipping");
            return;
        };
        let core = CoreLightmap::new(&device, &program).expect("the lightmap pipeline");
        for info in [
            LightmapInfo {
                sky_factor: 0.0,
                block_factor: 1.4,
                night_vision: 0.0,
                darkness: 0.0,
                boss_darkening: 0.0,
                brightness: 1.0,
                block_light_tint: [1.0, 0.847, 0.549],
                sky_light_color: [0.675, 0.376, 0.804],
                ambient_color: [0.247, 0.278, 0.247],
                night_vision_color: [0.6; 3],
            },
            LightmapInfo {
                sky_factor: 0.7,
                block_factor: 1.45,
                night_vision: 0.3,
                darkness: 0.1,
                boss_darkening: 0.5,
                brightness: 0.5,
                block_light_tint: [1.0, 0.6, 0.3],
                sky_light_color: [0.8, 0.8, 1.0],
                ambient_color: [0.04; 3],
                night_vision_color: [0.6; 3],
            },
        ] {
            let target = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: 16,
                    height: 16,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            // Rows of 256 bytes, as copies need.
            let read = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 256 * 16,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let (buffer, data) = core.uniforms([0; 48], &info, None);
            queue.write_buffer(buffer, 0, &data);
            let mut encoder = device.create_command_encoder(&Default::default());
            core.draw(&mut encoder, &target.create_view(&Default::default()));
            encoder.copy_texture_to_buffer(
                target.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &read,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(256),
                        rows_per_image: Some(16),
                    },
                },
                target.size(),
            );
            queue.submit([encoder.finish()]);
            read.slice(..)
                .map_async(wgpu::MapMode::Read, |r| r.unwrap());
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let gpu = read.slice(..).get_mapped_range().unwrap().to_vec();
            let native = native_lightmap(&info);
            for sky in 0..16 {
                for block in 0..16 {
                    let n = &native[(sky * 16 + block) * 4..][..4];
                    let g = &gpu[sky * 256 + block * 4..][..4];
                    // The native path truncates, the GPU rounds.
                    assert!(
                        n.iter().zip(g).all(|(a, b)| a.abs_diff(*b) <= 1),
                        "sky {sky} block {block}: native {n:?} shader {g:?}"
                    );
                }
            }
        }
    }
}
