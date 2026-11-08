//! Polytone GPU particle types with their own core shader (wiki "GPU Particles", `shader`): each
//! type's live records go to its shader as the vertex format, the same record on the quad's four
//! corners, and the shader places, colours and kills them on the graphics card. Types without a
//! shader are built on the CPU with the engine's particles (`particles::polytone_gpu`).

use crate::particles::polytone_gpu::{
    GPU_BLOCK, GPU_BLOCK_SIZE, GpuShader, ShaderBatch, ShaderVertex,
};
use horizon_polytone::particles::RenderType;
use std::sync::Arc;

/// Each vanilla block's offset in a pipeline's uniform buffer (`core_shader::BLOCKS` order), as
/// `core_lines::uniforms` lays them out, and their sizes.
const BLOCK_OFFSETS: [u64; 4] = [0, 256, 512, 768];
const BLOCK_SIZES: [u64; 4] = [48, 64, 160, 48];
/// Where the type's own blocks start (`polytone:gpu_particle.glsl`, then one per uniform), 256
/// bytes apart.
const OTHER_OFFSET: u64 = 1024;
/// A pack uniform's block: one float, padded as std140 pads a block.
const FLOAT_BLOCK: u64 = 16;
/// The heightmap `Sampler1` holds: this many columns a side around the camera.
pub const HEIGHTMAP_SIZE: u32 = 128;

/// The record as vertex attributes 0 to 5 (`polytone_gpu::GLUE`).
const ATTRIBUTES: [wgpu::VertexAttribute; 6] = wgpu::vertex_attr_array![
    0 => Float32x3,
    1 => Float32x3,
    2 => Float32x2,
    3 => Float32x4,
    4 => Unorm8x4,
    5 => Sint32x2,
];

/// One type's pipeline over its shader, with its uniforms and its texture.
struct Pipeline {
    pipeline: wgpu::RenderPipeline,
    group: wgpu::BindGroup,
    uniform: wgpu::Buffer,
    /// Each block after the vanilla ones: the type's knobs, or the pack uniform of that name.
    other: Vec<Other>,
}

enum Other {
    Knobs,
    Uniform(String),
}

impl Other {
    fn size(&self) -> u64 {
        match self {
            Self::Knobs => GPU_BLOCK_SIZE as u64,
            Self::Uniform(_) => FLOAT_BLOCK,
        }
    }
}

/// A draw of this frame: a pipeline and its quads in the record buffer.
struct Draw {
    pipeline: usize,
    first: u32,
    quads: u32,
}

pub(super) struct CoreParticles {
    /// By type: `None` for a shader that failed to build (logged once).
    pipelines: Vec<(Arc<GpuShader>, Option<Pipeline>)>,
    vertices: wgpu::Buffer,
    heightmap: wgpu::Texture,
    heightmap_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    blank: wgpu::TextureView,
    draws: Vec<Draw>,
}

impl CoreParticles {
    pub(super) fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let heightmap = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Polytone GPU particle heightmap"),
            size: wgpu::Extent3d {
                width: HEIGHTMAP_SIZE,
                height: HEIGHTMAP_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let blank = texture(device, queue, 1, 1, &[255; 4]);
        Self {
            pipelines: Vec::new(),
            vertices: vertex_buffer(device, 1 << 16),
            heightmap_view: heightmap.create_view(&Default::default()),
            heightmap,
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("Polytone GPU particles"),
                ..Default::default()
            }),
            blank: blank.create_view(&Default::default()),
            draws: Vec::new(),
        }
    }

    /// The motion-blocking heights `Sampler1` holds: [`HEIGHTMAP_SIZE`] columns a side, x along
    /// a row and z down the rows, each the height above the world's bottom.
    pub(super) fn set_heightmap(&self, queue: &wgpu::Queue, heights: &[u16]) {
        let size = HEIGHTMAP_SIZE as usize;
        if heights.len() != size * size {
            return;
        }
        // Wiki "GPU Particles": r is the low byte, g the high byte.
        let texels: Vec<u8> = heights
            .iter()
            .flat_map(|h| {
                let [low, high] = h.to_le_bytes();
                [low, high, 0, 255]
            })
            .collect();
        queue.write_texture(
            self.heightmap.as_image_copy(),
            &texels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(HEIGHTMAP_SIZE * 4),
                rows_per_image: Some(HEIGHTMAP_SIZE),
            },
            self.heightmap.size(),
        );
    }

    /// This frame's batches: the pipelines they need, their uniforms and records uploaded.
    /// `vanilla` is the four vanilla blocks as `core_lines::uniforms` lays them out.
    pub(super) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        lightmap: (&wgpu::TextureView, &wgpu::Sampler),
        batches: &[ShaderBatch],
        vanilla: &[u8],
    ) {
        self.draws.clear();
        let quads: usize = batches.iter().map(|b| b.vertices.len() / 4).sum();
        if quads == 0 {
            return;
        }
        let needed = (quads * 4 * std::mem::size_of::<ShaderVertex>()) as u64;
        if self.vertices.size() < needed {
            self.vertices = vertex_buffer(device, needed.next_power_of_two());
        }
        let mut vertices: Vec<ShaderVertex> = Vec::with_capacity(quads * 4);
        for batch in batches {
            let index = self.pipeline(device, queue, format, lightmap, &batch.shader);
            let Some(pipeline) = &self.pipelines[index].1 else {
                continue;
            };
            let mut data = vec![0u8; OTHER_OFFSET as usize + 256 * pipeline.other.len()];
            let n = vanilla.len().min(OTHER_OFFSET as usize);
            data[..n].copy_from_slice(&vanilla[..n]);
            for (i, other) in pipeline.other.iter().enumerate() {
                let at = OTHER_OFFSET as usize + 256 * i;
                match other {
                    Other::Knobs => data[at..at + GPU_BLOCK_SIZE].copy_from_slice(&batch.block),
                    // An expression that failed, or a uniform the batch lacks, reads 0.
                    Other::Uniform(name) => {
                        let value = batch
                            .uniforms
                            .iter()
                            .find(|(n, _)| n == name)
                            .map_or(0.0, |(_, v)| *v);
                        data[at..at + 4].copy_from_slice(&value.to_le_bytes());
                    }
                }
            }
            queue.write_buffer(&pipeline.uniform, 0, &data);
            let count = batch.vertices.len() / 4;
            self.draws.push(Draw {
                pipeline: index,
                first: (vertices.len() / 4) as u32,
                quads: count as u32,
            });
            vertices.extend_from_slice(&batch.vertices[..count * 4]);
        }
        if !vertices.is_empty() {
            queue.write_buffer(&self.vertices, 0, bytemuck::cast_slice(&vertices));
        }
    }

    /// Draw this frame's batches in the world pass, after the translucent particles, with the
    /// renderer's quad indices (`indices`, `index_quads` quads: two triangles over each four
    /// corners), which stay bound. Whether any drew: the caller then sets its own pipeline,
    /// vertex buffer and group 0 again.
    pub(super) fn draw(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        indices: &wgpu::Buffer,
        index_quads: u32,
    ) -> bool {
        if self.draws.is_empty() || index_quads == 0 {
            return false;
        }
        pass.set_vertex_buffer(0, self.vertices.slice(..));
        pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
        for draw in &self.draws {
            let Some(pipeline) = &self.pipelines[draw.pipeline].1 else {
                continue;
            };
            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(0, &pipeline.group, &[]);
            // More quads than indices: the same indices over the next corners.
            let mut done = 0;
            while done < draw.quads {
                let n = (draw.quads - done).min(index_quads);
                pass.draw_indexed(0..n * 6, ((draw.first + done) * 4) as i32, 0..1);
                done += n;
            }
        }
        true
    }

    /// The pipeline for `shader`, built the first time a batch asks for it.
    fn pipeline(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        lightmap: (&wgpu::TextureView, &wgpu::Sampler),
        shader: &Arc<GpuShader>,
    ) -> usize {
        if let Some(i) = self
            .pipelines
            .iter()
            .position(|(s, _)| Arc::ptr_eq(s, shader))
        {
            return i;
        }
        // The shaders of packs no longer loaded: only this cache holds them.
        self.pipelines.retain(|(s, _)| Arc::strong_count(s) > 1);
        let built = self.build(device, queue, format, lightmap, shader);
        self.pipelines.push((shader.clone(), built));
        self.pipelines.len() - 1
    }

    fn build(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        lightmap: (&wgpu::TextureView, &wgpu::Sampler),
        shader: &GpuShader,
    ) -> Option<Pipeline> {
        let program = &shader.program;
        let Some(other) = program
            .other
            .iter()
            .map(|name| {
                if name == GPU_BLOCK {
                    Some(Other::Knobs)
                } else if shader.uniforms.iter().any(|(n, _)| n == name) {
                    Some(Other::Uniform(name.clone()))
                } else {
                    None
                }
            })
            .collect::<Option<Vec<Other>>>()
        else {
            log::warn!(
                "custom particle shader {} is not used: it reads a block or sampler that is \
                 neither polytone:gpu_particle.glsl's nor one of its uniforms",
                shader.id
            );
            return None;
        };
        // A pack's shader is untrusted: the backend failing to compile it is an internal error.
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
        let module = |label, module: &naga::Module| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(module.clone())),
            })
        };
        let vertex = module("Polytone GPU particle (vertex)", &program.vertex);
        let fragment = module("Polytone GPU particle (fragment)", &program.fragment);
        let buffer_entry = |binding, size| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(size),
            },
            count: None,
        };
        // Vanilla's blocks and Sampler0 to 2 (`core_shader::BLOCKS`, `SAMPLERS`), then the
        // type's own blocks.
        let mut entries: Vec<wgpu::BindGroupLayoutEntry> = (0..4)
            .map(|b| buffer_entry(b, BLOCK_SIZES[b as usize]))
            .collect();
        for texture in [4, 6, 8] {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: texture,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            });
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: texture + 1,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            });
        }
        for (i, o) in other.iter().enumerate() {
            entries.push(buffer_entry(crate::core_shader::other_binding(i), o.size()));
        }
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Polytone GPU particle"),
            entries: &entries,
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Polytone GPU particle uniforms"),
            size: OTHER_OFFSET + 256 * other.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let own = shader.texture.as_ref().map(|(w, h, pixels)| {
            texture(device, queue, *w, *h, pixels).create_view(&Default::default())
        });
        let buffer = |binding, offset, size| wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &uniform,
                offset,
                size: wgpu::BufferSize::new(size),
            }),
        };
        let mut resources: Vec<wgpu::BindGroupEntry> = (0..4)
            .map(|i| buffer(i as u32, BLOCK_OFFSETS[i], BLOCK_SIZES[i]))
            .collect();
        for (binding, view, sampler) in [
            (4, own.as_ref().unwrap_or(&self.blank), &self.sampler),
            (6, &self.heightmap_view, &self.sampler),
            (8, lightmap.0, lightmap.1),
        ] {
            resources.push(wgpu::BindGroupEntry {
                binding,
                resource: wgpu::BindingResource::TextureView(view),
            });
            resources.push(wgpu::BindGroupEntry {
                binding: binding + 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            });
        }
        for (i, o) in other.iter().enumerate() {
            resources.push(buffer(
                crate::core_shader::other_binding(i),
                OTHER_OFFSET + 256 * i as u64,
                o.size(),
            ));
        }
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Polytone GPU particle"),
            layout: &layout,
            entries: &resources,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Polytone GPU particle"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        // Wiki "GPU Particles": the two translucent types blend (the additive one src alpha
        // and one); anything else is alpha cutout, the shader's own discard, with depth write.
        let (blend, depth_write) = match shader.render_type {
            RenderType::Translucent => (Some(wgpu::BlendState::ALPHA_BLENDING), false),
            RenderType::AdditiveTranslucent => (
                Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::SrcAlpha,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent::OVER,
                }),
                false,
            ),
            _ => (None, true),
        };
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Polytone GPU particle"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vertex,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<ShaderVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &ATTRIBUTES,
                })],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(depth_write),
                depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &fragment,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let internal = pollster::block_on(internal.pop());
        if let Some(e) = internal.or(pollster::block_on(validation.pop())) {
            log::warn!("custom particle shader {} is not used: {e}", shader.id);
            return None;
        }
        Some(Pipeline {
            pipeline,
            group,
            uniform,
            other,
        })
    }
}

fn vertex_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Polytone GPU particle records"),
        size,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
    pixels: &[u8],
) -> wgpu::Texture {
    wgpu::util::DeviceExt::create_texture_with_data(
        device,
        queue,
        &wgpu::TextureDescriptor {
            label: Some("Polytone GPU particle texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        pixels,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particles::polytone_gpu::GLUE;
    use glam::Mat4;

    const INCLUDES: [(&str, &str); 2] = [
        (
            "minecraft:shaders/include/dynamictransforms.glsl",
            "layout(std140) uniform DynamicTransforms {\n    mat4 ModelViewMat;\n    mat4 TextureMat;\n    vec4 ColorModulator;\n    vec3 ModelOffset;\n};\n",
        ),
        (
            "minecraft:shaders/include/projection.glsl",
            "layout(std140) uniform Projection {\n    mat4 ProjMat;\n};\n",
        ),
    ];

    fn include(id: &str) -> Option<String> {
        if id == crate::particles::polytone_gpu::GPU_PARTICLE_INCLUDE {
            return Some(crate::particles::polytone_gpu::GPU_PARTICLE_GLSL.to_owned());
        }
        INCLUDES
            .iter()
            .find(|(path, _)| *path == id)
            .map(|(_, text)| (*text).to_owned())
    }

    /// A custom shader in the wiki's 1.21.11 form: the record's quad around `Position + Origin`,
    /// `Params.x` its half-width, the corner from `gl_VertexIndex & 3`.
    const VSH: &str = "#version 330\n#extension GL_ARB_separate_shader_objects : require\n\
        #include <minecraft:dynamictransforms.glsl>\n#include <minecraft:projection.glsl>\n\
        #include <polytone:gpu_particle.glsl>\n\
        layout(location = 0) in vec3 Position;\nlayout(location = 1) in vec3 Velocity;\n\
        layout(location = 2) in vec2 SpawnLife;\nlayout(location = 3) in vec4 Params;\n\
        layout(location = 4) in vec4 Color;\nlayout(location = 5) in ivec2 UV2;\n\
        layout(location = 0) out vec4 vertexColor;\n\
        void main() {\n    int corner = gl_VertexIndex & 3;\n\
            vec2 side = vec2(corner == 1 || corner == 2 ? 1.0 : -1.0, corner >= 2 ? 1.0 : -1.0);\n\
            vec3 at = Position + Origin + vec3(side * Params.x, 0.0);\n\
            gl_Position = ProjMat * ModelViewMat * vec4(at, 1.0);\n\
            vertexColor = Color;\n}\n";

    /// The record's colour times the pack uniform `Brightness`, a block of its own.
    const FSH: &str = "#version 330\n#extension GL_ARB_separate_shader_objects : require\n\
        layout(std140) uniform Brightness {\n    float brightnessValue;\n};\n\
        layout(location = 0) in vec4 vertexColor;\nlayout(location = 0) out vec4 fragColor;\n\
        void main() {\n    fragColor = vec4(vertexColor.rgb * brightnessValue, 1.0);\n}\n";

    fn shader(fragment: &str, uniforms: &[&str]) -> Arc<GpuShader> {
        let provided = |name: &str| matches!(name, GPU_BLOCK | "Brightness");
        let program =
            crate::core_shader::translate_core_with(VSH, fragment, &[], &include, &GLUE, &provided)
                .expect("the shaders translate");
        Arc::new(GpuShader {
            id: "test:particle".to_owned(),
            program: Arc::new(program),
            texture: None,
            render_type: RenderType::Opaque,
            uniforms: uniforms
                .iter()
                .map(|name| {
                    let program = horizon_polytone::expr::Program::compile("0.5").unwrap();
                    ((*name).to_owned(), program)
                })
                .collect(),
        })
    }

    fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = crate::gfx::test_gpu::instance();
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
        pollster::block_on(adapter.request_device(&Default::default())).ok()
    }

    const SIZE: u32 = 64;

    /// One red record at the batch's `Origin`, 2 blocks ahead of the camera, a block wide, with
    /// `Brightness` 0.5, drawn on a 64 by 64 target: the RGBA rows read back, and whether it drew.
    fn draw(device: &wgpu::Device, queue: &wgpu::Queue, shader: Arc<GpuShader>) -> (Vec<u8>, bool) {
        let mut block = [0u8; GPU_BLOCK_SIZE];
        block[..12].copy_from_slice(bytemuck::cast_slice(&[0.0f32, 0.0, -2.0]));
        let mut vertex: ShaderVertex = bytemuck::Zeroable::zeroed();
        vertex.spawn_life = [0.0, 100.0];
        vertex.params = [0.5, 0.0, 0.0, 0.0];
        vertex.color = [255, 0, 0, 255];
        let batch = ShaderBatch {
            shader,
            vertices: vec![vertex; 4],
            block,
            uniforms: vec![("Brightness".to_owned(), 0.5)],
        };
        draw_batch(device, queue, batch)
    }

    fn draw_batch(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batch: ShaderBatch,
    ) -> (Vec<u8>, bool) {
        let target = |format, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: SIZE,
                    height: SIZE,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | usage,
                view_formats: &[],
            })
        };
        let color = target(
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::COPY_SRC,
        );
        let depth = target(
            wgpu::TextureFormat::Depth32Float,
            wgpu::TextureUsages::empty(),
        );
        let lightmap = texture(device, queue, 1, 1, &[255; 4]).create_view(&Default::default());
        let lightmap_sampler = device.create_sampler(&Default::default());
        let view = super::super::View {
            view: Mat4::IDENTITY,
            projection: super::super::world_projection(70.0, 1.0, 64.0),
            aspect: 1.0,
            tan_half_fov: 35f32.to_radians().tan(),
        };
        let vanilla = super::super::core_lines::uniforms(
            [0; 48],
            &view,
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            1.0,
            [0.0; 4],
            [0.0; 6],
        );
        let mut particles = CoreParticles::new(device, queue);
        let indices = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[0u32, 1, 2, 0, 2, 3]),
                usage: wgpu::BufferUsages::INDEX,
            },
        );
        particles.prepare(
            device,
            queue,
            wgpu::TextureFormat::Rgba8Unorm,
            (&lightmap, &lightmap_sampler),
            &[batch],
            &vanilla,
        );
        let read = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(SIZE * SIZE * 4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        let drew = {
            let color_view = color.create_view(&Default::default());
            let depth_view = depth.create_view(&Default::default());
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            particles.draw(&mut pass, &indices, 1)
        };
        encoder.copy_texture_to_buffer(
            color.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &read,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(SIZE * 4),
                    rows_per_image: Some(SIZE),
                },
            },
            color.size(),
        );
        queue.submit([encoder.finish()]);
        read.slice(..)
            .map_async(wgpu::MapMode::Read, |r| r.unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let pixels = read.slice(..).get_mapped_range().unwrap().to_vec();
        (pixels, drew)
    }

    fn at(pixels: &[u8], x: u32, y: u32) -> [u8; 4] {
        let i = ((y * SIZE + x) * 4) as usize;
        pixels[i..i + 4].try_into().unwrap()
    }

    /// The record reaches the shader as its vertex format (position, size, colour), the knobs
    /// block carries `Origin` and the pack's uniform is its own block: a half-red quad in the
    /// middle, nothing at the edge.
    #[test]
    fn a_custom_shader_draws_its_records_with_its_uniforms() {
        let Some((device, queue)) = device() else {
            eprintln!("no GPU: skipping");
            return;
        };
        let (pixels, drew) = draw(&device, &queue, shader(FSH, &["Brightness"]));
        assert!(drew);
        let middle = at(&pixels, 32, 32);
        assert!(
            middle[0].abs_diff(128) <= 1 && middle[1..] == [0, 0, 255],
            "{middle:?}"
        );
        assert_eq!(at(&pixels, 2, 2), [0, 0, 0, 0]);
    }

    /// A shader reading a block that is neither the knobs nor one of its type's `uniforms` is
    /// not used: nothing draws, rather than failing at the draw.
    #[test]
    fn a_shader_with_an_unknown_block_is_not_used() {
        let Some((device, queue)) = device() else {
            return;
        };
        let (pixels, drew) = draw(&device, &queue, shader(FSH, &[]));
        assert!(!drew);
        assert_eq!(at(&pixels, 32, 32), [0, 0, 0, 0]);
    }
}
