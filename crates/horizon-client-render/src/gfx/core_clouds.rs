//! A resource pack's `core/clouds` shader (26.3's `CloudRenderer`, `RenderPipelines.CLOUDS` and
//! `FLAT_CLOUDS`): the clouds drawn by the pack's GLSL from the faces `CloudRenderer.buildMesh`
//! encodes (`CloudFaces`, an `R8_SINT` texel buffer, here a storage buffer of ints) and its
//! `CloudInfo` block, as a pack that shades or reshapes the clouds expects. Without one the
//! clouds draw natively (`clouds.rs`, `vs_cloud`).

use crate::core_shader::{CoreProgram, Glue};

/// `core/clouds`: no vertex input, the quad and its corner from `gl_VertexIndex`.
pub(crate) const GLUE: Glue = Glue {
    inputs: &[],
    prelude: "",
    setup: "",
    // 26.3's ProjMat already has wgpu's reversed 0 to 1 depth.
    finish: "",
    vertex_id: "int(gl_VertexIndex)",
};

/// The pipeline's own block and buffer, beside the vanilla blocks.
pub(crate) const INFO_BLOCK: &str = "CloudInfo";
pub(crate) const FACES: &str = "CloudFaces";

/// Each vanilla block's offset in the one buffer (`core_shader::BLOCKS` order), then CloudInfo's.
const BLOCK_OFFSETS: [u64; 4] = [0, 256, 512, 768];
const BLOCK_SIZES: [u64; 4] = [48, 64, 160, 48];
const INFO_OFFSET: u64 = 1024;
const INFO_SIZE: u64 = 48;

/// The packs' cloud pipelines (with and without back faces culled) and their buffers.
pub(super) struct CoreClouds {
    fancy: wgpu::RenderPipeline,
    flat: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    group: wgpu::BindGroup,
    uniform: wgpu::Buffer,
    faces: wgpu::Buffer,
    /// Ints the faces buffer holds.
    capacity: u64,
    faces_binding: u32,
    info_binding: Option<u32>,
    extra: super::expression_blocks::ExpressionBlocks,
    /// The faces written last.
    last: Vec<i32>,
    /// What they were built for: unchanged, they are not built again.
    key: Option<crate::clouds::PackFacesKey>,
}

impl CoreClouds {
    /// The pipelines for `program`; `None` when the pack's shader does not read the faces, reads
    /// a block it is not given, or does not compile for the backend.
    pub(super) fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        program: &CoreProgram,
    ) -> Option<Self> {
        let binding = |name: &str| {
            program
                .other
                .iter()
                .position(|n| n == name)
                .map(crate::core_shader::other_binding)
        };
        let faces_binding = binding(FACES)?;
        let info_binding = binding(INFO_BLOCK);
        let extra = super::expression_blocks::ExpressionBlocks::new(
            program,
            &[INFO_BLOCK, FACES],
            INFO_OFFSET + 256,
        );
        // A pack's shader is untrusted: the backend failing to compile it is an internal error.
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
        let module = |stage, module: &naga::Module| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("Resource pack clouds ({stage})")),
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
        let mut entries: Vec<wgpu::BindGroupLayoutEntry> =
            (0..4).map(|b| block(b, BLOCK_SIZES[b as usize])).collect();
        entries.extend(info_binding.map(|b| block(b, INFO_SIZE)));
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: faces_binding,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        entries.extend(extra.layout());
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Resource pack clouds"),
            entries: &entries,
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Resource pack clouds"),
            size: extra.size().max(INFO_OFFSET + 256),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let capacity = 3 * 1024;
        let faces = faces_buffer(device, capacity);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Resource pack clouds"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        // RenderPipelines.CLOUDS: translucent, the default depth test and writes, back faces
        // culled; FLAT_CLOUDS without culling.
        let pipeline = |label, cull_mode| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &vertex,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState {
                    cull_mode,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
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
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let fancy = pipeline("Resource pack clouds", Some(wgpu::Face::Back));
        let flat = pipeline("Resource pack flat clouds", None);
        let internal = pollster::block_on(internal.pop());
        if let Some(e) = internal.or(pollster::block_on(validation.pop())) {
            log::warn!("the resource pack's clouds shader is not used: {e}");
            return None;
        }
        let group = Self::bind(
            device,
            &layout,
            &uniform,
            &faces,
            faces_binding,
            info_binding,
            &extra,
        );
        Some(Self {
            fancy,
            flat,
            layout,
            group,
            uniform,
            faces,
            capacity,
            faces_binding,
            info_binding,
            extra,
            last: Vec::new(),
            key: None,
        })
    }

    fn bind(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        uniform: &wgpu::Buffer,
        faces: &wgpu::Buffer,
        faces_binding: u32,
        info_binding: Option<u32>,
        extra: &super::expression_blocks::ExpressionBlocks,
    ) -> wgpu::BindGroup {
        let range = |binding: u32, offset: u64, size: u64| wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: uniform,
                offset,
                size: wgpu::BufferSize::new(size),
            }),
        };
        let mut entries: Vec<wgpu::BindGroupEntry> = (0..4)
            .map(|i| range(i as u32, BLOCK_OFFSETS[i], BLOCK_SIZES[i]))
            .collect();
        entries.extend(info_binding.map(|b| range(b, INFO_OFFSET, INFO_SIZE)));
        entries.push(wgpu::BindGroupEntry {
            binding: faces_binding,
            resource: faces.as_entire_binding(),
        });
        entries.extend(extra.entries(uniform));
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Resource pack clouds"),
            layout,
            entries: &entries,
        })
    }

    /// The frame's blocks (`blocks`: the vanilla ones, as `core_lines::uniforms` makes them) and
    /// the clouds' faces and `CloudInfo` (`clouds::pack_faces`), uploaded; the faces built and
    /// uploaded only when they change (`CloudRenderer` rebuilds them when the camera's cell or
    /// side of the layer does).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare(
        &mut self,
        device: &wgpu::Device,
        uploads: &mut super::upload::Uploads,
        encoder: &mut wgpu::CommandEncoder,
        mut blocks: Vec<u8>,
        input: &crate::clouds::Input<'_>,
        frame: Option<&std::collections::BTreeMap<String, Vec<u8>>>,
    ) {
        let key = crate::clouds::pack_faces_key(input);
        if self.key.as_ref() != Some(&key) {
            self.key = Some(key);
            let faces = crate::clouds::pack_face_list(input);
            let needed = (faces.len() as u64).max(3);
            if needed > self.capacity {
                self.capacity = needed.next_power_of_two();
                self.faces = faces_buffer(device, self.capacity);
                self.group = Self::bind(
                    device,
                    &self.layout,
                    &self.uniform,
                    &self.faces,
                    self.faces_binding,
                    self.info_binding,
                    &self.extra,
                );
            }
            if !faces.is_empty() {
                uploads.buffer(encoder, &self.faces, 0, bytemuck::cast_slice(&faces));
            }
            self.last = faces;
        }
        blocks.resize(INFO_OFFSET as usize, 0);
        blocks.extend_from_slice(&crate::clouds::pack_info(input));
        self.extra.write(&mut blocks, frame);
        uploads.buffer(encoder, &self.uniform, 0, &blocks);
    }

    /// The faces written last, `quads` at a time from the renderer's quad indices. The caller
    /// sets its own pipeline and group 0 again after.
    pub(super) fn draw(&self, pass: &mut wgpu::RenderPass<'_>, fancy: bool, index_quads: u32) {
        let quads = self.quads();
        if quads == 0 {
            return;
        }
        pass.set_pipeline(if fancy { &self.fancy } else { &self.flat });
        pass.set_bind_group(0, &self.group, &[]);
        let mut first = 0;
        while first < quads {
            let count = (quads - first).min(index_quads);
            pass.draw_indexed(0..count * 6, (first * 4) as i32, 0..1);
            first += count;
        }
    }

    pub(super) fn quads(&self) -> u32 {
        (self.last.len() / 3) as u32
    }
}

fn faces_buffer(device: &wgpu::Device, ints: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Resource pack cloud faces"),
        size: ints * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// The clouds on the renderer: a pack's `core/clouds`, or the native mesh.
impl super::Renderer {
    /// The resource packs' `core/clouds` (`CoreShaders::clouds`), or the native clouds.
    pub fn set_clouds_shader(&mut self, program: Option<&crate::core_shader::CoreProgram>) {
        self.core_clouds =
            program.and_then(|p| CoreClouds::new(&self.device, self.output.frame.format, p));
    }

    /// Draws the pack's clouds into the world pass, leaving its index buffer and group 0 as the
    /// native draws expect them.
    pub(super) fn draw_core_clouds(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        frame: &super::Frame<'_>,
    ) {
        let (Some(core), Some(input)) = (&self.core_clouds, frame.clouds.as_ref()) else {
            return;
        };
        pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
        core.draw(pass, !input.fast, self.index_quads);
        pass.set_bind_group(0, &self.group, &[]);
    }
}
