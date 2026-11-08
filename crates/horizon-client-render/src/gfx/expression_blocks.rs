//! A pack core shader's Polytone blocks beyond the pipeline's own: its expression uniforms
//! (`shader_modifiers`, wiki "Expression Uniforms on Any Shader"), one float each, and
//! `PolyGlobals` (wiki "Extra Shader Uniforms"), at their [`CoreProgram::other`] bindings, written
//! each frame from the frame's blocks by name.

use crate::core_shader::CoreProgram;
use std::collections::BTreeMap;

/// The block of Polytone's globals, which a pack's core shader may declare.
pub(crate) const POLY_GLOBALS: &str = "PolyGlobals";

/// The blocks of one pipeline, laid out in its uniform buffer from `base`, 256 bytes apart.
#[derive(Debug, Default)]
pub(crate) struct ExpressionBlocks {
    /// Each block's binding and name.
    blocks: Vec<(u32, String)>,
    base: u64,
}

impl ExpressionBlocks {
    /// The blocks of `program` that are none of `own` (the pipeline's other blocks).
    pub(crate) fn new(program: &CoreProgram, own: &[&str], base: u64) -> Self {
        let blocks = program
            .other
            .iter()
            .enumerate()
            .filter(|(_, name)| !own.contains(&name.as_str()))
            .map(|(i, name)| (crate::core_shader::other_binding(i), name.clone()))
            .collect();
        Self { blocks, base }
    }

    /// A block's size: `PolyGlobals` its own, an expression uniform one float.
    fn block_size(name: &str) -> u64 {
        if name == POLY_GLOBALS { 176 } else { 16 }
    }

    fn offset(&self, i: usize) -> u64 {
        self.base + 256 * i as u64
    }

    /// The uniform buffer's size with these blocks after `base`.
    pub(crate) fn size(&self) -> u64 {
        self.offset(self.blocks.len())
    }

    pub(crate) fn layout(&self) -> impl Iterator<Item = wgpu::BindGroupLayoutEntry> + '_ {
        self.blocks
            .iter()
            .map(|(binding, name)| wgpu::BindGroupLayoutEntry {
                binding: *binding,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(Self::block_size(name)),
                },
                count: None,
            })
    }

    pub(crate) fn entries<'a>(
        &'a self,
        buffer: &'a wgpu::Buffer,
    ) -> impl Iterator<Item = wgpu::BindGroupEntry<'a>> + 'a {
        self.blocks
            .iter()
            .enumerate()
            .map(move |(i, (binding, name))| wgpu::BindGroupEntry {
                binding: *binding,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer,
                    offset: self.offset(i),
                    size: wgpu::BufferSize::new(Self::block_size(name)),
                }),
            })
    }

    /// `data` (the pipeline's own blocks) grown to the buffer's size, with each block's bytes
    /// from `frame` by its name (0 where the frame has none).
    pub(crate) fn write(&self, data: &mut Vec<u8>, frame: Option<&BTreeMap<String, Vec<u8>>>) {
        if self.blocks.is_empty() {
            return;
        }
        data.resize(self.size() as usize, 0);
        for (i, (_, name)) in self.blocks.iter().enumerate() {
            let at = self.offset(i) as usize;
            let size = Self::block_size(name) as usize;
            data[at..at + size].fill(0);
            if let Some(bytes) = frame.and_then(|f| f.get(name)) {
                let n = bytes.len().min(size);
                data[at..at + n].copy_from_slice(&bytes[..n]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_beyond_the_pipelines_own_are_written_by_name() {
        let program = CoreProgram {
            vertex: naga::Module::default(),
            fragment: naga::Module::default(),
            other: vec!["LightmapInfo".into(), "Tint".into(), "Fade".into()],
        };
        let blocks = ExpressionBlocks::new(&program, &["LightmapInfo"], 1024);
        // Bindings after the pipeline's own block's.
        let bindings: Vec<u32> = blocks.layout().map(|e| e.binding).collect();
        assert_eq!(
            bindings,
            [
                crate::core_shader::other_binding(1),
                crate::core_shader::other_binding(2)
            ]
        );
        assert_eq!(blocks.size(), 1024 + 512);
        let mut frame = BTreeMap::new();
        frame.insert("Fade".to_owned(), 0.5f32.to_le_bytes().to_vec());
        let mut data = vec![7u8; 100];
        blocks.write(&mut data, Some(&frame));
        assert_eq!(data.len(), 1536);
        assert_eq!(&data[..100], &[7u8; 100]);
        // Tint: none this frame; Fade: its value.
        assert_eq!(&data[1024..1040], &[0u8; 16]);
        assert_eq!(
            f32::from_le_bytes(data[1280..1284].try_into().unwrap()),
            0.5
        );
    }
}
