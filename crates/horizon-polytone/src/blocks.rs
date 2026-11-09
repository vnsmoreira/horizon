//! What a Polytone block modifier (`assets/<namespace>/polytone/block_modifiers/*.json`) changes in
//! how a block is drawn, beyond its colormap and emitters: its render offset (`offset_type`).

use serde_json::Value as Json;

/// `offset_type`: the random model offset a block gets (vanilla's `BlockBehaviour.OffsetType`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockOffset {
    /// No offset, even on a block vanilla offsets.
    None,
    /// Horizontal, as flowers.
    Xz,
    /// Horizontal and down, as short grass.
    Xyz,
}

impl BlockOffset {
    /// `"none"`, `"xz"` or `"xyz"`. The wiki's custom form (an object of maximum offsets per axis)
    /// is not read: it doesn't say what each field defaults to.
    pub fn parse(json: &Json) -> Result<Self, String> {
        match json.as_str().map(str::to_ascii_lowercase).as_deref() {
            Some("none") => Ok(Self::None),
            Some("xz") => Ok(Self::Xz),
            Some("xyz") => Ok(Self::Xyz),
            Some(other) => Err(format!("unknown offset_type {other:?}")),
            None => Err("a custom offset_type is not read yet".into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn offset_types_read_their_vanilla_names() {
        assert_eq!(BlockOffset::parse(&json!("xz")), Ok(BlockOffset::Xz));
        assert_eq!(BlockOffset::parse(&json!("XYZ")), Ok(BlockOffset::Xyz));
        assert_eq!(BlockOffset::parse(&json!("none")), Ok(BlockOffset::None));
        assert!(BlockOffset::parse(&json!("diagonal")).is_err());
        assert!(BlockOffset::parse(&json!({"max_y": 0.5})).is_err());
    }
}
