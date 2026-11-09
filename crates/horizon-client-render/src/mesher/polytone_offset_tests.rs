use super::*;

/// A Polytone block modifier's `offset_type` gives a block the offset vanilla gives blocks of that
/// `OffsetType`, with the block's own maximum offsets; `none` takes a vanilla offset away.
#[test]
fn polytone_offset_types_move_blocks_as_vanillas_offset_blocks() {
    let root =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.local/horizon-client/26.3");
    if !root.join("horizon-resources.json").exists() {
        return;
    }
    let assets = Assets::load(&root).unwrap();
    let render = assets.reload(&root).unwrap();
    let facts = BlockFacts::load(&root, assets.registry).unwrap();
    let visibility = crate::models::load_visibility(&root, assets.registry).unwrap();
    let vanilla = MeshTables::new(&assets, &render, &facts, visibility.clone()).unwrap();
    let pack =
        std::env::temp_dir().join(format!("horizon-polytone-offsets-{}", std::process::id()));
    let modifiers = pack.join("assets/minecraft/polytone/block_modifiers");
    std::fs::create_dir_all(&modifiers).unwrap();
    for (block, offset) in [
        ("dandelion", "none"),
        ("stone", "xz"),
        ("short_grass", "xz"),
        ("pointed_dripstone", "xyz"),
    ] {
        std::fs::write(
            modifiers.join(format!("{block}.json")),
            format!(r#"{{"offset_type": "{offset}"}}"#),
        )
        .unwrap();
    }
    let packed = MeshTables::new(&assets, &render, &facts, visibility)
        .unwrap()
        .with_polytone_block_offsets(&pack, &facts);
    let _ = std::fs::remove_dir_all(&pack);
    let state = |name| assets.registry.default_state(name).unwrap();
    let (x, z) = (13, -7);
    let before = |name| vanilla.offset(state(name), x, z);
    let after = |name| packed.offset(state(name), x, z);

    assert!(before("minecraft:dandelion").is_some());
    assert_eq!(after("minecraft:dandelion"), None);
    // Stone now moves as a flower does.
    assert_eq!(before("minecraft:stone"), None);
    assert_eq!(after("minecraft:stone"), before("minecraft:dandelion"));
    // Short grass (vanilla XYZ) keeps its horizontal offset and loses its vertical one.
    let [gx, gy, gz] = before("minecraft:short_grass").unwrap();
    assert!(gy < 0.0);
    assert_eq!(after("minecraft:short_grass"), Some([gx, 0.0, gz]));
    // Pointed dripstone (vanilla XZ, 1/8 block at most) also sinks as short grass does.
    let [dx, _, dz] = before("minecraft:pointed_dripstone").unwrap();
    assert_eq!(after("minecraft:pointed_dripstone"), Some([dx, gy, dz]));
}
