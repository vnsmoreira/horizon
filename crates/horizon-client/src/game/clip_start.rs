//! `clip_start_source`, moved out of `game.rs` unchanged.

/// `VoxelShape.clip` of a fluid source's shape for a ray that starts in the
/// source's own block `cell`: the face and point it hits, in `pick_fluid`'s
/// `Hit` terms. `FlowingFluid.getShape` is the whole block under the same
/// fluid and `getOwnHeight` (8/9) tall otherwise. A start inside the shape
/// (tested 1e-7 of the way along) hits at the start, on the face opposite
/// `Direction.getApproximateNearest(dir)`; from above it, a ray going down
/// enters through the top while still in this block.
pub(super) fn clip_start_source(
    eye: [f64; 3],
    dir: [f64; 3],
    reach: f64,
    cell: [i32; 3],
    same_above: bool,
) -> Option<([i32; 3], [f64; 3])> {
    let height = if same_above {
        1.0
    } else {
        f64::from(8.0f32 / 9.0f32)
    };
    let probe_y = eye[1] + dir[1] * reach * 1.0e-7;
    if probe_y - f64::from(cell[1]) < height {
        // Direction.values() order; the first of equal dot products wins.
        let normals: [[i32; 3]; 6] = [
            [0, -1, 0],
            [0, 1, 0],
            [0, 0, -1],
            [0, 0, 1],
            [-1, 0, 0],
            [1, 0, 0],
        ];
        let mut nearest = normals[2];
        let mut best = f64::NEG_INFINITY;
        for n in normals {
            let dot =
                f64::from(n[0]) * dir[0] + f64::from(n[1]) * dir[1] + f64::from(n[2]) * dir[2];
            if dot > best {
                best = dot;
                nearest = n;
            }
        }
        return Some((nearest.map(|v| -v), eye));
    }
    if dir[1] >= 0.0 {
        return None;
    }
    let t = (f64::from(cell[1]) + height - eye[1]) / dir[1];
    if t > reach {
        return None;
    }
    let point = [
        eye[0] + dir[0] * t,
        eye[1] + dir[1] * t,
        eye[2] + dir[2] * t,
    ];
    (point[0].floor() as i32 == cell[0] && point[2].floor() as i32 == cell[2])
        .then_some(([0, 1, 0], point))
}
