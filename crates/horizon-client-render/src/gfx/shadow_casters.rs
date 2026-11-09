//! `shadow_casters`, moved out of `gfx.rs` unchanged.
use super::*;

/// The camera's view `view_proj` (wgpu depth, reversed or not) stretched without end toward
/// `light` (a direction): the boxes whose shadow, cast away from the light, can fall into the
/// view. Its planes are the view's faces turned toward the light, and through each edge between
/// such a face and one turned away, the plane along the light; unused ones take in everything.
pub(super) fn shadow_casters(view_proj: Mat4, light: Vec3) -> [Frustum; 2] {
    let faces = frustum(view_proj);
    let toward = faces.map(|p| p.truncate().dot(light) >= 0.0);
    let inverse = view_proj.inverse();
    // Corner (x, y, z) of clip space's box; `frustum`'s faces at x -1 and 1 are 0 and 1, at
    // y -1 and 1 are 2 and 3, at depth 0 and 1 are 4 and 5.
    let corner = |c: [usize; 3]| {
        inverse.project_point3(Vec3::new(
            c[0] as f32 * 2.0 - 1.0,
            c[1] as f32 * 2.0 - 1.0,
            c[2] as f32,
        ))
    };
    let corners: Vec<[usize; 3]> = (0..8).map(|i| [i & 1, (i >> 1) & 1, i >> 2]).collect();
    let centre = corners.iter().map(|&c| corner(c)).sum::<Vec3>() / 8.0;
    let mut planes = [Vec4::W; 12];
    let mut n = 0;
    for (k, &face) in faces.iter().enumerate() {
        if toward[k] {
            planes[n] = face;
            n += 1;
        }
    }
    for a in &corners {
        for axis in 0..3 {
            if a[axis] == 1 {
                continue;
            }
            let mut b = *a;
            b[axis] = 1;
            // The edge from a to b lies on the faces of the two other axes.
            let [f, g] = [(axis + 1) % 3, (axis + 2) % 3].map(|o| o * 2 + a[o]);
            if toward[f] == toward[g] {
                continue;
            }
            let (p, q) = (corner(*a), corner(b));
            let normal = (q - p).cross(light);
            if normal.length_squared() < 1e-12 {
                continue;
            }
            let normal = normal.normalize();
            let side = if normal.dot(centre - p) < 0.0 {
                -1.0
            } else {
                1.0
            };
            // At most six such edges with at most four faces toward the light: 12 planes.
            if let Some(plane) = planes.get_mut(n) {
                *plane = (normal * side).extend(-normal.dot(p) * side);
                n += 1;
            }
        }
    }
    let [a, b] = [0, 6].map(|i| Frustum {
        planes: planes[i..i + 6].try_into().expect("six planes"),
    });
    [a, b]
}
