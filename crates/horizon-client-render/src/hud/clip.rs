//! `clip_polygon`, moved out of `hud.rs` unchanged.
use super::*;

/// Sutherland-Hodgman against one half plane, interpolating every attribute.
pub(super) fn clip_polygon(
    poly: &[HudVertex],
    axis: usize,
    limit: f32,
    keep_below: bool,
) -> Vec<HudVertex> {
    let inside = |v: &HudVertex| {
        if keep_below {
            v.pos[axis] < limit
        } else {
            v.pos[axis] >= limit
        }
    };
    let lerp = |a: &HudVertex, b: &HudVertex, t: f32| {
        let mix = |p: f32, q: f32| p + (q - p) * t;
        let mut v = *a;
        v.pos = [mix(a.pos[0], b.pos[0]), mix(a.pos[1], b.pos[1])];
        v.uv = [mix(a.uv[0], b.uv[0]), mix(a.uv[1], b.uv[1])];
        v.depth = mix(a.depth, b.depth);
        // A shaded shape (a gradient, `hud_kit::shapes`) keeps its colours across the cut.
        v.color = crate::hud_kit::shapes::mix_colour(a.color, b.color, t);
        v.glint_uv = [
            mix(a.glint_uv[0], b.glint_uv[0]),
            mix(a.glint_uv[1], b.glint_uv[1]),
        ];
        v.pos[axis] = limit;
        v
    };
    let mut out = Vec::with_capacity(poly.len() + 2);
    for i in 0..poly.len() {
        let (a, b) = (&poly[i], &poly[(i + 1) % poly.len()]);
        let (ia, ib) = (inside(a), inside(b));
        if ia {
            out.push(*a);
        }
        if ia != ib {
            let t = (limit - a.pos[axis]) / (b.pos[axis] - a.pos[axis]);
            out.push(lerp(a, b, t));
        }
    }
    out
}
