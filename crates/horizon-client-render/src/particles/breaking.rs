//! `breaking_position`, moved out of `particles.rs` unchanged.
use super::*;

/// ClientLevel.addBreakingParticles placement, using local outline boxes.
pub fn breaking_position(
    random: &mut impl RandomSource,
    pos: [i32; 3],
    face: [i32; 3],
    boxes: &[[f64; 6]],
) -> Option<[f64; 3]> {
    let mut bounds = *boxes.first()?;
    for b in &boxes[1..] {
        for i in 0..3 {
            bounds[i] = bounds[i].min(b[i]);
            bounds[i + 3] = bounds[i + 3].max(b[i + 3]);
        }
    }
    // Vanilla consumes all three doubles before replacing the hit-face coordinate.
    let mut p = array::from_fn(|i| {
        f64::from(pos[i])
            + random.next_double() * (bounds[i + 3] - bounds[i] - f64::from(0.2f32))
            + f64::from(0.1f32)
            + bounds[i]
    });
    for i in 0..3 {
        match face[i] {
            -1 => p[i] = f64::from(pos[i]) + bounds[i] - f64::from(0.1f32),
            1 => p[i] = f64::from(pos[i]) + bounds[i + 3] + f64::from(0.1f32),
            _ => {}
        }
    }
    Some(p)
}
