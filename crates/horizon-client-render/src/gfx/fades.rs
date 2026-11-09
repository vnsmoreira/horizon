//! `mark_fades`, moved out of `gfx.rs` unchanged.
use super::*;

/// GPU culling's fading sections brought up to `f.now`, before a frame's gate
/// and `prepare`: every recorded section anew when the flags were decided with
/// another chunk fade-in time (GPU culling just switched on, or Video Settings
/// changed it), else those whose fade was due to end, handed back to the GPU if
/// whole by then. A section's record (`FADING`) and its walk mark follow its
/// `gpu_fading`. `rows`: the level's sections.
pub(super) fn mark_fades(
    sections: &mut resident::Resident<SectionGpu>,
    records: &mut gpu_culling::Records,
    fades: &mut gpu_culling::Fades,
    rows: usize,
    f: FadeInputs<'_>,
) {
    let mut marks = Vec::new();
    if fades.fade_ms != Some(f.fade_ms) {
        fades.clear();
        fades.fade_ms = Some(f.fade_ms);
        for (key, s, _) in sections.iter_mut() {
            let Some(id) = s.gpu_id else {
                continue;
            };
            let fading = decide_fade(s, key, fades, f);
            if fading != s.gpu_fading {
                s.gpu_fading = fading;
                records.set_fading(id, fading);
                marks.push((key, s.listed(key, rows)));
            }
        }
    } else {
        while let Some(key) = fades.pop_due(f.now) {
            // Gone since, or no longer flagged.
            let Some(s) = sections.find_mut(key).filter(|s| s.gpu_fading) else {
                continue;
            };
            // Still fading: meshed again since, with its own end waited for
            // (`decide_fade`), or leaving or held, with none.
            if s.drawn_faded(key, f) {
                continue;
            }
            s.gpu_fading = false;
            records.set_fading(s.gpu_id.expect("flagged sections have records"), false);
            marks.push((key, s.listed(key, rows)));
        }
    }
    for (key, marked) in marks {
        sections.set_marked(key, marked);
    }
}
