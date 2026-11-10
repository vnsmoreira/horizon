//! GLSL that vanilla's GL compiler takes and naga's front end doesn't (#9386), rewritten into
//! what both take:
//! - a function with `sampler2D` parameters is copied once per set of samplers its callers pass
//!   (all of them a shader's global samplers), the copy reading those samplers by name;
//! - an unsized array (`const float[] w = float[](...)`, `float w[] = ...`) takes the size its
//!   initializer gives.

/// `text` with both rewrites, where they apply; anything they can't follow is left as it was.
pub(super) fn fixups(text: &str) -> String {
    size_unsized_arrays(&specialize_sampler_params(text))
}

fn is_ident(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// The identifier that ends right before byte `end` (whitespace skipped), and where it starts.
fn ident_before(text: &str, end: usize) -> Option<(usize, &str)> {
    let trimmed = text[..end].trim_end();
    let start = trimmed
        .char_indices()
        .rev()
        .take_while(|(_, c)| is_ident(*c))
        .last()?
        .0;
    let ident = &trimmed[start..];
    (!ident.is_empty() && !ident.starts_with(|c: char| c.is_ascii_digit()))
        .then_some((start, ident))
}

/// The byte of the bracket that closes the one opening at `open`.
fn closing(text: &str, open: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let (o, c) = match bytes[open] {
        b'(' => (b'(', b')'),
        b'{' => (b'{', b'}'),
        b'[' => (b'[', b']'),
        _ => return None,
    };
    let mut depth = 0usize;
    for (i, &b) in bytes.iter().enumerate().skip(open) {
        if b == o {
            depth += 1;
        } else if b == c {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
    }
    None
}

/// The comma-separated parts of `list` at its own nesting level.
fn split_top(list: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let (mut depth, mut start) = (0i32, 0);
    for (i, c) in list.char_indices() {
        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(&list[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    if !list[start..].trim().is_empty() || !parts.is_empty() {
        parts.push(&list[start..]);
    }
    parts
}

/// A function at the file's top level whose parameters include a `sampler2D`.
struct SamplerFn {
    name: String,
    /// The header from its return type to its `{` (or `;` for a prototype), and the body after.
    start: usize,
    end: usize,
    params: Vec<String>,
    /// The indices of the `sampler2D` parameters and their names.
    samplers: Vec<(usize, String)>,
    /// `None` for a prototype.
    body: Option<String>,
    return_type: String,
}

fn sampler_functions(text: &str) -> Vec<SamplerFn> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'{' => depth += 1,
            b'}' => depth -= 1,
            b'(' if depth == 0 => {
                if let (Some((name_at, name)), Some(close)) =
                    (ident_before(text, i), closing(text, i))
                    && let Some((type_at, return_type)) = ident_before(text, name_at)
                {
                    let params: Vec<String> = split_top(&text[i + 1..close])
                        .into_iter()
                        .map(|p| p.trim().to_owned())
                        .collect();
                    let samplers: Vec<(usize, String)> = params
                        .iter()
                        .enumerate()
                        .filter(|(_, p)| p.split_whitespace().any(|w| w == "sampler2D"))
                        .filter_map(|(n, p)| {
                            Some((n, p.split_whitespace().last()?.trim_matches(';').to_owned()))
                        })
                        .collect();
                    let after = text[close + 1..].trim_start();
                    let next = close + 1 + (text[close + 1..].len() - after.len());
                    if !samplers.is_empty() && (after.starts_with('{') || after.starts_with(';')) {
                        let (end, body) = if after.starts_with('{') {
                            let Some(body_end) = closing(text, next) else {
                                break;
                            };
                            (body_end + 1, Some(text[next..=body_end].to_owned()))
                        } else {
                            (next + 1, None)
                        };
                        out.push(SamplerFn {
                            name: name.to_owned(),
                            start: type_at,
                            end,
                            params,
                            samplers,
                            body,
                            return_type: return_type.to_owned(),
                        });
                        i = end;
                        continue;
                    }
                }
            }
            _ => {}
        }
        i += 1;
    }
    out
}

/// The calls to `name` outside `skip` (its own header): each call's span and arguments.
fn calls<'a>(
    text: &'a str,
    name: &str,
    skip: &[(usize, usize)],
) -> Vec<(usize, usize, Vec<&'a str>)> {
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(found) = text[from..].find(name) {
        let at = from + found;
        from = at + name.len();
        let before = text[..at].chars().next_back();
        let rest = &text[at + name.len()..];
        let open = at + name.len() + (rest.len() - rest.trim_start().len());
        if before.is_some_and(is_ident)
            || !text[open..].starts_with('(')
            || skip.iter().any(|(s, e)| (*s..*e).contains(&at))
        {
            continue;
        }
        let Some(close) = closing(text, open) else {
            continue;
        };
        out.push((at, close + 1, split_top(&text[open + 1..close])));
    }
    out
}

fn replace_ident(text: &str, from: &str, to: &str) -> String {
    super::replace_ident(text, from, to)
}

/// The byte after `uniform sampler2D <name>;`, where the shader declares that sampler.
fn sampler_declaration_end(text: &str, name: &str) -> Option<usize> {
    let mut from = 0;
    while let Some(found) = text[from..].find("sampler2D") {
        let at = from + found;
        from = at + "sampler2D".len();
        let before = text[..at].trim_end();
        if !before.ends_with("uniform")
            || before[..before.len() - 7]
                .chars()
                .next_back()
                .is_some_and(is_ident)
        {
            continue;
        }
        let rest = &text[from..];
        let ident = rest.trim_start();
        let len = ident.chars().take_while(|c| is_ident(*c)).count();
        if &ident[..len] != name {
            continue;
        }
        let after = ident[len..].trim_start();
        // A declaration at the shader's top level, not one inside a function's braces.
        let depth = text[..at].matches('{').count() as i64 - text[..at].matches('}').count() as i64;
        if after.starts_with(';') && depth == 0 {
            return Some(text.len() - after.len() + 1);
        }
    }
    None
}

/// The samplers a call passes for `f`'s `sampler2D` parameters, in order.
fn chosen_samplers(f: &SamplerFn, args: &[&str]) -> Vec<String> {
    f.samplers
        .iter()
        .filter_map(|(n, _)| args.get(*n).map(|a| a.trim().to_owned()))
        .collect()
}

/// One pass per function with `sampler2D` parameters, until none is left or one can't be
/// followed (a caller passes something that is no sampler's name).
fn specialize_sampler_params(text: &str) -> String {
    let mut text = text.to_owned();
    for _ in 0..16 {
        let functions = sampler_functions(&text);
        // From the outside in: a function none of whose calls sits in another one still to be
        // copied, so each call passes a sampler's own name.
        let inside_another = |f: &SamplerFn, at: usize| {
            functions
                .iter()
                .any(|g| g.name != f.name && g.body.is_some() && (g.start..g.end).contains(&at))
        };
        let spans_of = |f: &SamplerFn| -> Vec<(usize, usize)> {
            functions
                .iter()
                .filter(|g| g.name == f.name)
                .map(|g| (g.start, g.end))
                .collect()
        };
        let Some(f) = functions.iter().find(|f| {
            f.body.is_some()
                && calls(&text, &f.name, &spans_of(f))
                    .iter()
                    .all(|(at, ..)| !inside_another(f, *at))
        }) else {
            break;
        };
        let spans = spans_of(f);
        let calls = calls(&text, &f.name, &spans);
        // Each call's samplers must be plain names (a global sampler, or a parameter of a
        // function to be copied in a later pass).
        let mut variants: Vec<Vec<String>> = Vec::new();
        let mut call_starts: Vec<usize> = Vec::new();
        let mut followable = true;
        for (at, _, args) in &calls {
            let chosen = chosen_samplers(f, args);
            if chosen.len() != f.samplers.len()
                || chosen
                    .iter()
                    .any(|a| a.is_empty() || !a.chars().all(is_ident))
            {
                followable = false;
                break;
            }
            call_starts.push(*at);
            if !variants.contains(&chosen) {
                variants.push(chosen);
            }
        }
        // Overloads (`tap(sampler2D, vec2)` and `tap(sampler2D, vec3)`) each keep their own
        // body; a call picks its samplers by position, so they must sit at the same places.
        let same_places = functions.iter().filter(|g| g.name == f.name).all(|g| {
            g.samplers
                .iter()
                .map(|s| s.0)
                .eq(f.samplers.iter().map(|s| s.0))
        });
        if !followable || !same_places {
            break;
        }
        // A body may call another overload of itself with its own sampler (vanilla's
        // `sampleNearest` calls the six-argument one): once its sampler parameters read a
        // sampler, those calls name that copy too, and their sets join the variants.
        let rewrite_own_calls = |body: &str| -> Option<(String, Vec<Vec<String>>)> {
            let mut body = body.to_owned();
            let mut found = Vec::new();
            let mut starts: Vec<usize> = self::calls(&body, &f.name, &[])
                .iter()
                .map(|c| c.0)
                .collect();
            starts.sort_unstable_by(|a, b| b.cmp(a));
            for start in starts {
                let (end, chosen, with) = {
                    let (_, end, args) = self::calls(&body, &f.name, &[])
                        .into_iter()
                        .find(|c| c.0 == start)?;
                    let chosen = chosen_samplers(f, &args);
                    let rest: Vec<&str> = args
                        .iter()
                        .enumerate()
                        .filter(|(n, _)| !f.samplers.iter().any(|(s, _)| s == n))
                        .map(|(_, a)| a.trim())
                        .collect();
                    let with = format!("{}_hz_{}({})", f.name, chosen.join("_"), rest.join(", "));
                    (end, chosen, with)
                };
                if chosen.len() != f.samplers.len()
                    || chosen
                        .iter()
                        .any(|a| a.is_empty() || !a.chars().all(is_ident))
                {
                    return None;
                }
                found.push(chosen);
                body.replace_range(start..end, &with);
            }
            Some((body, found))
        };
        let specialize = |g: &SamplerFn, chosen: &[String]| {
            let mut body = g.body.as_deref().unwrap_or_default().to_owned();
            for ((_, param), sampler) in g.samplers.iter().zip(chosen) {
                body = replace_ident(&body, param, sampler);
            }
            rewrite_own_calls(&body)
        };
        let same: Vec<&SamplerFn> = functions.iter().filter(|g| g.name == f.name).collect();
        let mut next = 0;
        while followable && next < variants.len() {
            let chosen = variants[next].clone();
            for g in same.iter().filter(|g| g.body.is_some()) {
                match specialize(g, &chosen) {
                    Some((_, inner)) => {
                        for v in inner {
                            if !variants.contains(&v) {
                                variants.push(v);
                            }
                        }
                    }
                    None => followable = false,
                }
            }
            next += 1;
            followable &= variants.len() <= 64;
        }
        if !followable {
            break;
        }
        // One copy of an overload per set of samplers its callers pass; the copies keep the
        // overload's other parameters, so the shader's compiler still tells them apart. A
        // prototype becomes the copies' prototypes, for a call above the definition.
        let signature = |g: &SamplerFn, chosen: &[String]| {
            let params: Vec<&str> = g
                .params
                .iter()
                .enumerate()
                .filter(|(n, _)| !g.samplers.iter().any(|(s, _)| s == n))
                .map(|(_, p)| p.as_str())
                .collect();
            format!(
                "{} {}_hz_{}({})",
                g.return_type,
                g.name,
                chosen.join("_"),
                params.join(", ")
            )
        };
        let copies_of = |g: &SamplerFn| -> String {
            variants
                .iter()
                .map(|chosen| match &g.body {
                    Some(_) => {
                        let body = specialize(g, chosen).map(|(b, _)| b).unwrap_or_default();
                        format!("{} {body}\n", signature(g, chosen))
                    }
                    None => format!("{};\n", signature(g, chosen)),
                })
                .collect()
        };
        // Rebuild from the back: the calls, then the function's own spans (each prototype and
        // definition becomes its copies). A call nested in another's arguments starts after it,
        // so it is rewritten first, and the call around it is read again from the text as it is
        // then. A copy reads its samplers by name, so definitions above their samplers'
        // declaration (vanilla's `sample_lightmap.glsl` is included above `uniform sampler2D
        // Sampler2;`) move after it, together and in their order, as long as no call comes first.
        let first_call = call_starts.iter().copied().min().unwrap_or(usize::MAX);
        let declared = variants
            .iter()
            .flatten()
            .filter_map(|s| sampler_declaration_end(&text, s))
            .max();
        let mut edits: Vec<(usize, Option<(usize, String)>)> =
            call_starts.into_iter().map(|at| (at, None)).collect();
        let (mut moved, mut move_to) = (String::new(), None);
        for g in &same {
            match (
                &g.body,
                declared.filter(|&end| end > g.start && end <= first_call),
            ) {
                (Some(_), Some(end)) => {
                    edits.push((g.start, Some((g.end, String::new()))));
                    moved.push_str(&copies_of(g));
                    move_to = Some(end);
                }
                _ => edits.push((g.start, Some((g.end, copies_of(g))))),
            }
        }
        if let Some(end) = move_to {
            edits.push((end, Some((end, format!("\n{moved}")))));
        }
        // Pack text is untrusted: edits that overlap, or offsets off the text, leave the shader
        // as it is (the translator then refuses it) instead of writing it wrong or panicking.
        let mut spans: Vec<(usize, usize)> = edits
            .iter()
            .filter_map(|(s, e)| e.as_ref().map(|(end, _)| (*s, *end)))
            .collect();
        spans.sort_unstable();
        if spans.windows(2).any(|w| w[1].0 < w[0].1) {
            break;
        }
        let before = text.clone();
        edits.sort_by_key(|(start, _)| std::cmp::Reverse(*start));
        let mut sound = true;
        for (start, edit) in edits {
            let (end, with) = match edit {
                Some(span) => span,
                None => {
                    let found = self::calls(&text, &f.name, &[])
                        .into_iter()
                        .find(|c| c.0 == start);
                    let Some((_, end, args)) = found else {
                        continue;
                    };
                    let rest: Vec<&str> = args
                        .iter()
                        .enumerate()
                        .filter(|(n, _)| !f.samplers.iter().any(|(s, _)| s == n))
                        .map(|(_, a)| a.trim())
                        .collect();
                    let copy = format!("{}_hz_{}", f.name, chosen_samplers(f, &args).join("_"));
                    (end, format!("{copy}({})", rest.join(", ")))
                }
            };
            if start > end || text.get(start..end).is_none() {
                sound = false;
                break;
            }
            text.replace_range(start..end, &with);
        }
        if !sound {
            text = before;
            break;
        }
    }
    text
}

/// Each array declared without a size that an initializer sizes: `T[] name = T[](a, b)` and
/// `T name[] = T[](a, b)` (or `T[2](...)`) are written with the size.
fn size_unsized_arrays(text: &str) -> String {
    let mut text = text.to_owned();
    let mut from = 0;
    while let Some(found) = text[from..].find('=') {
        let eq = from + found;
        from = eq + 1;
        // `==`, `<=`, `>=`, `!=`, `+=` ... are no initializers.
        let prev = text[..eq].chars().next_back();
        let next = text[eq + 1..].chars().next();
        if matches!(
            prev,
            Some('=' | '<' | '>' | '!' | '+' | '-' | '*' | '/' | '%' | '&' | '|' | '^')
        ) || next == Some('=')
        {
            continue;
        }
        // The constructor: `T [ n? ] (`.
        let rest = &text[eq + 1..];
        let ty_start = eq + 1 + (rest.len() - rest.trim_start().len());
        let ty_len = text[ty_start..]
            .chars()
            .take_while(|c| is_ident(*c))
            .count();
        if ty_len == 0 {
            continue;
        }
        let after_ty = &text[ty_start + ty_len..];
        let bracket = ty_start + ty_len + (after_ty.len() - after_ty.trim_start().len());
        if !text[bracket..].starts_with('[') {
            continue;
        }
        let Some(bracket_end) = closing(&text, bracket) else {
            continue;
        };
        let after_bracket = &text[bracket_end + 1..];
        let paren = bracket_end + 1 + (after_bracket.len() - after_bracket.trim_start().len());
        if !text[paren..].starts_with('(') {
            continue;
        }
        let Some(paren_end) = closing(&text, paren) else {
            continue;
        };
        let count = split_top(&text[paren + 1..paren_end]).len();
        // The declarator: `name []` or `T [] name`, between the statement's start and `=`.
        let statement_start = text[..eq].rfind([';', '{', '}', '\n']).map_or(0, |i| i + 1);
        let declarator = &text[statement_start..eq];
        let Some(empty) = declarator.rfind('[') else {
            continue;
        };
        let inside = &declarator[empty + 1..];
        let Some(close) = inside.find(']') else {
            continue;
        };
        if !inside[..close].trim().is_empty() {
            continue;
        }
        let size = count.to_string();
        // Sized in the constructor too, where it isn't.
        if text[bracket + 1..bracket_end].trim().is_empty() {
            text.replace_range(bracket + 1..bracket_end, &size);
        }
        let at = statement_start + empty + 1;
        text.replace_range(at..at + close, &size);
        from = eq + 1;
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sampler_parameter_becomes_a_copy_per_sampler_passed() {
        let text = "vec4 blur(sampler2D tex, vec2 uv);\n\
            vec4 blur(sampler2D tex, vec2 uv) { return texture(tex, uv) + texture(tex, uv * 2.0); }\n\
            void main() { vec4 a = blur(Sampler0, uv); vec4 b = blur( Sampler1 , uv + 1.0); vec4 c = blur(Sampler0, uv); }\n";
        let out = specialize_sampler_params(text);
        assert!(!out.contains("sampler2D"), "{out}");
        assert!(out.contains(
            "vec4 blur_hz_Sampler0(vec2 uv) { return texture(Sampler0, uv) + texture(Sampler0, uv * 2.0); }"
        ));
        assert!(out.contains("vec4 blur_hz_Sampler1(vec2 uv)"));
        assert!(out.contains("blur_hz_Sampler0(uv)"));
        assert!(out.contains("blur_hz_Sampler1(uv + 1.0)"));
        // One copy per sampler, not per call.
        assert_eq!(out.matches("vec4 blur_hz_Sampler0(").count(), 1);
    }

    #[test]
    fn a_sampler_passed_on_through_another_function_is_followed() {
        let text = "vec4 inner(sampler2D t, vec2 uv) { return texture(t, uv); }\n\
            vec4 outer(vec2 uv, sampler2D s) { return inner(s, uv); }\n\
            void main() { vec4 a = outer(uv, Sampler2); }\n";
        let out = specialize_sampler_params(text);
        assert!(!out.contains("sampler2D"), "{out}");
        assert!(out.contains("outer_hz_Sampler2(uv)"), "{out}");
        assert!(out.contains("inner_hz_Sampler2(uv)"), "{out}");
    }

    /// A call in another call's arguments is rewritten before the call around it, which is
    /// then read again: neither edit lands on the other's old text.
    #[test]
    fn a_call_nested_in_another_calls_arguments_is_rewritten_inside_out() {
        let text = "vec2 tap(sampler2D s, vec2 uv) { return texture(s, uv).xy; }\n\
            void main() { vec2 a = tap(Sampler0, tap(Sampler0, uv));\n\
            vec2 b = tap(   Sampler1   ,   tap(  Sampler0  ,     uv      )   ); }\n";
        let out = specialize_sampler_params(text);
        assert!(!out.contains("sampler2D") && !out.contains("tap("), "{out}");
        assert!(
            out.contains("vec2 a = tap_hz_Sampler0(tap_hz_Sampler0(uv));"),
            "{out}"
        );
        assert!(
            out.contains("vec2 b = tap_hz_Sampler1(tap_hz_Sampler0(uv)); }"),
            "{out}"
        );
        assert!(out.contains("vec2 tap_hz_Sampler0(vec2 uv) { return texture(Sampler0, uv).xy; }"));
        assert!(out.contains("vec2 tap_hz_Sampler1(vec2 uv) { return texture(Sampler1, uv).xy; }"));
    }

    /// Overloads that differ past their sampler each keep their own body in their copies.
    #[test]
    fn overloads_with_a_sampler_each_keep_their_own_body() {
        let text = "vec4 tap(sampler2D s, vec2 p) { return texture(s, p); }\n\
            vec4 tap(sampler2D s, vec3 p) { return texture(s, p.xy / p.z); }\n\
            void main() { vec4 a = tap(Sampler0, uv); vec4 b = tap(Sampler0, vec3(uv, 2.0)); }\n";
        let out = specialize_sampler_params(text);
        assert!(!out.contains("sampler2D"), "{out}");
        assert!(
            out.contains("vec4 tap_hz_Sampler0(vec2 p) { return texture(Sampler0, p); }"),
            "{out}"
        );
        assert!(
            out.contains("vec4 tap_hz_Sampler0(vec3 p) { return texture(Sampler0, p.xy / p.z); }"),
            "{out}"
        );
        assert_eq!(out.matches("vec4 tap_hz_Sampler0(").count(), 2, "{out}");
        assert!(out.contains("vec4 a = tap_hz_Sampler0(uv);"), "{out}");
        assert!(
            out.contains("vec4 b = tap_hz_Sampler0(vec3(uv, 2.0));"),
            "{out}"
        );
    }

    /// A copy reads its samplers by name, so it goes after a sampler declared below the helper
    /// (vanilla's world `text.vsh` includes `sample_lightmap.glsl` above `Sampler2`).
    #[test]
    fn a_copy_goes_after_its_samplers_declaration() {
        let text = "vec4 tap(sampler2D s, vec2 uv) { return texture(s, uv); }\n\
            uniform sampler2D Sampler2;\n\
            void main() { vec4 c = tap(Sampler2, vec2(0.5)); }\n";
        let out = specialize_sampler_params(text);
        let declared = out
            .find("uniform sampler2D Sampler2;")
            .expect("declaration kept");
        let copy = out
            .find("vec4 tap_hz_Sampler2(vec2 uv)")
            .expect("copy written");
        let call = out
            .find("tap_hz_Sampler2(vec2(0.5))")
            .expect("call rewritten");
        assert!(declared < copy && copy < call, "{out}");
        assert_eq!(out.matches("tap_hz_Sampler2(vec2 uv)").count(), 1, "{out}");
        assert!(!out.contains("tap(sampler2D"), "{out}");
    }

    /// A pack's text is untrusted: a sampler declared inside the helper itself (and text that
    /// isn't ASCII) must leave a shader the translator can refuse, never a panic.
    #[test]
    fn a_sampler_declared_inside_the_helper_never_panics() {
        let text = "vec4 tap(sampler2D s, vec2 uv) { é uniform sampler2D Sampler0; x return texture(s, uv); ééé }\n\
            void main() { fragColor = tap(Sampler0, vec2(0.0)); }\n";
        let out = specialize_sampler_params(text);
        assert!(out.is_char_boundary(out.len()));
        assert!(out.contains("tap_hz_Sampler0(vec2(0.0))"), "{out}");
    }

    /// Vanilla's `texture_sampling.glsl` shape: one overload calls the other with its own
    /// sampler. That call is rewritten too, and both copies come after the sampler's declaration
    /// in their own order, the called one first.
    #[test]
    fn an_overload_passing_its_sampler_to_another_is_followed() {
        let text = "vec4 sampleNearest(sampler2D source, vec2 uv, vec2 pixelSize, vec2 du, vec2 dv, vec2 texelSize) { return texture(source, uv + du + dv); }\n\
            vec4 sampleNearest(sampler2D source, vec2 uv, vec2 pixelSize) { return sampleNearest(source, uv, pixelSize, dFdx(uv), dFdy(uv), pixelSize); }\n\
            uniform sampler2D Sampler0;\n\
            void main() { vec4 c = sampleNearest(Sampler0, uv, px); }\n";
        let out = specialize_sampler_params(text);
        assert!(
            !out.contains("sampleNearest(") && !out.contains("sampler2D source"),
            "{out}"
        );
        let declared = out
            .find("uniform sampler2D Sampler0;")
            .expect("declaration");
        let six = out
            .find("vec4 sampleNearest_hz_Sampler0(vec2 uv, vec2 pixelSize, vec2 du, vec2 dv, vec2 texelSize)")
            .expect("six-argument copy");
        let three = out
            .find("vec4 sampleNearest_hz_Sampler0(vec2 uv, vec2 pixelSize) { return sampleNearest_hz_Sampler0(uv, pixelSize, dFdx(uv), dFdy(uv), pixelSize); }")
            .expect("three-argument copy calling the six-argument one");
        let call = out
            .find("vec4 c = sampleNearest_hz_Sampler0(uv, px);")
            .expect("call");
        assert!(declared < six && six < three && three < call, "{out}");
    }

    /// A prototype above `main` with the definition below it becomes the copies' prototypes, so
    /// the call still has a declaration above it.
    #[test]
    fn a_prototype_becomes_the_copies_prototypes() {
        let text = "uniform sampler2D Sampler0;\n\
            vec4 tap(sampler2D s, vec2 uv);\n\
            void main() { vec4 c = tap(Sampler0, uv); }\n\
            vec4 tap(sampler2D s, vec2 uv) { return texture(s, uv); }\n";
        let out = specialize_sampler_params(text);
        let prototype = out
            .find("vec4 tap_hz_Sampler0(vec2 uv);")
            .expect("prototype copy");
        let call = out.find("tap_hz_Sampler0(uv); }").expect("call");
        let definition = out
            .find("vec4 tap_hz_Sampler0(vec2 uv) { return texture(Sampler0, uv); }")
            .expect("definition copy");
        assert!(prototype < call && call < definition, "{out}");
    }

    #[test]
    fn unsized_arrays_take_their_initializers_size() {
        let text = "const float[] w = float[](0.25, 0.5, 0.25);\n\
            const vec2 offsets[] = vec2[](vec2(1.0, 0.0), vec2(0.0, 1.0));\n\
            float k[] = float[4](1.0, 2.0, 3.0, 4.0);\n\
            bool same = a == b;\n";
        let out = size_unsized_arrays(text);
        assert!(
            out.contains("const float[3] w = float[3](0.25, 0.5, 0.25);"),
            "{out}"
        );
        assert!(
            out.contains("const vec2 offsets[2] = vec2[2](vec2(1.0, 0.0), vec2(0.0, 1.0));"),
            "{out}"
        );
        assert!(
            out.contains("float k[4] = float[4](1.0, 2.0, 3.0, 4.0);"),
            "{out}"
        );
        assert!(out.contains("bool same = a == b;"));
    }
}
