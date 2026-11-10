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
        if !followable {
            break;
        }
        let body = f.body.as_deref().unwrap_or_default();
        let params: Vec<&str> = f
            .params
            .iter()
            .enumerate()
            .filter(|(n, _)| !f.samplers.iter().any(|(s, _)| s == n))
            .map(|(_, p)| p.as_str())
            .collect();
        let copies: String = variants
            .iter()
            .map(|chosen| {
                let mut body = body.to_owned();
                for ((_, param), sampler) in f.samplers.iter().zip(chosen) {
                    body = replace_ident(&body, param, sampler);
                }
                format!(
                    "{} {}_hz_{}({}) {}\n",
                    f.return_type,
                    f.name,
                    chosen.join("_"),
                    params.join(", "),
                    body
                )
            })
            .collect();
        // Rebuild from the back: the calls, then the function's own spans (its prototypes go,
        // its definition becomes the copies). A call nested in another's arguments starts
        // after it, so it is rewritten first, and the call around it is read again from the
        // text as it is then.
        let mut edits: Vec<(usize, Option<(usize, String)>)> =
            call_starts.into_iter().map(|at| (at, None)).collect();
        for g in functions.iter().filter(|g| g.name == f.name) {
            let with = if g.body.is_some() {
                copies.clone()
            } else {
                String::new()
            };
            edits.push((g.start, Some((g.end, with))));
        }
        edits.sort_by_key(|(start, _)| std::cmp::Reverse(*start));
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
            text.replace_range(start..end, &with);
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
