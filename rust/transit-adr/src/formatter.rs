//! Address formatting with country-code-aware compiled templates.
//!
//! Mirrors C++ `adr/formatter.h` + `formatter.cc`.
//!
//! The C++ uses `kainjow::mustache` with YAML-loaded country templates.
//! This Rust implementation uses a custom zero-regex, pre-compiled template
//! engine. Templates are parsed once into an instruction list; rendering is
//! a single linear pass with no allocations beyond the output string.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Address fields
// ---------------------------------------------------------------------------

/// Known template variable slots, array-indexed for O(1) lookup during render.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum Var {
    HouseNumber = 0,
    Road = 1,
    Neighbourhood = 2,
    Suburb = 3,
    Postcode = 4,
    City = 5,
    County = 6,
    State = 7,
    Country = 8,
    CountryCode = 9,
}

const _NUM_VARS: usize = 10;

/// Resolve a tag name to its `Var` slot.
fn parse_var(name: &str) -> Option<Var> {
    match name {
        "house_number" => Some(Var::HouseNumber),
        "road" => Some(Var::Road),
        "neighbourhood" => Some(Var::Neighbourhood),
        "suburb" => Some(Var::Suburb),
        "postcode" => Some(Var::Postcode),
        "city" => Some(Var::City),
        "county" => Some(Var::County),
        "state" => Some(Var::State),
        "country" => Some(Var::Country),
        "country_code" => Some(Var::CountryCode),
        _ => None,
    }
}

/// Address components for formatting.  C++ `formatter::address`.
///
/// Fields stored as `&str` slices to avoid copies when the caller already owns
/// the data (typical case: references into the typeahead string table).
#[derive(Debug, Clone, Default)]
pub struct FormatterAddress {
    pub house_number: String,
    pub road: String,
    pub neighbourhood: String,
    pub suburb: String,
    pub postcode: String,
    pub city: String,
    pub county: String,
    pub state: String,
    pub country: String,
    pub country_code: String,
}

impl FormatterAddress {
    /// Array-indexed field access — no branching during render.
    #[inline(always)]
    fn get(&self, var: Var) -> &str {
        match var {
            Var::HouseNumber => &self.house_number,
            Var::Road => &self.road,
            Var::Neighbourhood => &self.neighbourhood,
            Var::Suburb => &self.suburb,
            Var::Postcode => &self.postcode,
            Var::City => &self.city,
            Var::County => &self.county,
            Var::State => &self.state,
            Var::Country => &self.country,
            Var::CountryCode => &self.country_code,
        }
    }
}

// ---------------------------------------------------------------------------
// Compiled template IR
// ---------------------------------------------------------------------------

/// One instruction in a compiled template.
#[derive(Debug)]
enum Seg {
    /// Literal text — byte range `[start..end)` into `CompiledTemplate::source`.
    Lit(u32, u32),
    /// Variable substitution.
    Var(Var),
    /// `{{#first}}alt0||alt1||...{{/first}}` — try each alternative in order,
    /// emit the first that produces non-whitespace output.
    First(Vec<Vec<Seg>>),
}

/// A pre-compiled template ready for fast repeated rendering.
#[derive(Debug)]
struct CompiledTemplate {
    /// The template source after the `, ` → `\n` transform.
    source: Box<str>,
    /// Instruction list.
    segs: Vec<Seg>,
}

impl CompiledTemplate {
    /// Parse a raw template string (as loaded from YAML / config).
    ///
    /// The C++ does `boost::replace_all(address_template, ", ", "\n")` before
    /// rendering. We apply the same transform here at compile time.
    fn compile(raw: &str) -> Self {
        let source: String = raw.replace(", ", "\n");
        let segs = parse_segments(source.as_bytes(), 0, source.len());
        Self {
            source: source.into_boxed_str(),
            segs,
        }
    }

    /// Render this template with the given address fields into `buf`.
    /// Does NOT post-process (caller handles that).
    fn render_into(&self, addr: &FormatterAddress, buf: &mut String) {
        render_segs(&self.segs, &self.source, addr, buf);
    }
}

/// Render a segment list into `buf`.
fn render_segs(segs: &[Seg], source: &str, addr: &FormatterAddress, buf: &mut String) {
    for seg in segs {
        match seg {
            Seg::Lit(s, e) => {
                buf.push_str(&source[*s as usize..*e as usize]);
            }
            Seg::Var(v) => {
                buf.push_str(addr.get(*v));
            }
            Seg::First(alternatives) => {
                let checkpoint = buf.len();
                for alt in alternatives {
                    render_segs(alt, source, addr, buf);
                    // Check if the rendered portion has any non-whitespace.
                    let rendered = &buf[checkpoint..];
                    if rendered.bytes().any(|b| !b.is_ascii_whitespace()) {
                        break; // keep it, continue with remaining segments
                    }
                    buf.truncate(checkpoint); // rewind, try next
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Template parser — single-pass, zero-regex
// ---------------------------------------------------------------------------

/// Parse template bytes `src[pos..end]` into a segment list.
fn parse_segments(src: &[u8], pos: usize, end: usize) -> Vec<Seg> {
    parse_flat(src, pos, end)
}

/// Parse a flat (non-`||`-split) segment list.
fn parse_flat(src: &[u8], start: usize, end: usize) -> Vec<Seg> {
    let mut segs = Vec::new();
    let mut i = start;
    let mut lit_start = start;

    while i < end {
        if i + 1 < end && src[i] == b'{' && src[i + 1] == b'{' {
            // Flush pending literal.
            if i > lit_start {
                segs.push(Seg::Lit(lit_start as u32, i as u32));
            }

            let tag_start = i + 2;
            // Find closing `}}`.
            let close = find_close_braces(src, tag_start, end);
            let tag = std::str::from_utf8(&src[tag_start..close]).unwrap_or("");

            if tag == "#first" {
                // Find matching {{/first}}.
                let inner_start = close + 2;
                let inner_end = find_end_section(src, inner_start, end, b"/first");
                let alternatives = parse_first(src, inner_start, inner_end);
                segs.push(Seg::First(alternatives));
                // Skip past {{/first}}.
                i = inner_end;
                // Advance past the `{{/first}}` tag itself.
                if i + 1 < end && src[i] == b'{' && src[i + 1] == b'{' {
                    let c2 = find_close_braces(src, i + 2, end);
                    i = c2 + 2;
                }
                lit_start = i;
            } else if let Some(var) = parse_var(tag) {
                segs.push(Seg::Var(var));
                i = close + 2;
                lit_start = i;
            } else {
                // Unknown tag — treat as literal.
                i = close + 2;
                lit_start = i;
            }
        } else {
            i += 1;
        }
    }

    if end > lit_start {
        segs.push(Seg::Lit(lit_start as u32, end as u32));
    }

    segs
}

/// Parse the interior of a `{{#first}}...{{/first}}` block.
/// Splits by `||` and parses each alternative.
fn parse_first(src: &[u8], start: usize, end: usize) -> Vec<Vec<Seg>> {
    let ranges = split_alternatives(src, start, end);
    ranges
        .into_iter()
        .map(|(s, e)| parse_flat(src, s, e))
        .collect()
}

/// Split a byte range by `||`, returning (start, end) pairs.
fn split_alternatives(src: &[u8], start: usize, end: usize) -> Vec<(usize, usize)> {
    let mut result = Vec::new();
    let mut seg_start = start;
    let mut i = start;
    while i + 1 < end {
        if src[i] == b'|' && src[i + 1] == b'|' {
            result.push((seg_start, i));
            seg_start = i + 2;
            i += 2;
        } else {
            i += 1;
        }
    }
    result.push((seg_start, end));
    result
}

/// Find the position of `}}` starting from `pos`. Returns the index of the
/// first `}`.
fn find_close_braces(src: &[u8], pos: usize, end: usize) -> usize {
    let mut i = pos;
    while i + 1 < end {
        if src[i] == b'}' && src[i + 1] == b'}' {
            return i;
        }
        i += 1;
    }
    end // unclosed — treat rest as tag (won't crash)
}

/// Find the start of `{{/tag_name}}` in `src[pos..end]`.
/// Returns the byte index of the `{{` of the closing tag.
fn find_end_section(src: &[u8], pos: usize, end: usize, tag: &[u8]) -> usize {
    let mut i = pos;
    while i + 1 < end {
        if src[i] == b'{' && src[i + 1] == b'{' {
            let tag_start = i + 2;
            let close = find_close_braces(src, tag_start, end);
            let inner = &src[tag_start..close];
            if inner == tag {
                return i;
            }
            i = close + 2;
        } else {
            i += 1;
        }
    }
    end
}

// ---------------------------------------------------------------------------
// Post-processing — single pass, in-place
// ---------------------------------------------------------------------------

/// Apply the C++ post-processing to a rendered string:
/// 1. Collapse consecutive `\n` (keep at most one, remove trailing).
/// 2. Trim leading/trailing whitespace.
/// 3. Remove ` .` sequences.
/// 4. Replace `\n` with `, `.
/// 5. Final trim.
///
/// Operates on `buf` in-place with a single scan + one compaction.
fn postprocess(buf: &mut String) {
    // Phase 1: collapse newlines (mirrors the C++ loop exactly).
    // A newline is removed if:
    //   - It follows another newline (consecutive).
    //   - It is the last character.
    {
        let bytes = unsafe { buf.as_mut_vec() };
        let mut write = 0usize;
        let mut remove_next_nl = true;
        let len = bytes.len();

        for read in 0..len {
            let is_last = read + 1 == len;
            remove_next_nl |= is_last;

            if bytes[read] == b'\n' {
                if !remove_next_nl {
                    bytes[write] = b'\n';
                    write += 1;
                }
                remove_next_nl = true;
            } else {
                remove_next_nl = false;
                bytes[write] = bytes[read];
                write += 1;
            }
        }
        bytes.truncate(write);
    }

    // Phase 2: trim leading/trailing whitespace.
    trim_in_place(buf);

    // Phase 3: remove " ." (space-dot) sequences.
    {
        let bytes = unsafe { buf.as_mut_vec() };
        let mut write = 0usize;
        let len = bytes.len();
        let mut read = 0usize;
        while read < len {
            if read + 1 < len && bytes[read] == b' ' && bytes[read + 1] == b'.' {
                read += 2; // skip " ."
            } else {
                bytes[write] = bytes[read];
                write += 1;
                read += 1;
            }
        }
        bytes.truncate(write);
    }

    // Phase 4: replace `\n` with `, `.
    // This can grow the string so we use a proper replace.
    // Since newlines are rare (typically 0-3), this is fast.
    if buf.contains('\n') {
        *buf = buf.replace('\n', ", ");
    }

    // Phase 5: final trim.
    trim_in_place(buf);
}

/// Trim ASCII whitespace from both ends in-place (no allocation).
fn trim_in_place(s: &mut String) {
    let trimmed = s.trim();
    let start = trimmed.as_ptr() as usize - s.as_ptr() as usize;
    let len = trimmed.len();
    if start != 0 || len != s.len() {
        let bytes = unsafe { s.as_mut_vec() };
        if start > 0 {
            bytes.copy_within(start..start + len, 0);
        }
        bytes.truncate(len);
    }
}

// ---------------------------------------------------------------------------
// Formatter
// ---------------------------------------------------------------------------

/// Address formatter with pre-compiled country-specific templates.
/// C++ `adr::formatter`.
pub struct Formatter {
    /// Country code (2-char, lowercase) → compiled template.
    templates: HashMap<[u8; 2], CompiledTemplate>,
    /// Reusable render buffer to avoid per-call allocation.
    /// Caller must use `format()` (not shared across threads without mutex).
    render_buf: String,
}

impl Formatter {
    /// Create an empty formatter (no country templates loaded).
    pub fn new() -> Self {
        Self {
            templates: HashMap::new(),
            render_buf: String::with_capacity(256),
        }
    }

    /// Register a template for a country code.
    ///
    /// `country_code` should be a 2-character string (e.g. "DE", "US").
    /// `template` is the raw mustache-like template string as found in the
    /// worldwide.yaml configuration.
    pub fn add_template(&mut self, country_code: &str, template: &str) {
        if country_code.len() >= 2 {
            let key = [
                country_code.as_bytes()[0],
                country_code.as_bytes()[1],
            ];
            self.templates.insert(key, CompiledTemplate::compile(template));
        }
    }

    /// Format an address.  C++ `formatter::format()`.
    ///
    /// Looks up the country-specific template. Falls back to
    /// `"house_number road"` if no template is registered (matches C++ fallback).
    pub fn format(&mut self, addr: &FormatterAddress) -> String {
        let key = if addr.country_code.len() >= 2 {
            [
                addr.country_code.as_bytes()[0],
                addr.country_code.as_bytes()[1],
            ]
        } else {
            [0, 0]
        };

        if let Some(tmpl) = self.templates.get(&key) {
            self.render_buf.clear();
            tmpl.render_into(addr, &mut self.render_buf);
            postprocess(&mut self.render_buf);
            self.render_buf.clone()
        } else {
            // C++ fallback when no country template found.
            if addr.house_number.is_empty() {
                addr.road.clone()
            } else {
                let mut out = String::with_capacity(
                    addr.house_number.len() + 1 + addr.road.len(),
                );
                out.push_str(&addr.house_number);
                out.push(' ');
                out.push_str(&addr.road);
                out
            }
        }
    }

    /// Immutable format — creates a temporary buffer.
    /// Use when you can't take `&mut self`.
    pub fn format_immut(&self, addr: &FormatterAddress) -> String {
        let key = if addr.country_code.len() >= 2 {
            [
                addr.country_code.as_bytes()[0],
                addr.country_code.as_bytes()[1],
            ]
        } else {
            [0, 0]
        };

        if let Some(tmpl) = self.templates.get(&key) {
            let mut buf = String::with_capacity(128);
            tmpl.render_into(addr, &mut buf);
            postprocess(&mut buf);
            buf
        } else {
            if addr.house_number.is_empty() {
                addr.road.clone()
            } else {
                format!("{} {}", addr.house_number, addr.road)
            }
        }
    }
}

impl Default for Formatter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn addr(fields: &[(&str, &str)]) -> FormatterAddress {
        let mut a = FormatterAddress::default();
        for &(k, v) in fields {
            match k {
                "house_number" => a.house_number = v.to_string(),
                "road" => a.road = v.to_string(),
                "neighbourhood" => a.neighbourhood = v.to_string(),
                "suburb" => a.suburb = v.to_string(),
                "postcode" => a.postcode = v.to_string(),
                "city" => a.city = v.to_string(),
                "county" => a.county = v.to_string(),
                "state" => a.state = v.to_string(),
                "country" => a.country = v.to_string(),
                "country_code" => a.country_code = v.to_string(),
                _ => {}
            }
        }
        a
    }

    // --- Fallback (no template) ---

    #[test]
    fn format_fallback_road_only() {
        let mut f = Formatter::new();
        let a = addr(&[("road", "Hauptstraße")]);
        assert_eq!(f.format(&a), "Hauptstraße");
    }

    #[test]
    fn format_fallback_with_house_number() {
        let mut f = Formatter::new();
        let a = addr(&[("house_number", "42"), ("road", "Hauptstraße")]);
        assert_eq!(f.format(&a), "42 Hauptstraße");
    }

    // --- Simple variable substitution ---

    #[test]
    fn format_simple_template() {
        let mut f = Formatter::new();
        f.add_template("DE", "{{road}} {{house_number}}, {{postcode}} {{city}}");
        let a = addr(&[
            ("road", "Hauptstraße"),
            ("house_number", "42"),
            ("postcode", "64283"),
            ("city", "Darmstadt"),
            ("country_code", "DE"),
        ]);
        assert_eq!(f.format(&a), "Hauptstraße 42, 64283 Darmstadt");
    }

    // --- {{#first}} section ---

    #[test]
    fn format_first_picks_nonempty() {
        let mut f = Formatter::new();
        // Template: first non-empty of (postcode city) || (city) || (state)
        f.add_template(
            "XX",
            "{{road}}, {{#first}}{{postcode}} {{city}}||{{city}}||{{state}}{{/first}}",
        );
        let a = addr(&[
            ("road", "Main St"),
            ("postcode", "12345"),
            ("city", "Berlin"),
            ("country_code", "XX"),
        ]);
        assert_eq!(f.format(&a), "Main St, 12345 Berlin");
    }

    #[test]
    fn format_first_skips_empty_alt() {
        let mut f = Formatter::new();
        f.add_template(
            "XX",
            "{{road}}, {{#first}}{{postcode}} {{city}}||{{city}}||{{state}}{{/first}}",
        );
        // postcode and city both empty, state has value
        let a = addr(&[("road", "Main St"), ("state", "Bavaria"), ("country_code", "XX")]);
        assert_eq!(f.format(&a), "Main St, Bavaria");
    }

    #[test]
    fn format_first_all_empty() {
        let mut f = Formatter::new();
        f.add_template(
            "XX",
            "{{road}}, {{#first}}{{postcode}}||{{city}}||{{state}}{{/first}}",
        );
        let a = addr(&[("road", "Main St"), ("country_code", "XX")]);
        assert_eq!(f.format(&a), "Main St");
    }

    // --- Postprocessing: empty line collapse ---

    #[test]
    fn format_collapse_empty_lines() {
        let mut f = Formatter::new();
        // Template has `, ` which becomes `\n`. If city is empty, we get
        // consecutive newlines that should collapse.
        f.add_template("XX", "{{road}}, {{city}}, {{country}}");
        let a = addr(&[
            ("road", "Main St"),
            ("country", "Germany"),
            ("country_code", "XX"),
        ]);
        // "Main St\n\nGermany" → collapse → "Main St\nGermany" → "Main St, Germany"
        assert_eq!(f.format(&a), "Main St, Germany");
    }

    // --- Postprocessing: " ." removal ---

    #[test]
    fn format_removes_space_dot() {
        let mut f = Formatter::new();
        f.add_template("XX", "{{road}} .{{city}}");
        let a = addr(&[("road", "Main"), ("city", "Berlin"), ("country_code", "XX")]);
        // "Main .Berlin" → remove " ." → "MainBerlin"
        assert_eq!(f.format(&a), "MainBerlin");
    }

    // --- German address (real-world pattern) ---

    #[test]
    fn format_german_address() {
        let mut f = Formatter::new();
        // Typical German format: "Road HouseNumber\nPostcode City\nCountry"
        f.add_template(
            "DE",
            "{{road}} {{house_number}}, {{postcode}} {{city}}, {{country}}",
        );
        let a = addr(&[
            ("road", "Luisenplatz"),
            ("house_number", "5"),
            ("postcode", "64283"),
            ("city", "Darmstadt"),
            ("state", "Hessen"),
            ("country", "Deutschland"),
            ("country_code", "DE"),
        ]);
        assert_eq!(
            f.format(&a),
            "Luisenplatz 5, 64283 Darmstadt, Deutschland"
        );
    }

    // --- US address with {{#first}} ---

    #[test]
    fn format_us_address() {
        let mut f = Formatter::new();
        f.add_template(
            "US",
            "{{house_number}} {{road}}, {{#first}}{{city}}, {{state}} {{postcode}}||{{city}} {{postcode}}||{{city}}{{/first}}, {{country}}",
        );
        let a = addr(&[
            ("house_number", "1600"),
            ("road", "Pennsylvania Avenue"),
            ("city", "Washington"),
            ("state", "DC"),
            ("postcode", "20500"),
            ("country", "United States"),
            ("country_code", "US"),
        ]);
        assert_eq!(
            f.format(&a),
            "1600 Pennsylvania Avenue, Washington, DC 20500, United States"
        );
    }

    // --- Immutable format ---

    #[test]
    fn format_immut_matches_format() {
        let mut f = Formatter::new();
        f.add_template("DE", "{{road}} {{house_number}}, {{postcode}} {{city}}");
        let a = addr(&[
            ("road", "Hauptstraße"),
            ("house_number", "42"),
            ("postcode", "64283"),
            ("city", "Darmstadt"),
            ("country_code", "DE"),
        ]);
        let mutable_result = f.format(&a);
        let immutable_result = f.format_immut(&a);
        assert_eq!(mutable_result, immutable_result);
    }

    // --- Compile correctness ---

    #[test]
    fn compiled_template_segments() {
        let tmpl = CompiledTemplate::compile("{{road}} {{house_number}}");
        // Should have: Var(Road), Lit(" "), Var(HouseNumber)
        assert_eq!(tmpl.segs.len(), 3);
        assert!(matches!(tmpl.segs[0], Seg::Var(Var::Road)));
        assert!(matches!(tmpl.segs[1], Seg::Lit(_, _)));
        assert!(matches!(tmpl.segs[2], Seg::Var(Var::HouseNumber)));
    }

    // --- Postprocessing unit tests ---

    #[test]
    fn postprocess_empty() {
        let mut s = String::new();
        postprocess(&mut s);
        assert_eq!(s, "");
    }

    #[test]
    fn postprocess_no_change() {
        let mut s = "hello world".to_string();
        postprocess(&mut s);
        assert_eq!(s, "hello world");
    }

    #[test]
    fn postprocess_collapse_newlines() {
        let mut s = "a\n\n\nb".to_string();
        postprocess(&mut s);
        assert_eq!(s, "a, b");
    }

    #[test]
    fn postprocess_trailing_newline() {
        let mut s = "abc\n".to_string();
        postprocess(&mut s);
        assert_eq!(s, "abc");
    }

    #[test]
    fn postprocess_space_dot() {
        let mut s = "abc .def".to_string();
        postprocess(&mut s);
        assert_eq!(s, "abcdef");
    }

    #[test]
    fn postprocess_trim() {
        let mut s = "  hello  ".to_string();
        postprocess(&mut s);
        assert_eq!(s, "hello");
    }
}
