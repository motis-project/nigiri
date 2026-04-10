//! Area set display and language resolution.
//!
//! Mirrors C++ `adr/area_set.h` + `area_set.cc`.

use crate::typeahead::{find_lang, Typeahead};
use crate::types::*;

/// An area set with language-aware display.
/// C++ `adr::area_set`.
pub struct AreaSet<'a> {
    pub t: &'a Typeahead,
    pub languages: &'a [LanguageIdx],
    pub city_area_idx: Option<usize>,
    pub area_set_idx: AreaSetIdx,
    pub matched_mask: u32,
    pub matched_area_lang: AreaSetLang,
}

impl<'a> AreaSet<'a> {
    /// Get areas from the area set.
    pub fn get_areas(&self) -> &[AreaIdx] {
        let idx = self.area_set_idx.to_idx();
        if idx < self.t.area_sets.len() {
            &self.t.area_sets[idx]
        } else {
            &[]
        }
    }

    /// Find the best language index for an area name.
    /// C++ `area_set::get_area_lang_idx()`.
    pub fn get_area_lang_idx(&self, area: AreaIdx) -> i16 {
        let area_idx = area.to_idx();
        if area_idx >= self.t.area_name_lang.len() {
            return -1;
        }
        let area_langs = &self.t.area_name_lang[area_idx];
        // Search languages in reverse order (last = highest priority).
        for i in (0..self.languages.len()).rev() {
            let lang_idx = find_lang(area_langs, self.languages[i]);
            if lang_idx != -1 {
                return lang_idx;
            }
        }
        -1
    }

    /// Format the area set for display.
    pub fn format(&self) -> String {
        let areas = self.get_areas();
        let mut parts = Vec::new();

        for (i, &area) in areas.iter().enumerate() {
            let area_idx = area.to_idx();
            if area_idx >= self.t.area_admin_level.len() {
                continue;
            }
            let admin_lvl = self.t.area_admin_level[area_idx];
            if admin_lvl == TIMEZONE_ADMIN_LEVEL {
                continue;
            }

            let matched = ((1u32 << i) & self.matched_mask) != 0;
            let language_idx = if matched {
                self.matched_area_lang[i] as i16
            } else {
                self.get_area_lang_idx(area)
            };

            let name_idx = if language_idx < 0 {
                DEFAULT_LANG_IDX
            } else {
                language_idx as usize
            };

            let area_names = &self.t.area_names[area_idx];
            if name_idx >= area_names.len() {
                continue;
            }
            let str_idx = area_names[name_idx].to_idx();
            if str_idx >= self.t.strings.len() {
                continue;
            }
            let name = &self.t.strings[str_idx];

            let prefix = if matched { " *" } else { "" };
            parts.push(format!(
                "{}({}, {})",
                prefix,
                name,
                admin_lvl.0
            ));
        }

        format!(" [{}]", parts.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn area_set_empty() {
        let t = Typeahead::new();
        let aset = AreaSet {
            t: &t,
            languages: &[],
            city_area_idx: None,
            area_set_idx: AreaSetIdx(0),
            matched_mask: 0,
            matched_area_lang: [0u8; 32],
        };
        assert!(aset.get_areas().is_empty());
    }
}
