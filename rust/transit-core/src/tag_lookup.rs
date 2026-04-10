use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::{TransitError, TransitResult};
use crate::types::SourceIdx;

/// Separator used in GraphQL IDs between tag and GTFS ID.
pub const ID_SEPARATOR: char = ':';

/// Bidirectional mapping between dataset tags and source indices,
/// plus ID generation and parsing for the GraphQL API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagLookup {
    /// source_idx → tag string (e.g., 0 → "seq")
    src_to_tag: Vec<String>,
    /// tag string → source_idx (e.g., "seq" → SourceIdx(0))
    tag_to_src: HashMap<String, SourceIdx>,
}

/// Parsed components of a trip ID.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TripId {
    pub tag: String,
    pub start_date: String, // "YYYYMMDD"
    pub start_time: String, // "HH:MM"
    pub trip_id: String,    // GTFS trip_id (may contain underscores)
}

impl TagLookup {
    /// Create an empty TagLookup.
    pub fn new() -> Self {
        Self {
            src_to_tag: Vec::new(),
            tag_to_src: HashMap::new(),
        }
    }

    /// Register a dataset tag for a source index.
    pub fn register_tag(&mut self, tag: &str) -> SourceIdx {
        if let Some(&src) = self.tag_to_src.get(tag) {
            return src;
        }
        let src = SourceIdx(self.src_to_tag.len() as u32);
        self.src_to_tag.push(tag.to_string());
        self.tag_to_src.insert(tag.to_string(), src);
        src
    }

    /// Get the tag string for a source index.
    pub fn tag_for_source(&self, src: SourceIdx) -> Option<&str> {
        self.src_to_tag.get(src.0 as usize).map(|s| s.as_str())
    }

    /// Get the source index for a tag string.
    pub fn source_for_tag(&self, tag: &str) -> Option<SourceIdx> {
        self.tag_to_src.get(tag).copied()
    }

    /// Number of registered tags.
    pub fn tag_count(&self) -> usize {
        self.src_to_tag.len()
    }

    // --- ID generation ---

    /// Build a location ID: "{tag}:{gtfs_id}"
    pub fn format_location_id(&self, src: SourceIdx, gtfs_id: &str) -> Option<String> {
        let tag = self.src_to_tag.get(src.0 as usize)?;
        Some(format!("{}{}{}", tag, ID_SEPARATOR, gtfs_id))
    }

    /// Build a route ID: "{tag}:{gtfs_route_id}"
    pub fn format_route_id(&self, src: SourceIdx, gtfs_route_id: &str) -> Option<String> {
        let tag = self.src_to_tag.get(src.0 as usize)?;
        Some(format!("{}{}{}", tag, ID_SEPARATOR, gtfs_route_id))
    }

    /// Build a trip ID: "{tag}:{YYYYMMDD}_{HH:MM}_{gtfs_trip_id}"
    pub fn format_trip_id(
        &self,
        src: SourceIdx,
        date: &str,
        time: &str,
        gtfs_trip_id: &str,
    ) -> Option<String> {
        let tag = self.src_to_tag.get(src.0 as usize)?;
        Some(format!(
            "{}{}{}_{}_{}",
            tag, ID_SEPARATOR, date, time, gtfs_trip_id
        ))
    }

    // --- ID parsing ---

    /// Parse a location/route ID "tag:gtfs_id" → (SourceIdx, gtfs_id).
    pub fn parse_entity_id<'a>(&self, id: &'a str) -> TransitResult<(SourceIdx, &'a str)> {
        let (tag, gtfs_id) = id.split_once(ID_SEPARATOR).ok_or_else(|| {
            TransitError::InvalidId(format!("expected format 'tag{ID_SEPARATOR}id', got '{id}'"))
        })?;
        let src = self
            .tag_to_src
            .get(tag)
            .copied()
            .ok_or_else(|| TransitError::NotFound(format!("unknown tag '{tag}'")))?;
        Ok((src, gtfs_id))
    }

    /// Parse a trip ID "tag:YYYYMMDD_HH:MM_trip_id".
    pub fn parse_trip_id(&self, id: &str) -> TransitResult<TripId> {
        let (tag, rest) = id
            .split_once(ID_SEPARATOR)
            .ok_or_else(|| TransitError::InvalidId(id.to_string()))?;

        // Validate tag exists
        if !self.tag_to_src.contains_key(tag) {
            return Err(TransitError::NotFound(format!("unknown tag '{tag}'")));
        }

        // rest = "YYYYMMDD_HH:MM_trip_id"
        let (date, rest) = rest
            .split_once('_')
            .ok_or_else(|| TransitError::InvalidId(id.to_string()))?;

        // HH:MM contains a colon, so we split at the next underscore after the time
        let (time, trip_id) = rest
            .split_once('_')
            .ok_or_else(|| TransitError::InvalidId(id.to_string()))?;

        // Validate date: 8 digits
        if date.len() != 8 || !date.chars().all(|c| c.is_ascii_digit()) {
            return Err(TransitError::InvalidId(format!(
                "invalid date in trip ID: {date}"
            )));
        }

        // Validate time: contains ':'
        if !time.contains(':') {
            return Err(TransitError::InvalidId(format!(
                "invalid time in trip ID: {time}"
            )));
        }

        Ok(TripId {
            tag: tag.to_string(),
            start_date: date.to_string(),
            start_time: time.to_string(),
            trip_id: trip_id.to_string(),
        })
    }
}

impl Default for TagLookup {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_and_lookup_tags() {
        let mut lookup = TagLookup::new();
        let src0 = lookup.register_tag("seq");
        let src1 = lookup.register_tag("cns");
        assert_eq!(src0, SourceIdx(0));
        assert_eq!(src1, SourceIdx(1));

        assert_eq!(lookup.tag_for_source(src0), Some("seq"));
        assert_eq!(lookup.source_for_tag("seq"), Some(SourceIdx(0)));
        assert_eq!(lookup.source_for_tag("unknown"), None);

        // Re-registering returns same index
        assert_eq!(lookup.register_tag("seq"), SourceIdx(0));
        assert_eq!(lookup.tag_count(), 2);
    }

    #[test]
    fn format_location_id() {
        let mut lookup = TagLookup::new();
        lookup.register_tag("seq");
        assert_eq!(
            lookup.format_location_id(SourceIdx(0), "600011"),
            Some("seq:600011".to_string())
        );
        assert_eq!(lookup.format_location_id(SourceIdx(99), "x"), None);
    }

    #[test]
    fn format_trip_id() {
        let mut lookup = TagLookup::new();
        lookup.register_tag("seq");
        assert_eq!(
            lookup.format_trip_id(SourceIdx(0), "20240510", "14:30", "trip_123"),
            Some("seq:20240510_14:30_trip_123".to_string())
        );
    }

    #[test]
    fn parse_entity_id() {
        let mut lookup = TagLookup::new();
        lookup.register_tag("seq");

        let (src, gtfs_id) = lookup.parse_entity_id("seq:600011").unwrap();
        assert_eq!(src, SourceIdx(0));
        assert_eq!(gtfs_id, "600011");

        // Missing separator
        assert!(lookup.parse_entity_id("no_separator").is_err());

        // Unknown tag
        assert!(lookup.parse_entity_id("unk:123").is_err());
    }

    #[test]
    fn parse_trip_id_valid() {
        let mut lookup = TagLookup::new();
        lookup.register_tag("seq");

        let tid = lookup.parse_trip_id("seq:20240510_14:30_trip_123").unwrap();
        assert_eq!(tid.tag, "seq");
        assert_eq!(tid.start_date, "20240510");
        assert_eq!(tid.start_time, "14:30");
        assert_eq!(tid.trip_id, "trip_123");
    }

    #[test]
    fn parse_trip_id_with_underscores_in_trip_id() {
        let mut lookup = TagLookup::new();
        lookup.register_tag("seq");

        // trip_id itself can contain underscores — only the first two _ are split
        let tid = lookup
            .parse_trip_id("seq:20240510_14:30_trip_with_underscores")
            .unwrap();
        assert_eq!(tid.trip_id, "trip_with_underscores");
    }

    #[test]
    fn parse_trip_id_invalid() {
        let mut lookup = TagLookup::new();
        lookup.register_tag("seq");

        // Bad date
        assert!(lookup.parse_trip_id("seq:baddate_14:30_trip").is_err());
        // No separators
        assert!(lookup.parse_trip_id("seq:noseparator").is_err());
        // Unknown tag
        assert!(lookup.parse_trip_id("unk:20240510_14:30_trip").is_err());
    }

    #[test]
    fn roundtrip_serde() {
        let mut lookup = TagLookup::new();
        lookup.register_tag("seq");
        lookup.register_tag("cns");

        let json = serde_json::to_string(&lookup).unwrap();
        let deserialized: TagLookup = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.tag_count(), 2);
        assert_eq!(deserialized.tag_for_source(SourceIdx(0)), Some("seq"));
        assert_eq!(deserialized.source_for_tag("cns"), Some(SourceIdx(1)));
    }
}
