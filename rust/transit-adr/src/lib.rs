//! Address Data & Reverse Geocoding (ADR)
//!
//! Core algorithms for address search, fuzzy text matching, and scoring.
//! Provides UTF-8 normalization, n-gram indexing, SIFT4 approximate string
//! distance, hierarchical match scoring, and a suggestion engine for geocoding.

pub mod area_database;
pub mod area_set;
pub mod cache;
pub mod categories;
pub mod formatter;
pub mod guess;
pub mod import_context;
pub mod ngram;
pub mod normalize;
pub mod reverse;
pub mod score;
pub mod sift4;
pub mod suggestion;
pub mod suggestions;
pub mod typeahead;
pub mod types;
