//! Translation of osr/include/osr/geojson.h
//!
//! GeoJSON output formatting with complete serialization.
//! Provides utilities for converting OSR data structures to GeoJSON.

use std::collections::HashMap;

use serde_json::{json, Value};

use crate::types::*;
use crate::ways::{NodeProperties, WayProperties, Ways};
use crate::Point;

/// Convert point to GeoJSON coordinate array [lng, lat]
pub fn point_to_array(point: &Point) -> Value {
    let (lat, lng) = point.as_latlng();
    json!([lng, lat])
}

/// Convert point to GeoJSON Point geometry
pub fn point_to_geometry(point: &Point) -> Value {
    json!({
        "type": "Point",
        "coordinates": point_to_array(point)
    })
}

/// Convert point list to GeoJSON LineString geometry
pub fn linestring_to_geometry(points: &[Point]) -> Value {
    let coords: Vec<Value> = points.iter().map(point_to_array).collect();
    json!({
        "type": "LineString",
        "coordinates": coords
    })
}

/// Convert lat/lng tuples to GeoJSON LineString geometry
pub fn coords_to_linestring(coords: &[(f64, f64)]) -> Value {
    let points: Vec<Value> = coords.iter().map(|(lat, lng)| json!([lng, lat])).collect();
    json!({
        "type": "LineString",
        "coordinates": points
    })
}

/// GeoJSON feature with geometry and properties
#[derive(Debug, Clone)]
pub struct Feature {
    pub geometry: Value,
    pub properties: HashMap<String, Value>,
}

impl Feature {
    /// Create new feature with geometry
    pub fn new(geometry: Value) -> Self {
        Self {
            geometry,
            properties: HashMap::new(),
        }
    }

    /// Create point feature
    pub fn point(point: &Point) -> Self {
        Self::new(point_to_geometry(point))
    }

    /// Create linestring feature
    pub fn linestring(points: &[Point]) -> Self {
        Self::new(linestring_to_geometry(points))
    }

    /// Add property
    pub fn with_property(mut self, key: impl Into<String>, value: Value) -> Self {
        self.properties.insert(key.into(), value);
        self
    }

    /// Add string property
    pub fn with_str(self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.with_property(key, json!(value.into()))
    }

    /// Add numeric property
    pub fn with_num<T: Into<Value>>(self, key: impl Into<String>, value: T) -> Self {
        self.with_property(key, value.into())
    }

    /// Add boolean property
    pub fn with_bool(self, key: impl Into<String>, value: bool) -> Self {
        self.with_property(key, json!(value))
    }

    /// Convert to GeoJSON Value
    pub fn to_value(&self) -> Value {
        json!({
            "type": "Feature",
            "geometry": self.geometry,
            "properties": self.properties
        })
    }

    /// Convert to GeoJSON string
    pub fn to_geojson(&self) -> String {
        self.to_value().to_string()
    }
}

/// GeoJSON feature collection
#[derive(Debug, Clone)]
pub struct FeatureCollection {
    pub features: Vec<Feature>,
    pub metadata: HashMap<String, Value>,
}

impl FeatureCollection {
    /// Create empty collection
    pub fn new() -> Self {
        Self {
            features: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add feature
    pub fn add(&mut self, feature: Feature) {
        self.features.push(feature);
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Convert to GeoJSON Value
    pub fn to_value(&self) -> Value {
        json!({
            "type": "FeatureCollection",
            "metadata": self.metadata,
            "features": self.features.iter().map(|f| f.to_value()).collect::<Vec<_>>()
        })
    }

    /// Convert to GeoJSON string
    pub fn to_geojson(&self) -> String {
        self.to_value().to_string()
    }
}

impl Default for FeatureCollection {
    fn default() -> Self {
        Self::new()
    }
}

/// GeoJSON writer for ways and platforms
pub struct GeoJsonWriter<'a> {
    ways: &'a Ways,
    features: Vec<Feature>,
    nodes: std::collections::HashSet<NodeIdx>,
}

impl<'a> GeoJsonWriter<'a> {
    /// Create new writer
    pub fn new(ways: &'a Ways) -> Self {
        Self {
            ways,
            features: Vec::new(),
            nodes: std::collections::HashSet::new(),
        }
    }

    /// Write way as features (edges + geometry)
    pub fn write_way(&mut self, way_idx: WayIdx) {
        let nodes = self.ways.get_way_nodes(way_idx);
        if nodes.is_empty() {
            return;
        }

        let default_props = WayProperties::default();
        let props = self
            .ways
            .get_way_properties(way_idx)
            .unwrap_or(&default_props);
        let osm_way_id = self
            .ways
            .get_way_osm_id(way_idx)
            .map(|id| id.value())
            .unwrap_or(0);

        // Write individual edges
        for i in 0..nodes.len().saturating_sub(1) {
            let from = nodes[i];
            let to = nodes[i + 1];

            let from_pos = self.ways.get_node_pos(from);
            let to_pos = self.ways.get_node_pos(to);

            let edge_feature = Feature::linestring(&[from_pos, to_pos])
                .with_str("type", "edge")
                .with_num("osm_way_id", osm_way_id)
                .with_num("internal_id", way_idx.value())
                .with_bool("car", props.is_car_accessible())
                .with_bool("bike", props.is_bike_accessible())
                .with_bool("foot", props.is_foot_accessible())
                .with_bool("oneway_car", props.is_oneway_car())
                .with_bool("oneway_bike", props.is_oneway_bike())
                .with_num("from_level", props.from_level().to_float())
                .with_num("to_level", props.to_level().to_float());

            self.features.push(edge_feature);
        }

        // Write full geometry
        let polyline = self.ways.get_way_polyline(way_idx);
        if !polyline.is_empty() {
            let geom_feature = Feature::linestring(polyline)
                .with_str("type", "geometry")
                .with_num("osm_way_id", osm_way_id)
                .with_num("internal_id", way_idx.value())
                .with_bool("car", props.is_car_accessible())
                .with_bool("bike", props.is_bike_accessible())
                .with_bool("foot", props.is_foot_accessible());

            self.features.push(geom_feature);
        }

        // Track nodes
        for &node in nodes {
            self.nodes.insert(node);
        }
    }

    /// Write nodes with properties
    pub fn write_nodes(&mut self) {
        for &node in &self.nodes {
            let pos = self.ways.get_node_pos(node);
            let default_node_props = NodeProperties::default();
            let props = self
                .ways
                .get_node_properties(node)
                .unwrap_or(&default_node_props);

            let osm_node_id = self
                .ways
                .get_node_osm_id(node)
                .map(|id| id.value())
                .unwrap_or(0);

            let node_feature = Feature::point(&pos)
                .with_num("osm_node_id", osm_node_id)
                .with_num("internal_id", node.value())
                .with_bool("car", props.is_car_accessible())
                .with_bool("bike", props.is_bike_accessible())
                .with_bool("foot", props.is_walk_accessible())
                .with_bool("is_entrance", props.is_entrance())
                .with_bool("multi_level", props.is_multi_level());

            self.features.push(node_feature);
        }
    }

    /// Get feature collection
    pub fn to_feature_collection(self) -> FeatureCollection {
        let mut collection = FeatureCollection::new();
        collection.features = self.features;
        collection
    }

    /// Convert to GeoJSON string
    pub fn to_geojson(self) -> String {
        self.to_feature_collection().to_geojson()
    }
}

/// Convert a routing path to a GeoJSON FeatureCollection.
///
/// Each path segment becomes a GeoJSON Feature with LineString geometry and
/// properties: level, osm_way_id, cost, distance, mode.
///
/// Metadata always includes `mode_summary` (per-mode distance totals).
/// When `with_properties` is true, also includes `duration` and `distance`.
///
/// # Arguments
/// * `ways` - Street network data (for OSM way ID lookup)
/// * `path` - Routing result path
/// * `with_properties` - If true, include duration/distance in metadata
pub fn path_to_geojson(
    ways: &Ways,
    path: &crate::routing::Path,
    with_properties: bool,
) -> FeatureCollection {
    let mut collection = FeatureCollection::new();
    let mut mode_summary: HashMap<String, f64> = HashMap::new();

    if with_properties {
        collection.metadata.insert("duration".into(), json!(path.cost));
        collection
            .metadata
            .insert("distance".into(), json!(path.dist));
    }

    for segment in &path.segments {
        if segment.polyline.is_empty() {
            continue;
        }

        let osm_way_id: u64 = if segment.way.is_valid() {
            ways.get_way_osm_id(segment.way)
                .map(|id| id.value())
                .unwrap_or(0)
        } else {
            0
        };

        let seg_dist = segment.dist as f64;
        let mode_name = segment.mode.as_str();
        *mode_summary.entry(mode_name.to_string()).or_insert(0.0) += seg_dist;

        let feature = Feature::linestring(&segment.polyline)
            .with_num("level", segment.from_level.to_float() as f64)
            .with_num("osm_way_id", osm_way_id)
            .with_num("cost", segment.cost)
            .with_num("distance", segment.dist)
            .with_str("mode", mode_name);

        collection.add(feature);
    }

    collection.metadata.insert("mode_summary".into(), json!(mode_summary));
    collection
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_feature() {
        let point = Point::from_latlng(52.52, 13.405);
        let feature = Feature::point(&point);

        let value = feature.to_value();
        assert_eq!(value["type"], "Feature");
        assert_eq!(value["geometry"]["type"], "Point");
    }

    #[test]
    fn test_location_feature() {
        let point = Point::from_latlng(52.52, 13.405);
        let feature = Feature::point(&point).with_num("level", 1.0);

        assert!(feature.properties.contains_key("level"));
    }

    #[test]
    fn test_linestring_feature() {
        let points = vec![
            Point::from_latlng(52.52, 13.405),
            Point::from_latlng(52.53, 13.415),
        ];
        let feature = Feature::linestring(&points);

        let value = feature.to_value();
        assert_eq!(value["geometry"]["type"], "LineString");
        assert_eq!(
            value["geometry"]["coordinates"].as_array().unwrap().len(),
            2
        );
    }

    #[test]
    fn test_feature_collection() {
        let mut collection = FeatureCollection::new();
        let point = Point::from_latlng(52.52, 13.405);
        collection.add(Feature::point(&point));

        assert_eq!(collection.features.len(), 1);

        let geojson = collection.to_geojson();
        assert!(geojson.contains("FeatureCollection"));
    }

    #[test]
    fn test_point_to_array() {
        let point = Point::from_latlng(52.52, 13.405);
        let array = point_to_array(&point);

        let coords = array.as_array().unwrap();
        assert_eq!(coords.len(), 2);
        // GeoJSON uses [lng, lat] order
        assert!((coords[0].as_f64().unwrap() - 13.405).abs() < 0.001);
        assert!((coords[1].as_f64().unwrap() - 52.52).abs() < 0.001);
    }

    #[test]
    fn test_point_to_geometry() {
        let point = Point::from_latlng(52.52, 13.405);
        let geom = point_to_geometry(&point);

        assert_eq!(geom["type"], "Point");
        assert!(geom["coordinates"].is_array());
    }

    #[test]
    fn test_linestring_to_geometry() {
        let points = vec![
            Point::from_latlng(52.52, 13.405),
            Point::from_latlng(52.53, 13.415),
        ];
        let geom = linestring_to_geometry(&points);

        assert_eq!(geom["type"], "LineString");
        assert_eq!(geom["coordinates"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_feature_with_properties() {
        let point = Point::from_latlng(52.52, 13.405);
        let feature = Feature::point(&point)
            .with_str("name", "Test Point")
            .with_num("id", 123)
            .with_bool("active", true);

        assert_eq!(feature.properties.get("name").unwrap(), "Test Point");
        assert_eq!(feature.properties.get("id").unwrap(), &json!(123));
        assert_eq!(feature.properties.get("active").unwrap(), &json!(true));
    }

    #[test]
    fn test_feature_collection_with_metadata() {
        let collection = FeatureCollection::new()
            .with_metadata("version", json!("1.0"))
            .with_metadata("timestamp", json!(1234567890));

        assert_eq!(collection.metadata.get("version").unwrap(), "1.0");
    }

    #[test]
    fn test_geojson_writer() {
        use crate::ways::Ways;

        let mut ways = Ways::new();

        // Add nodes
        let pos1 = Point::from_latlng(52.52, 13.405);
        let pos2 = Point::from_latlng(52.53, 13.415);

        ways.add_node(OsmNodeIdx(1), pos1, crate::ways::NodeProperties::default());
        ways.add_node(OsmNodeIdx(2), pos2, crate::ways::NodeProperties::default());

        // Add way
        ways.add_way(
            OsmWayIdx(10),
            vec![NodeIdx::new(0), NodeIdx::new(1)],
            crate::ways::WayProperties::default(),
        );

        let mut writer = GeoJsonWriter::new(&ways);
        writer.write_way(WayIdx::new(0));
        writer.write_nodes();

        let collection = writer.to_feature_collection();

        // Should have edges + geometry + nodes
        assert!(!collection.features.is_empty());
    }

    #[test]
    fn test_geojson_serialization() {
        let point = Point::from_latlng(52.52, 13.405);
        let feature = Feature::point(&point).with_str("type", "test");

        let geojson = feature.to_geojson();

        // Should be valid JSON
        assert!(serde_json::from_str::<Value>(&geojson).is_ok());
    }

    #[test]
    fn test_coords_to_linestring() {
        let coords = vec![(52.52, 13.405), (52.53, 13.415)];
        let geom = coords_to_linestring(&coords);

        assert_eq!(geom["type"], "LineString");
        assert_eq!(geom["coordinates"].as_array().unwrap().len(), 2);
    }
}
