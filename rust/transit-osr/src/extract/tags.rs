//! Translation of osr/include/osr/extract/tags.h
//!
//! OSM tag interpretation and access rule evaluation.
//! Determines accessibility for different transport modes based on OSM tags.
//!
//! This is a 1:1 translation of the C++ tags parser.

use crate::types::*;

/// OSM object type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OsmObjType {
    Way,
    Node,
}

/// Access override type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Override {
    #[default]
    None,
    Whitelist, // Explicitly allowed
    Blacklist, // Explicitly forbidden
}

/// Parsed OSM tags with interpreted access rules
#[derive(Debug, Clone, Default)]
pub struct Tags {
    // Route relation
    pub is_route: bool,

    // Oneway restrictions
    pub oneway: bool,
    pub not_oneway_bike: bool,

    // Tag values
    pub barrier: String,
    pub motorcar: String,
    pub motor_vehicle: String,
    pub foot: String,
    pub bicycle: String,
    pub highway: String,
    pub cycleway: String,
    pub max_speed: String,
    pub name: String,
    pub ref_: String,
    pub access_conditional_no: String,

    // Flags
    pub sidewalk_separate: bool,
    pub is_destination: bool,
    pub landuse: bool,
    pub is_construction: bool,
    pub is_platform: bool,
    pub is_ramp: bool,
    pub is_elevator: bool,
    pub is_entrance: bool,
    pub is_parking: bool,
    pub is_incline_down: bool,
    pub toll: bool,

    // Access overrides
    pub vehicle: Override,
    pub access: Override,

    // Level information
    pub has_level: bool,
    pub level_bits: LevelBits,
}

impl Tags {
    /// Parse OSM tags from key-value pairs
    pub fn from_osm_tags<'a, I>(tags: I) -> Self
    where
        I: IntoIterator<Item = (&'a str, &'a str)>,
    {
        let mut t = Tags::default();
        let mut circular = false;
        let mut oneway_defined = false;

        for (key, value) in tags {
            t.process_tag_with_flags(key, value, &mut circular, &mut oneway_defined);
        }

        // Circular junctions default to oneway unless explicitly tagged otherwise
        if circular && !oneway_defined {
            t.oneway = true;
        }

        t
    }

    /// Check if accessible by foot (convenience wrapper)
    pub fn is_accessible_foot(&self) -> bool {
        is_accessible::<FootProfile>(self, OsmObjType::Way)
    }

    /// Check if accessible by bike (convenience wrapper)
    pub fn is_accessible_bike(&self) -> bool {
        is_accessible::<BikeProfile>(self, OsmObjType::Way)
    }

    /// Check if accessible by car (convenience wrapper)
    pub fn is_accessible_car(&self) -> bool {
        is_accessible::<CarProfile>(self, OsmObjType::Way)
    }

    /// Process a single tag (public for testing)
    fn _process_tag(&mut self, key: &str, value: &str) {
        let mut circular = false;
        let mut oneway_defined = false;
        self.process_tag_with_flags(key, value, &mut circular, &mut oneway_defined);
        // Note: circular/oneway_defined logic not applied in direct calls
    }

    /// Process a single tag with circular/oneway tracking
    fn process_tag_with_flags(
        &mut self,
        key: &str,
        value: &str,
        circular: &mut bool,
        oneway_defined: &mut bool,
    ) {
        match key {
            "ramp" => self.is_ramp |= value != "no",
            "type" => self.is_route |= value == "route",
            "parking" => self.is_parking = true,
            "amenity" => {
                self.is_parking |= value == "parking" || value == "parking_entrance";
            }
            "building" => {
                self.is_parking |= value == "parking";
                self.landuse = true;
            }
            "landuse" => self.landuse = true,
            "railway" => {
                self.landuse |= value == "station_area";
            }
            "oneway" => {
                *oneway_defined = true;
                self.oneway |= value == "yes";
            }
            "junction" => {
                self.oneway |= value == "roundabout";
                *circular |= value == "circular";
            }
            "oneway:bicycle" => {
                self.not_oneway_bike = value == "no";
            }
            "motor_vehicle" | "motor_vehicle:forward" => {
                self.motor_vehicle = value.to_string();
                self.is_destination |= value == "destination";
            }
            "foot" => self.foot = value.to_string(),
            "bicycle" => self.bicycle = value.to_string(),
            "highway" => {
                self.highway = value.to_string();
                if value == "elevator" {
                    self.is_elevator = true;
                }
                if value == "bus_stop" {
                    self.is_platform = true;
                }
            }
            "indoor:level" | "level" => {
                self.has_level = true;
                self.add_levels(value);
            }
            "name" => self.name = value.to_string(),
            "ref" => self.ref_ = value.to_string(),
            "entrance" => self.is_entrance = true,
            "sidewalk" | "sidewalk:both" | "sidewalk:left" | "sidewalk:right" => {
                if value == "separate" {
                    self.sidewalk_separate = true;
                }
            }
            "cycleway" => self.cycleway = value.to_string(),
            "motorcar" => {
                self.motorcar = value.to_string();
                self.is_destination |= value == "destination";
            }
            "barrier" => self.barrier = value.to_string(),
            "platform_edge" => self.is_platform = true,
            "public_transport" => {
                if value == "platform" || value == "stop_position" {
                    self.is_platform = true;
                }
            }
            "construction" => self.is_construction = true,
            "vehicle" => {
                self.vehicle = match value {
                    "private" | "delivery" | "no" => Override::Blacklist,
                    "destination" => {
                        self.is_destination = true;
                        Override::Whitelist
                    }
                    "permissive" | "yes" => Override::Whitelist,
                    _ => Override::None,
                };
            }
            "access" => {
                self.access = match value {
                    "no" | "agricultural" | "forestry" | "emergency" | "psv" | "private"
                    | "delivery" => Override::Blacklist,
                    "designated" | "dismount" | "customers" | "permissive" | "yes" => {
                        Override::Whitelist
                    }
                    _ => Override::None,
                };
            }
            "access:conditional" => {
                // Extract "no @ (time_range)" pattern
                if value.starts_with("no @ (") && value.ends_with(')') {
                    let start = "no @ (".len();
                    let end = value.len() - 1;
                    self.access_conditional_no = value[start..end].to_string();
                }
            }
            "maxspeed" => self.max_speed = value.to_string(),
            "toll" => self.toll = value == "yes",
            "incline" => {
                self.is_incline_down = value == "down" || value.starts_with('-');
            }
            _ => {}
        }
    }

    /// Parse level values (semicolon-separated floats)
    fn add_levels(&mut self, value: &str) {
        for level_str in value.split(';') {
            if let Ok(level_f) = level_str.trim().parse::<f32>() {
                let clamped = level_f.clamp(K_MIN_LEVEL, K_MAX_LEVEL);
                let lvl = Level::from_float(clamped);
                let bit_idx = lvl.to_idx() as u64;
                self.level_bits |= 1u64 << bit_idx;
            }
        }
    }

    /// Check if this is a valid platform (not under construction)
    pub fn is_platform(&self) -> bool {
        self.is_platform && !self.is_construction
    }
}

// ============================================================================
// Access Rule Evaluation
// ============================================================================

/// Check if object is accessible for a given profile
pub fn is_accessible<P: Profile>(tags: &Tags, obj_type: OsmObjType) -> bool {
    let override_ = P::access_override(tags);
    override_ == Override::Whitelist
        || (P::default_access(tags, obj_type) && override_ != Override::Blacklist)
}

/// Profile trait for access rules
pub trait Profile {
    fn access_override(tags: &Tags) -> Override;
    fn default_access(tags: &Tags, obj_type: OsmObjType) -> bool;
}

/// Foot/pedestrian access profile
pub struct FootProfile;

impl Profile for FootProfile {
    fn access_override(tags: &Tags) -> Override {
        if tags.is_route || tags.sidewalk_separate {
            return Override::Blacklist;
        }

        // Barrier checks
        match tags.barrier.as_str() {
            "yes" | "wall" | "fence" => return Override::Blacklist,
            _ => {}
        }

        // Foot-specific tags
        match tags.foot.as_str() {
            "no" | "private" | "use_sidepath" => return Override::Blacklist,
            "yes" | "permissive" | "designated" => return Override::Whitelist,
            _ => {}
        }

        // Platform/parking are accessible
        if tags.is_platform || tags.is_parking {
            return Override::Whitelist;
        }

        // General access restriction
        if tags.access == Override::Blacklist {
            return Override::Blacklist;
        }

        Override::None
    }

    fn default_access(tags: &Tags, obj_type: OsmObjType) -> bool {
        match obj_type {
            OsmObjType::Way => {
                if tags.is_elevator || tags.is_parking {
                    return true;
                }
                matches!(
                    tags.highway.as_str(),
                    "primary"
                        | "primary_link"
                        | "secondary"
                        | "secondary_link"
                        | "tertiary"
                        | "tertiary_link"
                        | "unclassified"
                        | "residential"
                        | "road"
                        | "living_street"
                        | "service"
                        | "track"
                        | "path"
                        | "steps"
                        | "pedestrian"
                        | "platform"
                        | "corridor"
                        | "footway"
                        | "pier"
                )
            }
            OsmObjType::Node => true,
        }
    }
}

/// Bicycle access profile
pub struct BikeProfile;

impl Profile for BikeProfile {
    fn access_override(tags: &Tags) -> Override {
        if tags.is_route {
            return Override::Blacklist;
        }

        // Barrier checks
        match tags.barrier.as_str() {
            "yes" | "wall" | "fence" => return Override::Blacklist,
            _ => {}
        }

        // Bicycle-specific tags
        match tags.bicycle.as_str() {
            "no" | "private" | "optional_sidepath" | "use_sidepath" => return Override::Blacklist,
            "yes" | "permissive" | "designated" => return Override::Whitelist,
            _ => {}
        }

        // General access restriction
        if tags.access == Override::Blacklist {
            return Override::Blacklist;
        }

        tags.vehicle
    }

    fn default_access(tags: &Tags, obj_type: OsmObjType) -> bool {
        match obj_type {
            OsmObjType::Way => matches!(
                tags.highway.as_str(),
                "cycleway"
                    | "primary"
                    | "primary_link"
                    | "secondary"
                    | "secondary_link"
                    | "tertiary"
                    | "tertiary_link"
                    | "residential"
                    | "unclassified"
                    | "living_street"
                    | "road"
                    | "service"
                    | "track"
                    | "path"
            ),
            OsmObjType::Node => true,
        }
    }
}

/// Car/motor vehicle access profile
pub struct CarProfile;

impl Profile for CarProfile {
    fn access_override(tags: &Tags) -> Override {
        if tags.access == Override::Blacklist || tags.is_route {
            return Override::Blacklist;
        }

        // Barrier checks (some barriers allow cars)
        if !tags.barrier.is_empty() {
            match tags.barrier.as_str() {
                "cattle_grid" | "border_control" | "toll_booth" | "sally_port" | "gate"
                | "lift_gate" | "no" | "entrance" | "height_restrictor" | "arch" => {}
                _ => return Override::Blacklist,
            }
        }

        // Check motor_vehicle tag
        let mv_override = get_tag_override(&tags.motor_vehicle);
        if mv_override != Override::None {
            return mv_override;
        }

        // Check motorcar tag
        let mc_override = get_tag_override(&tags.motorcar);
        if mc_override != Override::None {
            return mc_override;
        }

        // Parking is accessible
        if tags.is_parking {
            return Override::Whitelist;
        }

        tags.vehicle
    }

    fn default_access(tags: &Tags, obj_type: OsmObjType) -> bool {
        match obj_type {
            OsmObjType::Way => matches!(
                tags.highway.as_str(),
                "motorway"
                    | "motorway_link"
                    | "trunk"
                    | "trunk_link"
                    | "primary"
                    | "primary_link"
                    | "secondary"
                    | "secondary_link"
                    | "tertiary"
                    | "tertiary_link"
                    | "residential"
                    | "living_street"
                    | "unclassified"
                    | "service"
            ),
            OsmObjType::Node => true,
        }
    }
}

/// Parse access override from tag value
fn get_tag_override(tag_value: &str) -> Override {
    match tag_value {
        "private"
        | "optional_sidepath"
        | "agricultural"
        | "forestry"
        | "agricultural;forestry"
        | "permit"
        | "customers"
        | "delivery"
        | "no" => Override::Blacklist,
        "designated" | "permissive" | "yes" => Override::Whitelist,
        _ => Override::None,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tags_parsing() {
        let tags = Tags::from_osm_tags([
            ("highway", "residential"),
            ("name", "Main Street"),
            ("oneway", "yes"),
        ]);

        assert_eq!(tags.highway, "residential");
        assert_eq!(tags.name, "Main Street");
        assert!(tags.oneway);
    }

    #[test]
    fn test_level_parsing() {
        let tags = Tags::from_osm_tags([("level", "0;1;2")]);

        assert!(tags.has_level);
        assert_ne!(tags.level_bits, 0);
    }

    #[test]
    fn test_platform_detection() {
        let mut tags = Tags::default();
        tags.is_platform = true;
        tags.is_construction = false;
        assert!(tags.is_platform());

        tags.is_construction = true;
        assert!(!tags.is_platform());
    }

    #[test]
    fn test_foot_access_residential() {
        let tags = Tags::from_osm_tags([("highway", "residential")]);
        assert!(is_accessible::<FootProfile>(&tags, OsmObjType::Way));
    }

    #[test]
    fn test_foot_access_motorway() {
        let tags = Tags::from_osm_tags([("highway", "motorway")]);
        assert!(!is_accessible::<FootProfile>(&tags, OsmObjType::Way));
    }

    #[test]
    fn test_foot_access_footway() {
        let tags = Tags::from_osm_tags([("highway", "footway")]);
        assert!(is_accessible::<FootProfile>(&tags, OsmObjType::Way));
    }

    #[test]
    fn test_bike_access_cycleway() {
        let tags = Tags::from_osm_tags([("highway", "cycleway")]);
        assert!(is_accessible::<BikeProfile>(&tags, OsmObjType::Way));
    }

    #[test]
    fn test_bike_access_forbidden() {
        let tags = Tags::from_osm_tags([("highway", "residential"), ("bicycle", "no")]);
        assert!(!is_accessible::<BikeProfile>(&tags, OsmObjType::Way));
    }

    #[test]
    fn test_car_access_residential() {
        let tags = Tags::from_osm_tags([("highway", "residential")]);
        assert!(is_accessible::<CarProfile>(&tags, OsmObjType::Way));
    }

    #[test]
    fn test_car_access_footway() {
        let tags = Tags::from_osm_tags([("highway", "footway")]);
        assert!(!is_accessible::<CarProfile>(&tags, OsmObjType::Way));
    }

    #[test]
    fn test_access_override_whitelist() {
        let tags = Tags::from_osm_tags([("highway", "path"), ("foot", "yes")]);
        assert!(is_accessible::<FootProfile>(&tags, OsmObjType::Way));
    }

    #[test]
    fn test_access_override_blacklist() {
        let tags = Tags::from_osm_tags([("highway", "residential"), ("access", "private")]);
        assert!(!is_accessible::<FootProfile>(&tags, OsmObjType::Way));
    }

    #[test]
    fn test_vehicle_override() {
        let tags = Tags::from_osm_tags([("highway", "track"), ("vehicle", "no")]);
        assert!(!is_accessible::<BikeProfile>(&tags, OsmObjType::Way));
    }

    #[test]
    fn test_barrier_wall() {
        let tags = Tags::from_osm_tags([("barrier", "wall")]);
        assert!(!is_accessible::<FootProfile>(&tags, OsmObjType::Node));
    }

    #[test]
    fn test_oneway_detection() {
        let tags1 = Tags::from_osm_tags([("oneway", "yes")]);
        assert!(tags1.oneway);

        let tags2 = Tags::from_osm_tags([("junction", "roundabout")]);
        assert!(tags2.oneway);

        // Circular junctions default to oneway
        let tags3 = Tags::from_osm_tags([("junction", "circular")]);
        assert!(tags3.oneway);

        // But circular with explicit oneway=no should NOT be oneway
        let tags4 = Tags::from_osm_tags([("junction", "circular"), ("oneway", "no")]);
        assert!(!tags4.oneway);
    }

    #[test]
    fn test_incline_down() {
        let tags1 = Tags::from_osm_tags([("incline", "down")]);
        assert!(tags1.is_incline_down);

        let tags2 = Tags::from_osm_tags([("incline", "-5%")]);
        assert!(tags2.is_incline_down);
    }
}
