use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::error::TransitResult;

/// Top-level application configuration.
///
/// Maps 1:1 to the motis config structure, minus `tiles` (external tile server)
/// and with our own `server` section for the GraphQL endpoint.
#[derive(Debug, Default, Deserialize, Clone)]
pub struct Config {
    #[serde(default)]
    pub server: ServerConfig,
    pub osm: Option<PathBuf>,
    pub timetable: Option<TimetableConfig>,
    pub gbfs: Option<GbfsConfig>,
    pub prima: Option<PrimaConfig>,
    pub elevators: Option<ElevatorConfig>,
    pub street_routing: Option<StreetRoutingConfig>,
    pub limits: Option<LimitsConfig>,
    pub logging: Option<LoggingConfig>,
}

impl Config {
    /// Load configuration from a YAML file.
    pub fn load(path: impl AsRef<Path>) -> TransitResult<Self> {
        let contents = std::fs::read_to_string(path.as_ref())?;
        let config: Self = serde_yaml::from_str(&contents)?;
        Ok(config)
    }
}

// --- Server ---

#[derive(Debug, Deserialize, Clone)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    pub n_threads: Option<usize>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            n_threads: None,
        }
    }
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}
fn default_port() -> u16 {
    8080
}

// --- Timetable ---

#[derive(Debug, Deserialize, Clone)]
pub struct TimetableConfig {
    #[serde(default = "default_first_day")]
    pub first_day: String,
    #[serde(default = "default_num_days")]
    pub num_days: u16,
    #[serde(default)]
    pub with_shapes: bool,
    #[serde(default)]
    pub adjust_footpaths: bool,
    #[serde(default)]
    pub merge_dupes_intra_src: bool,
    #[serde(default)]
    pub merge_dupes_inter_src: bool,
    #[serde(default = "default_link_stop_distance")]
    pub link_stop_distance: u32,
    #[serde(default = "default_update_interval")]
    pub update_interval: u32,
    #[serde(default = "default_http_timeout")]
    pub http_timeout: u32,
    #[serde(default = "default_max_footpath_length")]
    pub max_footpath_length: u16,
    #[serde(default = "default_max_matching_distance")]
    pub max_matching_distance: f64,
    #[serde(default)]
    pub datasets: HashMap<String, DatasetConfig>,
    pub route_shapes: Option<RouteShapesConfig>,
}

fn default_first_day() -> String {
    "TODAY".to_string()
}
fn default_num_days() -> u16 {
    365
}
fn default_link_stop_distance() -> u32 {
    100
}
fn default_update_interval() -> u32 {
    60
}
fn default_http_timeout() -> u32 {
    30
}
fn default_max_footpath_length() -> u16 {
    15
}
fn default_max_matching_distance() -> f64 {
    25.0
}

#[derive(Debug, Deserialize, Clone)]
pub struct DatasetConfig {
    pub path: String,
    pub script: Option<String>,
    #[serde(default)]
    pub default_bikes_allowed: bool,
    #[serde(default)]
    pub default_cars_allowed: bool,
    #[serde(default)]
    pub extend_calendar: bool,
    pub rt: Option<Vec<RtFeedConfig>>,
    pub default_timezone: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct RtFeedConfig {
    pub url: String,
    pub headers: Option<HashMap<String, String>>,
    pub protocol: RtProtocol,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum RtProtocol {
    Gtfsrt,
    Auser,
    Siri,
    #[serde(rename = "siri_json")]
    SiriJson,
}

#[derive(Debug, Deserialize, Clone)]
pub struct RouteShapesConfig {
    #[serde(default = "default_shapes_mode")]
    pub mode: String,
    pub cache_dir: Option<PathBuf>,
    #[serde(default)]
    pub debug: bool,
}

fn default_shapes_mode() -> String {
    "osrm".to_string()
}

// --- GBFS ---

#[derive(Debug, Deserialize, Clone)]
pub struct GbfsConfig {
    #[serde(default)]
    pub feeds: HashMap<String, GbfsFeedConfig>,
    #[serde(default)]
    pub groups: Vec<Vec<String>>,
    #[serde(default)]
    pub restrictions: HashMap<String, String>,
    #[serde(default = "default_gbfs_ttl")]
    pub ttl: u32,
    pub proxy: Option<String>,
    #[serde(default = "default_gbfs_cache_size")]
    pub cache_size: u32,
}

fn default_gbfs_ttl() -> u32 {
    60
}
fn default_gbfs_cache_size() -> u32 {
    1024
}

#[derive(Debug, Deserialize, Clone)]
pub struct GbfsFeedConfig {
    pub url: String,
    pub r#type: Option<String>,
    pub headers: Option<HashMap<String, String>>,
}

// --- Prima (on-demand mobility) ---

#[derive(Debug, Deserialize, Clone)]
pub struct PrimaConfig {
    pub url: String,
    pub bounds: Option<Vec<[f64; 2]>>,
    pub ride_sharing_bounds: Option<Vec<[f64; 2]>>,
}

// --- Elevators ---

#[derive(Debug, Deserialize, Clone)]
pub struct ElevatorConfig {
    pub url: Option<String>,
    #[serde(default)]
    pub init: bool,
    #[serde(default)]
    pub osm_mapping: bool,
    #[serde(default = "default_elevator_http_timeout")]
    pub http_timeout: u32,
    pub headers: Option<HashMap<String, String>>,
}

fn default_elevator_http_timeout() -> u32 {
    30
}

// --- Street Routing ---

#[derive(Debug, Deserialize, Clone)]
pub struct StreetRoutingConfig {
    pub elevation_data_dir: Option<PathBuf>,
}

// --- Limits ---

#[derive(Debug, Deserialize, Clone)]
pub struct LimitsConfig {
    #[serde(default = "default_stoptimes_max")]
    pub stoptimes_max_results: u32,
    #[serde(default = "default_plan_max")]
    pub plan_max_results: u32,
    #[serde(default = "default_plan_max_window")]
    pub plan_max_search_window_minutes: u32,
    #[serde(default = "default_stops_max")]
    pub stops_max_results: u32,
    #[serde(default = "default_onetomany_max")]
    pub onetomany_max_many: u32,
    #[serde(default = "default_onetoall_max")]
    pub onetoall_max_results: u32,
    #[serde(default = "default_onetoall_travel")]
    pub onetoall_max_travel_minutes: u32,
    #[serde(default = "default_routing_timeout")]
    pub routing_max_timeout_seconds: u32,
}

fn default_stoptimes_max() -> u32 {
    256
}
fn default_plan_max() -> u32 {
    256
}
fn default_plan_max_window() -> u32 {
    5760
}
fn default_stops_max() -> u32 {
    2048
}
fn default_onetomany_max() -> u32 {
    128
}
fn default_onetoall_max() -> u32 {
    65535
}
fn default_onetoall_travel() -> u32 {
    90
}
fn default_routing_timeout() -> u32 {
    90
}

// --- Logging ---

#[derive(Debug, Deserialize, Clone)]
pub struct LoggingConfig {
    #[serde(default = "default_log_level")]
    pub log_level: String,
}

fn default_log_level() -> String {
    "info".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_minimal_config() {
        let yaml = r#"
server:
  port: 3000
"#;
        let config: Config = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.server.port, 3000);
        assert_eq!(config.server.host, "0.0.0.0");
        assert!(config.timetable.is_none());
    }

    #[test]
    fn parse_full_config() {
        let yaml = r#"
server:
  host: "127.0.0.1"
  port: 9090
  n_threads: 4
timetable:
  first_day: "2024-01-01"
  num_days: 30
  with_shapes: true
  adjust_footpaths: false
  merge_dupes_intra_src: true
  merge_dupes_inter_src: false
  link_stop_distance: 200
  update_interval: 120
  http_timeout: 60
  max_footpath_length: 20
  max_matching_distance: 50.0
  datasets:
    de:
      path: "/data/gtfs/de"
      default_bikes_allowed: true
      default_cars_allowed: false
      extend_calendar: true
      rt:
        - url: "https://example.com/gtfsrt"
          protocol: gtfsrt
limits:
  stoptimes_max_results: 128
  plan_max_results: 64
  routing_max_timeout_seconds: 30
logging:
  log_level: debug
"#;
        let config: Config = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.server.n_threads, Some(4));

        let tt = config.timetable.unwrap();
        assert_eq!(tt.first_day, "2024-01-01");
        assert_eq!(tt.num_days, 30);
        assert!(tt.with_shapes);
        assert_eq!(tt.link_stop_distance, 200);
        assert_eq!(tt.datasets.len(), 1);

        let de = &tt.datasets["de"];
        assert!(de.default_bikes_allowed);
        assert_eq!(de.rt.as_ref().unwrap().len(), 1);
        assert_eq!(de.rt.as_ref().unwrap()[0].protocol, RtProtocol::Gtfsrt);

        let limits = config.limits.unwrap();
        assert_eq!(limits.stoptimes_max_results, 128);
        assert_eq!(limits.routing_max_timeout_seconds, 30);
        // Defaults should fill in
        assert_eq!(limits.stops_max_results, 2048);

        let logging = config.logging.unwrap();
        assert_eq!(logging.log_level, "debug");
    }

    #[test]
    fn default_config() {
        let config = Config::default();
        assert_eq!(config.server.host, "0.0.0.0");
        assert_eq!(config.server.port, 8080);
    }
}
