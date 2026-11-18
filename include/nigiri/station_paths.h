#pragma once

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string_view>

#include "cista/mmap.h"

#include "geo/latlng.h"

#include "nigiri/footpath_restriction.h"
#include "nigiri/types.h"

namespace nigiri {

using station_path_idx_t =
    cista::strong<std::uint32_t, struct _station_path_idx>;
using station_piece_idx_t =
    cista::strong<std::uint32_t, struct _station_piece_idx>;
using station_node_idx_t =
    cista::strong<std::uint32_t, struct _station_node_idx>;
using station_string_idx_t =
    cista::strong<std::uint32_t, struct _station_string_idx>;
using station_entrance_idx_t =
    cista::strong<std::uint32_t, struct _station_entrance_idx>;
using station_entrance_path_idx_t =
    cista::strong<std::uint32_t, struct _station_entrance_path_idx>;

// Point of a station graph: only kept as polyline point / step boundary.
struct station_node {
  bool has_pos() const { return !std::isnan(pos_.lat_); }
  bool has_level() const { return !std::isnan(level_); }

  geo::latlng pos_;  // NaN if unknown
  float level_;  // NaN if unknown
};

enum class station_piece_mode : std::uint8_t {
  kWalkway,
  kStairs,
  kTravelator,
  kEscalator,
  kElevator,
  kFareGate,
  kExitGate
};

// One edge of a station graph as used by the paths.
struct station_piece {
  station_node_idx_t from_;
  station_node_idx_t to_;
  station_string_idx_t signpost_;  // invalid = none
  std::uint32_t seconds_;  // walking + fixed at the default profile
  footpath_restriction restriction_;  // this edge only
  station_piece_mode mode_;
};

// Route through a station graph behind a restricted footpath: identified by
// its (base) stops and its restriction record.
struct station_path {
  location_idx_t from_;
  location_idx_t to_;
  footpath_restriction restriction_;
  std::uint32_t walk_seconds_;
  std::uint32_t fixed_seconds_;
};

struct station_entrance {
  station_node_idx_t node_;
  station_string_idx_t name_;
};

// Route between a station entrance and a stop: first mile (entrance -> stop)
// or last mile (stop -> entrance).
struct station_entrance_path {
  station_entrance_idx_t entrance_;
  bool is_exit_;  // false: entrance -> l_, true: l_ -> entrance
  location_idx_t l_;  // base location
  footpath_restriction restriction_;
  std::uint32_t walk_seconds_;
  std::uint32_t fixed_seconds_;
};

// Geometry and steps of the footpaths through station graphs. Separate
// mmapped files next to the timetable (not part of tt.bin, not on the GPU);
// always written, empty without station graphs.
struct station_paths {
  station_paths(std::filesystem::path, cista::mmap::protection);

  cista::mmap mm(char const* file);

  // The path for the footpath from -> to (base locations) with this record.
  std::optional<station_path_idx_t> find(location_idx_t from,
                                         location_idx_t to,
                                         footpath_restriction const&) const;

  std::string_view get_string(station_string_idx_t) const;

  // Entrance paths of an entrance: first mile (is_exit = false) or last mile
  // (is_exit = true).
  std::span<station_entrance_path const> entrance_paths(station_entrance_idx_t,
                                                        bool is_exit) const;

  station_entrance_path_idx_t get_idx(station_entrance_path const&) const;

  // Stops that are reached from the street only through entrances: they have
  // entrance paths in both directions and no stop_access=1.
  bool is_entrance_only(location_idx_t) const;

  cista::mmap::protection mode_;
  std::filesystem::path p_;

  mm_vec_map<station_node_idx_t, station_node> nodes_;
  mm_vec_map<station_piece_idx_t, station_piece> pieces_;
  mm_vec_map<station_path_idx_t, station_path> paths_;  // by (from, to)
  mm_vecvec<station_path_idx_t, station_piece_idx_t, std::uint64_t>
      path_pieces_;
  mm_vec_map<station_entrance_idx_t, station_entrance> entrances_;
  // sorted by (entrance, is_exit, location)
  mm_vec_map<station_entrance_path_idx_t, station_entrance_path>
      entrance_paths_;
  mm_vecvec<station_entrance_path_idx_t, station_piece_idx_t, std::uint64_t>
      entrance_path_pieces_;
  mm_vec<location_idx_t> entrance_only_;  // sorted
  mm_vecvec<station_string_idx_t, char, std::uint64_t> strings_;
};

}  // namespace nigiri
