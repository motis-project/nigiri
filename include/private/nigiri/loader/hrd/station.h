#pragma once

#include <string_view>

#include "geo/latlng.h"

#include "nigiri/loader/hrd/eva_number.h"
#include "nigiri/loader/hrd/parser_config.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::loader::hrd {

struct hrd_location {
  location_idx_t idx_;
  eva_number id_;
  string name_;
  geo::latlng pos_;
  location_type type_;
  vector<eva_number> children_;
  hash_set<eva_number> equivalent_;
  hash_map<eva_number, duration_t> footpaths_out_;
};

using location_map_t = hash_map<eva_number, hrd_location>;

location_map_t parse_stations(config const& c,
                              source_idx_t,
                              timetable& tt,
                              std::string_view station_names_file,
                              std::string_view station_coordinates_file,
                              std::string_view station_metabhf_file);

}  // namespace nigiri::loader::hrd
