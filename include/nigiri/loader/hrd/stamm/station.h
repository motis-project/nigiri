#pragma once

#include <string_view>

#include "geo/latlng.h"

#include "nigiri/loader/hrd/eva_number.h"
#include "nigiri/loader/hrd/parser_config.h"
#include "nigiri/loader/hrd/stamm/timezone.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::loader::hrd {

struct stamm;

struct hrd_location {
  location_idx_t idx_;
  eva_number id_;
  std::string name_;
  geo::latlng pos_;
  vector<eva_number> children_;
  hash_set<eva_number> equivalent_;
  hash_map<eva_number, u8_minutes> footpaths_out_;
};

using location_map_t = hash_map<eva_number, hrd_location>;

location_map_t parse_stations(config const&,
                              source_idx_t,
                              timetable&,
                              stamm&,
                              std::string_view station_names_file,
                              std::string_view station_coordinates_file,
                              std::string_view station_metabhf_file);

}  // namespace nigiri::loader::hrd
