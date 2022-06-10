#pragma once

#include <string_view>

#include "geo/latlng.h"

#include "nigiri/loader/hrd/eva_number.h"
#include "nigiri/loader/hrd/parser_config.h"
#include "nigiri/types.h"

namespace nigiri::loader::hrd {

struct hrd_location {
  eva_number id_;
  string name_;
  geo::latlng pos_;
  location_type type_;
  vector<eva_number> children_;
  vector<eva_number> equivalent_;
};

hash_map<eva_number, hrd_location> parse_stations(
    config const&,
    std::string_view station_names_file,
    std::string_view station_coordinates_file,
    std::string_view station_metabhf_file);

}  // namespace nigiri::loader::hrd
