//
// Created by mirko on 10/11/23.
//

#pragma once
#include "nigiri/loader/gtfs/agency.h"
#include "nigiri/loader/gtfs/tz_map.h"
#include "nigiri/timetable.h"
#include <pugixml.hpp>

namespace nigiri::loader::netex {

struct route_operator {
  std::string longName;
  std::string shortName;
};

// operatorMap maps from operatorID to the short and long name
void read_resource_frame(
    timetable& tt,
    const pugi::xml_document& doc,
    hash_map<std::string_view, provider_idx_t>& operatorMap);
}  // namespace nigiri::loader::netex