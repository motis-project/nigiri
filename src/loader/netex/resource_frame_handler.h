//
// Created by mirko on 10/11/23.
//

#pragma once
#include "nigiri/loader/gtfs/agency.h"
#include "nigiri/loader/gtfs/tz_map.h"
#include "nigiri/timetable.h"
#include <pugixml.hpp>

namespace nigiri::loader::netex {
void processResourceFrame(timetable& t,
                          const pugi::xml_node& frame,
                          gtfs::tz_map& timezones,
                          gtfs::agency_map_t& agencyMap);
}