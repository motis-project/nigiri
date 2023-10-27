#pragma once

#include "nigiri/timetable.h"
#include <pugixml.hpp>
namespace nigiri::loader::netex {
// This file parses the TimetableFrame & registers the routes in the timetable

void parse_service_journeys(timetable& tt, const pugi::xml_document& doc);
}  // namespace nigiri::loader::netex