#pragma once

#include "pugixml.hpp"

#include "utl/parser/cstr.h"

#include "nigiri/timetable.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::loader::netex {

int resolve_day_idx(timetable const& tt, utl::cstr const& date);

void read_operating_periods(
    timetable const& tt,
    pugi::xml_document& doc,
    hash_map<std::string_view, bitfield>& operating_periods);

void read_service_calendar(
    timetable const& tt,
    pugi::xml_document& doc,
    hash_map<std::string_view, bitfield>& calendar,
    hash_map<std::string_view, bitfield>& operating_period);

}  // namespace nigiri::loader::netex