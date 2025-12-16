#pragma once

#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/noon_offsets.h"
#include "nigiri/loader/gtfs/services.h"
#include "nigiri/loader/gtfs/stop.h"
#include "nigiri/loader/gtfs/trip.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::loader::gtfs {

using booking_rules_t = hash_map<std::string, booking_rule_idx_t>;
using location_groups_t = hash_map<std::string, location_group_idx_t>;
using flex_areas_t = hash_map<std::string, flex_area_idx_t>;
using stop_seq_map_t = hash_map<std::vector<flex_stop_t>, flex_stop_seq_idx_t>;

flex_areas_t parse_flex_areas(timetable&,
                              translator&,
                              source_idx_t,
                              std::string_view file_content);

void parse_location_group_stops(timetable&,
                                std::string_view file_content,
                                location_groups_t const&,
                                stops_map_t const&);

location_groups_t parse_location_groups(timetable&,
                                        translator&,
                                        std::string_view file_content);

booking_rules_t parse_booking_rules(timetable&,
                                    translator&,
                                    std::string_view file_content,
                                    traffic_days_t const&,
                                    hash_map<bitfield, bitfield_idx_t>&);

void expand_flex_trip(timetable&,
                      hash_map<bitfield, bitfield_idx_t>&,
                      stop_seq_map_t&,
                      noon_offset_hours_t const& noon_offsets,
                      interval<date::sys_days> const& selection,
                      trip const&);

}  // namespace nigiri::loader::gtfs