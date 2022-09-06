#pragma once

#include "nigiri/loader/dir.h"
#include "nigiri/loader/hrd/eva_number.h"
#include "nigiri/loader/hrd/parser_config.h"
#include "nigiri/loader/hrd/stamm/attribute.h"
#include "nigiri/loader/hrd/stamm/bitfield.h"
#include "nigiri/loader/hrd/stamm/category.h"
#include "nigiri/loader/hrd/stamm/direction.h"
#include "nigiri/loader/hrd/stamm/provider.h"
#include "nigiri/loader/hrd/stamm/station.h"
#include "nigiri/loader/hrd/stamm/timezone.h"
#include "nigiri/loader/hrd/stamm/track.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::loader::hrd {

using direction_info_t =
    variant<utl::cstr /* custom string */, eva_number /* eva number */>;

struct stamm {
  stamm(config const&, timetable&, dir const&);
  stamm(timetable&, timezone_map_t&&);

  location_idx_t resolve_location(eva_number) const;
  category const* resolve_category(utl::cstr) const;
  trip_direction_idx_t resolve_direction(direction_info_t const&);
  bitfield resolve_bitfield(unsigned) const;
  provider_idx_t resolve_provider(utl::cstr);
  attribute_idx_t resolve_attribute(utl::cstr) const;
  std::pair<timezone_idx_t, tz_offsets> const& get_tz(eva_number) const;
  location_idx_t resolve_track(track_rule_key const&, day_idx_t) const;

private:
  location_map_t locations_;
  category_map_t categories_;
  provider_map_t providers_;
  attribute_map_t attributes_;
  direction_map_t directions_;
  bitfield_map_t bitfields_;
  track_rule_map_t track_rules_;
  track_location_map_t track_locations_;
  timezone_map_t timezones_;
  timetable& tt_;

  hash_map<string, trip_direction_idx_t> string_directions_;
  hash_map<eva_number, trip_direction_idx_t> eva_directions_;
  hash_map<string, line_idx_t> lines_;
};

}  // namespace nigiri::loader::hrd