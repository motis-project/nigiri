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

  interval<std::chrono::sys_days> get_date_range() const;
  location_idx_t resolve_location(eva_number) const;
  category const* resolve_category(utl::cstr) const;
  trip_direction_idx_t resolve_direction(direction_info_t const&);
  bitfield resolve_bitfield(unsigned) const;
  provider_idx_t resolve_provider(utl::cstr);
  attribute_idx_t resolve_attribute(utl::cstr) const;
  std::pair<timezone_idx_t, tz_offsets> const& get_tz(eva_number) const;
  location_idx_t resolve_track(track_rule_key const&,
                               minutes_after_midnight_t,
                               day_idx_t) const;
  trip_line_idx_t resolve_line(std::string_view s);

private:
  friend std::uint64_t hash(config const&,
                            dir const&,
                            std::uint64_t const seed);
  static std::vector<file> load_files(config const&, dir const&);

  location_map_t locations_;
  category_map_t categories_;
  provider_map_t providers_;
  attribute_map_t attributes_;
  direction_map_t directions_;
  bitfield_map_t bitfields_;
  tracks tracks_;
  timezone_map_t timezones_;
  interval<std::chrono::sys_days> date_range_;
  timetable& tt_;

  hash_map<string, trip_direction_idx_t> string_directions_;
  hash_map<eva_number, trip_direction_idx_t> eva_directions_;
  hash_map<string, trip_line_idx_t> lines_;
};

}  // namespace nigiri::loader::hrd
