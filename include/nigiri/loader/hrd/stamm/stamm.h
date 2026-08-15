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
#include "nigiri/loader/hrd/stamm/transfer_times.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::loader::hrd {

using direction_info_t =
    variant<utl::cstr /* custom string */, eva_number /* eva number */>;

struct stamm {
  stamm(config const&, timetable&, dir const&, source_idx_t);
  stamm(timetable&, timezone_map_t&&, source_idx_t);

  interval<std::chrono::sys_days> get_date_range() const;
  location_idx_t resolve_location(eva_number) const;
  category const* resolve_category(utl::cstr) const;
  translation_idx_t resolve_direction(direction_info_t const&);
  bitfield resolve_bitfield(unsigned) const;
  provider_idx_t resolve_provider(utl::cstr);
  attribute_idx_t resolve_attribute(utl::cstr) const;
  std::pair<timezone_idx_t, tz_offsets> const& get_tz(eva_number) const;
  location_idx_t resolve_track(track_rule_key const&,
                               minutes_after_midnight_t,
                               day_idx_t) const;
  trip_line_idx_t resolve_line(std::string_view s);

  // transfer time rules (UMSTEIGV/UMSTEIGL/UMSTEIGZ)
  bool has_transfer_rules() const {
    return transfer_times_.has_pair_rules();
  }
  void scan_transfer_events(config const&, std::string_view fplan_content);
  void build_transfer_groups();
  location_idx_t resolve_transfer_group(location_idx_t const base,
                                        stop_attrs const& attrs,
                                        std::size_t const day) const {
    return transfer_groups_.resolve(base, attrs, day);
  }

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
  transfer_times transfer_times_;
  transfer_groups transfer_groups_;
  timezone_map_t timezones_;
  interval<std::chrono::sys_days> date_range_;
  timetable& tt_;
  source_idx_t src_;

  hash_map<string, translation_idx_t> string_directions_;
  hash_map<string, trip_line_idx_t> lines_;
};

}  // namespace nigiri::loader::hrd
