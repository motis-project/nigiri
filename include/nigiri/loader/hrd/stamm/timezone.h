#pragma once

#include <chrono>
#include <string_view>

#include "nigiri/loader/hrd/eva_number.h"
#include "nigiri/loader/hrd/parser_config.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::loader::hrd {

using timezone_map_t =
    std::map<eva_number, std::pair<timezone_idx_t, tz_offsets>>;

inline bool is_local_time_in_season(
    tz_offsets::season const& season,
    unixtime_t const day,
    duration_t const local_minutes_after_midnight) {
  using std::chrono::sys_days;
  auto const local_time = day + local_minutes_after_midnight;
  return season.begin_ + season.season_begin_mam_ <= local_time &&
         local_time < season.end_ + season.season_end_mam_;
}

inline std::tuple<duration_t, duration_t, bool> local_mam_to_utc_mam(
    tz_offsets const& tz,
    unixtime_t const day,
    duration_t const local_mam,
    bool const first = false) {
  auto const active_season_it =
      utl::find_if(tz.seasons_, [&](tz_offsets::season const& season) {
        return is_local_time_in_season(season, day, local_mam);
      });
  auto const tz_offset = active_season_it == end(tz.seasons_)
                             ? tz.offset_
                             : active_season_it->offset_;
  auto utc_mam = local_mam - tz_offset;

  auto const local_time = day + local_mam;

  auto valid = true;
  auto const is_season = active_season_it != end(tz.seasons_);
  if (is_season) {
    auto const local_season_begin = active_season_it->begin_ +
                                    active_season_it->season_begin_mam_ +
                                    (active_season_it->offset_ - tz.offset_);
    valid = local_time > local_season_begin;
    if (!valid) {
      utc_mam += 1_hours;
    }
  }

  if (first && utc_mam >= 1_days) {
    return {duration_t{utc_mam.count() % 1440},
            (utc_mam.count() / 1440) * 1_days, valid};
  } else if (utc_mam < duration_t{0}) {
    return {utc_mam + 1_days, -1_days, valid};
  } else {
    return {utc_mam, 0_days, valid};
  }
}

timezone_map_t parse_timezones(config const&, timetable&, std::string_view);

}  // namespace nigiri::loader::hrd
