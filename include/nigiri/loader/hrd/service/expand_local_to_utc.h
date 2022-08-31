#pragma once

#include "nigiri/loader/hrd/service.h"
#include "nigiri/loader/hrd/service/ref_service.h"
#include "nigiri/loader/hrd/timezone.h"
#include "nigiri/types.h"

namespace nigiri::loader::hrd {

template <typename Fn>
void to_local_time(timezone_map_t const& timezones,
                   interval<std::chrono::sys_days> const& interval,
                   ref_service const& s,
                   Fn&& consumer) {
  struct duration_hash {
    cista::hash_t operator()(vector<duration_t> const& v) const {
      auto h = cista::BASE_HASH;
      for (auto const& el : v) {
        h = cista::hash_combine(h, el.count());
      }
      return h;
    }
  };

  auto utc_time_traffic_days =
      hash_map<vector<duration_t>, bitfield, duration_hash>{};
  auto const local_times = s.get_stop_times();
  auto const stop_timezones = s.get_stop_timezones(timezones);
  auto const first_day = interval.from_ + kBaseDayOffset;
  auto const last_day = interval.to_ - kBaseDayOffset;
  auto utc_service_times = vector<duration_t>{};
  utc_service_times.resize(
      static_cast<vector<duration_t>::size_type>(s.stops_.size() * 2U - 2U));
  for (auto day = first_day; day <= last_day; day += std::chrono::days{1}) {
    auto const day_idx = static_cast<size_t>((day - first_day).count());
    if (!s.traffic_days_.test(day_idx)) {
      continue;
    }

    auto const [first_utc, first_offset, first_valid] = local_mam_to_utc_mam(
        stop_timezones.front(), day, local_times.front(), true);

    if (!first_valid) {
      log(log_lvl::error, "loader.hrd.service",
          "first departure local to utc failed, ignoring: {}, local_time={}, "
          "stop={:07}, day={}",
          s.origin_, local_times.front(), to_idx(s.stops_.front().eva_num_),
          day);
      continue;
    }

    auto i = 0U;
    auto pred = duration_t{0};
    auto fail = false;
    for (auto const [local_time, tz] : utl::zip(local_times, stop_timezones)) {
      auto const [utc_mam, offset, valid] = local_mam_to_utc_mam(
          tz, day + first_offset, local_time - first_offset);
      if (offset != 0_days || pred > utc_mam || !valid) {
        log(log_lvl::error, "loader.hrd.service",
            "local to utc failed, ignoring: {}, day={}, time={}, offset={}, "
            "pred={}, utc_mam={}, valid={}",
            s.origin_, day, local_time, offset, pred, utc_mam, valid);
        fail = true;
        break;
      }

      utc_service_times[i++] = utc_mam;
      pred = utc_mam;
    }

    if (!fail) {
      utc_time_traffic_days[utc_service_times].set(
          kBaseDayOffset.count() + day_idx +
          static_cast<size_t>(first_offset / 1_days));
    }
  }

  for (auto const& [times, traffic_days] : utc_time_traffic_days) {
    consumer(service{s, times, traffic_days});
  }
}

}  // namespace nigiri::loader::hrd
