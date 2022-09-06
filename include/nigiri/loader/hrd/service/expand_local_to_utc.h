#pragma once

#include "nigiri/loader/hrd/service/ref_service.h"
#include "nigiri/loader/hrd/service/service.h"
#include "nigiri/loader/hrd/stamm/timezone.h"
#include "nigiri/types.h"

namespace nigiri::loader::hrd {

template <typename Fn>
void to_local_time(service_store const& store,
                   stamm const& st,
                   interval<std::chrono::sys_days> const& tt_interval,
                   ref_service const& s,
                   Fn&& consumer) {
  using key_t = std::pair<std::basic_string<duration_t>,
                          std::basic_string<location_idx_t>>;
  auto utc_time_traffic_days = hash_map<key_t, bitfield>{};

  auto const local_times = s.local_times(store);
  auto const stop_timezones = s.get_stop_timezones(store, st);
  auto const first_day = tt_interval.from_ + kBaseDayOffset;
  auto const last_day = tt_interval.to_ - kBaseDayOffset;

  key_t key;
  auto& utc_service_times = key.first;
  auto& stop_seq = key.second;
  utc_service_times.resize(static_cast<vector<duration_t>::size_type>(
      s.split_info_.stop_range().size() * 2U - 2U));
  stop_seq.resize(s.stops(store).size());

  for (auto day = first_day; day <= last_day; day += std::chrono::days{1}) {
    auto const day_idx = static_cast<size_t>((day - first_day).count());
    if (!s.local_traffic_days().test(day_idx)) {
      continue;
    }

    auto const [_, first_day_offset, first_valid] = local_mam_to_utc_mam(
        stop_timezones.front(), day, local_times.front(), true);

    if (!first_valid) {
      log(log_lvl::error, "loader.hrd.service.utc",
          "first departure local to utc failed for {}: local_time={}, day={}",
          s.origin(store), local_times.front(), day);
      continue;
    }

    auto i = 0U;
    auto pred = duration_t{0};
    auto fail = false;
    for (auto const& [local_time, tz] : utl::zip(local_times, stop_timezones)) {
      auto const [utc_mam, day_offset, valid] = local_mam_to_utc_mam(
          tz, day + first_day_offset, local_time - first_day_offset);
      if (day_offset != 0_days || pred > utc_mam || !valid) {
        log(log_lvl::error, "loader.hrd.service.utc",
            "local to utc failed, ignoring: {}, day={}, time={}, offset={}, "
            "pred={}, utc_mam={}, valid={}",
            s.origin(store), day, local_time, day_offset, pred, utc_mam, valid);
        fail = true;
        break;
      }

      utc_service_times[i] = utc_mam;

      auto const train_num = 0U;
      auto const provider = provider_idx_t::invalid();
      auto const local_mam = duration_t{local_time.count() % 1440};
      auto const local_day = day_idx_t{day_idx + (local_time.count() / 1440)};
      stop_seq[i] =
          st.resolve_track(track_rule_key{st.resolve_location(stop.eva_num_),
                                          train_num, provider, local_mam},
                           local_day);

      ++i;

      pred = utc_mam;
    }

    if (fail) {
      continue;
    }

    auto const traffic_day = kBaseDayOffset.count() + day_idx +
                             static_cast<size_t>(first_day_offset / 1_days);
    auto const it = utc_time_traffic_days.find(utc_service_times);
    if (it == end(utc_time_traffic_days)) {
      utc_time_traffic_days.emplace(utc_service_times, bitfield{})
          .first->second.set(traffic_day);
    } else {
      it->second.set(traffic_day);
    }
  }

  for (auto& [times, traffic_days] : utc_time_traffic_days) {
    consumer(ref_service{s, std::move(times.first), std::move(times.second),
                         traffic_days});
  }
}

}  // namespace nigiri::loader::hrd
