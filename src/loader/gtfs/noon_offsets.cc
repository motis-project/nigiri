#include "nigiri/loader/gtfs/noon_offsets.h"

#include "nigiri/timetable.h"

namespace nigiri::loader::gtfs {

duration_t get_noon_offset(date::local_days const days,
                           date::time_zone const* tz) {
  auto const abs_zoned_time = date::zoned_time<std::chrono::minutes>(
      tz, days + std::chrono::minutes{12 * 60});
  return duration_t{abs_zoned_time.get_info().offset.count() / 60};
}

noon_offset_hours_t precompute_noon_offsets(timetable const& tt,
                                            agency_map_t const& agencies) {
  auto const tt_interval = tt.internal_interval_days();
  auto ret = noon_offset_hours_t{};

  for (auto const& [id, provider_idx] : agencies) {
    if (provider_idx == provider_idx_t::invalid()) {
      continue;
    }

    auto const tz_idx = tt.providers_[provider_idx].tz_;
    if (ret.size() <= to_idx(tz_idx)) {
      ret.resize(static_cast<timezone_idx_t::value_t>(to_idx(tz_idx) + 1U));
    }

    if (ret[tz_idx].has_value()) {
      continue;  // already previously computed
    }

    ret[tz_idx] = std::array<duration_t, kMaxDays>{};
    auto const tz = reinterpret_cast<date::time_zone const*>(
        tt.locations_.timezones_[tz_idx]
            .as<pair<string, void const*>>()
            .second);
    for (auto day = tt_interval.from_; day != tt_interval.to_;
         day += std::chrono::days{1}) {
      if (!tt_interval.contains(day)) {
        continue;
      }

      auto const day_idx =
          static_cast<std::size_t>((day - tt_interval.from_).count());
      assert(day_idx < kMaxDays);
      (*ret[tz_idx])[day_idx] =
          get_noon_offset(date::local_days{date::year_month_day{day}}, tz);
    }
  }

  return ret;
}

}  // namespace nigiri::loader::gtfs