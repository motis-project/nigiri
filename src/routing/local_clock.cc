#include "nigiri/routing/local_clock.h"

#include <algorithm>

#include "date/tz.h"

#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

local_clock::local_clock(timetable const& tt, rt_timetable const* rtt)
    : tt_{tt}, rtt_{rtt}, offsets_(tt.timezones_.size()) {}

int local_clock::mam(unixtime_t const t,
                     location_idx_t const l,
                     timezone_idx_t const trip_tz) {
  auto const loc_tz = to_idx(l) < tt_.n_locations()  // else an rt virt
                          ? tt_.locations_.location_timezones_[l]
                          : timezone_idx_t::invalid();
  auto const tz = loc_tz != timezone_idx_t::invalid() ? loc_tz : trip_tz;
  auto off = duration_t{0};
  if (tz != timezone_idx_t::invalid()) {
    auto& o = offsets_[to_idx(tz)];
    if (t < o.from_ || t >= o.to_) {
      auto const* const z = to_time_zone(tt_.timezones_[tz]);
      if (z == nullptr) {  // fixed seasons (HRD): not cached
        o = {.offset_ =
                 to_local_time(tt_.timezones_[tz], t).time_since_epoch() -
                 t.time_since_epoch()};
      } else {
        auto const info = z->get_info(t);
        o = {.from_ = std::chrono::ceil<duration_t>(info.begin),
             .to_ = std::chrono::floor<duration_t>(info.end),
             .offset_ = std::chrono::duration_cast<duration_t>(info.offset)};
      }
    }
    off = o.offset_;
  }
  auto const m = (t + off).time_since_epoch().count() % 1440;
  return static_cast<int>(m < 0 ? m + 1440 : m);
}

timezone_idx_t local_clock::trip_tz(transport_idx_t const t) const {
  auto const providers = tt_.transport_section_providers_[t];
  return providers.empty() ? timezone_idx_t::invalid()
                           : tt_.providers_[providers.front()].tz_;
}

timezone_idx_t local_clock::trip_tz(rt_transport_idx_t const t) const {
  auto const src = rtt_->rt_transport_src_[t];
  auto const it = std::lower_bound(
      begin(tt_.providers_), end(tt_.providers_), src,
      [](provider const& a, source_idx_t const b) { return a.src_ < b; });
  return it != end(tt_.providers_) && it->src_ == src
             ? it->tz_
             : timezone_idx_t::invalid();
}

}  // namespace nigiri::routing
