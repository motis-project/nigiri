#include "nigiri/rt/gtfsrt_resolve_run.h"

#include "nigiri/common/day_list.h"
#include "nigiri/common/split_duration.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/rt/trip_update.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::rt {

std::pair<date::days, duration_t> split(duration_t const i) {
  auto const a = i.count() / 1440;
  auto const b = i.count() % 1440;
  return {date::days{a}, duration_t{b}};
}

void resolve_static(date::sys_days const today,
                    timetable const& tt,
                    source_idx_t const src,
                    transit_realtime::TripDescriptor const& td,
                    run& r,
                    trip_idx_t& trip) {
  resolve_static(today, tt, src, td, [&](run const& a, trip_idx_t const b) {
    r = a;
    trip = b;
    return utl::continue_t::kBreak;
  });
}

void resolve_rt(rt_timetable const& rtt,
                run& output,
                std::string_view rt_changed_trip_id) {
  auto const it = rtt.static_trip_lookup_.find(output.t_);
  if (it != end(rtt.static_trip_lookup_)) {
    output.rt_ = it->second;
    return;
  }
  auto const rt_add_idx = rtt.additional_trip_ids_.find(rt_changed_trip_id);
  if (rt_add_idx.has_value()) {
    output.rt_ = rtt.additional_trips_lookup_[*rt_add_idx];
    if (output.stop_range_.size() == 0) {
      output.stop_range_ = {
          static_cast<stop_idx_t>(0U),
          static_cast<stop_idx_t>(
              rtt.rt_transport_location_seq_.at(output.rt_).size())};
    }
  }
}

std::pair<run, trip_idx_t> gtfsrt_resolve_run(
    date::sys_days const today,
    timetable const& tt,
    rt_timetable const* rtt,
    source_idx_t const src,
    transit_realtime::TripDescriptor const& td,
    std::string_view rt_changed_trip_id) {
  auto r = run{};
  trip_idx_t trip;
  resolve_static(today, tt, src, td, r, trip);
  if (rtt != nullptr) {
    resolve_rt(*rtt, r,
               rt_changed_trip_id.empty() ? td.trip_id() : rt_changed_trip_id);
  }
  return {r, trip};
}

}  // namespace nigiri::rt