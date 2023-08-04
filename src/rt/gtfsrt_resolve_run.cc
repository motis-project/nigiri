#include "nigiri/rt/gtfsrt_resolve_run.h"

#include "utl/parser/arg_parser.h"

#include "nigiri/loader/gtfs/parse_date.h"
#include "nigiri/loader/gtfs/parse_time.h"
#include "nigiri/common/day_list.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/rt/trip_update.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::rt {

void resolve_static(date::sys_days const today,
                    timetable const& tt,
                    source_idx_t const src,
                    transit_realtime::TripDescriptor const& td,
                    run& r,
                    trip_idx_t& trip) {
  using loader::gtfs::hhmm_to_min;
  using loader::gtfs::parse_date;

  auto const& trip_id = td.trip_id();
  auto const lb = std::lower_bound(
      begin(tt.trip_id_to_idx_), end(tt.trip_id_to_idx_), trip_id,
      [&](pair<trip_id_idx_t, trip_idx_t> const& a, string const& b) {
        return std::tuple(tt.trip_id_src_[a.first],
                          tt.trip_id_strings_[a.first].view()) <
               std::tuple(src, std::string_view{b});
      });

  auto const start_date = td.has_start_date()
                              ? std::make_optional(parse_date(
                                    utl::parse<unsigned>(td.start_date())))
                              : std::nullopt;
  auto const start_time = td.has_start_time()
                              ? std::make_optional(hhmm_to_min(td.start_time()))
                              : std::nullopt;

  auto const id_matches = [&](trip_id_idx_t const t_id_idx) {
    std::cout << "MATCH: " << tt.trip_id_strings_[t_id_idx].view()
              << " == " << trip_id << ": " << std::boolalpha
              << (tt.trip_id_strings_[t_id_idx].view() == trip_id) << "\n";
    return tt.trip_id_src_[t_id_idx] == src &&
           tt.trip_id_strings_[t_id_idx].view() == trip_id;
  };

  if (lb == end(tt.trip_id_to_idx_) || !id_matches(lb->first)) {
    std::cout << "ID MISMATCH\n";
    return;
  }

  for (auto i = lb; i != end(tt.trip_id_to_idx_) && id_matches(i->first); ++i) {
    std::cout << "TRIP TRANSPORTS: "
              << tt.trip_transport_ranges_[i->second].size() << "\n";
    for (auto const [t, stop_range] : tt.trip_transport_ranges_[i->second]) {
      auto const gtfs_static_dep =
          tt.event_mam(t, stop_range.from_, event_type::kDep).as_duration() +
          tt.transport_first_dep_offset_[t];

      if (start_time.has_value() && gtfs_static_dep != *start_time) {
        std::cout << "TIME MISMATCH: gtfs_static_dep=" << gtfs_static_dep
                  << ", start_time=" << *start_time << "\n";
        continue;
      }

      auto const utc_dep =
          tt.event_mam(t, stop_range.from_, event_type::kDep).as_duration();
      auto const day_offset = date::days{static_cast<int>(std::floor(
          static_cast<float>(utc_dep.count() - gtfs_static_dep.count()) /
          1440U))};
      std::cout << "offset=" << utc_dep.count() - gtfs_static_dep.count()
                << "\n";
      std::cout << "utc_dep=" << utc_dep.count() / 60 << ":"
                << utc_dep.count() % 60 << "\n";
      std::cout << "gtfs_static_dep=" << (gtfs_static_dep.count() / 60) << ":"
                << (gtfs_static_dep.count() % 60) << "\n";
      std::cout << "day_offset: " << gtfs_static_dep.count() << " "
                << utc_dep.count() << " " << day_offset << "\n";
      auto const day_idx = ((start_date.has_value() ? *start_date : today) -
                            day_offset - tt.internal_interval_days().from_)
                               .count();
      if (day_idx > kMaxDays || day_idx < 0) {
        std::cout << "DAY OUT OF RANGE: " << day_idx << "\n";
        continue;
      }

      auto const& traffic_days = tt.bitfields_[tt.transport_traffic_days_[t]];
      if (traffic_days.test(static_cast<std::size_t>(day_idx))) {
        r.t_ = transport{t, day_idx_t{day_idx}};
        r.stop_range_ = stop_range;
        trip = i->second;
        return;
      }

      auto const day =
          tt.internal_interval_days().from_ + day_idx * std::chrono::days{1};
      std::cout << "DAY INACTIVE: " << day << " [=day_idx " << day_idx
                << "], active="
                << day_list{traffic_days, tt.internal_interval_days().from_}
                << "\n";
    }
  }
}

void resolve_rt(rt_timetable const& rtt, run& output) {
  auto const it = rtt.static_trip_lookup_.find(output.t_);
  if (it != end(rtt.static_trip_lookup_)) {
    output.rt_ = it->second;
  }
}

std::pair<run, trip_idx_t> gtfsrt_resolve_run(
    date::sys_days const today,
    timetable const& tt,
    rt_timetable& rtt,
    source_idx_t const src,
    transit_realtime::TripDescriptor const& td) {
  auto r = run{};
  trip_idx_t trip;
  resolve_static(today, tt, src, td, r, trip);
  resolve_rt(rtt, r);
  return {r, trip};
}

}  // namespace nigiri::rt