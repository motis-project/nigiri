#pragma once

#include <string_view>

#include "utl/parser/arg_parser.h"

#include "nigiri/loader/gtfs/parse_date.h"
#include "nigiri/loader/gtfs/parse_time.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/rt/run.h"
#include "nigiri/timetable.h"

#include "gtfsrt/gtfs-realtime.pb.h"

namespace nigiri::rt {

std::pair<date::days, duration_t> split(duration_t);

template <typename Fn>
void resolve_static(date::sys_days const today,
                    timetable const& tt,
                    source_idx_t const src,
                    transit_realtime::TripDescriptor const& td,
                    Fn&& fn) {
  using loader::gtfs::hhmm_to_min;
  using loader::gtfs::parse_date;

  auto const& trip_id = td.trip_id();
  auto const lb = std::lower_bound(
      begin(tt.trip_id_to_idx_), end(tt.trip_id_to_idx_), trip_id,
      [&](pair<trip_id_idx_t, trip_idx_t> const& a, auto&& b) {
        return std::tuple{tt.trip_id_src_[a.first],
                          tt.trip_id_strings_[a.first].view()} <
               std::tuple{src, static_cast<std::string_view>(b)};
      });

  auto const start_date = td.has_start_date()
                              ? std::make_optional(parse_date(
                                    utl::parse<unsigned>(td.start_date())))
                              : std::nullopt;
  auto const start_time = td.has_start_time()
                              ? std::make_optional(hhmm_to_min(td.start_time()))
                              : std::nullopt;

  auto const id_matches = [&](trip_id_idx_t const t_id_idx) {
    return tt.trip_id_src_[t_id_idx] == src &&
           tt.trip_id_strings_[t_id_idx].view() == trip_id;
  };

  for (auto i = lb; i != end(tt.trip_id_to_idx_) && id_matches(i->first); ++i) {
    for (auto const [t, stop_range] : tt.trip_transport_ranges_[i->second]) {
      auto const [first_dep_offset, tz_offset] =
          tt.transport_first_dep_offset_[t].to_offset();
      auto const utc_dep =
          tt.event_mam(t, stop_range.from_, event_type::kDep).as_duration();
      auto const gtfs_static_dep = utc_dep + first_dep_offset + tz_offset;
      auto const [gtfs_static_dep_day, gtfs_static_dep_time] =
          split(gtfs_static_dep);
      auto const [start_time_day, start_time_time] =
          start_time.has_value() ? split(*start_time)
                                 : std::pair{date::days{0U}, duration_t{0U}};

      if (start_time.has_value() && gtfs_static_dep_time != start_time_time) {
        continue;
      }

      auto const start_time_day_offset =
          start_time.has_value() ? gtfs_static_dep_day - start_time_day
                                 : date::days{0U};

      auto const day_idx =
          ((start_date.has_value() ? *start_date : today) + first_dep_offset -
           start_time_day_offset - tt.internal_interval_days().from_)
              .count();

      if (day_idx > kMaxDays || day_idx < 0) {
        continue;
      }

      auto const& traffic_days = tt.bitfields_[tt.transport_traffic_days_[t]];
      if (traffic_days.test(static_cast<std::size_t>(day_idx))) {
        auto const r = run{.t_ = transport{t, day_idx_t{day_idx}},
                           .stop_range_ = stop_range};
        if (fn(r, i->second) == utl::continue_t::kBreak) {
          return;
        }
      }
    }
  }
}

std::pair<run, trip_idx_t> gtfsrt_resolve_run(
    date::sys_days const today,
    timetable const&,
    rt_timetable const*,
    source_idx_t,
    transit_realtime::TripDescriptor const&,
    std::string_view rt_changed_trip_id = {});

void resolve_rt(rt_timetable const&, run&, std::string_view trip_id);

}  // namespace nigiri::rt