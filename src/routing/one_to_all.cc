#include "nigiri/routing/one_to_all.h"

#include <utility>
#include <vector>

#include "nigiri/routing/limits.h"
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/routing/start_times.h"
#include "nigiri/types.h"
#include "utl/verify.h"

namespace nigiri::routing {

day_idx_t make_base(timetable const& tt, start_time_t start_time) {
  auto const search_interval = std::visit(
      utl::overloaded{
          [](interval<unixtime_t> const start_interval) {
            return start_interval;
          },
          [](unixtime_t const t) { return interval<unixtime_t>{t, t}; }},
      start_time);
  return day_idx_t{
      std::chrono::duration_cast<date::days>(
          std::chrono::round<std::chrono::days>(
              search_interval.from_ +
              ((search_interval.to_ - search_interval.from_) / 2)) -
          tt.internal_interval().from_)
          .count()};
}

unixtime_t make_start_time(start_time_t start_time) {
  return std::visit(
      utl::overloaded{[](interval<unixtime_t> const start_interval) {
                        return start_interval.from_;
                      },
                      [](unixtime_t const t) { return t; }},
      start_time);
}

template <direction SearchDir, via_offset_t Vias>
void run_raptor(raptor<SearchDir, true, Vias, search_mode::reachable>&& algo,
                std::vector<start>&& starts,
                query const& q) {
  auto results = pareto_set<journey>{};
  utl::equal_ranges_linear(
      starts,
      [](start const& a, start const& b) {
        return a.time_at_start_ == b.time_at_start_;
      },
      [&](auto&& from_it, auto&& to_it) {
        algo.next_start_time();
        auto const start_time = from_it->time_at_start_;
        for (auto const& s : it_range{from_it, to_it}) {
          trace("init: time_at_start={}, time_at_stop={} at {}\n",
                s.time_at_start_, s.time_at_stop_, location_idx_t{s.stop_});
          algo.add_start(s.stop_, s.time_at_stop_);
        }

        /*
         * Upper bound: Search journeys faster than 'worst_time_at_dest'
         * It will not find journeys with the same duration
         */
        constexpr auto const kEpsilon = duration_t{1};
        auto const worst_time_at_dest =
            start_time +
            (SearchDir == direction::kForward ? 1 : -1) * (q.max_travel_time_) +
            kEpsilon;

        algo.execute(start_time, q.max_transfers_, worst_time_at_dest,
                     q.prf_idx_, results);
      });
}

template <direction SearchDir, via_offset_t Vias>
raptor_state one_to_all(timetable const& tt,
                        rt_timetable const* rtt,
                        query const& q) {
  auto state = raptor_state{};

  auto is_dest = bitvec::max(tt.n_locations());
  auto is_via = std::array<bitvec, kMaxVias>{};
  auto dist_to_dest = std::vector<std::uint16_t>{};
  auto lb = std::vector<std::uint16_t>(tt.n_locations(), 0U);
  auto const base = make_base(tt, q.start_time_);

  auto r = raptor<SearchDir, true, Vias, search_mode::reachable>{
      tt,
      rtt,
      state,
      is_dest,
      is_via,
      dist_to_dest,
      q.td_dest_,
      lb,
      q.via_stops_,
      base,
      q.allowed_claszes_,
      q.require_bike_transport_,
      true,  // is_wheelchair
      q.transfer_time_settings_};

  auto starts = std::vector<start>{};
  {
    auto const add_ontrip = true;
    get_starts(SearchDir, tt, rtt, q.start_time_, q.start_, q.td_start_,
               q.max_start_offset_, q.start_match_mode_, q.use_start_footpaths_,
               starts, add_ontrip, q.prf_idx_, q.transfer_time_settings_);
  }

  run_raptor(std::move(r), std::move(starts), q);

  return state;
}

template <direction SearchDir>
raptor_state one_to_all(timetable const& tt,
                        rt_timetable const* rtt,
                        query const& q) {
  auto const via_count = q.via_stops_.size();
  utl::verify(via_count <= kMaxVias, "too many via stops: {}, limit: {}",
              via_count, kMaxVias);

  static_assert(kMaxVias == 2,
                "one_to_all.h needs to be adjusted for kMaxVias");

  switch (via_count) {
    case 0: return one_to_all<SearchDir, 0>(tt, rtt, q); break;
    case 1: return one_to_all<SearchDir, 1>(tt, rtt, q); break;
    case 2: return one_to_all<SearchDir, 2>(tt, rtt, q); break;
  }
  std::unreachable();
}

template raptor_state one_to_all<direction::kForward>(timetable const&,
                                                      rt_timetable const*,
                                                      query const&);
template raptor_state one_to_all<direction::kBackward>(timetable const&,
                                                       rt_timetable const*,
                                                       query const&);

}  // namespace nigiri::routing