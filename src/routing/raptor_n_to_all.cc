#include "nigiri/routing/raptor_n_to_all.h"

#include <algorithm>
#include <utility>
#include <variant>

#include "utl/verify.h"
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/routing/search.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

// The one-to-all RAPTOR is instantiated here and not in raptor_search.cc:
// that file already holds so many RAPTOR variants that adding two more makes
// MSVC run into C1128 (too many sections in one object file).

namespace nigiri::routing {

namespace {

template <bool Rt>
interval<unixtime_t> run_search(timetable const& tt,
                                rt_timetable const* rtt,
                                raptor_state& state,
                                query q) {
  using algo_t = raptor<direction::kForward, Rt, via_offset_t{0U},
                        search_mode::kOneToAll>;
  auto search_state = routing::search_state{};
  return search<direction::kForward, algo_t>{tt,    rtt, search_state,
                                             state, std::move(q)}
  .execute()
  .interval_;
}

}  // namespace

n_to_all_result raptor_n_to_all(timetable const& tt,
                                rt_timetable const* rtt,
                                query q,
                                bool const with_profile,
                                bool const with_aggregates,
                                std::vector<std::pair<location_idx_t, std::uint8_t>> const&
                                    origin_of_start) {
  utl::verify(q.via_stops_.empty(), "raptor_n_to_all: via stops not supported");
  q.sanitize(tt);

  auto result = n_to_all_result{};
  result.state_.track_min_ = true;  // raptor then fills state_.min_
  result.state_.track_profile_ = with_profile;  // ... and state_.profile_
  result.state_.track_agg_ = with_aggregates;  // ... and state_.agg_
  if (with_aggregates) {
    // the window of the aggregates is the requested one, not the one search<>
    // may widen; it is closed: minutes 0..len
    auto const w =
        std::holds_alternative<interval<unixtime_t>>(q.start_time_)
            ? std::get<interval<unixtime_t>>(q.start_time_)
            : interval<unixtime_t>{std::get<unixtime_t>(q.start_time_),
                                   std::get<unixtime_t>(q.start_time_)};
    result.state_.agg_.t0_ = w.from_;
    result.state_.agg_.len_ = static_cast<int>((w.to_ - w.from_).count());
    result.state_.agg_.t_max_ = static_cast<int>(q.max_travel_time_.count());

    if (!origin_of_start.empty()) {
      utl::verify(result.state_.agg_.len_ + 1 < 65535,
                  "raptor_n_to_all: window too long for the counters");
      auto n = std::uint32_t{0U};
      auto& map = result.state_.agg_.origin_of_;
      map.assign(tt.n_locations(), agg_state::kNoOrigin);
      for (auto const& [l, o] : origin_of_start) {
        utl::verify(o != agg_state::kNoOrigin, "raptor_n_to_all: origin index 255");
        map[to_idx(l)] = o;
        n = std::max(n, static_cast<std::uint32_t>(o) + 1U);
      }
      result.state_.agg_.n_origins_ = n;
    }
  }
  result.searched_ = rtt == nullptr
                         ? run_search<false>(tt, nullptr, result.state_,
                                             std::move(q))
                         : run_search<true>(tt, rtt, result.state_,
                                            std::move(q));
  if (with_aggregates) {
    result.state_.agg_.finalize(result.state_.profile_start_);
  }
  return result;
}

}  // namespace nigiri::routing