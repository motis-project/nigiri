#include "nigiri/routing/raptor_n_to_all.h"

#include <utility>

#include "utl/verify.h"

#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/routing/search.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"


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
                                query q) {
  utl::verify(q.via_stops_.empty(), "raptor_n_to_all: via stops not supported");
  q.sanitize(tt);

  auto result = n_to_all_result{};
  result.state_.track_min_ = true;  // raptor then fills state_.min_
  result.searched_ = rtt == nullptr
                         ? run_search<false>(tt, nullptr, result.state_,
                                             std::move(q))
                         : run_search<true>(tt, rtt, result.state_,
                                            std::move(q));
  return result;
}

}  // namespace nigiri::routing