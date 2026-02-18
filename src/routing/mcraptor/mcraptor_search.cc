/*
* This file is based on the mcraptor from PumarPap https://github.com/motis-project/nigiri/pull/183
*/

#include "nigiri/routing/mcraptor/mcraptor_search.h"
#include "nigiri/routing/mcraptor/mcraptor.h"
#include <utility>
#include "date/date.h"
#include "utl/to_vec.h"

namespace nigiri::routing::da {

template <direction SearchDir>
routing_result mcraptor_search(
    timetable const& tt,
    rt_timetable const* rtt,
    search_state& s_state,
    raptor_state& r_state,
    query q,
    std::optional<std::chrono::seconds> const timeout,
    std::vector<std::vector<std::pair<int, double>>> arr_dist) {
  //  if (rtt != nullptr) {
  //    return {};
  //  }

  using algo_t = mcraptor<>;
  r_state.arr_dist_ = std::move(arr_dist);
  return search<SearchDir, algo_t>{tt,      rtt,          s_state,
                                   r_state, std::move(q), timeout}
      .execute();
}

template routing_result mcraptor_search<direction::kForward>(
    timetable const&,
    rt_timetable const*,
    search_state&,
    raptor_state&,
    query,
    std::optional<std::chrono::seconds>,
    std::vector<std::vector<std::pair<int, double>>> arr_dist);

template routing_result mcraptor_search<direction::kBackward>(
    timetable const&,
    rt_timetable const*,
    search_state&,
    raptor_state&,
    query,
    std::optional<std::chrono::seconds>,
    std::vector<std::vector<std::pair<int, double>>> arr_dist);

}  // namespace nigiri::routing::da