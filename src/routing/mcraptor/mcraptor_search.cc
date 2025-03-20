#include "nigiri/routing/mcraptor/mcraptor_search.h"

#include <string>
#include <utility>
#include "date/date.h"
#include "utl/to_vec.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/mcraptor/mcraptor.h"

namespace nigiri::routing {

routing_result<raptor_stats> mcraptor_search(
    timetable const& tt,
    rt_timetable const* rtt,
    search_state& s_state,
    raptor_state& r_state,
    query q,
    direction const search_dir,
    std::optional<std::chrono::seconds> const timeout) {
  if (rtt != nullptr) {
    return {};
  }

  using algo_t = mcraptor<>;
  return search<direction::kForward, algo_t>{
      tt, rtt, s_state,r_state, std::move(q), timeout}
        .execute();
}

bool results_are_equal(timetable const& tt, routing_result<raptor_stats> const& result_1, routing_result<raptor_stats> const& result_2) {

  pareto_set<journey> res_1 = *result_1.journeys_;
  pareto_set<journey> res_2 = *result_2.journeys_;

  std::stringstream ss_result_1;
  ss_result_1 << "\n";
  for (auto const& x : res_1) {
    x.print(ss_result_1, tt);
    ss_result_1 << "\n\n";
  }
  std::stringstream ss_result_2;
  ss_result_2 << "\n";
  for (auto const& x : res_2) {
    x.print(ss_result_2, tt);
    ss_result_2 << "\n\n";
  }

  return ss_result_1.str() == ss_result_2.str();
}

}  // namespace nigiri::routing