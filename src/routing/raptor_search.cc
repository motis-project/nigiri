#include "nigiri/routing/raptor_search.h"

#include <utility>

#include "utl/verify.h"

#include "nigiri/routing/via_search.h"

namespace nigiri::routing {

namespace {

template <direction SearchDir, via_offset_t Vias>
routing_result<raptor_stats> raptor_search_with_vias(
    timetable const& tt,
    rt_timetable const* rtt,
    search_state& s_state,
    raptor_state& r_state,
    query q,
    std::optional<std::chrono::seconds> const timeout) {

  if (rtt == nullptr) {
    using algo_t = raptor<SearchDir, false, Vias>;
    return search<SearchDir, algo_t>{tt,      rtt,          s_state,
                                     r_state, std::move(q), timeout}
        .execute();
  } else {
    using algo_t = raptor<SearchDir, true, Vias>;
    return search<SearchDir, algo_t>{tt,      rtt,          s_state,
                                     r_state, std::move(q), timeout}
        .execute();
  }
}

template <direction SearchDir>
routing_result<raptor_stats> raptor_search_with_dir(
    timetable const& tt,
    rt_timetable const* rtt,
    search_state& s_state,
    raptor_state& r_state,
    query q,
    std::optional<std::chrono::seconds> const timeout) {
  sanitize_via_stops(tt, q);
  utl::verify(q.via_stops_.size() <= kMaxVias,
              "too many via stops: {}, limit: {}", q.via_stops_.size(),
              kMaxVias);

  static_assert(kMaxVias == 3,
                "raptor_search.cc needs to be adjusted for kMaxVias");

  switch (q.via_stops_.size()) {
    case 0:
      return raptor_search_with_vias<SearchDir, 0>(tt, rtt, s_state, r_state,
                                                   std::move(q), timeout);
    case 1:
      return raptor_search_with_vias<SearchDir, 1>(tt, rtt, s_state, r_state,
                                                   std::move(q), timeout);
    case 2:
      return raptor_search_with_vias<SearchDir, 2>(tt, rtt, s_state, r_state,
                                                   std::move(q), timeout);
    case 3:
      return raptor_search_with_vias<SearchDir, 3>(tt, rtt, s_state, r_state,
                                                   std::move(q), timeout);
  }
  std::unreachable();
}

}  // namespace

routing_result<raptor_stats> raptor_search(
    timetable const& tt,
    rt_timetable const* rtt,
    search_state& s_state,
    raptor_state& r_state,
    query q,
    direction const search_dir,
    std::optional<std::chrono::seconds> const timeout) {
  if (search_dir == direction::kForward) {
    return raptor_search_with_dir<direction::kForward>(
        tt, rtt, s_state, r_state, std::move(q), timeout);
  } else {
    return raptor_search_with_dir<direction::kBackward>(
        tt, rtt, s_state, r_state, std::move(q), timeout);
  }
}

}  // namespace nigiri::routing
