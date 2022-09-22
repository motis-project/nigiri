#pragma once

#include <cinttypes>

#include "nigiri/routing/query.h"
#include "nigiri/routing/routing_time.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::routing {

struct search_state;

template <direction SearchDir>
struct raptor {
public:
  raptor(timetable& tt, search_state& state, query q);

  void route();
  void reconstruct(unixtime_t const start_at_start);

private:
  static constexpr auto const kFwd = (SearchDir == direction::kForward);
  static constexpr auto const kBwd = (SearchDir == direction::kBackward);
  static constexpr auto const kInvalidTime =
      kFwd ? routing_time::max() : routing_time::min();

  bool is_better(auto a, auto b);
  bool is_better_or_eq(auto a, auto b);
  auto get_best(auto a, auto b);

  routing_time time_at_stop(transport const& t,
                            unsigned const stop_idx,
                            event_type const ev_type);
  transport get_earliest_transport(unsigned const k,
                                   route_idx_t const r,
                                   unsigned const stop_idx,
                                   location_idx_t const l_idx);
  void update_route(unsigned const k, route_idx_t const r);
  void update_footpaths(unsigned const k);

  unsigned end_k() const;

  void rounds();
  void force_print_state(char const* comment = "");
  void print_state(char const* comment = "");

  timetable const& tt_;
  std::uint16_t n_days_;
  query q_;
  routing_time time_at_destination_{kInvalidTime};
  search_state& state_;
};

}  // namespace nigiri::routing
