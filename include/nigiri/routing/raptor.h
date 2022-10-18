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

private:
  static constexpr auto const kFwd = (SearchDir == direction::kForward);
  static constexpr auto const kBwd = (SearchDir == direction::kBackward);

  bool is_better(auto a, auto b);
  bool is_better_or_eq(auto a, auto b);
  auto get_best(auto a, auto b);

  location_idx_t parent_of(location_idx_t);

  routing_time time_at_stop(transport const& t,
                            unsigned const stop_idx,
                            event_type const ev_type);
  transport get_earliest_transport(unsigned const k,
                                   route_idx_t const r,
                                   unsigned const stop_idx,
                                   location_idx_t const l_idx);
  bool update_route(unsigned const k, route_idx_t const r);
  void update_footpaths(unsigned const k);

  unsigned end_k() const;

  void rounds();
  void force_print_state(char const* comment = "");
  void print_state(char const* comment = "");

  void reconstruct(unixtime_t const start_at_start);

  void set_time_at_destination(unsigned round_k);

  timetable const& tt_;
  std::uint16_t n_days_;
  query q_;
  routing_time time_at_destination_{kInvalidTime<SearchDir>};
  search_state& state_;
};

}  // namespace nigiri::routing
