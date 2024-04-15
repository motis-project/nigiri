#pragma once

#include <ctime>
#include <random>

#include "geo/point_rtree.h"

#include "nigiri/query_generator/generator_settings.h"
#include "nigiri/query_generator/transport_mode.h"
#include "nigiri/routing/query.h"

namespace nigiri {
struct timetable;
}  // namespace nigiri

namespace nigiri::query_generation {

constexpr auto const kMaxGenAttempts = 1000U;

struct generator {
  explicit generator(timetable const&, generator_settings const&);
  explicit generator(timetable const&,
                     generator_settings const&,
                     std::uint32_t seed);

  // randomize a point in time within the timetable
  unixtime_t random_time();

  // randomize a location that is active during the interval
  // for the given event type
  std::optional<location_idx_t> random_active_location(
      interval<unixtime_t> const&, event_type);

  // use start transport mode to randomize coordinates near a location
  geo::latlng pos_near_start(location_idx_t);

  // uses dest transport mode to randomize coordinates near a location
  geo::latlng pos_near_dest(location_idx_t);

  // randomize a transport and one of its stops that allows the given event type
  std::pair<transport, stop_idx_t> random_transport_active_stop(event_type);

  // convenience functions for query generation inside nigiri
  std::optional<routing::query> random_pretrip_query();
  std::optional<routing::query> random_ontrip_query();

  timetable const& tt_;
  generator_settings const& s_;
  std::uint32_t seed_;

private:
  transport_idx_t random_transport_idx();
  day_idx_t random_day();

  std::optional<day_idx_t> random_active_day(transport_idx_t);
  std::optional<stop_idx_t> random_active_stop(transport_idx_t, event_type);

  geo::latlng random_point_in_range(
      geo::latlng const&, std::uniform_int_distribution<std::uint32_t>&);

  interval<day_idx_t> unix_to_day_itv(interval<unixtime_t> const&);
  std::uint16_t tt_n_days();

  routing::query make_query() const;

  void add_offsets_for_pos(std::vector<routing::offset>&,
                           geo::latlng const&,
                           query_generation::transport_mode const&);

  // R-Tree
  geo::point_rtree locations_rtree_;

  // RNG
  std::mt19937 rng_;

  // Distributions
  std::uniform_int_distribution<location_idx_t::value_t> location_d_;
  std::uniform_int_distribution<unixtime_t::rep> time_d_;
  std::uniform_int_distribution<transport_idx_t::value_t> transport_d_;
  std::uniform_int_distribution<day_idx_t::value_t> day_d_;
  std::uniform_int_distribution<std::uint32_t> start_mode_range_d_;
  std::uniform_int_distribution<std::uint32_t> dest_mode_range_d_;
  std::uniform_int_distribution<std::uint16_t> bearing_d_{0, 359};
};

}  // namespace nigiri::query_generation