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

  std::optional<routing::query> random_pretrip_query();

  // use start transport mode to randomize coordinates near a location
  geo::latlng pos_near_start(location_idx_t);

  // uses dest transport mode to randomize coordinates near a location
  geo::latlng pos_near_dest(location_idx_t);

  std::pair<transport, stop_idx_t> random_transport_active_stop();

  timetable const& tt_;
  generator_settings const& s_;
  std::uint32_t seed_;

private:
  void init_geo(generator_settings const& settings);

  location_idx_t random_location();
  route_idx_t random_route(location_idx_t);
  transport_idx_t random_transport();
  transport_idx_t random_transport(route_idx_t);
  stop_idx_t get_stop_idx(transport_idx_t, location_idx_t) const;

  std::optional<stop_idx_t> random_active_stop(transport_idx_t);

  bool can_dep(transport_idx_t, stop_idx_t) const;
  std::optional<day_idx_t> random_active_day(transport_idx_t);
  std::optional<interval<unixtime_t>> get_start_interval(location_idx_t);

  bool arr_in_itv(transport_idx_t,
                  stop_idx_t,
                  interval<unixtime_t> const&) const;
  bool is_active_dest(location_idx_t, interval<unixtime_t> const&) const;

  geo::latlng random_point_in_range(
      geo::latlng const&, std::uniform_int_distribution<std::uint32_t>&);

  std::uint16_t tt_n_days() const;

  routing::query make_query() const;

  void add_offsets_for_pos(std::vector<routing::offset>&,
                           geo::latlng const&,
                           query_generation::transport_mode const&) const;

  // R-Tree
  geo::point_rtree locations_rtree_;
  std::vector<size_t> locs_in_bbox;

  // RNG
  std::mt19937 rng_;

  // Distributions
  std::uniform_int_distribution<location_idx_t::value_t> location_d_;
  std::uniform_int_distribution<size_t> locs_in_bbox_d_;
  std::uniform_int_distribution<transport_idx_t::value_t> transport_d_;
  std::uniform_int_distribution<day_idx_t::value_t> day_d_;
  std::uniform_int_distribution<std::uint32_t> start_mode_range_d_;
  std::uniform_int_distribution<std::uint32_t> dest_mode_range_d_;
  std::uniform_int_distribution<std::uint16_t> bearing_d_{0, 359};
};

}  // namespace nigiri::query_generation