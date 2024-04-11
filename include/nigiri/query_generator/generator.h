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

constexpr auto const kMaxGenAttempts = 10000U;

constexpr auto const kTimeOfDayWeights = std::array<int, 24>{
    1,  // 01: 00:00 - 01:00
    1,  // 02: 01:00 - 02:00
    1,  // 03: 02:00 - 03:00
    1,  // 04: 03:00 - 04:00
    1,  // 05: 04:00 - 05:00
    2,  // 06: 05:00 - 06:00
    3,  // 07: 06:00 - 07:00
    4,  // 08: 07:00 - 08:00
    4,  // 09: 08:00 - 09:00
    3,  // 10: 09:00 - 10:00
    2,  // 11: 10:00 - 11:00
    2,  // 12: 11:00 - 12:00
    2,  // 13: 12:00 - 13:00
    2,  // 14: 13:00 - 14:00
    3,  // 15: 14:00 - 15:00
    4,  // 16: 15:00 - 16:00
    4,  // 17: 16:00 - 17:00
    4,  // 18: 17:00 - 18:00
    4,  // 19: 18:00 - 19:00
    3,  // 20: 19:00 - 20:00
    2,  // 21: 20:00 - 21:00
    1,  // 22: 21:00 - 22:00
    1,  // 23: 22:00 - 23:00
    1  // 24: 23:00 - 24:00
};

struct generator {
  explicit generator(timetable const&, generator_settings const&);

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
  std::pair<transport, stop_idx_t> random_transport_active_stop(event_type et);

  // convenience functions for query generation inside nigiri
  std::optional<routing::query> random_pretrip_query();
  std::optional<routing::query> random_ontrip_query();

  timetable const& tt_;
  generator_settings const& s_;

private:
  transport_idx_t random_transport_idx();
  day_idx_t random_day();

  std::optional<day_idx_t> random_active_day(transport_idx_t);
  std::optional<stop_idx_t> random_active_stop(transport_idx_t, event_type);

  geo::latlng random_point_in_range(
      geo::latlng const&, std::uniform_int_distribution<std::uint32_t>&);

  interval<day_idx_t> unix_to_day_interval(interval<unixtime_t> const&);
  std::uint16_t tt_n_days();

  routing::query make_query() const;

  void add_offsets_for_pos(std::vector<routing::offset>&,
                           geo::latlng const&,
                           query_generation::transport_mode const&);

  // R-Tree
  geo::point_rtree locations_rtree_;

  // RNG
  std::random_device rd_;
  std::mt19937 rng_{rd_()};

  // Distributions
  std::uniform_int_distribution<location_idx_t::value_t> location_d_;
  std::uniform_int_distribution<date::sys_days::rep> date_d_;
  std::uniform_int_distribution<transport_idx_t::value_t> transport_d_;
  std::uniform_int_distribution<day_idx_t::value_t> day_d_;
  std::uniform_int_distribution<std::uint32_t> start_mode_range_d_;
  std::uniform_int_distribution<std::uint32_t> dest_mode_range_d_;
  std::discrete_distribution<std::uint16_t> hours_d_{begin(kTimeOfDayWeights),
                                                     end(kTimeOfDayWeights)};
  std::uniform_int_distribution<std::uint16_t> minutes_d_{0, 59};
  std::uniform_int_distribution<std::uint16_t> bearing_d_{0, 359};
};

}  // namespace nigiri::query_generation