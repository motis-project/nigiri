#pragma once

#include <ctime>
#include "nigiri/query_generator/transport_mode.h"
#include "nigiri/routing/clasz_mask.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/location_match_mode.h"
#include "nigiri/routing/query.h"
#include "nigiri/special_stations.h"
#include "nigiri/timetable.h"
#include <random>
#include "geo/point_rtree.h"

namespace nigiri {
struct timetable;
}  // namespace nigiri

namespace nigiri::routing {
struct query;
}  // namespace nigiri::routing

namespace nigiri::query_generation {

struct on_trip_export {
  trip_idx_t trip_idx_;
  transport transport_;
  std::string trip_stop_id_;  // the intermediate stop along the trip
  unixtime_t unixtime_arr_stop_;  // the arrival time at the intermediate stop
};

struct query_generator {
  explicit query_generator(timetable const& tt) : tt_(tt) { init_rng(); }

  // initializes the RNG as well as some distributions, should
  // be called after adjusting public options
  void init_rng();

  // Public interface
  geo::latlng random_start_pos();
  geo::latlng random_dest_pos();
  std::string random_stop_id();
  unixtime_t random_time();
  on_trip_export random_on_trip();

  // Generate queries from within nigiri
  routing::query random_pretrip_query();
  routing::query random_ontrip_query();

  // Public options
  timetable const& tt_;
  duration_t interval_size_{60U};
  routing::location_match_mode start_match_mode_{
      routing::location_match_mode::kIntermodal};
  routing::location_match_mode dest_match_mode_{
      routing::location_match_mode::kIntermodal};
  transport_mode start_mode_{kWalk};
  transport_mode dest_mode_{kWalk};
  bool use_start_footpaths_{true};
  std::uint8_t max_transfers_{routing::kMaxTransfers};
  unsigned min_connection_count_{0U};
  bool extend_interval_earlier_{false};
  bool extend_interval_later_{false};
  profile_idx_t prf_idx_{0};
  routing::clasz_mask_t allowed_claszes_{routing::all_clasz_allowed()};

private:
  std::pair<transport, stop_idx_t> random_transport_and_stop_idx();

  geo::latlng random_point_in_range(
      geo::latlng const&, std::uniform_int_distribution<std::uint32_t>&);

  location_idx_t random_location();

  transport_idx_t random_transport_idx();

  std::int32_t tt_n_days();

  day_idx_t random_day();

  std::optional<day_idx_t> random_active_day(transport_idx_t);

  stop_idx_t random_stop(transport_idx_t);

  void add_offsets_for_pos(std::vector<routing::offset>&,
                           geo::latlng const&,
                           query_generation::transport_mode const&);

  void init_query(routing::query&);

  void add_time(routing::query&);

  void add_starts(routing::query&);

  void add_dests(routing::query&);

  // R-tree
  geo::point_rtree locations_rtree_;

  // Distributions
  std::uniform_int_distribution<std::uint32_t> location_d_;
  std::uniform_int_distribution<std::uint32_t> date_d_;
  std::uniform_int_distribution<std::uint32_t> transport_d_;
  std::uniform_int_distribution<std::uint16_t> day_d_;
  std::uniform_int_distribution<std::uint32_t> start_mode_range_d_;
  std::uniform_int_distribution<std::uint32_t> dest_mode_range_d_;

  // Static
  inline static bool rng_initialized_ = false;
  inline static std::random_device rd_;
  inline static std::mt19937 rng_;
  inline static std::uniform_int_distribution<std::uint32_t> bearing_d_{0, 359};
  constexpr static int time_of_day_weights_[] = {
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
  inline static std::discrete_distribution<std::uint32_t> hours_d_{
      std::begin(time_of_day_weights_), std::end(time_of_day_weights_)};
  inline static std::uniform_int_distribution<std::uint32_t> minutes_d_{0, 59};
};

}  // namespace nigiri::query_generation