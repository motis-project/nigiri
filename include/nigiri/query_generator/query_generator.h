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

static bool rng_initialized = false;

struct query_generator {
  query_generator(timetable const& tt) : tt_(tt) { init_rng(); }

  void init_rng();

  geo::latlng random_start_pos();

  geo::latlng random_dest_pos();

  // returns a time point within the timetable [unixtime in minutes]
  unixtime_t random_time();

  routing::query random_pretrip_query();

  routing::query random_ontrip_query();

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
  geo::latlng random_point_in_range(
      geo::latlng const&, std::uniform_int_distribution<std::uint32_t>&);

  location_idx_t random_location();

  transport_idx_t random_transport_idx();

  std::int32_t tt_n_days();

  day_idx_t random_day();

  std::optional<day_idx_t> random_active_day(transport_idx_t const);

  stop_idx_t random_stop(transport_idx_t const);

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
  std::uniform_int_distribution<std::int32_t> time_d_;
  std::uniform_int_distribution<std::uint32_t> transport_d_;
  std::uniform_int_distribution<std::uint16_t> day_d_;
  std::uniform_int_distribution<std::uint32_t> start_mode_range_d_;
  std::uniform_int_distribution<std::uint32_t> dest_mode_range_d_;

  // Static
  inline static std::random_device rd_;
  inline static std::mt19937 rng_;
  inline static std::uniform_int_distribution<std::uint32_t> bearing_d_{0, 359};
};

}  // namespace nigiri::query_generation