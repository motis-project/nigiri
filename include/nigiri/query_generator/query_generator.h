#pragma once

#include <ctime>
#include "nigiri/query_generator/transport_mode.h"
#include "nigiri/routing/clasz_mask.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/location_match_mode.h"
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

namespace nigiri::query_generator {

static bool rng_initialized = false;

struct query_generator {

  query_generator(timetable const& tt,
                  duration_t const interval_size,
                  routing::location_match_mode const start_match_mode,
                  routing::location_match_mode const destination_match_mode,
                  transport_mode const& start_mode,
                  transport_mode const& dest_mode,
                  bool const use_start_footpaths,
                  std::uint8_t const max_transfers,
                  unsigned const min_connection_count,
                  bool const extend_interval_earlier,
                  bool const extend_interval_later,
                  profile_idx_t const prf_idx,
                  routing::clasz_mask_t const allowed_claszes)
      : tt_(tt),
        interval_size_(interval_size),
        start_match_mode_(start_match_mode),
        dest_match_mode_(destination_match_mode),
        start_mode_(start_mode),
        dest_mode_(dest_mode),
        use_start_footpaths_(use_start_footpaths),
        max_transfers_(max_transfers),
        min_connection_count_(min_connection_count),
        extend_interval_earlier_(extend_interval_earlier),
        extend_interval_later_(extend_interval_later),
        prf_idx_(prf_idx),
        allowed_claszes_(allowed_claszes),
        location_d_{
            static_cast<std::uint32_t>(special_station::kSpecialStationsSize),
            tt.n_locations()},
        time_d_{tt.external_interval().from_.time_since_epoch().count(),
                tt.external_interval().to_.time_since_epoch().count()},
        start_mode_range_d_{10, start_mode_.range_},
        dest_mode_range_d_{10, dest_mode_.range_} {
    if (start_match_mode == routing::location_match_mode::kIntermodal ||
        destination_match_mode == routing::location_match_mode::kIntermodal) {
      locations_rtree_ = geo::make_point_rtree(tt_.locations_.coordinates_);
    }
    if (!rng_initialized) {
      rng_ = std::mt19937(rd_());
      rng_.seed(static_cast<unsigned long>(std::time(nullptr)));
      rng_initialized = true;
    }
  }

  routing::query random_query();

private:
  geo::latlng random_point_in_range(
      geo::latlng const&, std::uniform_int_distribution<std::uint32_t>&);
  location_idx_t random_location();
  unixtime_t random_time();

  timetable const& tt_;
  duration_t const interval_size_;
  routing::location_match_mode const start_match_mode_;
  routing::location_match_mode const dest_match_mode_;
  transport_mode const& start_mode_;
  transport_mode const& dest_mode_;
  bool const use_start_footpaths_;
  std::uint8_t const max_transfers_;
  unsigned const min_connection_count_;
  bool const extend_interval_earlier_;
  bool const extend_interval_later_;
  profile_idx_t const prf_idx_;
  routing::clasz_mask_t const allowed_claszes_;

  // R-tree
  geo::point_rtree locations_rtree_;

  // Distributions
  std::uniform_int_distribution<std::uint32_t> location_d_;
  std::uniform_int_distribution<std::int32_t> time_d_;
  std::uniform_int_distribution<std::uint32_t> start_mode_range_d_;
  std::uniform_int_distribution<std::uint32_t> dest_mode_range_d_;
  std::uniform_int_distribution<std::uint32_t> bearing_d_{0, 359};

  // Static
  static std::random_device rd_;
  static std::mt19937 rng_;
};

}  // namespace nigiri::query_generator