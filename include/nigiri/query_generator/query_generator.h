#pragma once

#include <ctime>
#include "nigiri/routing/clasz_mask.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/location_match_mode.h"
#include "nigiri/special_stations.h"
#include "nigiri/timetable.h"
#include <random>

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
                  duration_t interval_size,
                  routing::location_match_mode const start_match_mode,
                  routing::location_match_mode const destination_match_mode,
                  bool use_start_footpaths,
                  std::uint8_t max_transfers,
                  unsigned min_connection_count,
                  bool extend_interval_earlier,
                  bool extend_interval_later,
                  profile_idx_t prf_idx,
                  routing::clasz_mask_t allowed_claszes)
      : tt_(tt),
        location_d_{
            static_cast<std::uint32_t>(special_station::kSpecialStationsSize),
            tt.n_locations()},
        time_d_{tt.external_interval().from_.time_since_epoch().count(),
                tt.external_interval().to_.time_since_epoch().count()},
        interval_size_(interval_size),
        start_match_mode_(start_match_mode),
        dest_match_mode_(destination_match_mode),
        use_start_footpaths_(use_start_footpaths),
        max_transfers_(max_transfers),
        min_connection_count_(min_connection_count),
        extend_interval_earlier_(extend_interval_earlier),
        extend_interval_later_(extend_interval_later),
        prf_idx_(prf_idx),
        allowed_claszes_(allowed_claszes) {
    if (!rng_initialized) {
      rng_ = std::mt19937(rd_());
      rng_.seed(static_cast<unsigned long>(std::time(nullptr)));
      rng_initialized = true;
    }
  }

  routing::query random_query();

private:
  location_idx_t random_location();
  unixtime_t random_time();

  timetable const& tt_;
  std::uniform_int_distribution<std::uint32_t> location_d_;
  std::uniform_int_distribution<std::int32_t> time_d_;
  duration_t interval_size_;
  routing::location_match_mode start_match_mode_;
  routing::location_match_mode dest_match_mode_;
  bool use_start_footpaths_;
  std::uint8_t max_transfers_;
  unsigned min_connection_count_;
  bool extend_interval_earlier_;
  bool extend_interval_later_;
  profile_idx_t prf_idx_;
  routing::clasz_mask_t allowed_claszes_;

  static std::random_device rd_;
  static std::mt19937 rng_;
};

}  // namespace nigiri::query_generator