#include "nigiri/query_generator/query_generator.h"

#include "nigiri/routing/query.h"
#include "nigiri/types.h"

namespace nigiri::query_generator {

location_idx_t query_generator::random_location() {
  return location_idx_t{location_d_(rng_)};
}

unixtime_t query_generator::random_time() {
  return unixtime_t{duration_t{time_d_(rng_)}};
}

routing::query query_generator::random_query() {
  routing::query q;

  auto const start_time = random_time();
  if (interval_size_.count() == 0) {
    q.start_time_ = start_time;
  } else {
    q.start_time_ =
        interval<unixtime_t>{start_time, start_time + interval_size_};
  }

  q.start_match_mode_ = start_match_mode_;
  q.dest_match_mode_ = dest_match_mode_;
  q.use_start_footpaths_ = use_start_footpaths_;

  auto const source = random_location();
  auto const destination = random_location();
  // draw circle around source/destination if
  // start_match_mode/destination_match_mode is intermodal find random
  // coordinates within that circle -> start/end coordinates find all stations
  // in a circle around start/end coordinates add them to start/destination
  // offsets

  return q;
}

}  // namespace nigiri::query_generator