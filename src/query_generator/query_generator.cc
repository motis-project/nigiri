#include "nigiri/query_generator/query_generator.h"

#include <cmath>

#include "nigiri/routing/query.h"
#include "nigiri/types.h"

#include "geo/constants.h"
#include "geo/latlng.h"

namespace nigiri::query_generator {

void add_locs_from_vec(std::vector<routing::offset> const& o_vec,
                       std::vector<size_t> l_vec) {
  for (auto const l : l_vec) {
  }
}

geo::latlng query_generator::random_point_in_range(
    geo::latlng const& c, std::uniform_int_distribution<std::uint32_t>& d) {
  return geo::destination_point(c, static_cast<double>(d(rng_)),
                                static_cast<double>(bearing_d_(rng_)));
}

location_idx_t query_generator::random_location() {
  return location_idx_t{location_d_(rng_)};
}

unixtime_t query_generator::random_time() {
  return unixtime_t{duration_t{time_d_(rng_)}};
}

routing::query query_generator::random_query() {
  routing::query q;
  q.start_match_mode_ = start_match_mode_;
  q.dest_match_mode_ = dest_match_mode_;
  q.use_start_footpaths_ = use_start_footpaths_;
  q.max_transfers_ = max_transfers_;
  q.min_connection_count_ = min_connection_count_;
  q.extend_interval_earlier_ = extend_interval_earlier_;
  q.extend_interval_later_ = extend_interval_later_;
  q.prf_idx_ = prf_idx_;
  q.allowed_claszes_ = allowed_claszes_;

  // start time or interval
  auto const start_time = random_time();
  if (interval_size_.count() == 0) {
    q.start_time_ = start_time;
  } else {
    q.start_time_ =
        interval<unixtime_t>{start_time, start_time + interval_size_};
  }

  // randomize source
  auto source = random_location();
  if (start_match_mode_ == routing::location_match_mode::kIntermodal) {
    auto const random_start_pos = random_point_in_range(
        tt_.locations_.coordinates_[source], start_mode_range_d_);
    auto const l_in_range =
        locations_rtree_.in_radius(random_start_pos, start_mode_range_d_.max());
  }

  // randomize destination
  auto destination = random_location();
  if (dest_match_mode_ == routing::location_match_mode::kIntermodal) {
    auto const random_dest_pos = random_point_in_range(
        tt_.locations_.coordinates_[destination], dest_mode_range_d_);
  }
  // draw circle around source/destination if
  // start_match_mode/destination_match_mode is intermodal find random
  // coordinates within range

  return q;
}

}  // namespace nigiri::query_generator