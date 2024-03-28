#include "nigiri/query_generator/query_generator.h"

#include <cmath>

#include "nigiri/routing/ontrip_train.h"
#include "nigiri/routing/query.h"
#include "nigiri/types.h"

#include "geo/constants.h"
#include "geo/latlng.h"

namespace nigiri::query_generation {

void query_generator::init_rng() {
  location_d_ = std::uniform_int_distribution<std::uint32_t>{
      static_cast<std::uint32_t>(special_station::kSpecialStationsSize),
      tt_.n_locations() - 1};
  date_d_ = std::uniform_int_distribution<std::uint32_t>{
      static_cast<std::uint32_t>(
          tt_.date_range_.from_.time_since_epoch().count()),
      static_cast<std::uint32_t>(
          tt_.date_range_.to_.time_since_epoch().count()) -
          1U};
  transport_d_ = std::uniform_int_distribution<std::uint32_t>{
      0U, tt_.transport_traffic_days_.size() - 1};
  day_d_ = std::uniform_int_distribution<std::uint16_t>{
      kTimetableOffset.count(),
      static_cast<std::uint16_t>(kTimetableOffset.count() + tt_n_days() - 1)};
  start_mode_range_d_ =
      std::uniform_int_distribution<std::uint32_t>{10, start_mode_.range()};
  dest_mode_range_d_ =
      std::uniform_int_distribution<std::uint32_t>{10, dest_mode_.range()};
  if (start_match_mode_ == routing::location_match_mode::kIntermodal ||
      dest_match_mode_ == routing::location_match_mode::kIntermodal) {
    locations_rtree_ = geo::make_point_rtree(tt_.locations_.coordinates_);
  }
  if (!rng_initialized_) {
    rng_ = std::mt19937(rd_());
    rng_.seed(static_cast<unsigned long>(std::time(nullptr)));
    rng_initialized_ = true;
  }
}

geo::latlng query_generator::random_start_pos() {
  return random_point_in_range(tt_.locations_.coordinates_[random_location()],
                               start_mode_range_d_);
}

geo::latlng query_generator::random_dest_pos() {
  return random_point_in_range(tt_.locations_.coordinates_[random_location()],
                               dest_mode_range_d_);
}

std::string query_generator::random_stop_id() {
  auto const& random_location_id = tt_.locations_.ids_[random_location()];
  return {random_location_id.begin(), random_location_id.end()};
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
  return unixtime_t{std::chrono::minutes{
      date_d_(rng_) * 1440U + hours_d_(rng_) * 60U + minutes_d_(rng_)}};
}

transport_idx_t query_generator::random_transport_idx() {
  return transport_idx_t{transport_d_(rng_)};
}

stop_idx_t query_generator::random_stop(transport_idx_t const tr_idx) {
  std::uniform_int_distribution<stop_idx_t> stop_d{
      1U,
      static_cast<stop_idx_t>(
          tt_.route_location_seq_[tt_.transport_route_[tr_idx]].size() - 1)};
  return stop_idx_t{stop_d(rng_)};
}

std::int32_t query_generator::tt_n_days() {
  return tt_.date_range_.size().count();
}

day_idx_t query_generator::random_day() { return day_idx_t{day_d_(rng_)}; }

std::optional<day_idx_t> query_generator::random_active_day(
    nigiri::transport_idx_t const tr_idx) {
  // try randomize
  auto const& bf = tt_.bitfields_[tt_.transport_traffic_days_[tr_idx]];
  for (auto i = 0U; i < 100U; ++i) {
    auto const d_idx = random_day();
    if (bf.test(d_idx.v_)) {
      return d_idx;
    }
  }
  // look from beginning
  for (auto d = static_cast<std::uint16_t>(kTimetableOffset.count());
       d < tt_n_days(); ++d) {
    if (bf.test(d)) {
      return day_idx_t{d};
    }
  }
  // no active day found
  return std::nullopt;
}

void query_generator::add_offsets_for_pos(
    std::vector<routing::offset>& o,
    geo::latlng const& pos,
    query_generation::transport_mode const& mode) {
  auto const locs_in_range = locations_rtree_.in_radius(pos, mode.range());
  for (auto const loc : locs_in_range) {
    auto const duration = duration_t{
        static_cast<std::int16_t>(
            geo::distance(pos,
                          tt_.locations_.coordinates_[location_idx_t{loc}]) /
            (mode.speed_ * 60)) +
        1};
    o.emplace_back(location_idx_t{loc}, duration, mode.mode_id_);
  }
}

void query_generator::init_query(routing::query& q) {
  q.start_match_mode_ = start_match_mode_;
  q.dest_match_mode_ = dest_match_mode_;
  q.use_start_footpaths_ = use_start_footpaths_;
  q.max_transfers_ = max_transfers_;
  q.min_connection_count_ = min_connection_count_;
  q.extend_interval_earlier_ = extend_interval_earlier_;
  q.extend_interval_later_ = extend_interval_later_;
  q.prf_idx_ = prf_idx_;
  q.allowed_claszes_ = allowed_claszes_;
}

void query_generator::add_time(routing::query& q) {
  auto const start_time = random_time();
  if (interval_size_.count() == 0) {
    q.start_time_ = start_time;
  } else {
    q.start_time_ =
        interval<unixtime_t>{start_time, start_time + interval_size_};
  }
}

void query_generator::add_starts(routing::query& q) {
  if (start_match_mode_ == routing::location_match_mode::kIntermodal) {
    add_offsets_for_pos(q.start_, random_start_pos(), start_mode_);
  } else {
    q.start_.emplace_back(random_location(), 0_minutes, 0U);
  }
}

void query_generator::add_dests(routing::query& q) {
  if (dest_match_mode_ == routing::location_match_mode::kIntermodal) {
    add_offsets_for_pos(q.destination_, random_dest_pos(), dest_mode_);
  } else {
    q.destination_.emplace_back(random_location(), 0_minutes, 0U);
  }
}

routing::query query_generator::random_pretrip_query() {

  routing::query q;
  init_query(q);
  add_time(q);
  add_starts(q);
  add_dests(q);

  return q;
}

routing::query query_generator::random_ontrip_query() {

  routing::query q;
  init_query(q);

  auto const tr_idx = random_transport_idx();
  auto const d_idx = random_active_day(tr_idx);
  if (d_idx.has_value()) {
    transport const tr{tr_idx, d_idx.value()};
    auto const s = random_stop(tr_idx);
    routing::generate_ontrip_train_query(tt_, tr, s, q);
  } else {
    std::cout << "WARNING: Could not find active day for transport\n";
  }

  add_dests(q);

  return q;
}

}  // namespace nigiri::query_generation