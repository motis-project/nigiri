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

geo::latlng query_generator::random_active_pos(
    interval<unixtime_t> const& unix_iv, event_type const et) {
  return random_point_in_range(
      tt_.locations_.coordinates_[random_active_location(unix_iv, et)],
      et == event_type::kDep ? start_mode_range_d_ : dest_mode_range_d_);
}

geo::latlng query_generator::random_point_in_range(
    geo::latlng const& c, std::uniform_int_distribution<std::uint32_t>& d) {
  return geo::destination_point(c, static_cast<double>(d(rng_)),
                                static_cast<double>(bearing_d_(rng_)));
}

location_idx_t query_generator::random_location() {
  return location_idx_t{location_d_(rng_)};
}

location_idx_t query_generator::random_active_location(
    interval<unixtime_t> const& unix_iv, event_type const et) {

  // find relevent day_idx of the interval, may be multiple days
  auto const day_idx_iv = unix_to_day_interval(unix_iv);
  auto const from_tod = unix_iv.from_.time_since_epoch() % 1440;
  auto const to_tod = unix_iv.to_.time_since_epoch() % 1440;

  std::optional<location_idx_t> active_location = std::nullopt;
  while (!active_location.has_value()) {
    auto const transport_idx = random_transport_idx();
    auto const stop_idx = random_active_stop(transport_idx, et);
    auto const event_delta = tt_.event_mam(transport_idx, stop_idx, et);

    // check if event happens during the interval
    for (auto day_idx : day_idx_iv) {
      if (!tt_.bitfields_[tt_.transport_traffic_days_[transport_idx]].test(
              day_idx.v_ - event_delta.days_)) {
        continue;  // skip if transport not active on this day
      }
      if (day_idx == day_idx_iv.from_ && event_delta.mam_ < from_tod.count()) {
        continue;  // skip if event is before interval begin on first day
      }
      if (day_idx == day_idx_iv.to_ - 1 && to_tod.count() < event_delta.mam_) {
        continue;  // skip if event_tod is after interval end on last day
      }
      // transport is active and event time of day lies within the interval
      active_location =
          stop{tt_.route_location_seq_[tt_.transport_route_[transport_idx]]
                                      [stop_idx]}
              .location_idx();
      break;
    }
  }

  return active_location.value();
}

interval<day_idx_t> query_generator::unix_to_day_interval(
    interval<unixtime_t> const& iv) {
  auto const start_day_idx =
      tt_.day_idx(std::chrono::floor<std::chrono::days>(iv.from_));
  auto const end_day_idx = tt_.day_idx(
      std::chrono::floor<std::chrono::days>(iv.to_) + std::chrono::days(1));
  return {start_day_idx, end_day_idx};
}

unixtime_t query_generator::random_time() {
  return unixtime_t{std::chrono::minutes{
      date_d_(rng_) * 1440U + hours_d_(rng_) * 60U + minutes_d_(rng_)}};
}

on_trip_export query_generator::random_on_trip() {
  auto const [tr, stop_idx] = random_transport_active_stop(event_type::kArr);
  auto const merged_trips_idx =
      tt_.transport_to_trip_section_[tr.t_idx_].size() == 1
          ? tt_.transport_to_trip_section_[tr.t_idx_][0]  // all sections belong
                                                          // to the same trip
          : tt_.transport_to_trip_section_[tr.t_idx_][stop_idx];
  auto const trip_idx =
      tt_.merged_trips_[merged_trips_idx][0];  // 0 until more than one trip is
                                               // merged in the transport
  auto const trip_stop =
      stop{tt_.route_location_seq_[tt_.transport_route_[tr.t_idx_]][stop_idx]}
          .location_idx();
  auto const unixtime_arr_stop = tt_.event_time(tr, stop_idx, event_type::kArr);
  return {.trip_idx_ = trip_idx,
          .transport_ = tr,
          .trip_stop_ = trip_stop,
          .unixtime_arr_stop_ = unixtime_arr_stop};
}

std::pair<transport, stop_idx_t> query_generator::random_transport_active_stop(
    event_type const et) {
  transport tr;
  auto const tr_idx = random_transport_idx();
  auto const day_idx = random_active_day(tr_idx);
  if (day_idx.has_value()) {
    tr.t_idx_ = tr_idx;
    tr.day_ = day_idx.value();
  }
  auto const stop = random_active_stop(tr_idx, et);
  return {tr, stop};
}

transport_idx_t query_generator::random_transport_idx() {
  return transport_idx_t{transport_d_(rng_)};
}

stop_idx_t query_generator::random_active_stop(transport_idx_t const tr_idx,
                                               event_type const et) {
  std::uniform_int_distribution<stop_idx_t> stop_d{
      static_cast<stop_idx_t>(et == event_type::kDep ? 0U : 1U),
      static_cast<stop_idx_t>(
          tt_.route_location_seq_[tt_.transport_route_[tr_idx]].size() - 1)};
  while (true) {
    auto const stop_idx = stop_idx_t{stop_d(rng_)};
    if (et == event_type::kDep &&
        stop{tt_.route_location_seq_[tt_.transport_route_[tr_idx]][stop_idx]}
            .in_allowed()) {
      return stop_idx;
    } else if (et == event_type::kArr &&
               stop{tt_.route_location_seq_[tt_.transport_route_[tr_idx]]
                                           [stop_idx]}
                   .out_allowed()) {
      return stop_idx;
    }
  }
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

routing::query query_generator::random_pretrip_query() {

  routing::query q;
  init_query(q);

  auto const start_time = random_time();
  auto const start_interval =
      interval<unixtime_t>{start_time, start_time + interval_size_};
  auto const dest_interval =
      interval<unixtime_t>{start_time, start_time + duration_t{1440U}};
  if (interval_size_.count() == 0) {
    q.start_time_ = start_time;
  } else {
    q.start_time_ = start_interval;
  }

  if (start_match_mode_ == routing::location_match_mode::kIntermodal) {
    add_offsets_for_pos(q.start_,
                        random_active_pos(start_interval, event_type::kDep),
                        start_mode_);
  } else {
    q.start_.emplace_back(
        random_active_location(start_interval, event_type::kDep), 0_minutes,
        0U);
  }

  if (dest_match_mode_ == routing::location_match_mode::kIntermodal) {
    add_offsets_for_pos(q.destination_,
                        random_active_pos(dest_interval, event_type::kArr),
                        dest_mode_);
  } else {
    q.destination_.emplace_back(
        random_active_location(dest_interval, event_type::kArr), 0_minutes, 0U);
  }

  return q;
}

routing::query query_generator::random_ontrip_query() {

  routing::query q;
  init_query(q);

  auto const [tr, stop_idx] = random_transport_active_stop(event_type::kArr);
  routing::generate_ontrip_train_query(tt_, tr, stop_idx, q);

  auto const dest_interval = interval<unixtime_t>{
      std::get<unixtime_t>(q.start_time_),
      std::get<unixtime_t>(q.start_time_) + duration_t{1440U}};

  if (dest_match_mode_ == routing::location_match_mode::kIntermodal) {
    add_offsets_for_pos(q.destination_,
                        random_active_pos(dest_interval, event_type::kArr),
                        dest_mode_);
  } else {
    q.destination_.emplace_back(
        random_active_location(dest_interval, event_type::kArr), 0_minutes, 0U);
  }

  return q;
}

}  // namespace nigiri::query_generation