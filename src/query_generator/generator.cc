#include "nigiri/query_generator/generator.h"

#include "nigiri/types.h"

#include "geo/latlng.h"

namespace nigiri::query_generation {

generator::generator(timetable const& tt, generator_settings const& settings)
    : tt_(tt),
      s_(settings),
      location_d_{
          static_cast<std::uint32_t>(special_station::kSpecialStationsSize),
          tt_.n_locations() - 1},
      date_d_{static_cast<std::uint32_t>(
                  tt_.date_range_.from_.time_since_epoch().count()),
              static_cast<std::uint32_t>(
                  tt_.date_range_.to_.time_since_epoch().count()) -
                  1U},
      transport_d_{0U, tt_.transport_traffic_days_.size() - 1},
      day_d_{kTimetableOffset.count(),
             static_cast<std::uint16_t>(kTimetableOffset.count() + tt_n_days() -
                                        1)},
      start_mode_range_d_{10, s_.start_mode_.range()},
      dest_mode_range_d_{10, s_.dest_mode_.range()} {
  if (!rng_initialized_) {
    rng_ = std::mt19937(rd_());
    rng_.seed(static_cast<std::uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    rng_initialized_ = true;
  }
}

unixtime_t generator::random_time() {
  return unixtime_t{std::chrono::minutes{
      date_d_(rng_) * 1440U + hours_d_(rng_) * 60U + minutes_d_(rng_)}};
}

std::optional<location_idx_t> generator::random_active_location(
    interval<unixtime_t> const& unix_iv, event_type const et) {

  // find relevant day_idx of the interval, may be multiple days
  auto const day_idx_iv = unix_to_day_interval(unix_iv);
  auto const from_tod = unix_iv.from_.time_since_epoch() % 1440;
  auto const to_tod = unix_iv.to_.time_since_epoch() % 1440;

  for (auto i = 0U; i < max_gen_attempts_; ++i) {

    auto const transport_idx = random_transport_idx();
    auto const stop_idx = random_active_stop(transport_idx, et);
    if (!stop_idx.has_value()) {
      continue;
    }

    auto const event_delta = tt_.event_mam(transport_idx, stop_idx.value(), et);

    // check if event happens during the interval
    for (auto day_idx : day_idx_iv) {
      // skip if transport is not active on this day
      if (!tt_.bitfields_[tt_.transport_traffic_days_[transport_idx]].test(
              day_idx.v_ - event_delta.days_)) {
        continue;
      }
      // skip if event's time of day lies before interval begin on first day
      if (day_idx == day_idx_iv.from_ && event_delta.mam_ < from_tod.count()) {
        continue;
      }
      // skip if event's time of day lies after interval end on last day
      if (day_idx == day_idx_iv.to_ - 1 && to_tod.count() < event_delta.mam_) {
        continue;
      }

      // transport is active and event time of day lies within the interval
      return stop{tt_.route_location_seq_[tt_.transport_route_[transport_idx]]
                                         [stop_idx.value()]}
          .location_idx();
    }
  }

  return std::nullopt;
}

geo::latlng generator::pos_near_start(location_idx_t const loc_idx) {
  auto const loc_pos = tt_.locations_.coordinates_[loc_idx];
  return random_point_in_range(loc_pos, start_mode_range_d_);
}

geo::latlng generator::pos_near_dest(location_idx_t const loc_idx) {
  auto const loc_pos = tt_.locations_.coordinates_[loc_idx];
  return random_point_in_range(loc_pos, dest_mode_range_d_);
}

std::pair<transport, stop_idx_t> generator::random_transport_active_stop(
    event_type const et) {
  transport tr;
  std::optional<stop_idx_t> stop_idx = std::nullopt;
  while (!stop_idx.has_value()) {
    auto const tr_idx = random_transport_idx();
    auto const day_idx = random_active_day(tr_idx);
    if (!day_idx.has_value()) {
      continue;
    }
    tr.t_idx_ = tr_idx;
    tr.day_ = day_idx.value();

    stop_idx = random_active_stop(tr_idx, et);
  }
  return {tr, stop_idx.value()};
}

transport_idx_t generator::random_transport_idx() {
  return transport_idx_t{transport_d_(rng_)};
}

day_idx_t generator::random_day() { return day_idx_t{day_d_(rng_)}; }

std::optional<day_idx_t> generator::random_active_day(
    nigiri::transport_idx_t const tr_idx) {
  // try randomize
  auto const& bf = tt_.bitfields_[tt_.transport_traffic_days_[tr_idx]];
  for (auto i = 0U; i < max_gen_attempts_; ++i) {
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

std::optional<stop_idx_t> generator::random_active_stop(
    transport_idx_t const tr_idx, event_type const et) {

  // distribution for stop index
  // initial stop is only valid for departure events
  // final stop is only valid for arrival events
  std::uniform_int_distribution<stop_idx_t> stop_d{
      static_cast<stop_idx_t>(et == event_type::kDep ? 0U : 1U),
      static_cast<stop_idx_t>(
          tt_.route_location_seq_[tt_.transport_route_[tr_idx]].size() -
          (et == event_type::kArr ? 1U : 2U))};

  for (auto i = 0U; i < max_gen_attempts_; ++i) {

    // randomize stop index
    auto const stop_idx = stop_idx_t{stop_d(rng_)};

    switch (et) {
      case event_type::kDep: {
        if (stop{
                tt_.route_location_seq_[tt_.transport_route_[tr_idx]][stop_idx]}
                .in_allowed()) {
          return stop_idx;
        }
        break;
      }
      case event_type::kArr: {
        if (stop{
                tt_.route_location_seq_[tt_.transport_route_[tr_idx]][stop_idx]}
                .out_allowed()) {
          return stop_idx;
        }
        break;
      }
    }
  }
  return std::nullopt;
}

geo::latlng generator::random_point_in_range(
    geo::latlng const& c, std::uniform_int_distribution<std::uint32_t>& d) {
  return geo::destination_point(c, static_cast<double>(d(rng_)),
                                static_cast<double>(bearing_d_(rng_)));
}

interval<day_idx_t> generator::unix_to_day_interval(
    interval<unixtime_t> const& iv) {
  auto const start_day_idx =
      tt_.day_idx(std::chrono::floor<std::chrono::days>(iv.from_));
  auto const end_day_idx = tt_.day_idx(
      std::chrono::floor<std::chrono::days>(iv.to_) + std::chrono::days(1));
  return {start_day_idx, end_day_idx};
}

std::uint16_t generator::tt_n_days() {
  return static_cast<std::uint16_t>(tt_.date_range_.size().count());
}

}  // namespace nigiri::query_generation