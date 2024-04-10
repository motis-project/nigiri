#include "nigiri/query_generator/generator.h"

#include "nigiri/routing/location_match_mode.h"
#include "nigiri/routing/ontrip_train.h"
#include "nigiri/special_stations.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::query_generation {

generator::generator(timetable const& tt, generator_settings const& settings)
    : tt_{tt},
      s_{settings},
      location_d_{
          static_cast<std::uint32_t>(special_station::kSpecialStationsSize),
          tt_.n_locations() - 1},
      date_d_{tt_.date_range_.from_.time_since_epoch().count(),

              tt_.date_range_.to_.time_since_epoch().count() -
                  date::sys_days::rep{1U}},
      transport_d_{0U, tt_.transport_traffic_days_.size() - 1U},
      day_d_{kTimetableOffset.count(),
             static_cast<day_idx_t::value_t>(kTimetableOffset.count() +
                                             tt_n_days() - 1)},
      start_mode_range_d_{10, s_.start_mode_.range()},
      dest_mode_range_d_{10, s_.dest_mode_.range()} {
  if (settings.start_match_mode_ == routing::location_match_mode::kIntermodal ||
      settings.dest_match_mode_ == routing::location_match_mode::kIntermodal) {
    locations_rtree_ = geo::make_point_rtree(tt_.locations_.coordinates_);
  }
}

unixtime_t generator::random_time() {
  return unixtime_t{std::chrono::minutes{
      date_d_(rng_) * 1440 + hours_d_(rng_) * 60 + minutes_d_(rng_)}};
}

std::optional<location_idx_t> generator::random_active_location(
    interval<unixtime_t> const& unix_iv, event_type const et) {

  // find relevant day_idx of the interval, may be multiple days
  auto const day_idx_iv = unix_to_day_interval(unix_iv);
  auto const from_tod = unix_iv.from_.time_since_epoch() % 1440;
  auto const to_tod = unix_iv.to_.time_since_epoch() % 1440;

  for (auto i = 0U; i < kMaxGenAttempts; ++i) {

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
  for (auto i = 0U; i < kMaxGenAttempts; ++i) {
    auto const d_idx = random_day();
    if (bf.test(d_idx.v_)) {
      return d_idx;
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
      (et == event_type::kDep ? stop_idx_t{0U} : stop_idx_t{1U}),
      static_cast<stop_idx_t>(
          tt_.route_location_seq_[tt_.transport_route_[tr_idx]].size() -
          (et == event_type::kArr ? 1U : 2U))};

  for (auto i = 0U; i < kMaxGenAttempts; ++i) {

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

routing::query generator::make_query() const {
  routing::query q;
  q.start_match_mode_ = s_.start_match_mode_;
  q.dest_match_mode_ = s_.dest_match_mode_;
  q.use_start_footpaths_ = s_.use_start_footpaths_;
  q.max_transfers_ = s_.max_transfers_;
  q.min_connection_count_ = s_.min_connection_count_;
  q.extend_interval_earlier_ = s_.extend_interval_earlier_;
  q.extend_interval_later_ = s_.extend_interval_later_;
  q.prf_idx_ = s_.prf_idx_;
  q.allowed_claszes_ = s_.allowed_claszes_;
  return q;
}

void generator::add_offsets_for_pos(
    std::vector<routing::offset>& o,
    geo::latlng const& pos,
    query_generation::transport_mode const& mode) {
  for (auto const loc : locations_rtree_.in_radius(pos, mode.range())) {
    auto const duration = duration_t{
        static_cast<std::int16_t>(
            geo::distance(pos,
                          tt_.locations_.coordinates_[location_idx_t{loc}]) /
            (mode.speed_ * 60)) +
        1};
    o.emplace_back(location_idx_t{loc}, duration, mode.mode_id_);
  }
}

std::optional<routing::query> generator::random_pretrip_query() {

  for (auto i = 0U; i < kMaxGenAttempts; ++i) {
    // start with a new query
    auto q = make_query();

    // randomize time
    auto const start_time = random_time();
    auto const start_interval =
        interval<unixtime_t>{start_time, start_time + s_.interval_size_};
    auto const dest_interval =
        interval<unixtime_t>{start_time, start_time + duration_t{1440U}};
    if (s_.interval_size_.count() == 0) {
      q.start_time_ = start_time;
    } else {
      q.start_time_ = start_interval;
    }

    // randomize start location
    auto const start_loc_idx =
        random_active_location(start_interval, event_type::kDep);
    if (!start_loc_idx.has_value()) {
      continue;
    }

    // randomize destination location
    auto const dest_loc_idx =
        random_active_location(dest_interval, event_type::kArr);
    if (!dest_loc_idx.has_value()) {
      continue;
    }

    if (start_loc_idx.value() == dest_loc_idx.value()) {
      continue;
    }

    // add start(s) to query
    if (s_.start_match_mode_ == routing::location_match_mode::kIntermodal) {
      add_offsets_for_pos(q.start_, pos_near_start(start_loc_idx.value()),
                          s_.start_mode_);
    } else {
      q.start_.emplace_back(start_loc_idx.value(), 0_minutes, 0U);
    }

    // add destination(s) to query
    if (s_.dest_match_mode_ == routing::location_match_mode::kIntermodal) {
      add_offsets_for_pos(q.destination_, pos_near_dest(dest_loc_idx.value()),
                          s_.dest_mode_);
    } else {
      q.destination_.emplace_back(dest_loc_idx.value(), 0_minutes, 0U);
    }

    return q;
  }

  std::cout << "WARNING: failed to generate a valid query after "
            << kMaxGenAttempts << " attempts\n";

  return std::nullopt;
}

std::optional<routing::query> generator::random_ontrip_query() {

  for (auto i = 0U; i < kMaxGenAttempts; ++i) {
    // start with a new query
    auto q = make_query();

    // randomize transport and stop index
    auto const [tr, stop_idx] = random_transport_active_stop(event_type::kArr);

    // generate ontrip train query for transport and stop index
    routing::generate_ontrip_train_query(tt_, tr, stop_idx, q);

    // set interval at destination
    auto const dest_interval = interval<unixtime_t>{
        std::get<unixtime_t>(q.start_time_),
        std::get<unixtime_t>(q.start_time_) + duration_t{1440U}};

    // randomize destination location
    auto const dest_loc_idx =
        random_active_location(dest_interval, event_type::kArr);
    if (!dest_loc_idx.has_value()) {
      continue;
    }

    auto const start_loc_idx =
        stop{tt_.route_location_seq_[tt_.transport_route_[tr.t_idx_]][stop_idx]}
            .location_idx();
    if (start_loc_idx == dest_loc_idx.value()) {
      continue;
    }

    // add destination(s) to query
    if (s_.dest_match_mode_ == routing::location_match_mode::kIntermodal) {
      add_offsets_for_pos(q.destination_, pos_near_dest(dest_loc_idx.value()),
                          s_.dest_mode_);
    } else {
      q.destination_.emplace_back(dest_loc_idx.value(), 0_minutes, 0U);
    }

    return q;
  }

  std::cout << "WARNING: failed to generate a valid query after "
            << kMaxGenAttempts << " attempts\n";

  return std::nullopt;
}

}  // namespace nigiri::query_generation