#include "nigiri/query_generator/generator.h"

#include "nigiri/logging.h"
#include "nigiri/routing/location_match_mode.h"
#include "nigiri/routing/ontrip_train.h"
#include "nigiri/special_stations.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::query_generation {

generator::generator(timetable const& tt, generator_settings const& settings)
    : tt_{tt},
      s_{settings},
      seed_{std::random_device{}()},
      rng_{seed_},
      location_d_{
          static_cast<std::uint32_t>(special_station::kSpecialStationsSize),
          tt_.n_locations() - 1},
      transport_d_{0U, tt_.transport_traffic_days_.size() - 1U},
      day_d_{kTimetableOffset.count(),
             static_cast<day_idx_t::value_t>(kTimetableOffset.count() +
                                             tt_n_days() - 1)},
      start_mode_range_d_{10, s_.start_mode_.range()},
      dest_mode_range_d_{10, s_.dest_mode_.range()} {
  init_geo(settings);
}

generator::generator(timetable const& tt,
                     generator_settings const& settings,
                     std::uint32_t seed)
    : tt_{tt},
      s_{settings},
      seed_{seed},
      rng_{seed_},
      location_d_{
          static_cast<std::uint32_t>(special_station::kSpecialStationsSize),
          tt_.n_locations() - 1U},
      transport_d_{0U, tt_.transport_traffic_days_.size() - 1U},
      day_d_{kTimetableOffset.count(),
             static_cast<day_idx_t::value_t>(kTimetableOffset.count() +
                                             tt_n_days() - 1)},
      start_mode_range_d_{10, s_.start_mode_.range()},
      dest_mode_range_d_{10, s_.dest_mode_.range()} {
  init_geo(settings);
}

void generator::init_geo(generator_settings const& settings) {
  if (settings.start_match_mode_ == routing::location_match_mode::kIntermodal ||
      settings.dest_match_mode_ == routing::location_match_mode::kIntermodal ||
      settings.bbox_.has_value()) {
    locations_rtree_ = geo::make_point_rtree(tt_.locations_.coordinates_);
    if (settings.bbox_.has_value()) {
      locs_in_bbox = locations_rtree_.within(s_.bbox_.value());
      locs_in_bbox_d_ =
          std::uniform_int_distribution<size_t>{0U, locs_in_bbox.size() - 1U};
    }
  }
}

std::optional<routing::query> generator::random_pretrip_query() {
  for (auto i = 0U; i != kMaxGenAttempts; ++i) {
    auto q = make_query();

    auto const start_loc_idx = random_location();
    if (tt_.location_routes_[start_loc_idx].empty()) {
      continue;
    }

    // derive start itv from start
    auto const start_itv = get_start_interval(start_loc_idx);
    if (!start_itv.has_value()) {
      continue;
    }

    // randomize destination location
    auto const dest_itv = interval<unixtime_t>{
        start_itv.value().from_, start_itv.value().from_ + 1_days};
    std::optional<location_idx_t> dest_loc_idx;
    for (auto j = 0U; j != kMaxGenAttempts; ++j) {
      dest_loc_idx = random_location();
      while (start_loc_idx == dest_loc_idx) {
        dest_loc_idx = random_location();
      }
      if (is_active_dest(dest_loc_idx.value(), dest_itv)) {
        break;
      } else {
        dest_loc_idx = std::nullopt;
      }
    }
    if (!dest_loc_idx.has_value()) {
      continue;
    }

    // found start, time and destination

    // add start(s) to query
    if (s_.start_match_mode_ == routing::location_match_mode::kIntermodal) {
      add_offsets_for_pos(q.start_, pos_near_start(start_loc_idx),
                          s_.start_mode_);
    } else {
      q.start_.emplace_back(start_loc_idx, 0_minutes, 0U);
    }

    // add time to query
    if (s_.interval_size_.count() == 0) {
      q.start_time_ = start_itv.value().from_;
    } else {
      q.start_time_ = start_itv.value();
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

  log(log_lvl::info, "query_generator.random_pretrip",
      "WARNING: failed to generate a valid query after {} attempts",
      kMaxGenAttempts);

  return std::nullopt;
}

std::pair<transport, stop_idx_t> generator::random_transport_active_stop() {
  transport tpt;
  std::optional<stop_idx_t> stop_idx = std::nullopt;
  while (!stop_idx.has_value()) {
    auto const tpt_idx = random_transport();
    auto const day_idx = random_active_day(tpt_idx);
    if (!day_idx.has_value()) {
      continue;
    }
    tpt.t_idx_ = tpt_idx;
    tpt.day_ = day_idx.value();

    stop_idx = random_active_stop(tpt_idx);
  }
  return {tpt, stop_idx.value()};
}

location_idx_t generator::random_location() {
  if (s_.bbox_.has_value()) {
    return location_idx_t{locs_in_bbox[locs_in_bbox_d_(rng_)]};
  } else {
    return location_idx_t{location_d_(rng_)};
  }
}

route_idx_t generator::random_route(location_idx_t const loc_idx) {
  auto const& routes = tt_.location_routes_[loc_idx];
  auto routes_d =
      std::uniform_int_distribution<std::uint32_t>{0U, routes.size() - 1U};
  return routes[routes_d(rng_)];
}

transport_idx_t generator::random_transport() {
  return transport_idx_t{transport_d_(rng_)};
}

transport_idx_t generator::random_transport(route_idx_t const route_idx) {
  auto const transports = tt_.route_transport_ranges_[route_idx];
  auto transports_d = std::uniform_int_distribution<transport_idx_t::value_t>{
      transports.from_.v_, transports.to_.v_ - 1};
  return transport_idx_t{transports_d(rng_)};
}

stop_idx_t generator::get_stop_idx(transport_idx_t const tpt_idx,
                                   location_idx_t const loc_idx) const {
  auto const stops = tt_.route_location_seq_[tt_.transport_route_[tpt_idx]];
  for (auto i = 0U; i != stops.size(); ++i) {
    if (stop{stops[i]}.location_idx() == loc_idx) {
      return static_cast<stop_idx_t>(i);
    }
  }
  assert(false);
  return std::numeric_limits<stop_idx_t>::max();
}

std::optional<stop_idx_t> generator::random_active_stop(
    transport_idx_t const tpt_idx) {

  // distribution for stop index
  // initial stop is not valid for arrival events
  std::uniform_int_distribution<stop_idx_t> stop_d{
      stop_idx_t{1U},
      static_cast<stop_idx_t>(
          tt_.route_location_seq_[tt_.transport_route_[tpt_idx]].size() - 1U)};

  auto const random_stop = [&]() { return stop_idx_t{stop_d(rng_)}; };

  auto const can_exit = [&](stop_idx_t const& stop_idx) {
    auto const s =
        stop{tt_.route_location_seq_[tt_.transport_route_[tpt_idx]][stop_idx]};
    return s.out_allowed();
  };

  auto const is_in_bbox = [&](stop_idx_t const& stop_idx) {
    return !s_.bbox_.has_value() ||
           s_.bbox_.value().contains(
               tt_.locations_.coordinates_[stop{
                   tt_.route_location_seq_[tt_.transport_route_[tpt_idx]]
                                          [stop_idx]}
                                               .location_idx()]);
  };

  auto const is_valid = [&](stop_idx_t const& stop_idx) {
    return can_exit(stop_idx) && is_in_bbox(stop_idx);
  };

  // try randomize
  for (auto i = 0U; i < 10; ++i) {
    auto const stop_idx = random_stop();
    if (is_valid(stop_idx)) {
      return stop_idx;
    }
  }

  // fallback: linear search from random stop
  auto stop_idx = random_stop();
  // search stops after randomized stop
  auto stop_idx_itv = interval<stop_idx_t>{
      stop_idx, static_cast<stop_idx_t>(
                    stop_d.max() + 1U)};  // +1 since distribution endpoints
                                          // are [a,b] and interval's are [a,b)
  auto found_stop =
      std::find_if(begin(stop_idx_itv), end(stop_idx_itv), is_valid);
  if (found_stop != end(stop_idx_itv)) {
    return *found_stop;
  }
  // search stops until randomized stop
  stop_idx_itv = interval<stop_idx_t>{stop_d.min(), stop_idx};
  found_stop = std::find_if(begin(stop_idx_itv), end(stop_idx_itv), is_valid);
  if (found_stop != end(stop_idx_itv)) {
    return *found_stop;
  }

  // no active stop found
  return std::nullopt;
}

bool generator::can_dep(transport_idx_t const tpt_idx,
                        stop_idx_t stp_idx) const {
  auto const loc_seq = tt_.route_location_seq_[tt_.transport_route_[tpt_idx]];
  return stp_idx < loc_seq.size() - 1 && stop{loc_seq[stp_idx]}.in_allowed();
}

std::optional<day_idx_t> generator::random_active_day(
    nigiri::transport_idx_t const tr_idx) {
  auto const random_day = [&]() { return day_idx_t{day_d_(rng_)}; };

  auto const& bf = tt_.bitfields_[tt_.transport_traffic_days_[tr_idx]];
  auto const is_active = [&bf](day_idx_t const d) { return bf.test(d.v_); };

  // try randomize
  for (auto i = 0U; i < 10; ++i) {
    auto const day_idx = random_day();
    if (is_active(day_idx)) {
      return day_idx;
    }
  }

  // fallback: linear search from random day
  auto const day_idx = random_day();
  // search days after randomized day
  auto day_idx_itv = interval<day_idx_t>{
      day_idx, day_idx_t{kTimetableOffset.count() + tt_n_days()}};
  auto found_day =
      std::find_if(begin(day_idx_itv), end(day_idx_itv), is_active);
  if (found_day != end(day_idx_itv)) {
    return *found_day;
  }
  // search days until randomized day
  day_idx_itv =
      interval<day_idx_t>{day_idx_t{kTimetableOffset.count()}, day_idx};
  found_day = std::find_if(begin(day_idx_itv), end(day_idx_itv), is_active);
  if (found_day != end(day_idx_itv)) {
    return *found_day;
  }

  // no active day found
  return std::nullopt;
}

std::optional<interval<unixtime_t>> generator::get_start_interval(
    location_idx_t const loc_idx) {
  auto const route_idx = random_route(loc_idx);
  auto const tpt_idx = random_transport(route_idx);
  auto const stp_idx = get_stop_idx(tpt_idx, loc_idx);
  if (!can_dep(tpt_idx, stp_idx)) {
    return std::nullopt;
  }
  auto const day_idx = random_active_day(tpt_idx);
  if (!day_idx.has_value()) {
    return std::nullopt;
  }

  auto const dep_time = tt_.event_time(transport{tpt_idx, day_idx.value()},
                                       stp_idx, event_type::kDep);
  return interval<unixtime_t>{dep_time,
                              dep_time + duration_t{s_.interval_size_}};
}

bool generator::arr_in_itv(transport_idx_t const tpt_idx,
                           stop_idx_t const stp_idx,
                           interval<unixtime_t> const& itv) const {
  auto const& bf = tt_.bitfields_[tt_.transport_traffic_days_[tpt_idx]];
  for (auto day_idx = day_idx_t{kTimetableOffset.count()};
       day_idx != tt_n_days(); ++day_idx) {
    if (bf.test(day_idx.v_)) {
      if (itv.contains(
              tt_.event_time({tpt_idx, day_idx}, stp_idx, event_type::kArr))) {
        return true;
      }
    }
  }
  return false;
}

bool generator::is_active_dest(location_idx_t const loc,
                               interval<nigiri::unixtime_t> const& itv) const {
  for (auto const& route_idx : tt_.location_routes_[loc]) {
    for (auto const tpt_idx : tt_.route_transport_ranges_[route_idx]) {
      auto const& loc_seq = tt_.route_location_seq_[route_idx];
      for (auto stp_idx = stop_idx_t{1U}; stp_idx != loc_seq.size();
           ++stp_idx) {
        auto const stp = stop{loc_seq[stp_idx]};
        if (stp.out_allowed() && stp.location_idx() == loc &&
            arr_in_itv(tpt_idx, stp_idx, itv)) {
          return true;
        }
      }
    }
  }
  return false;
}

geo::latlng generator::pos_near_start(location_idx_t const loc_idx) {
  auto const loc_pos = tt_.locations_.coordinates_[loc_idx];
  return random_point_in_range(loc_pos, start_mode_range_d_);
}

geo::latlng generator::pos_near_dest(location_idx_t const loc_idx) {
  auto const loc_pos = tt_.locations_.coordinates_[loc_idx];
  return random_point_in_range(loc_pos, dest_mode_range_d_);
}

geo::latlng generator::random_point_in_range(
    geo::latlng const& c, std::uniform_int_distribution<std::uint32_t>& d) {
  return geo::destination_point(c, static_cast<double>(d(rng_)),
                                static_cast<double>(bearing_d_(rng_)));
}

std::uint16_t generator::tt_n_days() const {
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
    query_generation::transport_mode const& mode) const {
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

}  // namespace nigiri::query_generation