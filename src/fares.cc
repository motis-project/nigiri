#include "nigiri/fares.h"

#include "utl/pairwise.h"
#include "utl/parser/cstr.h"
#include "utl/to_vec.h"

#include "nigiri/loader/gtfs/noon_offsets.h"
#include "nigiri/routing/journey.h"
#include "nigiri/rt/frun.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

location_idx_t parent(timetable const& tt, location_idx_t const l) {
  return tt.locations_.parents_[l] == location_idx_t::invalid()
             ? l
             : tt.locations_.parents_[l];
}

network_idx_t get_network(rt::frun const& a) {
  auto const& tt = *a.tt_;
  auto const src = tt.trip_id_src_[tt.trip_ids_[a.trip_idx()].front()];
  auto const& f = tt.fares_[src];
  auto const route_id = tt.trip_route_id_[a.trip_idx()];
  auto const it = f.route_networks_.find(route_id);
  return it == end(f.route_networks_) ? network_idx_t::invalid() : it->second;
}

vecvec<location_idx_t, area_idx_t>::const_bucket get_areas(
    timetable const& tt, location_idx_t const l) {
  if (auto const l_areas = tt.location_areas_.at(l); !l_areas.empty()) {
    return l_areas;
  } else {
    return tt.location_areas_.at(parent(tt, l));
  }
}

bool join(timetable const& tt,
          journey::leg const& a_l,
          journey::leg const& b_l) {
  auto const r_a = std::get<journey::run_enter_exit>(a_l.uses_);
  auto const r_b = std::get<journey::run_enter_exit>(b_l.uses_);
  auto const a = rt::frun{tt, nullptr, r_a.r_};
  auto const b = rt::frun{tt, nullptr, r_b.r_};

  auto const src = tt.trip_id_src_[tt.trip_ids_[a.trip_idx()].front()];
  auto const& fare = tt.fares_[src];

  // Leg join rules require network to be defined.
  auto const network_a = get_network(a);
  auto const network_b = get_network(b);
  if (network_a == network_idx_t::invalid() ||
      network_b == network_idx_t::invalid()) {
    return false;
  }

  // Find first fare leg rule regarding both networks.
  auto const it = std::lower_bound(
      begin(fare.fare_leg_join_rules_), end(fare.fare_leg_join_rules_),
      fares::fare_leg_join_rule{.from_network_id_ = network_a,
                                .to_network_id_ = network_b,
                                .from_stop_id_ = location_idx_t{0U},
                                .to_stop_id_ = location_idx_t{0U}});

  // Search for matching stops.
  auto const from = a[r_a.stop_range_.to_ - 1U].get_location_idx();
  auto const from_station = parent(tt, from);
  auto const to = b[r_b.stop_range_.from_].get_location_idx();
  auto const to_station = parent(tt, to);
  while (it != end(fare.fare_leg_join_rules_) &&
         it->from_network_id_ == network_a && it->to_network_id_ == network_b) {
    if ((from_station == to_station &&  //
         it->from_stop_id_ == location_idx_t::invalid() &&
         it->to_stop_id_ == location_idx_t::invalid()) ||
        ((it->from_stop_id_ == from_station || it->from_stop_id_ == from) &&
         (it->to_stop_id_ == to_station || it->to_stop_id_ == to))) {
      return true;
    }
  }

  return false;
}

std::vector<journey::leg> get_transit_legs(journey const& j) {
  auto transit_legs = std::vector<journey::leg>{};
  for (auto const& l : j.legs_) {
    if (std::holds_alternative<journey::run_enter_exit>(l.uses_)) {
      transit_legs.push_back(l);
    }
  }
  return transit_legs;
}

using joined_legs_t = std::vector<std::vector<journey::leg>>;

joined_legs_t join_legs(timetable const& tt,
                        std::vector<journey::leg> const& transit_legs) {
  auto const has_equal_src = [&](journey::leg const& a_l,
                                 journey::leg const& b_l) {
    auto const a =
        rt::frun{tt, nullptr, std::get<journey::run_enter_exit>(a_l.uses_).r_};
    auto const b =
        rt::frun{tt, nullptr, std::get<journey::run_enter_exit>(b_l.uses_).r_};

    auto const a_id_idx = tt.trip_ids_[a.trip_idx()].front();
    auto const b_id_idx = tt.trip_ids_[b.trip_idx()].front();

    return tt.trip_id_src_[a_id_idx] != tt.trip_id_src_[b_id_idx];
  };

  auto joined_legs = std::vector<std::vector<journey::leg>>{};
  utl::equal_ranges_linear(
      transit_legs, has_equal_src,
      [&](std::vector<journey::leg>::const_iterator const from_it,
          std::vector<journey::leg>::const_iterator const to_it) {
        auto join_from = from_it;
        auto pred = from_it;
        for (auto it = std::next(from_it); it != to_it; ++it, ++pred) {
          if (join(tt, *pred, *it)) {
            continue;
          }
          joined_legs.emplace_back(join_from, it);
          join_from = it;
        }
        joined_legs.emplace_back(join_from, to_it);
      });
  return joined_legs;
}

timeframe_group_idx_t match_timeframe(timetable const& tt,
                                      fares const& f,
                                      location_idx_t const l,
                                      transport_idx_t const t,
                                      unixtime_t const time) {
  auto const stop_tz = tt.locations_.location_timezones_.at(l);
  auto const& tz = tt.locations_.timezones_.at(
      stop_tz == timezone_idx_t::invalid()
          ? tt.providers_[tt.transport_section_providers_.at(t).at(0)].tz_
          : stop_tz);
  auto const base_day = std::chrono::time_point_cast<date::days>(
      to_local_time(tz, tt.internal_interval_days().from_));
  for (auto i = timeframe_group_idx_t{0U}; i != f.timeframes_.size(); ++i) {
    for (auto const& tf : f.timeframes_[i]) {
      auto const local_time = to_local_time(tz, time);
      auto const day = std::chrono::time_point_cast<date::days>(local_time);
      auto const day_idx = static_cast<std::size_t>((day - base_day).count());
      if (day + tf.start_time_ <= local_time &&
          local_time < day + tf.end_time_ &&  //
          day_idx < tf.service_.size() && tf.service_.test(day_idx)) {
        return i;
      }
    }
  }
  return timeframe_group_idx_t::invalid();
}

std::pair<source_idx_t, std::optional<fares::fare_leg_rule>> match_leg_rule(
    timetable const& tt, std::vector<journey::leg> const& joined_legs) {
  auto const& first = joined_legs.front();
  auto const& last = joined_legs.back();

  auto const first_r = std::get<journey::run_enter_exit>(first.uses_).r_;
  auto const first_trip = rt::frun{tt, nullptr, first_r};

  auto const last_r = std::get<journey::run_enter_exit>(last.uses_).r_;
  auto const last_trip = rt::frun{tt, nullptr, last_r};

  auto const from = first_trip[first_r.stop_range_.from_];
  auto const to = last_trip[last_r.stop_range_.to_ - 1U];

  auto const src = tt.trip_id_src_[tt.trip_ids_[first_trip.trip_idx()].front()];
  auto const& fare = tt.fares_[src];

  auto const from_network = get_network(first_trip);
  auto const to_network = get_network(last_trip);
  auto const network =
      from_network == to_network ? from_network : network_idx_t::invalid();
  auto const from_tf =
      match_timeframe(tt, fare, from.get_location_idx(), from.fr_->t_.t_idx_,
                      from.time(event_type::kDep));
  auto const to_tf =
      match_timeframe(tt, fare, to.get_location_idx(), to.fr_->t_.t_idx_,
                      to.time(event_type::kArr));

  auto const try_match = [&](auto&& fn) -> std::optional<fares::fare_leg_rule> {
    for (auto const from_area : get_areas(tt, from.get_location_idx())) {
      for (auto const to_area : get_areas(tt, to.get_location_idx())) {
        auto const matched =
            fn(fares::fare_leg_rule{.network_id_ = network,
                                    .from_area_id_ = from_area,
                                    .to_area_id_ = to_area,
                                    .from_timeframe_group_id_ = from_tf,
                                    .to_timeframe_group_id_ = to_tf});
        if (matched.has_value()) {
          return matched;
        }
      }
    }
    return std::nullopt;
  };

  auto const exact_match = try_match([&](fares::fare_leg_rule const& r) {
    auto const it = utl::find(fare.fare_leg_rules_, r);
    auto const exact_match_found = it != end(fare.fare_leg_rules_);
    return exact_match_found ? std::optional{*it} : std::nullopt;
  });

  if (exact_match.has_value()) {
    return {src, exact_match};
  }

  auto const fuzzy_match =
      try_match([&](fares::fare_leg_rule const& r)
                    -> std::optional<fares::fare_leg_rule> {
        for (auto const& rule : fare.fare_leg_rules_) {
          if (rule.fuzzy_matches(r)) {
            return std::optional{rule};
          }
        }
        return std::nullopt;
      });

  return {src, fuzzy_match};
}

std::vector<fare_leg> compute_price(timetable const& tt, journey const& j) {
  auto const transit_legs = get_transit_legs(j);
  auto const joined_legs = join_legs(tt, transit_legs);
  return utl::to_vec(joined_legs,
                     [&](std::vector<journey::leg> const& joined_leg) {
                       auto const [src, rule] = match_leg_rule(tt, joined_leg);
                       return fare_leg{src, joined_leg, rule};
                     });
}

}  // namespace nigiri::routing