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

std::vector<area_idx_t> get_areas(timetable const& tt, location_idx_t const l) {
  auto v = std::vector<area_idx_t>{};
  if (auto const l_areas = tt.location_areas_.at(l); !l_areas.empty()) {
    v = utl::to_vec(l_areas);
  } else {
    v = utl::to_vec(tt.location_areas_.at(parent(tt, l)));
  }
  if (v.empty()) {
    v.emplace_back(area_idx_t::invalid());
  }
  return v;
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

    return tt.trip_id_src_[a_id_idx] == tt.trip_id_src_[b_id_idx];
  };

  auto joined_legs = std::vector<std::vector<journey::leg>>{};
  utl::equal_ranges_linear(
      transit_legs, has_equal_src,
      [&](std::vector<journey::leg>::const_iterator const from_it,
          std::vector<journey::leg>::const_iterator const to_it) {
        utl::verify(std::distance(from_it, to_it) != 0U,
                    "invalid zero-size range");

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

std::pair<source_idx_t, std::vector<fares::fare_leg_rule>> match_leg_rule(
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

  auto matching_rules = std::vector<fares::fare_leg_rule>{};
  for (auto const from_area : get_areas(tt, from.get_location_idx())) {
    for (auto const to_area : get_areas(tt, to.get_location_idx())) {
      auto const x = fares::fare_leg_rule{.network_id_ = network,
                                          .from_area_id_ = from_area,
                                          .to_area_id_ = to_area,
                                          .from_timeframe_group_id_ = from_tf,
                                          .to_timeframe_group_id_ = to_tf};
      for (auto const& r : fare.fare_leg_rules_) {
        if (r.fuzzy_matches(x)) {
          matching_rules.push_back(r);
        }
      }
    }
  }
  return {src, matching_rules};
}

bool matches(fares::fare_transfer_rule const& r,
             fare_leg const& from,
             fare_leg const& a,
             fare_leg const& b) {
  using duration_limit_type = fares::fare_transfer_rule::duration_limit_type;

  utl::verify(!from.joined_leg_.empty(), "from no joined leg");
  utl::verify(!a.joined_leg_.empty(), "a no joined leg");
  utl::verify(!b.joined_leg_.empty(), "b no joined leg");

  if (a.rule_.empty() || b.rule_.empty()) {
    return false;
  }

  auto const& curr = a.rule_.front();
  auto const& next = b.rule_.front();

  auto const get_start_time = [&]() {
    switch (r.duration_limit_type_) {
      case duration_limit_type::kCurrDepNextDep:
      case duration_limit_type::kCurrDepNextArr:
        return from.joined_leg_.front().dep_time_;

      case duration_limit_type::kCurrArrNextArr:
      case duration_limit_type::kCurrArrNextDep:
        return from.joined_leg_.back().arr_time_;
    }
    std::unreachable();
  };

  auto const get_end_time = [&]() {
    switch (r.duration_limit_type_) {
      case duration_limit_type::kCurrDepNextDep:
      case duration_limit_type::kCurrArrNextDep:
        return b.joined_leg_.front().dep_time_;

      case duration_limit_type::kCurrArrNextArr:
      case duration_limit_type::kCurrDepNextArr:
        return b.joined_leg_.back().arr_time_;
    }
    std::unreachable();
  };

  return (r.duration_limit_ == fares::fare_transfer_rule::kNoDurationLimit ||
          r.duration_limit_ >= (get_end_time() - get_start_time())) &&
         (r.from_leg_group_ == leg_group_idx_t ::invalid() ||
          r.from_leg_group_ == curr.leg_group_idx_) &&
         (r.to_leg_group_ == leg_group_idx_t::invalid() ||
          r.to_leg_group_ == next.leg_group_idx_);
}

std::vector<fare_transfer> join_transfers(
    timetable const& tt, std::vector<fare_leg> const& fare_legs) {
  auto transfers = std::vector<fare_transfer>{};
  utl::equal_ranges_linear(
      fare_legs,
      [](fare_leg const& a, fare_leg const& b) { return a.src_ == b.src_; },
      [&](std::vector<fare_leg>::const_iterator const from_it,
          std::vector<fare_leg>::const_iterator const to_it) {
        auto const size = static_cast<unsigned>(std::distance(from_it, to_it));
        utl::verify(size != 0U, "invalid zero-size range");

        if (size == 1U) {
          transfers.push_back({std::nullopt, {*from_it}});
          return;
        }

        auto const& fares = tt.fares_[from_it->src_];
        auto last_matched = false;
        for (auto it = from_it, next = std::next(from_it); next != to_it;
             ++it, ++next) {
          utl::verify(it >= from_it && it < to_it, "error 1");
          utl::verify(next >= from_it && next < to_it, "error 2");
          auto const match_it =
              utl::find_if(fares.fare_transfer_rules_,
                           [&](fares::fare_transfer_rule const& r) {
                             return matches(r, *it, *it, *next);
                           });
          if (match_it == end(fares.fare_transfer_rules_)) {
            last_matched = false;
            transfers.push_back({std::nullopt, {*it}});
            continue;
          }

          last_matched = true;
          auto matched = std::vector<fare_leg>{*it};
          auto remaining_transfers = match_it->transfer_count_;
          auto const from = it;
          for (; next != to_it &&
                 remaining_transfers != 0;  // -1=infinite will not reach 0
               ++it, ++next, --remaining_transfers) {
            if (!matches(*match_it, *from, *it, *next)) {
              break;
            }
            matched.emplace_back(*next);
          }

          transfers.push_back({*match_it, std::move(matched)});

          // last one could not be matched by this rule
          // -> go back and try with all rules again
          --it;
          --next;
        }

        if (!last_matched) {
          transfers.push_back({std::nullopt, {*std::next(from_it, size - 1U)}});
        }
      });
  return transfers;
}

std::vector<fare_transfer> compute_price(timetable const& tt,
                                         journey const& j) {
  return join_transfers(
      tt, utl::to_vec(join_legs(tt, get_transit_legs(j)),
                      [&](std::vector<journey::leg> const& joined_leg) {
                        auto const [src, rules] =
                            match_leg_rule(tt, joined_leg);
                        return fare_leg{src, joined_leg, rules};
                      }));
}

}  // namespace nigiri::routing