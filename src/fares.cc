#include "nigiri/fares.h"

#include <ranges>

#include "utl/pairwise.h"
#include "utl/parser/cstr.h"
#include "utl/to_vec.h"

#include "nigiri/loader/gtfs/noon_offsets.h"
#include "nigiri/routing/journey.h"
#include "nigiri/rt/frun.h"
#include "nigiri/timetable.h"

namespace nigiri {

using routing::journey;

bool contains(auto&& range, auto&& needle) {
  return std::ranges::find(range, needle) != std::end(range);
}

std::ostream& operator<<(
    std::ostream& out, fares::fare_transfer_rule::fare_transfer_type const t) {
  using fare_transfer_type = fares::fare_transfer_rule::fare_transfer_type;
  switch (t) {
    case fare_transfer_type::kAPlusAB: return out << "A+AB";
    case fare_transfer_type::kAPlusABPlusB: return out << "A+AB+B";
    case fare_transfer_type::kAB: return out << "AB";
  }
  std::unreachable();
}

std::ostream& operator<<(std::ostream& out,
                         fares::fare_media::fare_media_type t) {
  using fare_media_type = fares::fare_media::fare_media_type;
  switch (t) {
    case fare_media_type::kNone: return out << "NONE";
    case fare_media_type::kPaper: return out << "PAPER";
    case fare_media_type::kCard: return out << "CARD";
    case fare_media_type::kContactless: return out << "CONTACTLESS";
    case fare_media_type::kApp: return out << "APP";
  }
  std::unreachable();
}

std::ostream& operator<<(std::ostream& out, fares::fare_leg_rule const& r) {
  return out << "FROM_AREA=" << r.from_area_ << ", TO_AREA=" << r.to_area_
             << ", NETWORK=" << r.network_
             << ", FROM_TIMEFRAME_GROUP=" << r.from_timeframe_group_
             << ", TO_TIMEFRAME_GROUP=" << r.to_timeframe_group_;
}

auto fares::fare_leg_rule::match_members() const {
  return std::tie(network_, from_area_, to_area_, from_timeframe_group_,
                  to_timeframe_group_);
}

bool operator==(fares::fare_leg_rule const& a, fares::fare_leg_rule const& b) {
  return a.match_members() == b.match_members();
}

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
          routing::journey::leg const& a_l,
          routing::journey::leg const& b_l) {
  auto const r_a = std::get<routing::journey::run_enter_exit>(a_l.uses_);
  auto const r_b = std::get<routing::journey::run_enter_exit>(b_l.uses_);
  auto const a = rt::frun{tt, nullptr, r_a.r_};
  auto const b = rt::frun{tt, nullptr, r_b.r_};

  if (!a.is_scheduled() || !b.is_scheduled()) {
    return false;
  }

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
      fares::fare_leg_join_rule{.from_network_ = network_a,
                                .to_network_ = network_b,
                                .from_stop_ = location_idx_t{0U},
                                .to_stop_ = location_idx_t{0U}});

  // Search for matching stops.
  auto const from = a[r_a.stop_range_.to_ - 1U].get_location_idx();
  auto const from_station = parent(tt, from);
  auto const to = b[r_b.stop_range_.from_].get_location_idx();
  auto const to_station = parent(tt, to);
  while (it != end(fare.fare_leg_join_rules_) &&
         it->from_network_ == network_a && it->to_network_ == network_b) {
    if ((from_station == to_station &&  //
         it->from_stop_ == location_idx_t::invalid() &&
         it->to_stop_ == location_idx_t::invalid()) ||
        ((it->from_stop_ == from_station || it->from_stop_ == from) &&
         (it->to_stop_ == to_station || it->to_stop_ == to))) {
      return true;
    }
  }

  return false;
}

std::vector<journey::leg const*> get_transit_legs(journey const& j) {
  auto transit_legs = std::vector<journey::leg const*>{};
  for (auto const& l : j.legs_) {
    if (std::holds_alternative<journey::run_enter_exit>(l.uses_)) {
      transit_legs.push_back(&l);
    }
  }
  return transit_legs;
}

using joined_legs_t = std::vector<std::vector<journey::leg const*>>;

joined_legs_t join_legs(timetable const& tt,
                        std::vector<journey::leg const*> const& transit_legs) {
  auto const has_equal_src = [&](journey::leg const* a_l,
                                 journey::leg const* b_l) {
    auto const a =
        rt::frun{tt, nullptr, std::get<journey::run_enter_exit>(a_l->uses_).r_};
    auto const b =
        rt::frun{tt, nullptr, std::get<journey::run_enter_exit>(b_l->uses_).r_};

    if (!a.is_scheduled() || !b.is_scheduled()) {
      return a_l == b_l;
    }

    auto const a_id_idx = tt.trip_ids_[a.trip_idx()].front();
    auto const b_id_idx = tt.trip_ids_[b.trip_idx()].front();

    return tt.trip_id_src_[a_id_idx] == tt.trip_id_src_[b_id_idx];
  };

  auto joined_legs = joined_legs_t{};
  utl::equal_ranges_linear(
      transit_legs, has_equal_src,
      [&](std::vector<journey::leg const*>::const_iterator const from_it,
          std::vector<journey::leg const*>::const_iterator const to_it) {
        utl::verify(std::distance(from_it, to_it) != 0U,
                    "invalid zero-size range");

        auto join_from = from_it;
        auto pred = from_it;
        for (auto it = std::next(from_it); it != to_it; ++it, ++pred) {
          if (join(tt, **pred, **it)) {
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
    timetable const& tt,
    rt_timetable const* rtt,
    effective_fare_leg_t const& joined_legs) {
  auto const& first = joined_legs.front();
  auto const& last = joined_legs.back();

  auto const first_r = std::get<journey::run_enter_exit>(first->uses_).r_;
  auto const first_trip = rt::frun{tt, rtt, first_r};

  auto const last_r = std::get<journey::run_enter_exit>(last->uses_).r_;
  auto const last_trip = rt::frun{tt, rtt, last_r};

  auto const from = first_trip[first_r.stop_range_.from_];
  auto const to = last_trip[last_r.stop_range_.to_ - 1U];

  auto const src =
      first_trip.is_scheduled()
          ? tt.trip_id_src_[tt.trip_ids_[first_trip.trip_idx()].front()]
          : first_trip.id().src_;
  auto const& fare = tt.fares_[src];

  if (!first_trip.is_scheduled() || !last_trip.is_scheduled()) {
    return {src, std::vector<fares::fare_leg_rule>{}};
  }

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

  namespace sv = std::views;
  auto concrete_network =
      fare.fare_leg_rules_ |
      sv::transform([](auto const& r) { return r.network_; }) |
      sv::filter([](auto const n) { return n != network_idx_t::invalid(); });
  auto concrete_from =
      fare.fare_leg_rules_ |
      sv::transform([](auto const& r) { return r.from_area_; }) |
      sv::filter([](auto const a) { return a != area_idx_t::invalid(); });
  auto concrete_to =
      fare.fare_leg_rules_ |
      sv::transform([](auto const& r) { return r.to_area_; }) |
      sv::filter([](auto const a) { return a != area_idx_t::invalid(); });

  auto matching_rules = std::vector<fares::fare_leg_rule>{};
  for (auto const from_area : get_areas(tt, from.get_location_idx())) {
    for (auto const to_area : get_areas(tt, to.get_location_idx())) {
      auto const x = fares::fare_leg_rule{.network_ = network,
                                          .from_area_ = from_area,
                                          .to_area_ = to_area,
                                          .from_timeframe_group_ = from_tf,
                                          .to_timeframe_group_ = to_tf};
      for (auto const& r : fare.fare_leg_rules_) {
        auto const matches =
            ((r.network_ == network_idx_t::invalid() &&
              !contains(concrete_network, x.network_)) ||
             r.network_ == x.network_) &&
            ((r.from_area_ == area_idx_t::invalid() &&
              !contains(concrete_from, x.from_area_)) ||
             r.from_area_ == x.from_area_) &&
            ((r.to_area_ == area_idx_t::invalid() &&
              !contains(concrete_to, x.to_area_)) ||
             r.to_area_ == x.to_area_) &&
            (r.from_timeframe_group_ == timeframe_group_idx_t::invalid() ||
             r.from_timeframe_group_ == x.from_timeframe_group_) &&
            (r.to_timeframe_group_ == timeframe_group_idx_t::invalid() ||
             r.to_timeframe_group_ == x.to_timeframe_group_);

        if (matches) {
          matching_rules.push_back(r);
        }
      }
    }
  }
  utl::sort(matching_rules, [&](fares::fare_leg_rule const& a,
                                fares::fare_leg_rule const& b) {
    auto const& ap = fare.fare_products_[a.fare_product_];
    auto const& bp = fare.fare_products_[b.fare_product_];
    auto const a_rider_not_default =
        ap.rider_category_ == rider_category_idx_t::invalid() ||
        !fare.rider_categories_[ap.rider_category_].is_default_fare_category_;
    auto const b_rider_not_default =
        bp.rider_category_ == rider_category_idx_t::invalid() ||
        !fare.rider_categories_[bp.rider_category_].is_default_fare_category_;
    return std::tuple{-a.rule_priority_, a_rider_not_default,
                      ap.rider_category_, ap.amount_,
                      tt.strings_.get(ap.name_)} <
           std::tuple{-b.rule_priority_, b_rider_not_default,
                      bp.rider_category_, bp.amount_,
                      tt.strings_.get(bp.name_)};
  });

  if (!matching_rules.empty()) {
    auto const highest_prio = matching_rules.front().rule_priority_;
    std::erase_if(matching_rules, [&](fares::fare_leg_rule const x) {
      return x.rule_priority_ < highest_prio;
    });
  }

  return {src, matching_rules};
}

bool matches(fares::fare_transfer_rule const& r,
             fare_leg const& from,
             fare_leg const& a,
             fare_leg const& b,
             auto&& concrete_from,
             auto&& concrete_to) {
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
        return from.joined_leg_.front()->dep_time_;

      case duration_limit_type::kCurrArrNextArr:
      case duration_limit_type::kCurrArrNextDep:
        return from.joined_leg_.back()->arr_time_;
    }
    std::unreachable();
  };

  auto const get_end_time = [&]() {
    switch (r.duration_limit_type_) {
      case duration_limit_type::kCurrDepNextDep:
      case duration_limit_type::kCurrArrNextDep:
        return b.joined_leg_.front()->dep_time_;

      case duration_limit_type::kCurrArrNextArr:
      case duration_limit_type::kCurrDepNextArr:
        return b.joined_leg_.back()->arr_time_;
    }
    std::unreachable();
  };

  // TODO do not match from_leg_group/to_leg_group if another rule has it
  return (r.duration_limit_ == fares::fare_transfer_rule::kNoDurationLimit ||
          r.duration_limit_ >= (get_end_time() - get_start_time())) &&
         ((r.from_leg_group_ == leg_group_idx_t ::invalid() &&
           !contains(concrete_from, curr.leg_group_idx_)) ||
          r.from_leg_group_ == curr.leg_group_idx_) &&
         ((r.to_leg_group_ == leg_group_idx_t::invalid() &&
           !contains(concrete_to, next.leg_group_idx_)) ||
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

        namespace sv = std::views;
        auto concrete_from =
            fares.fare_transfer_rules_ |
            sv::transform([](auto const& r) { return r.from_leg_group_; }) |
            sv::filter([](auto a) { return a != leg_group_idx_t::invalid(); });
        auto concrete_to =
            fares.fare_transfer_rules_ |
            sv::transform([](auto const& r) { return r.to_leg_group_; }) |
            sv::filter([](auto a) { return a != leg_group_idx_t::invalid(); });

        auto last_matched = false;
        for (auto it = from_it, next = std::next(from_it); next != to_it;
             ++it, ++next) {
          utl::verify(it >= from_it && it < to_it, "curr it not in range");
          utl::verify(next >= from_it && next < to_it, "next it not in range");

          auto const match_it = utl::find_if(
              fares.fare_transfer_rules_,
              [&](fares::fare_transfer_rule const& r) {
                return matches(r, *it, *it, *next, concrete_from, concrete_to);
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
            if (!matches(*match_it, *from, *it, *next, concrete_from,
                         concrete_to)) {
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

std::vector<fare_transfer> get_fares(timetable const& tt,
                                     rt_timetable const* rtt,
                                     journey const& j) {
  return join_transfers(
      tt, utl::to_vec(join_legs(tt, get_transit_legs(j)),
                      [&](effective_fare_leg_t const& joined_leg) {
                        auto const [src, rules] =
                            match_leg_rule(tt, rtt, joined_leg);
                        return fare_leg{src, joined_leg, rules};
                      }));
}

}  // namespace nigiri