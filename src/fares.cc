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

// #define NIGIRI_FARES_DEBUG
#ifdef NIGIRI_FARES_DEBUG
#define trace(...) fmt::println(__VA_ARGS__)
#else
#define trace(...)
#endif

struct journey_leg {
  friend std::ostream& operator<<(std::ostream& out, journey_leg const& l) {
    l.l_.print(out, l.tt_, l.rtt_);
    return out;
  }

  journey::leg const& l_;
  timetable const& tt_;
  rt_timetable const* rtt_;
};

struct leg_group {
  friend std::ostream& operator<<(std::ostream& out, leg_group const x) {
    auto const& [tt, f, g] = x;
    return out << (g == leg_group_idx_t::invalid()
                       ? "ANY"
                       : tt.strings_.get(f.leg_group_name_[g]));
  }

  timetable const& tt_;
  fares const& f_;
  leg_group_idx_t g_;
};

struct transfer_rule {
  friend std::ostream& operator<<(std::ostream& out, transfer_rule const& x) {
    auto const& [tt, f, r] = x;
    return out << "(transfer_type=" << r.fare_transfer_type_
               << ", from_leg_group=" << leg_group{tt, f, r.from_leg_group_}
               << ", to_leg_group=" << leg_group{tt, f, r.to_leg_group_}
               << ", duration_limit="
               << (r.duration_limit_ ==
                           fares::fare_transfer_rule::kNoDurationLimit
                       ? "NO_LIMIT"
                       : fmt::to_string(r.duration_limit_))
               << ", duration_limit_type=" << r.duration_limit_type_
               << ", transfer_count=" << static_cast<int>(r.transfer_count_)
               << ", product="
               << (r.fare_product_ == fare_product_idx_t::invalid()
                       ? "NONE"
                       : tt.strings_.get(f.fare_product_id_[r.fare_product_]))
               << ")";
  }

  timetable const& tt_;
  fares const& f_;
  fares::fare_transfer_rule const& r_;
};

struct area_set {
  friend std::ostream& operator<<(std::ostream& out, area_set const& x) {
    if (x.area_set_ == area_set_idx_t::invalid()) {
      return out << "-";
    }
    out << "[";
    auto first = true;
    for (auto const& area : x.f_.area_sets_[x.area_set_]) {
      if (first) {
        first = false;
      } else {
        out << ", ";
      }
      out << x.tt_.strings_.get(x.tt_.areas_[area].id_);
    }
    return out << "]";
  }

  timetable const& tt_;
  fares const& f_;
  area_set_idx_t const& area_set_;
};

struct leg_rule {
  friend std::ostream& operator<<(std::ostream& out, leg_rule const& x) {
    auto const [tt, f, r] = x;
    return out
           << "(from_area="
           << (r.from_area_ == area_idx_t::invalid()
                   ? "ANY"
                   : tt.strings_.get(tt.areas_[r.from_area_].id_))
           << ", to_area="
           << (r.to_area_ == area_idx_t::invalid()
                   ? "ANY"
                   : tt.strings_.get(tt.areas_[r.to_area_].id_))
           << ", network="
           << (r.network_ == network_idx_t::invalid()
                   ? "ANY"
                   : tt.strings_.get(f.networks_[r.network_].id_))
           << ", from_timeframe_group="
           << (r.from_timeframe_group_ == timeframe_group_idx_t::invalid()
                   ? "ANY"
                   : tt.strings_.get(f.timeframe_id_[r.from_timeframe_group_]))
           << ", to_timeframe_group="
           << (r.to_timeframe_group_ == timeframe_group_idx_t::invalid()
                   ? "ANY"
                   : tt.strings_.get(f.timeframe_id_[r.to_timeframe_group_]))
           << ", contains_exactly_area_set="
           << area_set{tt, f, r.contains_exactly_area_set_id_}
           << ", contains_area_set=" << area_set{tt, f, r.contains_area_set_id_}
           << ", product="
           << (r.fare_product_ == fare_product_idx_t::invalid()
                   ? "-"
                   : tt.strings_.get(f.fare_product_id_[r.fare_product_]))
           << ")";
  }

  timetable const& tt_;
  fares const& f_;
  fares::fare_leg_rule const& r_;
};

std::ostream& operator<<(std::ostream& out,
                         fares::fare_transfer_rule::duration_limit_type t) {
  using duration_limit_type = fares::fare_transfer_rule::duration_limit_type;
  switch (t) {
    case duration_limit_type::kCurrDepNextArr: return out << "CurrDepNextArr";
    case duration_limit_type::kCurrDepNextDep: return out << "CurrDepNextDep";
    case duration_limit_type::kCurrArrNextDep: return out << "CurrArrNextDep";
    case duration_limit_type::kCurrArrNextArr: return out << "CurrArrNextArr";
  }
  std::unreachable();
}

}  // namespace nigiri

template <>
struct fmt::formatter<nigiri::leg_group> : ostream_formatter {};

template <>
struct fmt::formatter<nigiri::leg_rule> : ostream_formatter {};

template <>
struct fmt::formatter<nigiri::transfer_rule> : ostream_formatter {};

template <>
struct fmt::formatter<nigiri::journey_leg> : ostream_formatter {};

namespace nigiri {

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

vecvec<location_idx_t, area_idx_t>::const_bucket get_areas(
    timetable const& tt, location_idx_t const l) {
  auto const l_areas = tt.location_areas_.at(l);
  return l_areas.empty() ? tt.location_areas_.at(parent(tt, l)) : l_areas;
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

  // Search for matching join rule matching both stops.
  auto const from = a[r_a.stop_range_.to_ - 1U].get_location_idx();
  auto const from_station = parent(tt, from);
  auto const to = b[r_b.stop_range_.from_].get_location_idx();
  auto const to_station = parent(tt, to);
  return utl::find_if(
             fare.fare_leg_join_rules_,
             [&](fares::fare_leg_join_rule const& jr) {
               auto const networks_match =
                   jr.from_network_ == network_a && jr.to_network_ == network_b;
               auto const stops_match =
                   (jr.from_stop_ == location_idx_t::invalid() &&
                    jr.to_stop_ == location_idx_t::invalid()) ||
                   ((jr.from_stop_ == from_station || jr.from_stop_ == from) &&
                    (jr.to_stop_ == to_station || jr.to_stop_ == to));
               return networks_match && stops_match;
             }) != end(fare.fare_leg_join_rules_);
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
        trace(
            "TIMEFRAME MATCH: local_time={}, time={}, day={}, day_idx={}, "
            "timeframe_group_id={}, service={}, service_id={}",
            fmt::streamed(local_time), fmt::streamed(time), fmt::streamed(day),
            day_idx, tt.strings_.get(f.timeframe_id_[i]), tf.service_,
            tt.strings_.get(tf.service_id_));
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
#ifdef NIGIRI_FARES_DEBUG
  trace("EFFECTIVE LEG");
  for (auto const& l : joined_legs) {
    trace("{}", journey_leg{*l, tt, rtt});
  }
  trace("\n");
#endif

  auto const& first = joined_legs.front();
  auto const& last = joined_legs.back();

  auto const [first_r, first_stop_range] =
      std::get<journey::run_enter_exit>(first->uses_);
  auto const first_trip = rt::frun{tt, rtt, first_r};

  auto const [last_r, last_stop_range] =
      std::get<journey::run_enter_exit>(last->uses_);
  auto const last_trip = rt::frun{tt, rtt, last_r};

  auto const from = first_trip[first_stop_range.from_];
  auto const to = last_trip[last_stop_range.to_ - 1];

  auto const src =
      first_trip.is_scheduled()
          ? tt.trip_id_src_[tt.trip_ids_[first_trip.trip_idx()].front()]
          : first_trip.id().src_;
  auto const& f = tt.fares_[src];

  if (!first_trip.is_scheduled() || !last_trip.is_scheduled()) {
    return {src, std::vector<fares::fare_leg_rule>{}};
  }

  auto const from_network = get_network(first_trip);
  auto const to_network = get_network(last_trip);
  auto const network =
      from_network == to_network ? from_network : network_idx_t::invalid();

  trace("from: {}", fmt::streamed(from));
  auto const from_tf =
      match_timeframe(tt, f, from.get_location_idx(), from.fr_->t_.t_idx_,
                      from.time(event_type::kDep));

  trace("  to: {}", fmt::streamed(to));
  auto const to_tf =
      match_timeframe(tt, f, to.get_location_idx(), to.fr_->t_.t_idx_,
                      to.time(event_type::kArr));

  namespace sv = std::views;
  auto concrete_network =
      f.fare_leg_rules_ |
      sv::transform([](auto const& r) { return r.network_; }) |
      sv::filter([](auto const n) { return n != network_idx_t::invalid(); });
  auto concrete_from =
      f.fare_leg_rules_ |
      sv::transform([](auto const& r) { return r.from_area_; }) |
      sv::filter([](auto const a) { return a != area_idx_t::invalid(); });
  auto concrete_to =
      f.fare_leg_rules_ |
      sv::transform([](auto const& r) { return r.to_area_; }) |
      sv::filter([](auto const a) { return a != area_idx_t::invalid(); });

  auto const has_area = [&](area_idx_t const x) {
    for (auto const& l : joined_legs) {
      auto const ree = std::get<journey::run_enter_exit>(l->uses_);
      auto const fr = rt::frun{tt, rtt, ree.r_};
      auto const a = static_cast<stop_idx_t>(ree.stop_range_.from_);
      auto const b = static_cast<stop_idx_t>(ree.stop_range_.to_);
      for (auto i = a; i < b; ++i) {
        auto const stop_areas = get_areas(tt, fr[i].get_location_idx());
        if (utl::find(stop_areas, x) != end(stop_areas)) {
          return true;
        }
      }
    }
    return false;
  };

  auto const has_other_area =
      [&](vecvec<area_set_idx_t, area_idx_t>::const_bucket const& exact_areas) {
        for (auto const& l : joined_legs) {
          auto const ree = std::get<journey::run_enter_exit>(l->uses_);
          auto const fr = rt::frun{tt, rtt, ree.r_};
          auto const a = static_cast<stop_idx_t>(ree.stop_range_.from_);
          auto const b = static_cast<stop_idx_t>(ree.stop_range_.to_);
          for (auto i = a; i < b; ++i) {
            auto const stop_areas = get_areas(tt, fr[i].get_location_idx());
            trace("areas of {}: {}", fr[i].get_location().name_,
                  stop_areas | std::views::transform([&](area_idx_t const x) {
                    return tt.strings_.get(tt.areas_[x].name_);
                  }));
            auto const contains_other_area =
                utl::any_of(stop_areas, [&](area_idx_t const x) {
                  return std::ranges::find(exact_areas, x) == end(exact_areas);
                });
            if (contains_other_area) {
              return true;
            }
          }
        }
        return false;
      };

  auto const for_each_area = [&](location_idx_t const l, auto&& fn) {
    auto const areas = get_areas(tt, l);
    if (areas.empty()) {
      fn(area_idx_t::invalid());
    } else {
      for (auto const a : areas) {
        fn(a);
      }
    }
  };

  auto matching_rules = std::vector<fares::fare_leg_rule>{};
  for_each_area(from.get_location_idx(), [&](area_idx_t const from_area) {
    for_each_area(to.get_location_idx(), [&](area_idx_t const to_area) {
      auto const x = fares::fare_leg_rule{.network_ = network,
                                          .from_area_ = from_area,
                                          .to_area_ = to_area,
                                          .from_timeframe_group_ = from_tf,
                                          .to_timeframe_group_ = to_tf};
      for (auto const& r : f.fare_leg_rules_) {
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
             r.to_timeframe_group_ == x.to_timeframe_group_) &&
            (r.contains_area_set_id_ == area_set_idx_t::invalid() ||
             utl::all_of(f.area_sets_[r.contains_area_set_id_], has_area)) &&
            (r.contains_exactly_area_set_id_ == area_set_idx_t::invalid() ||
             (utl::all_of(f.area_sets_[r.contains_exactly_area_set_id_],
                          has_area) &&
              !has_other_area(f.area_sets_[r.contains_exactly_area_set_id_])));

        if (matches) {
          trace("RULE MATCH\n\t\tRULE = {}\n\t\tLEG = {}\n", leg_rule{tt, f, r},
                leg_rule{tt, f, x});
          matching_rules.push_back(r);
        } else {
          trace("NO MATCH\n\t\tRULE = {}\n\t\tLEG = {}", leg_rule{tt, f, r},
                leg_rule{tt, f, x});

          auto const criteria =
              std::initializer_list<std::pair<char const*, bool>>{
                  {"network", (r.network_ == network_idx_t::invalid() &&
                               !contains(concrete_network, x.network_)) ||
                                  r.network_ == x.network_},
                  {
                      "from_area",
                      (r.from_area_ == area_idx_t::invalid() &&
                       !contains(concrete_from, x.from_area_)) ||
                          r.from_area_ == x.from_area_,
                  },
                  {
                      "to_area",
                      (r.to_area_ == area_idx_t::invalid() &&
                       !contains(concrete_from, x.to_area_)) ||
                          r.to_area_ == x.to_area_,
                  },
                  {"from_timeframe",
                   r.from_timeframe_group_ ==
                           timeframe_group_idx_t::invalid() ||
                       r.from_timeframe_group_ == x.from_timeframe_group_},
                  {"to_timeframe",
                   r.to_timeframe_group_ == timeframe_group_idx_t::invalid() ||
                       r.to_timeframe_group_ == x.to_timeframe_group_},
                  {"contains_area_set",
                   r.contains_area_set_id_ == area_set_idx_t::invalid() ||
                       utl::all_of(f.area_sets_[r.contains_area_set_id_],
                                   has_area)},
                  {"contains_exactly_area_set",
                   r.contains_exactly_area_set_id_ ==
                           area_set_idx_t::invalid() ||
                       (utl::all_of(
                            f.area_sets_[r.contains_exactly_area_set_id_],
                            has_area) &&
                        !has_other_area(
                            f.area_sets_[r.contains_exactly_area_set_id_]))}};
          for (auto const& [criterion, matched] : criteria) {
            if (!matched) {
              trace("    {} -> NO MATCH", criterion);
            }
          }
        }
      }
    });
  });
  utl::sort(matching_rules, [&](fares::fare_leg_rule const& a,
                                fares::fare_leg_rule const& b) {
    if (a.fare_product_ == fare_product_idx_t::invalid() ||
        b.fare_product_ == fare_product_idx_t::invalid()) {
      return a.rule_priority_ > b.rule_priority_;
    }
    auto const ap = f.fare_products_[a.fare_product_].front();
    auto const bp = f.fare_products_[b.fare_product_].front();
    auto const a_rider_not_default =
        ap.rider_category_ == rider_category_idx_t::invalid() ||
        !f.rider_categories_[ap.rider_category_].is_default_fare_category_;
    auto const b_rider_not_default =
        bp.rider_category_ == rider_category_idx_t::invalid() ||
        !f.rider_categories_[bp.rider_category_].is_default_fare_category_;
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

bool matches([[maybe_unused]] timetable const& tt,
             [[maybe_unused]] fares const& f,
             fares::fare_transfer_rule const& r,
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

  auto const& curr_rule = a.rule_.front();
  auto const& next_rule = b.rule_.front();

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

  auto const transfer_limit_ok =
      (r.duration_limit_ == fares::fare_transfer_rule::kNoDurationLimit ||
       r.duration_limit_ >= (get_end_time() - get_start_time()));
  auto const from_leg_group_matches =
      ((r.from_leg_group_ == leg_group_idx_t::invalid() &&
        !contains(concrete_from, curr_rule.leg_group_idx_)) ||
       r.from_leg_group_ == curr_rule.leg_group_idx_);
  auto const to_leg_group_matches =
      ((r.to_leg_group_ == leg_group_idx_t::invalid() &&
        !contains(concrete_to, next_rule.leg_group_idx_)) ||
       r.to_leg_group_ == next_rule.leg_group_idx_);

  if (!transfer_limit_ok) {
    trace(
        "      transfer limit exceeded: end_time={}, start_time={}, "
        "duration={}, limit={} (no_limit={})",
        get_start_time(), get_end_time(), get_end_time() - get_start_time(),
        r.duration_limit_,
        r.duration_limit_ == fares::fare_transfer_rule::kNoDurationLimit);
  }
  if (!from_leg_group_matches) {
    trace(
        "      from leg group mismatch\n"
        "        r.from_leg_group: {}\n"
        "        curr_rule.leg_group: {}\n"
        "        concrete_from: {}",
        leg_group{tt, f, r.from_leg_group_},
        leg_group{tt, f, curr_rule.leg_group_idx_},
        concrete_from | std::views::transform(
                            [&](auto&& x) { return leg_group{tt, f, x}; }));
  }
  if (!to_leg_group_matches) {
    trace(
        "      to leg group mismatch\n"
        "        r.to_leg_group: {}\n"
        "        next_rule.leg_group: {}\n"
        "        concrete_to: {}",
        leg_group{tt, f, r.to_leg_group_},
        leg_group{tt, f, next_rule.leg_group_idx_},
        concrete_to | std::views::transform(
                          [&](auto&& x) { return leg_group{tt, f, x}; }));
  }
  if (transfer_limit_ok && from_leg_group_matches && to_leg_group_matches) {
    trace("      rule matched!");
  }

  return transfer_limit_ok && from_leg_group_matches && to_leg_group_matches;
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

        auto const& f = tt.fares_[from_it->src_];

        namespace sv = std::views;
        auto concrete_from =
            f.fare_transfer_rules_ |
            sv::transform([](auto const& r) { return r.from_leg_group_; }) |
            sv::filter([](auto a) { return a != leg_group_idx_t::invalid(); });
        auto concrete_to =
            f.fare_transfer_rules_ |
            sv::transform([](auto const& r) { return r.to_leg_group_; }) |
            sv::filter([](auto a) { return a != leg_group_idx_t::invalid(); });

        auto last_matched = false;
        for (auto it = from_it, next = std::next(from_it); next != to_it;
             ++it, ++next) {
          utl::verify(it >= from_it && it < to_it, "curr it not in range");
          utl::verify(next >= from_it && next < to_it, "next it not in range");

          auto const match_it = utl::find_if(
              f.fare_transfer_rules_, [&](fares::fare_transfer_rule const& r) {
                return matches(tt, f, r, *it, *it, *next, concrete_from,
                               concrete_to);
              });
          if (match_it == end(f.fare_transfer_rules_)) {
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
            if (!matches(tt, f, *match_it, *from, *it, *next, concrete_from,
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
