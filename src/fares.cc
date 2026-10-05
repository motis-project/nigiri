#include "nigiri/fares.h"

#include <cmath>
#include <algorithm>
#include <array>
#include <iterator>
#include <limits>
#include <ranges>

#include "utl/enumerate.h"
#include "utl/helpers/algorithm.h"
#include "utl/to_vec.h"

#include "nigiri/routing/journey.h"
#include "nigiri/rt/frun.h"
#include "nigiri/timetable.h"

// #define NIGIRI_FARES_DEBUG
#ifdef NIGIRI_FARES_DEBUG
#define trace(...) fmt::println(__VA_ARGS__)
#else
#define trace(...)
#endif

namespace nigiri {

using routing::journey;

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
  auto const& tz = tt.timezones_.at(
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
            trace("areas of {}: {}", fmt::streamed(fr[i].get_loc()),
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

  auto const area_set_matches = [&](fares::fare_leg_rule const& r) {
    return (r.contains_area_set_id_ == area_set_idx_t::invalid() ||
            utl::all_of(f.area_sets_[r.contains_area_set_id_], has_area)) &&
           (r.contains_exactly_area_set_id_ == area_set_idx_t::invalid() ||
            (utl::all_of(f.area_sets_[r.contains_exactly_area_set_id_],
                         has_area) &&
             !has_other_area(f.area_sets_[r.contains_exactly_area_set_id_])));
  };

  auto const in_set = [](auto const& s, auto const v) {
    return s.find(v) != s.end();
  };

  auto const num_opts = [&](auto const val, auto const& concrete) {
    return val != std::decay_t<decltype(val)>::invalid() &&
                   !in_set(concrete, val)  //  true for feeds with rule priority
               ? 2U
               : 1U;
  };
  auto const num_tf_opts = [](auto const val) {
    return val != std::decay_t<decltype(val)>::invalid() ? 2U : 1U;
  };

  namespace sv = std::views;
  auto matching_rules = std::vector<fares::fare_leg_rule>{};
  for_each_area(from.get_location_idx(), [&](area_idx_t const from_area) {
    for_each_area(to.get_location_idx(), [&](area_idx_t const to_area) {
      auto const x = fares::fare_leg_rule{.network_ = network,
                                          .from_area_ = from_area,
                                          .to_area_ = to_area,
                                          .from_timeframe_group_ = from_tf,
                                          .to_timeframe_group_ = to_tf};

      auto const nets = std::array{x.network_, network_idx_t::invalid()} |
                        sv::take(num_opts(x.network_, f.concrete_networks_));
      auto const froms =
          std::array{x.from_area_, area_idx_t::invalid()} |
          sv::take(num_opts(x.from_area_, f.concrete_from_areas_));
      auto const tos = std::array{x.to_area_, area_idx_t::invalid()} |
                       sv::take(num_opts(x.to_area_, f.concrete_to_areas_));
      auto const from_tfs = std::array{x.from_timeframe_group_,
                                       timeframe_group_idx_t::invalid()} |
                            sv::take(num_tf_opts(x.from_timeframe_group_));
      auto const to_tfs =
          std::array{x.to_timeframe_group_, timeframe_group_idx_t::invalid()} |
          sv::take(num_tf_opts(x.to_timeframe_group_));

      for (auto const nw : nets) {
        for (auto const fa : froms) {
          for (auto const ta : tos) {
            for (auto const ftg : from_tfs) {
              for (auto const ttg : to_tfs) {
                auto const it = f.fare_leg_rule_keys_.find(
                    fares::fare_leg_rule_key{.network_ = nw,
                                             .from_area_ = fa,
                                             .to_area_ = ta,
                                             .from_timeframe_group_ = ftg,
                                             .to_timeframe_group_ = ttg});
                if (it == end(f.fare_leg_rule_keys_)) {
                  continue;
                }
                for (auto const& r : f.fare_leg_rule_groups_[it->second]) {
                  if (area_set_matches(r)) {
                    trace("RULE MATCH\n\t\tRULE = {}\n\t\tLEG = {}\n",
                          leg_rule{tt, f, r}, leg_rule{tt, f, x});
                    matching_rules.push_back(r);
                  }
                }
              }
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

// Leg rules of a fare leg that are covered by a transfer rule's leg group.
// An unset leg group covers all leg groups not named by any transfer rule.
std::vector<fares::fare_leg_rule> covered_rules(
    std::vector<fares::fare_leg_rule> const& rules,
    leg_group_idx_t const leg_group,
    auto&& concrete) {
  auto covered = std::vector<fares::fare_leg_rule>{};
  std::ranges::copy_if(rules, std::back_inserter(covered),
                       [&](fares::fare_leg_rule const& x) {
                         return leg_group == leg_group_idx_t::invalid()
                                    ? !contains(concrete, x.leg_group_idx_)
                                    : leg_group == x.leg_group_idx_;
                       });
  return covered;
}

// Returns the leg rules of `a` and `b` taking part in the transfer if the
// transfer rule `r` applies to the transfer from `a` to `b`. Leg rules not
// covered by the transfer rule's from/to leg groups are thinned out.
using transfer_match_t = std::pair<std::vector<fares::fare_leg_rule>,
                                   std::vector<fares::fare_leg_rule>>;
std::optional<transfer_match_t> matches([[maybe_unused]] timetable const& tt,
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
  auto from_rules = covered_rules(a.rule_, r.from_leg_group_, concrete_from);
  auto to_rules = covered_rules(b.rule_, r.to_leg_group_, concrete_to);

  if (!transfer_limit_ok) {
    trace(
        "      transfer limit exceeded: end_time={}, start_time={}, "
        "duration={}, limit={} (no_limit={})",
        get_start_time(), get_end_time(), get_end_time() - get_start_time(),
        r.duration_limit_,
        r.duration_limit_ == fares::fare_transfer_rule::kNoDurationLimit);
  }
  if (from_rules.empty()) {
    trace(
        "      from leg group mismatch\n"
        "        r.from_leg_group: {}\n"
        "        curr leg groups: {}\n"
        "        concrete_from: {}",
        leg_group{tt, f, r.from_leg_group_},
        a.rule_ | std::views::transform([&](auto&& x) {
          return leg_group{tt, f, x.leg_group_idx_};
        }),
        concrete_from | std::views::transform(
                            [&](auto&& x) { return leg_group{tt, f, x}; }));
  }
  if (to_rules.empty()) {
    trace(
        "      to leg group mismatch\n"
        "        r.to_leg_group: {}\n"
        "        next leg groups: {}\n"
        "        concrete_to: {}",
        leg_group{tt, f, r.to_leg_group_},
        b.rule_ | std::views::transform([&](auto&& x) {
          return leg_group{tt, f, x.leg_group_idx_};
        }),
        concrete_to | std::views::transform(
                          [&](auto&& x) { return leg_group{tt, f, x}; }));
  }

  if (!transfer_limit_ok || from_rules.empty() || to_rules.empty()) {
    return std::nullopt;
  }
  trace("      rule matched!");
  return transfer_match_t{std::move(from_rules), std::move(to_rules)};
}

// A traveler profile: rider category (invalid = default category) and fare
// media (invalid = any media). Products are applicable to a profile if one of
// their variants matches.
struct profile {
  bool operator==(profile const&) const = default;
  rider_category_idx_t rider_category_{rider_category_idx_t::invalid()};
  fare_media_idx_t media_{fare_media_idx_t::invalid()};
};

bool is_applicable(fares const& f,
                   fares::fare_product const& p,
                   profile const& pr) {
  auto const has_default_category =
      utl::any_of(f.rider_categories_,
                  [](auto const& c) { return c.is_default_fare_category_; });
  auto const rider_ok =
      p.rider_category_ == rider_category_idx_t::invalid() ||
      (pr.rider_category_ == rider_category_idx_t::invalid()
           ? !has_default_category || f.rider_categories_[p.rider_category_]
                                          .is_default_fare_category_
           : p.rider_category_ == pr.rider_category_);
  // "no fare media" products (e.g. cash, no ticket needed) apply to everyone
  auto const media_ok = p.media_ == fare_media_idx_t::invalid() ||
                        f.fare_media_[p.media_].type_ ==
                            fares::fare_media::fare_media_type::kNone ||
                        pr.media_ == fare_media_idx_t::invalid() ||
                        p.media_ == pr.media_;
  return rider_ok && media_ok;
}

// Cheapest applicable variant of a fare product, infinity if not applicable.
// Unset product = free.
float product_cost(fares const& f,
                   fare_product_idx_t const p,
                   profile const& pr) {
  if (p == fare_product_idx_t::invalid()) {
    return 0.F;
  }
  auto min = std::numeric_limits<float>::infinity();
  for (auto const& x : f.fare_products_[p]) {
    if (is_applicable(f, x, pr)) {
      min = std::min(min, x.amount_);
    }
  }
  return min;
}

float leg_cost(fares const& f, fare_leg const& l, profile const& pr) {
  return l.rule_.empty()
             ? 0.F
             : std::ranges::min(l.rule_ | std::views::transform([&](auto&& r) {
                                  return product_cost(f, r.fare_product_, pr);
                                }));
}

// Cost of a sequence of legs joined by transfer rule r.
// Assumption for sequences with more than one transfer: the transfer product
// is paid once for AB (e.g. day pass) and once per transfer otherwise.
float transfer_cost(fares const& f,
                    fares::fare_transfer_rule const& r,
                    std::vector<fare_leg> const& legs,
                    profile const& pr) {
  using fare_transfer_type = fares::fare_transfer_rule::fare_transfer_type;
  auto const ab = product_cost(f, r.fare_product_, pr);
  auto const n_transfers = static_cast<float>(legs.size() - 1U);
  auto const sum = [&](auto&& range) {
    auto x = 0.F;
    for (auto const& l : range) {
      x += leg_cost(f, l, pr);
    }
    return x;
  };
  switch (r.fare_transfer_type_) {
    case fare_transfer_type::kAB: return ab;
    case fare_transfer_type::kAPlusAB:
      return leg_cost(f, legs.front(), pr) + n_transfers * ab;
    case fare_transfer_type::kAPlusABPlusB: return sum(legs) + n_transfers * ab;
  }
  std::unreachable();
}

// Longest sequence of legs starting at `from` joined by transfer rule r with
// the leg rules thinned out to the ones taking part in the transfers.
// Empty if r does not apply to the first transfer.
template <typename It>
std::vector<fare_leg> build_transfer(timetable const& tt,
                                     fares const& f,
                                     fares::fare_transfer_rule const& r,
                                     It const from,
                                     It const end,
                                     auto&& concrete_from,
                                     auto&& concrete_to) {
  auto legs = std::vector<fare_leg>{*from};
  auto remaining_transfers = r.transfer_count_;
  for (auto next = std::next(from);
       next != end && remaining_transfers != 0;  // -1=infinite won't reach 0
       ++next, --remaining_transfers) {
    // legs.back() is the current leg thinned out to the leg rules covered by
    // the previous transfer (or the unthinned first leg)
    auto m = matches(tt, f, r, *from, legs.back(), *next, concrete_from,
                     concrete_to);
    if (!m.has_value()) {
      break;
    }
    legs.back().rule_ = std::move(m->first);
    auto next_leg = *next;
    next_leg.rule_ = std::move(m->second);
    legs.emplace_back(std::move(next_leg));
  }
  if (legs.size() == 1U) {
    legs.clear();
  }
  return legs;
}

// Fare transfer variants of the legs (same source): candidates are one
// standalone entry per leg and one entry per transfer rule and start leg.
// A candidate is a variant if it is part of the cheapest cover of the legs
// for some traveler profile (rider category, fare media) - a transfer is not
// necessarily cheaper than separate leg products (e.g. free legs on weekends
// vs. day pass transfer product) and vice versa. The cover for the default
// traveler (default rider category, any media) is flagged as main and listed
// first.
std::vector<fare_transfer> join_transfers(
    timetable const& tt, std::vector<fare_leg> const& fare_legs) {
  auto transfers = std::vector<fare_transfer>{};
  utl::equal_ranges_linear(
      fare_legs,
      [](fare_leg const& a, fare_leg const& b) { return a.src_ == b.src_; },
      [&](std::vector<fare_leg>::const_iterator const from_it,
          std::vector<fare_leg>::const_iterator const to_it) {
        auto const n = static_cast<std::size_t>(std::distance(from_it, to_it));
        utl::verify(n != 0U, "invalid zero-size range");

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

        auto candidates = std::vector<fare_transfer>{};
        auto standalone = std::vector<std::size_t>(n);  // candidate per leg
        auto chains = std::vector<std::vector<std::size_t>>(n);  // per start
        auto profiles = std::vector<profile>{{}};  // default traveler first
        auto const add_profiles = [&](fare_product_idx_t const p) {
          if (p == fare_product_idx_t::invalid()) {
            return;
          }
          for (auto const& x : f.fare_products_[p]) {
            auto const pr = profile{x.rider_category_, x.media_};
            if (utl::find(profiles, pr) == end(profiles)) {
              profiles.push_back(pr);
            }
          }
        };
        for (auto i = 0U; i != n; ++i) {
          auto const leg_it = std::next(from_it, i);
          for (auto const& r : leg_it->rule_) {
            add_profiles(r.fare_product_);
          }

          standalone[i] = candidates.size();
          candidates.push_back({std::nullopt, {*leg_it}});

          for (auto const& r : f.fare_transfer_rules_) {
            auto legs = build_transfer(tt, f, r, leg_it, to_it, concrete_from,
                                       concrete_to);
            if (!legs.empty()) {
              add_profiles(r.fare_product_);
              chains[i].push_back(candidates.size());
              candidates.push_back({r, std::move(legs)});
            }
          }
        }

        // Cheapest cover of legs [i, n) by standalone legs and transfers.
        struct cover {
          float cost_;
          std::size_t candidate_;
        };
        auto used = std::vector<bool>(candidates.size());
        for (auto const [p, pr] : utl::enumerate(profiles)) {
          auto best = std::vector<cover>(n + 1U, {0.F, 0U});
          for (auto i = n; i-- != 0U;) {
            best[i] = {
                leg_cost(f, *std::next(from_it, i), pr) + best[i + 1U].cost_,
                standalone[i]};
            for (auto const c : chains[i]) {
              auto const& t = candidates[c];
              auto const cost = transfer_cost(f, *t.rule_, t.legs_, pr) +
                                best[i + t.legs_.size()].cost_;
              // on ties: prefer transfers over standalone legs, earlier rules
              // over later rules
              auto const is_better = best[i].candidate_ == standalone[i]
                                         ? cost <= best[i].cost_
                                         : cost < best[i].cost_;
              if (is_better) {
                best[i] = {cost, c};
              }
            }
          }

          auto const is_main = p == 0U;
          if (!is_main && std::isinf(best[0].cost_)) {
            continue;  // no cover for this profile
          }
          for (auto i = 0U; i != n;) {
            auto& t = candidates[best[i].candidate_];
            used[best[i].candidate_] = true;
            t.main_ |= is_main;
            i += t.legs_.size();
          }
        }

        auto variants = std::vector<fare_transfer>{};
        for (auto const [i, c] : utl::enumerate(candidates)) {
          if (used[i]) {
            variants.emplace_back(std::move(c));
          }
        }
        std::ranges::stable_partition(variants,
                                      [](auto const& t) { return t.main_; });
        transfers.insert(end(transfers), std::move_iterator{begin(variants)},
                         std::move_iterator{end(variants)});
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
