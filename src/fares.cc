#include "nigiri/fares.h"

#include <ranges>

#include "fmt/ranges.h"

#include "utl/concat.h"
#include "utl/enumerate.h"
#include "utl/pairwise.h"
#include "utl/parser/cstr.h"
#include "utl/to_vec.h"

#include "nigiri/loader/gtfs/noon_offsets.h"
#include "nigiri/routing/journey.h"
#include "nigiri/rt/frun.h"
#include "nigiri/timetable.h"

// #define trace(...) fmt::println(__VA_ARGS__)
#define trace(...)

namespace nigiri {

using routing::journey;

std::string to_string(timetable const& tt,
                      std::vector<fare_transfer> const& x) {
  auto ss = std::stringstream{};

  for (auto const& transfer : x) {
    if (transfer.rule_.has_value()) {
      ss << "FARE TRANSFER START\n";
      auto const f = tt.fares_[transfer.legs_.front().src_];
      auto const r = *transfer.rule_;
      auto const product =
          r.fare_product_ == fare_product_idx_t::invalid()
              ? ""
              : tt.strings_.get(f.fare_products_[r.fare_product_].name_);
      ss << "TRANSFER PRODUCT: " << product << "\n";
      ss << "RULE: " << r.fare_transfer_type_ << "\n";
    }
    for (auto const& l : transfer.legs_) {
      ss << "FARE LEG:\n";
      auto first = true;
      for (auto const& jl : l.joined_leg_) {
        if (first) {
          first = false;
        } else {
          ss << "** JOINED WITH\n";
        }
        jl->print(ss, tt);
      }
      ss << "PRODUCTS\n";
      for (auto const& r : l.rules_) {
        auto const& f = tt.fares_[l.src_];
        auto const& p = f.fare_products_[r.fare_product_];
        ss << tt.strings_.get(p.name_) << " [priority=" << r.rule_priority_
           << "]: " << p.amount_ << " " << tt.strings_.get(p.currency_code_)
           << ", fare_media_name="
           << (p.media_ == fare_media_idx_t::invalid()
                   ? ""
                   : tt.strings_.get(f.fare_media_[p.media_].name_))
           << ", fare_type="
           << (p.media_ == fare_media_idx_t::invalid()
                   ? fares::fare_media::fare_media_type::kNone
                   : f.fare_media_[p.media_].type_)
           << ", ride_category="
           << (p.rider_category_ == rider_category_idx_t::invalid()
                   ? ""
                   : tt.strings_.get(
                         f.rider_categories_[p.rider_category_].name_))
           << "\n";
      }
      ss << "\n\n";
    }
    if (transfer.rule_.has_value()) {
      ss << "FARE TRANSFER END\n\n";
    }
  }
  return ss.str();
}

bool contains(auto&& range, auto&& needle) {
  return std::ranges::find(range, needle) != std::end(range);
}

auto key(fares::fare_transfer_rule const& x) {
  return std::tie(x.from_leg_group_, x.to_leg_group_, x.transfer_count_,
                  x.duration_limit_, x.duration_limit_type_,
                  x.fare_transfer_type_);
}

bool operator<(fares::fare_transfer_rule const& a,
               fares::fare_transfer_rule const& b) {
  return key(a) < key(b);
}

bool operator==(fares::fare_transfer_rule const& a,
                fares::fare_transfer_rule const& b) {
  return key(a) == key(b);
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

bool fare_props::matches(fare_props const o) const {
  return (rider_category_ == rider_category_idx_t::invalid() ||
          rider_category_ == o.rider_category_);
}

fare_props fares::fare_leg_rule::props(fares const& f) const {
  return {f.fare_products_[fare_product_].rider_category_,
          f.fare_products_[fare_product_].media_};
}

fare_props fares::fare_transfer_rule::props(fares const& f) const {
  return fare_product_ == fare_product_idx_t::invalid()
             ? fare_props{rider_category_idx_t::invalid(),
                          fare_media_idx_t::invalid()}
             : fare_props{f.fare_products_[fare_product_].rider_category_,
                          f.fare_products_[fare_product_].media_};
}

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
                       : tt.strings_.get(f.fare_products_[r.fare_product_].id_))
               << ")";
  }

  timetable const& tt_;
  fares const& f_;
  fares::fare_transfer_rule const& r_;
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
           << ", product="
           << tt.strings_.get(f.fare_products_[r.fare_product_].id_) << ")";
  }

  timetable const& tt_;
  fares const& f_;
  fares::fare_leg_rule const& r_;
};

float price(fares const& f, fare_product_idx_t const p) {
  return p == fare_product_idx_t::invalid() ? 0.F : f.fare_products_[p].amount_;
}

auto fares::fare_leg_rule::match_members() const {
  return std::tie(network_, from_area_, to_area_, from_timeframe_group_,
                  to_timeframe_group_);
}

bool operator==(fares::fare_leg_rule const&,
                fares::fare_leg_rule const&) = default;

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
    timetable const& tt, effective_fare_leg_t const& joined_legs) {
  auto const& first = joined_legs.front();
  auto const& last = joined_legs.back();

  auto const first_r = std::get<journey::run_enter_exit>(first->uses_).r_;
  auto const first_trip = rt::frun{tt, nullptr, first_r};

  auto const last_r = std::get<journey::run_enter_exit>(last->uses_).r_;
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

bool fare_transfer_matches([[maybe_unused]] timetable const& tt,
                           fares const& f,
                           fares::fare_transfer_rule const& r,
                           fare_leg const& from,
                           fare_leg const& next,
                           fares::fare_leg_rule const& from_rule,
                           fares::fare_leg_rule const& curr_rule,
                           fares::fare_leg_rule const& next_rule,
                           auto&& concrete_from,
                           auto&& concrete_to) {
  using duration_limit_type = fares::fare_transfer_rule::duration_limit_type;

  utl::verify(!from.joined_leg_.empty(), "from no joined leg");
  utl::verify(!next.joined_leg_.empty(), "b no joined leg");

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
        return next.joined_leg_.front()->dep_time_;

      case duration_limit_type::kCurrArrNextArr:
      case duration_limit_type::kCurrDepNextArr:
        return next.joined_leg_.back()->arr_time_;
    }
    std::unreachable();
  };

  auto const props_match = r.props(f).matches(curr_rule.props(f)) &&
                           r.props(f).matches(next_rule.props(f)) &&
                           curr_rule.props(f).matches(next_rule.props(f)) &&
                           from_rule.props(f).matches(next_rule.props(f));
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

  if (!props_match) {
    trace("      no props match");
  }
  if (!transfer_limit_ok) {
    trace("      transfer limit exceeded");
  }
  if (!from_leg_group_matches) {
    trace(
        "      from leg group mismatch\n"
        "        r.from_leg_group: {}\n"
        "        curr_rule.leg_group: {}\n"
        "        concrete_from: {}\n",
        fmt::streamed(leg_group{tt, f, r.from_leg_group_}),
        fmt::streamed(leg_group{tt, f, curr_rule.leg_group_idx_}),
        concrete_from | std::views::transform([&](auto&& x) {
          return fmt::streamed(leg_group{tt, f, x});
        }));
  }
  if (!to_leg_group_matches) {
    trace(
        "      to leg group mismatch\n"
        "        r.to_leg_group: {}\n"
        "        next_rule.leg_group: {}\n"
        "        concrete_to: {}",
        fmt::streamed(leg_group{tt, f, r.to_leg_group_}),
        fmt::streamed(leg_group{tt, f, next_rule.leg_group_idx_}),
        concrete_to | std::views::transform([&](auto&& x) {
          return fmt::streamed(leg_group{tt, f, x});
        }));
  }

  return props_match && transfer_limit_ok && from_leg_group_matches &&
         to_leg_group_matches;
}

std::optional<fares::fare_leg_rule> find_fare_transfer_match(
    timetable const& tt,
    fares const& f,
    fares::fare_transfer_rule const& r,
    fare_leg const& from,
    fare_leg const& next,
    fares::fare_leg_rule const& from_rule,
    fares::fare_leg_rule const& curr_rule,
    auto&& concrete_from,
    auto&& concrete_to) {
  for (auto const next_rule : next.rules_) {
    if (fare_transfer_matches(tt, f, r, from, next, from_rule, curr_rule,
                              next_rule, concrete_from, concrete_to)) {
      trace("    {}: match\n", fmt::streamed(leg_rule{tt, f, next_rule}));
      return next_rule;
    }
    trace("    {}: NO match\n", fmt::streamed(leg_rule{tt, f, next_rule}));
  }
  return std::nullopt;
}

fare_leg filter_rule(fares::fare_transfer_rule const& transfer_rule,
                     fare_leg const& l,
                     std::optional<fares::fare_leg_rule> const& leg_rule,
                     std::size_t const i) {
  using fare_transfer_type = fares::fare_transfer_rule::fare_transfer_type;
  auto copy = l;
  copy.rules_ =
      leg_rule.has_value() && ((transfer_rule.fare_transfer_type_ ==
                                    fare_transfer_type::kAPlusABPlusB ||
                                (transfer_rule.fare_transfer_type_ ==
                                     fare_transfer_type::kAPlusAB &&
                                 i == 0)))
          ? std::vector<fares::fare_leg_rule>{*leg_rule}
          : std::vector<fares::fare_leg_rule>{};
  return copy;
}

struct joined_leg {
  friend std::ostream& operator<<(std::ostream& out, joined_leg const& l) {
    auto const& [tt, joined_leg] = l;
    return out << location{tt, joined_leg.front()->from_} << "-"
               << location{tt, joined_leg.back()->to_};
  }

  timetable const& tt_;
  std::vector<journey::leg const*> const& joined_leg_;
};

float price(fares const& f,
            std::vector<fare_transfer> const& v,
            fare_props const& p) {
  auto const find_matching_rule =
      [&](std::vector<fares::fare_leg_rule> const& rules)
      -> std::optional<fares::fare_leg_rule> {
    {  // Try with specific criteria
      auto const it = utl::find_if(rules, [&](fares::fare_leg_rule const& x) {
        return x.props(f).matches(p);
      });
      if (it != end(rules)) {
        return *it;
      }
    }

    {  // Try with default criteria
      auto const it = utl::find_if(rules, [&](fares::fare_leg_rule const& x) {
        auto const rc = f.fare_products_[x.fare_product_].rider_category_;
        return rc == rider_category_idx_t::invalid() ||
               f.rider_categories_[rc].is_default_fare_category_;
      });
      if (it != end(rules)) {
        return *it;
      }
    }

    return std::nullopt;
  };

  auto sum = 0.F;

  for (auto const& [rule, legs] : v) {
    auto leg_sum = 0.F;
    for (auto const& l : legs) {
      if (l.rules_.empty()) {
        continue;
      }

      auto const matching_rule = find_matching_rule(l.rules_);
      if (matching_rule) {
        leg_sum += f.fare_products_[matching_rule->fare_product_].amount_;
        continue;
      }

      return std::numeric_limits<float>::infinity();
    }

    if (rule.has_value()) {
      if (!rule->props(f).matches(p)) {
        return std::numeric_limits<float>::infinity();
      }
      auto const rule_sum =
          (rule->fare_product_ == fare_product_idx_t::invalid()
               ? 0.F
               : f.fare_products_[rule->fare_product_].amount_);
      switch (rule->fare_transfer_type_) {
        case fares::fare_transfer_rule::fare_transfer_type::kAPlusAB:
          utl::verify(!legs.front().rules_.empty(), "empty rules");
          sum += rule_sum;
          sum += f.fare_products_[legs.front().rules_.front().fare_product_]
                     .amount_;
          break;
        case fares::fare_transfer_rule::fare_transfer_type::kAPlusABPlusB:
          sum += rule_sum;
          sum += leg_sum;
          break;
        case fares::fare_transfer_rule::fare_transfer_type::kAB:
          sum += rule_sum;
          break;
      }
    } else {
      sum += leg_sum;
    }
  }
  return sum;
}

std::vector<std::vector<fare_transfer>> join_transfers(
    timetable const& tt, std::vector<fare_leg> const& fare_legs) {
  auto all_transfers = std::vector<std::vector<fare_transfer>>{};
  utl::equal_ranges_linear(
      fare_legs,
      [](fare_leg const& a, fare_leg const& b) { return a.src_ == b.src_; },
      [&](std::vector<fare_leg>::const_iterator const from_it,
          std::vector<fare_leg>::const_iterator const to_it) {
        auto const size = static_cast<unsigned>(std::distance(from_it, to_it));
        utl::verify(size != 0U, "invalid zero-size range");

        auto transfers = std::vector<fare_transfer>{};
        if (size == 1U) {
          transfers.push_back({{}, {*from_it}});
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

        struct queue_entry {
          std::vector<fare_transfer> transfers_;
          std::vector<fare_leg>::const_iterator it_;
        };
        [[maybe_unused]] auto const to_string = [&](queue_entry const& e) {
          return fmt::format(
              "#transfers={}, transfers={}, it={}", e.transfers_.size(),
              e.transfers_ | std::views::transform([&](fare_transfer const& x) {
                return x.legs_ | std::views::transform([&](fare_leg const& l) {
                         return fmt::streamed(joined_leg{tt, l.joined_leg_});
                       });
              }),
              fmt::streamed(joined_leg{tt, e.it_->joined_leg_}));
        };

        auto q = std::vector<queue_entry>{{.transfers_ = {}, .it_ = from_it}};
        auto alternatives = std::vector<std::vector<fare_transfer>>{};
        while (!q.empty()) {
          auto curr = q.back();
          q.resize(q.size() - 1U);

          trace("DEQUEUE: {}", to_string(curr));

          utl::verify(curr.it_ != to_it && std::next(curr.it_) != to_it,
                      "invalid iterator");

          auto has_match = false;
          for (auto const [r_i, r] : utl::enumerate(f.fare_transfer_rules_)) {
            for (auto const [from_i, from_rule] :
                 utl::enumerate(curr.it_->rules_)) {
              for (auto const [second_i, second_rule] :
                   utl::enumerate(std::next(curr.it_)->rules_)) {
                auto const initial_match = fare_transfer_matches(
                    tt, f, r, *curr.it_, *std::next(curr.it_), from_rule,
                    from_rule, second_rule, concrete_from, concrete_to);

                trace(
                    "{}/{}, {}/{}, {}/{}  {}initial match [{} => {}]:\n"
                    "  transfer={}\n"
                    "  from_rule={}\n"
                    "  second_rule={}"
                    "{}",
                    r_i, f.fare_transfer_rules_.size(), from_i,
                    curr.it_->rules_.size(), second_i,
                    std::next(curr.it_)->rules_.size(),
                    !initial_match ? "NO " : "",
                    fmt::streamed(joined_leg{tt, curr.it_->joined_leg_}),
                    fmt::streamed(
                        joined_leg{tt, std::next(curr.it_)->joined_leg_}),
                    fmt::streamed(transfer_rule{tt, f, r}),
                    fmt::streamed(leg_rule{tt, f, from_rule}),
                    fmt::streamed(leg_rule{tt, f, second_rule}),
                    initial_match ? "" : "\n");

                if (!initial_match) {
                  continue;
                }

                has_match = true;

                auto const from = curr.it_;
                auto it = from;
                auto next = std::next(it);
                auto matched =
                    std::vector<fare_leg>{filter_rule(r, *it, from_rule, 0U)};
                auto remaining_transfers =
                    r.transfer_count_ > 0 ? r.transfer_count_ : -1;
                auto pred_rule = from_rule;
                for (; next != to_it && remaining_transfers != 0;
                     ++it, ++next, --remaining_transfers) {
                  auto const match =
                      remaining_transfers > 0
                          ? std::nullopt
                          : find_fare_transfer_match(
                                tt, f, r, *from, *next, from_rule, pred_rule,
                                concrete_from, concrete_to);
                  if (remaining_transfers > 0 || match.has_value()) {
                    trace("  NEXT MATCH [TO: {}]",
                          fmt::streamed(joined_leg{tt, next->joined_leg_}));
                    matched.emplace_back(
                        filter_rule(r, *next, match, matched.size()));
                    pred_rule = *match;
                  } else {
                    trace("  NO MATCH [TO: {}]",
                          fmt::streamed(joined_leg{tt, next->joined_leg_}));
                    break;
                  }
                }

                auto copy = curr.transfers_;
                copy.push_back(fare_transfer{r, std::move(matched)});

                if (next == to_it) {
                  // End reached. Write alternative.
                  trace("FINISHED -> WRITE ALTERNATIVE\n");
                  alternatives.emplace_back(std::move(copy));
                } else if (std::next(next) == to_it) {
                  // Single leg left, no transfer to match.
                  copy.push_back(fare_transfer{{}, {*next}});
                  trace("SINGLE LEG LEFT -> WRITE ALTERNATIVE [#rules={}]\n",
                        next->rules_.size());
                  alternatives.emplace_back(std::move(copy));
                } else {
                  auto e =
                      queue_entry{.transfers_ = std::move(copy), .it_ = next};
                  trace("NOT FINISHED [remaining_transfers={}] -> ENQUEUE {}\n",
                        remaining_transfers, to_string(e));
                  q.push_back(std::move(e));
                }
              }
            }
          }

          if (!has_match) {
            auto copy = curr.transfers_;
            copy.push_back(fare_transfer{{}, {*curr.it_}});
            if (std::next(curr.it_, 2) == to_it) {
              trace("NO MATCH, FINISHED [TWO LEFT] -> WRITE ALTERNATIVE\n");
              copy.push_back(fare_transfer{{}, {*std::next(curr.it_)}});
              alternatives.emplace_back(std::move(copy));
            } else if (std::next(curr.it_) == to_it) {
              trace("NO MATCH, FINISHED [ONE LEFT] -> WRITE ALTERNATIVE\n");
              alternatives.emplace_back(std::move(copy));
            } else {
              trace("NO MATCH -> ENQUEUE\n\n");
              q.push_back(
                  {.transfers_ = std::move(copy), .it_ = std::next(curr.it_)});
            }
          }
        }

        utl::verify(!alternatives.empty(), "no alternatives");

        auto combinations =
            hash_set<std::pair<rider_category_idx_t, fare_media_idx_t>>{};
        for (auto const& a : alternatives) {
          for (auto const& x : a) {
            if (x.rule_.has_value() &&
                x.rule_->fare_product_ != fare_product_idx_t::invalid()) {
              auto const& p = f.fare_products_[x.rule_->fare_product_];
              combinations.emplace(p.rider_category_, p.media_);
            }
            for (auto const& l : x.legs_) {
              for (auto const& r : l.rules_) {
                if (r.fare_product_ != fare_product_idx_t::invalid()) {
                  auto const& p = f.fare_products_[r.fare_product_];
                  combinations.emplace(p.rider_category_, p.media_);
                }
              }
            }
          }
        }

        auto cheapest_alternatives = std::vector<std::vector<fare_transfer>>{};

        trace("COMBINATIONS:\n");
        for (auto const& [r, m] : combinations) {
          trace("combination: {} ++ {}\n",
                (r == rider_category_idx_t::invalid()
                     ? "-"
                     : tt.strings_.get(f.rider_categories_[r].name_)),
                (m == fare_media_idx_t::invalid()
                     ? "-"
                     : tt.strings_.get(f.fare_media_[m].name_)));
          utl::sort(alternatives, [&](std::vector<fare_transfer> const& a,
                                      std::vector<fare_transfer> const& b) {
            return price(f, a, {r, m}) < price(f, b, {r, m});
          });

          for (auto const& a : alternatives) {
            auto const p = price(f, a, {r, m});
            if ((p < std::numeric_limits<float>::infinity())) {
              trace("PRICE: {}\n", p);
              trace("{}\n", nigiri::to_string(tt, a));
            }
          }

          cheapest_alternatives.push_back(alternatives.front());
        }

        utl::concat(all_transfers, cheapest_alternatives);
      });

  return all_transfers;
}

std::vector<std::vector<fare_transfer>> get_fares(timetable const& tt,
                                                  journey const& j) {
  return join_transfers(
      tt, utl::to_vec(join_legs(tt, get_transit_legs(j)),
                      [&](effective_fare_leg_t const& joined_leg) {
                        auto const [src, rules] =
                            match_leg_rule(tt, joined_leg);
                        return fare_leg{src, joined_leg, rules};
                      }));
}

}  // namespace nigiri