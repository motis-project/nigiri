#pragma once

#include <compare>
#include <cstdint>
#include <algorithm>
#include <optional>
#include <ranges>
#include <span>
#include <tuple>
#include <vector>

#include "cista/reflection/comparable.h"

#include "nigiri/common/interval.h"
#include "nigiri/footpath.h"
#include "nigiri/routing/limits.h"
#include "nigiri/types.h"

namespace nigiri {

struct timetable;

struct preferred_transfer {
  location_idx_t to_{location_idx_t::invalid()};
  trip_idx_t from_trip_{trip_idx_t::invalid()};
  trip_idx_t to_trip_{trip_idx_t::invalid()};
  route_id_idx_t from_route_{route_id_idx_t::invalid()};
  route_id_idx_t to_route_{route_id_idx_t::invalid()};
};

using transfer_rule_specificity_t = std::uint8_t;

struct transfer_rule {
  location_idx_t stop(event_type const ev) const {
    return ev == event_type::kArr ? from_stop_ : to_stop_;
  }
  route_id_idx_t route(event_type const ev) const {
    return ev == event_type::kArr ? from_route_ : to_route_;
  }
  trip_idx_t trip(event_type const ev) const {
    return ev == event_type::kArr ? from_trip_ : to_trip_;
  }
  bool is_qualified(event_type const ev) const {
    return route(ev) != route_id_idx_t::invalid() ||
           trip(ev) != trip_idx_t::invalid();
  }
  bool is_qualified() const {
    return is_qualified(event_type::kArr) || is_qualified(event_type::kDep);
  }
  bool is_forbidden() const { return duration_ == footpath::kBan; }

  location_idx_t from_stop_{location_idx_t::invalid()};
  location_idx_t to_stop_{location_idx_t::invalid()};
  route_id_idx_t from_route_{route_id_idx_t::invalid()};
  route_id_idx_t to_route_{route_id_idx_t::invalid()};
  trip_idx_t from_trip_{trip_idx_t::invalid()};
  trip_idx_t to_trip_{trip_idx_t::invalid()};
  source_idx_t src_{source_idx_t::invalid()};
  duration_t duration_{0};
  transfer_rule_specificity_t specificity_{0U};
};

struct transfer_rule_side_idx {
  transfer_rule_side_idx() = default;
  constexpr transfer_rule_side_idx(transfer_rule_idx_t const rule,
                                   event_type const ev)
      : v_{(to_idx(rule) << 1U) | (ev == event_type::kArr ? 0U : 1U)} {}

  constexpr transfer_rule_idx_t rule() const {
    return transfer_rule_idx_t{v_ >> 1U};
  }
  constexpr event_type ev_type() const {
    return (v_ & 1U) == 0U ? event_type::kArr : event_type::kDep;
  }

  auto operator<=>(transfer_rule_side_idx const&) const = default;
  bool operator==(transfer_rule_side_idx const&) const = default;

  auto cista_members() { return std::tie(v_); }

  std::uint32_t v_{0U};
};

template <typename Qualifier>
struct qualified_side {
  CISTA_COMPARABLE()
  Qualifier qualifier_;
  location_idx_t l_;
  transfer_rule_side_idx side_;
};

template <typename Qualifier>
using qualified_sides_t = vector<qualified_side<Qualifier>>;

template <typename Qualifier>
std::span<qualified_side<Qualifier> const> rules_of(
    qualified_sides_t<Qualifier> const& v, Qualifier const qualifier) {
  auto const r = std::ranges::equal_range(
      v, qualifier, {}, &qualified_side<Qualifier>::qualifier_);
  return {r.begin(), r.end()};
}

template <typename Qualifier>
auto rules_at(std::span<qualified_side<Qualifier> const> rules,
              location_idx_t const l) {
  return std::ranges::equal_range(rules, l, {},
                                  &qualified_side<Qualifier>::l_) |
         std::views::transform(&qualified_side<Qualifier>::side_);
}

template <typename SortedPairs, typename Key>
auto values_of(SortedPairs const& v, Key const key) {
  using entry = std::ranges::range_value_t<SortedPairs>;
  return std::ranges::equal_range(v, key, {}, &entry::first) |
         std::views::transform(&entry::second);
}

struct transfer_rules {
  vector_map<transfer_rule_idx_t, transfer_rule> rules_;

  qualified_sides_t<trip_idx_t> trip_rules_;
  qualified_sides_t<pair<source_idx_t, route_id_idx_t>> route_rules_;

  vector<pair<location_idx_t, transfer_rule_side_idx>> stop_rules_;
  bitvec_map<trip_idx_t> trip_has_rule_;

  vector_map<location_idx_t, duration_t> max_rule_duration_;
  vector<pair<location_idx_t, location_idx_t>> to_stops_, from_stops_;
  bitvec_map<location_idx_t> is_rule_location_;
};

using window_t = interval<duration_t>;

constexpr auto const kNever = duration_t{routing::kMaxTravelTime};

inline void narrow(window_t& w,
                   duration_t const gap,
                   duration_t const expected) {
  if (gap >= expected) {
    w.to_ = std::min(w.to_, gap + duration_t{1});
  } else {
    w.from_ = std::max(w.from_, gap + duration_t{1});
  }
}

inline std::optional<duration_t> pick_duration(window_t const& w,
                                               duration_t const fallback) {
  if (w.contains(fallback)) {
    return fallback;
  }
  if (w.contains(kNever)) {
    return kNever;
  }
  return w.from_ < std::min(w.to_, footpath::kBan) ? std::optional{w.from_}
                                                   : std::nullopt;
}

using signature_t = std::vector<transfer_rule_side_idx>;

bool has_transfer_rules(timetable const&, transport_idx_t);

void get_stop_signature(timetable const&,
                        transport_idx_t,
                        stop_idx_t,
                        location_idx_t base,
                        signature_t&);

struct section_rules {
  bool empty() const { return trips_.empty() && routes_.empty(); }

  std::span<qualified_side<trip_idx_t> const> trips_;
  std::span<qualified_side<pair<source_idx_t, route_id_idx_t>> const> routes_;
};

section_rules get_section_rules(timetable const&, merged_trips_idx_t);

void add_stop_rules(timetable const&,
                    section_rules const&,
                    event_type,
                    location_idx_t base,
                    signature_t&);

transfer_rule_idx_t get_change_rule(
    timetable const&,
    std::span<transfer_rule_side_idx const> from_sig,
    std::span<transfer_rule_side_idx const> to_sig,
    location_idx_t from_base,
    location_idx_t to_base);

}  // namespace nigiri
