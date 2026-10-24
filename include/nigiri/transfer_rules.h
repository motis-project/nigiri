#pragma once

#include <compare>
#include <cstdint>
#include <algorithm>
#include <ranges>
#include <span>
#include <tuple>
#include <vector>

#include "cista/reflection/comparable.h"

#include "nigiri/footpath.h"
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
           trip(ev) != trip_idx_t::invalid() ||
           (listed_ & (ev == event_type::kArr ? 1U : 2U)) != 0U;
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

  // side qualified by the trips or routes the loader listed in
  // trip_rules_ or route_rules_ (HRDF); bit 0: from side, bit 1: to side
  std::uint8_t listed_{0U};
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
  bool empty() const { return rules_.empty(); }

  vector_map<transfer_rule_idx_t, transfer_rule> rules_;

  qualified_sides_t<trip_idx_t> trip_rules_;
  qualified_sides_t<pair<source_idx_t, route_id_idx_t>> route_rules_;

  vector<pair<location_idx_t, transfer_rule_side_idx>> stop_rules_;
};

}  // namespace nigiri
