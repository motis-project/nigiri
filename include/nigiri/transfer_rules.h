#pragma once

#include <compare>
#include <cstdint>
#include <algorithm>
#include <ranges>
#include <tuple>

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
  bool from_qualified() const {
    return from_route_ != route_id_idx_t::invalid() ||
           from_trip_ != trip_idx_t::invalid();
  }
  bool to_qualified() const {
    return to_route_ != route_id_idx_t::invalid() ||
           to_trip_ != trip_idx_t::invalid();
  }

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
                                   bool const is_from)
      : v_{(to_idx(rule) << 1U) | (is_from ? 0U : 1U)} {}

  constexpr transfer_rule_idx_t rule() const {
    return transfer_rule_idx_t{v_ >> 1U};
  }
  constexpr bool is_from() const { return (v_ & 1U) == 0U; }

  auto operator<=>(transfer_rule_side_idx const&) const = default;
  bool operator==(transfer_rule_side_idx const&) const = default;

  auto cista_members() { return std::tie(v_); }

  std::uint32_t v_{0U};
};

template <typename SortedPairs, typename Key>
auto values_of(SortedPairs const& v, Key const key) {
  using entry = std::ranges::range_value_t<SortedPairs>;
  return std::ranges::equal_range(v, key, {}, &entry::first) |
         std::views::transform(&entry::second);
}

struct transfer_rules {
  bool empty() const { return rules_.empty(); }

  vector_map<transfer_rule_idx_t, transfer_rule> rules_;

  vector<pair<trip_idx_t, transfer_rule_side_idx>> trip_rules_;
  vector<pair<route_id_idx_t, transfer_rule_side_idx>> route_rules_;

  vector<pair<location_idx_t, transfer_rule_side_idx>> stop_rules_;

  vector<pair<transfer_rule_side_idx, location_idx_t>> rule_virts_;

  vector<pair<location_idx_t, transfer_rule_side_idx>> virt_rules_;
};

bool covers(timetable const&, location_idx_t stop, location_idx_t base);

}  // namespace nigiri
