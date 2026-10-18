#include "nigiri/transfer_rule_sides.h"

#include <algorithm>
#include <iterator>
#include <span>
#include <tuple>
#include <vector>

#include "utl/equal_ranges_linear.h"
#include "utl/erase_duplicates.h"
#include "utl/helpers/algorithm.h"
#include "utl/to_vec.h"

#include "nigiri/footpath.h"
#include "nigiri/stop.h"
#include "nigiri/timetable.h"

namespace nigiri {

bool has_transfer_rules(timetable const& tt, transport_idx_t const t) {
  auto const& tr = tt.transfer_rules_;
  return utl::any_of(tt.transport_to_trip_section_[t], [&](auto const s) {
    return utl::any_of(tt.merged_trips_[s], [&](trip_idx_t const trip) {
      return !std::ranges::empty(values_of(tr.trip_rules_, trip)) ||
             !std::ranges::empty(
                 values_of(tr.route_rules_, tt.trip_route_id_[trip]));
    });
  });
}

void get_section_sides(timetable const& tt,
                       merged_trips_idx_t const section,
                       std::vector<transfer_rule_side_idx>& sides) {
  auto const& tr = tt.transfer_rules_;
  sides.clear();
  for (auto const trip : tt.merged_trips_[section]) {
    for (auto const s : values_of(tr.trip_rules_, trip)) {
      sides.push_back(s);
    }
    auto const src = tt.trip_id_src_[tt.trip_ids_[trip].front()];
    for (auto const s : values_of(tr.route_rules_, tt.trip_route_id_[trip])) {
      if (tr.rules_[s.rule()].src_ == src) {
        sides.push_back(s);
      }
    }
  }
}

void add_stop_sides(timetable const& tt,
                    std::span<transfer_rule_side_idx const> section_sides,
                    event_type const ev,
                    location_idx_t const base,
                    std::vector<transfer_rule_side_idx>& sig) {
  for (auto const s : section_sides) {
    if (s.ev_type() == ev &&
        is_rule_stop(tt, tt.transfer_rules_.rules_[s.rule()].stop(ev), base)) {
      sig.push_back(s);
    }
  }
}

// The from sides of the arriving trips' rules, the to sides of the departing
// trips' rules (block_id / stay seated: they are not the same trips).
void get_stop_signature(timetable const& tt,
                        transport_idx_t const t,
                        stop_idx_t const i,
                        location_idx_t const base,
                        std::vector<transfer_rule_side_idx>& sig) {
  auto const seq = tt.route_location_seq_[tt.transport_route_[t]];
  auto const stp = stop{seq[i]};
  auto const sections = tt.transport_to_trip_section_[t];
  auto const section = [&](stop_idx_t const k) {
    return sections[sections.size() == 1U ? 0U : k];
  };
  auto sides = std::vector<transfer_rule_side_idx>{};
  sig.clear();
  if (i != 0U && stp.out_allowed()) {
    get_section_sides(tt, section(static_cast<stop_idx_t>(i - 1U)), sides);
    add_stop_sides(tt, sides, event_type::kArr, base, sig);
  }
  if (i + 1U != seq.size() && stp.in_allowed()) {
    get_section_sides(tt, section(i), sides);
    add_stop_sides(tt, sides, event_type::kDep, base, sig);
  }
  utl::erase_duplicates(sig);
}

duration_t get_transfer_time(timetable const& tt,
                             std::span<transfer_rule_side_idx const> sig,
                             location_idx_t const base) {
  auto const& rules = tt.transfer_rules_.rules_;
  auto best = transfer_rule_idx_t::invalid();
  for (auto const s : sig) {
    auto const rule = s.rule();
    if (is_applicable(tt, {rule, event_type::kArr}, sig, base) &&
        is_applicable(tt, {rule, event_type::kDep}, sig, base)) {
      best = best == transfer_rule_idx_t::invalid()
                 ? rule
                 : get_more_specific(tt, best, rule);
    }
  }
  return best != transfer_rule_idx_t::invalid()
             ? rules[best].duration_
             : to_fp_duration(tt.locations_.transfer_time_[base]);
}

transfer_rule_idx_t get_change_rule(
    timetable const& tt,
    std::span<transfer_rule_side_idx const> from,
    std::span<transfer_rule_side_idx const> to,
    location_idx_t const from_base,
    location_idx_t const to_base) {
  auto best = transfer_rule_idx_t::invalid();
  auto const keep = [&](transfer_rule_idx_t const r) {
    if (!is_same_station(tt, tt.transfer_rules_.rules_[r], from_base,
                         to_base)) {
      return;
    }
    best = best == transfer_rule_idx_t::invalid()
               ? r
               : get_more_specific(tt, best, r);
  };
  for (auto const side : from) {
    if (side.ev_type() == event_type::kArr &&
        is_applicable(tt, {side.rule(), event_type::kDep}, to, to_base)) {
      keep(side.rule());
    }
  }
  for (auto const side : to) {
    if (side.ev_type() == event_type::kDep &&
        is_applicable(tt, {side.rule(), event_type::kArr}, from, from_base)) {
      keep(side.rule());
    }
  }
  return best;
}

hash_map<location_idx_t, std::vector<transfer_rule_idx_t>> get_global_rules(
    timetable const& tt) {
  auto const& tr = tt.transfer_rules_;
  auto global_rules =
      hash_map<location_idx_t, std::vector<transfer_rule_idx_t>>{};
  if (utl::none_of(tr.rules_,
                   [](transfer_rule const& r) { return r.is_global(); })) {
    return global_rules;
  }

  // the route ids (attribute groups) arriving / departing at each station
  using event_t =
      std::tuple<location_idx_t, route_id_idx_t, source_idx_t, event_type>;
  auto events = hash_set<event_t>{};
  for (auto r = route_idx_t{0U}; r != tt.n_routes(); ++r) {
    auto const seq = tt.route_location_seq_[r];
    for (auto const t : tt.route_transport_ranges_[r]) {
      if (!has_transfer_rules(tt, t)) {
        continue;
      }
      auto const sections = tt.transport_to_trip_section_[t];
      for (auto i = stop_idx_t{0U}; i != seq.size(); ++i) {
        auto const stp = stop{seq[i]};
        auto const station = tt.locations_.get_root_idx(stp.location_idx());
        if (tr.is_global_excluded(station)) {
          continue;
        }
        auto const add = [&](std::size_t const section, event_type const ev) {
          for (auto const trip :
               tt.merged_trips_[sections[sections.size() == 1U ? 0U
                                                               : section]]) {
            if (auto const route = tt.trip_route_id_[trip];
                route != route_id_idx_t::invalid()) {
              events.emplace(station, route,
                             tt.trip_id_src_[tt.trip_ids_[trip].front()], ev);
            }
          }
        };
        if (i != 0U && stp.out_allowed()) {
          add(i - 1U, event_type::kArr);
        }
        if (i + 1U != seq.size() && stp.in_allowed()) {
          add(i, event_type::kDep);
        }
      }
    }
  }

  auto sorted = utl::to_vec(events);
  events = {};
  utl::sort(sorted);
  auto arr = std::vector<transfer_rule_idx_t>{};
  auto dep = std::vector<transfer_rule_idx_t>{};
  utl::equal_ranges_linear(
      sorted,
      [](event_t const& a, event_t const& b) {
        return std::get<0>(a) == std::get<0>(b);
      },
      [&](auto const from, auto const to) {
        arr.clear();
        dep.clear();
        for (auto it = from; it != to; ++it) {
          auto const [_, route, src, ev] = *it;
          for (auto const s : values_of(tr.route_rules_, route)) {
            auto const& rule = tr.rules_[s.rule()];
            if (rule.is_global() && rule.src_ == src && s.ev_type() == ev) {
              (ev == event_type::kArr ? arr : dep).push_back(s.rule());
            }
          }
        }
        utl::erase_duplicates(arr);
        utl::erase_duplicates(dep);
        auto both = std::vector<transfer_rule_idx_t>{};
        std::ranges::set_intersection(arr, dep, std::back_inserter(both));
        if (!both.empty()) {
          global_rules.emplace(std::get<0>(*from), std::move(both));
        }
      });
  return global_rules;
}

}  // namespace nigiri
