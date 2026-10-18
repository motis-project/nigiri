#include "nigiri/transfer_rule_sides.h"

#include <algorithm>
#include <span>
#include <vector>

#include "utl/erase_duplicates.h"
#include "utl/helpers/algorithm.h"

#include "nigiri/footpath.h"
#include "nigiri/stop.h"
#include "nigiri/timetable.h"

namespace nigiri {

bool has_transfer_rules(timetable const& tt, transport_idx_t const t) {
  auto const& tr = tt.transfer_rules_;
  return utl::any_of(tt.transport_section_providers_[t],
                     [&](provider_idx_t const p) {
                       return tr.has_provider_rules(p);
                     }) ||
         utl::any_of(tt.transport_to_trip_section_[t], [&](auto const s) {
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

void add_provider_sides(timetable const& tt,
                        provider_idx_t const p,
                        event_type const ev,
                        location_idx_t const base,
                        std::vector<transfer_rule_side_idx>& sig) {
  for (auto const s : values_of(tt.transfer_rules_.provider_rules_,
                                pair{p, tt.locations_.get_root_idx(base)})) {
    if (s.ev_type() == ev) {
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
  auto const providers = tt.transport_section_providers_[t];
  auto const section = [&](stop_idx_t const k) {
    return sections[sections.size() == 1U ? 0U : k];
  };
  auto sides = std::vector<transfer_rule_side_idx>{};
  auto const add = [&](stop_idx_t const k, event_type const ev) {
    get_section_sides(tt, section(k), sides);
    add_stop_sides(tt, sides, ev, base, sig);
    if (!providers.empty()) {
      add_provider_sides(tt, providers[providers.size() == 1U ? 0U : k], ev,
                         base, sig);
    }
  };
  sig.clear();
  if (i != 0U && stp.out_allowed()) {
    add(static_cast<stop_idx_t>(i - 1U), event_type::kArr);
  }
  if (i + 1U != seq.size() && stp.in_allowed()) {
    add(i, event_type::kDep);
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

}  // namespace nigiri
