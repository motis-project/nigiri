#include "nigiri/transfer_rules.h"

#include <algorithm>
#include <span>
#include <tuple>
#include <vector>

#include "utl/erase_duplicates.h"
#include "utl/helpers/algorithm.h"

#include "nigiri/stop.h"
#include "nigiri/timetable.h"

namespace nigiri {

pair<source_idx_t, route_id_idx_t> route_id_of(timetable const& tt,
                                               trip_idx_t const trip) {
  return {tt.trip_id_src_[trip], tt.trip_route_id_[trip]};
}

bool has_transfer_rules(timetable const& tt, transport_idx_t const t) {
  auto const& tr = tt.transfer_rules_;
  return utl::any_of(tt.transport_to_trip_section_[t], [&](auto const s) {
    return utl::any_of(tt.merged_trips_[s], [&](trip_idx_t const trip) {
      return tr.trip_has_rule_.test(trip);
    });
  });
}

section_rules get_section_rules(timetable const& tt,
                                merged_trips_idx_t const section) {
  auto const& tr = tt.transfer_rules_;
  auto const trip = tt.merged_trips_[section].front();
  return {.trips_ = rules_of(tr.trip_rules_, trip),
          .routes_ = rules_of(tr.route_rules_, route_id_of(tt, trip))};
}

void add_stop_rules(timetable const& tt,
                    section_rules const& section,
                    event_type const ev,
                    location_idx_t const base,
                    signature_t& sig) {
  auto const& tr = tt.transfer_rules_;
  auto const root = tt.locations_.get_root_idx(base);
  auto const add = [&](transfer_rule_side_idx const s) {
    auto const& r = tr.rules_[s.rule()];
    if (s.ev_type() == ev &&
        tt.locations_.is_self_or_parent(r.stop(ev), base)) {
      sig.push_back(s);
    }
  };
  for (auto const s : rules_at(section.trips_, root)) {
    add(s);
  }
  for (auto const s : rules_at(section.routes_, root)) {
    add(s);
  }
}

void get_stop_signature(timetable const& tt,
                        transport_idx_t const t,
                        stop_idx_t const i,
                        location_idx_t const base,
                        signature_t& sig) {
  auto const seq = tt.route_location_seq_[tt.transport_route_[t]];
  auto const stp = stop{seq[i]};
  auto const sections = tt.transport_to_trip_section_[t];
  auto const rules = [&](unsigned const k) {
    return get_section_rules(tt, sections[sections.size() == 1U ? 0U : k]);
  };
  sig.clear();
  if (i != 0U && stp.out_allowed()) {
    add_stop_rules(tt, rules(i - 1U), event_type::kArr, base, sig);
  }
  if (i + 1U != seq.size() && stp.in_allowed()) {
    add_stop_rules(tt, rules(i), event_type::kDep, base, sig);
  }
  utl::erase_duplicates(sig);
}

transfer_rule_idx_t get_more_specific(timetable const& tt,
                                      transfer_rule_idx_t const a,
                                      transfer_rule_idx_t const b) {
  // Compare with rule index as tie breaker
  // -> deterministic ordering accross imports
  // -> loader and real-time updates choose the same rule
  auto const& rules = tt.transfer_rules_.rules_;
  return std::tie(rules[a].specificity_, a) < std::tie(rules[b].specificity_, b)
             ? b
             : a;
}

bool is_applicable(timetable const& tt,
                   transfer_rule_side_idx const side,
                   std::span<transfer_rule_side_idx const> sig,
                   location_idx_t const base) {
  auto const& r = tt.transfer_rules_.rules_[side.rule()];
  return r.is_qualified(side.ev_type())
             ? std::ranges::binary_search(sig, side)
             : tt.locations_.is_self_or_parent(r.stop(side.ev_type()), base);
}

transfer_rule_idx_t get_change_rule(
    timetable const& tt,
    std::span<transfer_rule_side_idx const> from_sig,
    std::span<transfer_rule_side_idx const> to_sig,
    location_idx_t const from_base,
    location_idx_t const to_base) {
  auto best = transfer_rule_idx_t::invalid();
  auto const keep = [&](transfer_rule_idx_t const r) {
    best = best == transfer_rule_idx_t::invalid()
               ? r
               : get_more_specific(tt, best, r);
  };
  for (auto const side : from_sig) {
    if (side.ev_type() == event_type::kArr &&
        is_applicable(tt, {side.rule(), event_type::kDep}, to_sig, to_base)) {
      keep(side.rule());
    }
  }
  for (auto const side : to_sig) {
    if (side.ev_type() == event_type::kDep &&
        is_applicable(tt, {side.rule(), event_type::kArr}, from_sig,
                      from_base)) {
      keep(side.rule());
    }
  }
  return best;
}

}  // namespace nigiri
