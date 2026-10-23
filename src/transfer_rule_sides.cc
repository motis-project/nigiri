#include "nigiri/transfer_rule_sides.h"

#include <algorithm>
#include <span>
#include <vector>

#include "utl/erase_duplicates.h"
#include "utl/helpers/algorithm.h"

#include "nigiri/stop.h"
#include "nigiri/timetable.h"

namespace nigiri {

pair<source_idx_t, route_id_idx_t> route_id_of(timetable const& tt,
                                               trip_idx_t const trip) {
  return {tt.trip_id_src_[tt.trip_ids_[trip].front()], tt.trip_route_id_[trip]};
}

bool has_transfer_rules(timetable const& tt, transport_idx_t const t) {
  auto const& tr = tt.transfer_rules_;
  return utl::any_of(tt.transport_to_trip_section_[t], [&](auto const s) {
    return utl::any_of(tt.merged_trips_[s], [&](trip_idx_t const trip) {
      return !rules_of(tr.trip_rules_, trip).empty() ||
             !rules_of(tr.route_rules_, route_id_of(tt, trip)).empty();
    });
  });
}

void get_section_rules(timetable const& tt,
                       merged_trips_idx_t const section,
                       section_rules& out) {
  out.trips_.clear();
  out.routes_.clear();

  auto const& tr = tt.transfer_rules_;
  for (auto const trip : tt.merged_trips_[section]) {
    auto const trip_rules = rules_of(tr.trip_rules_, trip);
    if (!trip_rules.empty()) {
      out.trips_.push_back(trip_rules);
    }

    auto const route_rules = rules_of(tr.route_rules_, route_id_of(tt, trip));
    if (!route_rules.empty()) {
      out.routes_.push_back(route_rules);
    }
  }
}

void add_stop_rules(timetable const& tt,
                    section_rules const& section,
                    event_type const ev,
                    location_idx_t const base,
                    std::vector<transfer_rule_side_idx>& sig) {
  auto const& tr = tt.transfer_rules_;
  auto const root = tt.locations_.get_root_idx(base);
  auto const add = [&](transfer_rule_side_idx const s) {
    auto const& r = tr.rules_[s.rule()];
    if (s.ev_type() == ev &&
        tt.locations_.is_self_or_parent(r.stop(ev), base)) {
      sig.push_back(s);
    }
  };
  for (auto const rules : section.trips_) {
    for (auto const s : rules_at(rules, root)) {
      add(s);
    }
  }
  for (auto const rules : section.routes_) {
    for (auto const s : rules_at(rules, root)) {
      add(s);
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
  auto rules = section_rules{};
  sig.clear();
  if (i != 0U && stp.out_allowed()) {
    get_section_rules(tt, sections[sections.size() == 1U ? 0U : i - 1U], rules);
    add_stop_rules(tt, rules, event_type::kArr, base, sig);
  }
  if (i + 1U != seq.size() && stp.in_allowed()) {
    get_section_rules(tt, sections[sections.size() == 1U ? 0U : i], rules);
    add_stop_rules(tt, rules, event_type::kDep, base, sig);
  }
  utl::erase_duplicates(sig);
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
