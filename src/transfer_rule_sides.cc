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
  return utl::any_of(tt.transport_to_trip_section_[t], [&](auto const s) {
    return utl::any_of(tt.merged_trips_[s], [&](trip_idx_t const trip) {
      return !rules_of(tr.trip_rules_, trip).empty() ||
             !rules_of(tr.route_rules_, tt.trip_route_id_[trip]).empty();
    });
  });
}

void get_section_rules(timetable const& tt,
                       merged_trips_idx_t const section,
                       section_rules& out) {
  auto const& tr = tt.transfer_rules_;
  out.trips_.clear();
  out.routes_.clear();
  out.route_srcs_.clear();
  for (auto const trip : tt.merged_trips_[section]) {
    if (auto const rules = rules_of(tr.trip_rules_, trip); !rules.empty()) {
      out.trips_.push_back(rules);
    }
    if (auto const rules = rules_of(tr.route_rules_, tt.trip_route_id_[trip]);
        !rules.empty()) {
      out.routes_.push_back(rules);
      out.route_srcs_.push_back(tt.trip_id_src_[tt.trip_ids_[trip].front()]);
    }
  }
}

void add_stop_sides(timetable const& tt,
                    section_rules const& section,
                    event_type const ev,
                    location_idx_t const base,
                    std::vector<transfer_rule_side_idx>& sig) {
  auto const& tr = tt.transfer_rules_;
  auto const station = tt.locations_.get_root_idx(base);
  auto const add = [&](transfer_rule_side_idx const s, source_idx_t const src) {
    auto const& r = tr.rules_[s.rule()];
    if (s.ev_type() == ev &&
        (src == source_idx_t::invalid() || r.src_ == src) &&
        tt.locations_.is_self_or_parent(r.stop(ev), base)) {
      sig.push_back(s);
    }
  };
  for (auto const rules : section.trips_) {
    for (auto const s : rules_at(rules, station)) {
      add(s, source_idx_t::invalid());
    }
  }
  for (auto k = 0U; k != section.routes_.size(); ++k) {
    for (auto const s : rules_at(section.routes_[k], station)) {
      add(s, section.route_srcs_[k]);
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
    add_stop_sides(tt, rules, event_type::kArr, base, sig);
  }
  if (i + 1U != seq.size() && stp.in_allowed()) {
    get_section_rules(tt, sections[sections.size() == 1U ? 0U : i], rules);
    add_stop_sides(tt, rules, event_type::kDep, base, sig);
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
