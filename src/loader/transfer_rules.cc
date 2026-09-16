#include "nigiri/loader/transfer_rules.h"

#include <vector>

#include "utl/verify.h"

#include "nigiri/timetable.h"

namespace nigiri::loader {

void write_transfer_rules(
    timetable& tt,
    hash_map<transfer_pair, candidate> const& most_specific,
    vector_map<rule_idx_t, duration_t> const& durations,
    location_idx_t const first_virt,
    bool const rule_hubs) {
  // The value a base's own hubs hand out. A banned base gets no hubs (see
  // build_hubs), so nothing but a ban equals its value and only bans are left
  // to it - which is right, a ban is derived by having nothing.
  auto const base_time = [&](location_idx_t const base) {
    return from_transfer_time(tt.locations_.transfer_time_[base]);
  };

  // A transfer x -> y inside one base can be left out if the base's own hubs
  // derive it: it takes exactly the base's transfer time (the only value the
  // derivation produces), x's own change time is not longer than the base's
  // (the derivation also yields x -> x at the base's time and would undercut
  // it), and the split lets x reach y (hub_split). A pair of one location
  // with itself is never a cell: the derivation it could undercut starts at
  // that location, and its own change time already bars it.
  //
  // A rule can also state one value for a whole cross product of locations -
  // typically a stop pair whose sides both carry virtual locations. That costs
  // |X| * |Y| transfers, where one hub covers it in |X| + |Y| edges. The cells
  // are grouped by their rule here, the hubs follow below.
  auto base_split = hub_split{};
  struct cross_rule {
    hash_set<location_idx_t> x_, y_;
  };
  auto cross = hash_map<rule_idx_t, cross_rule>{};
  for (auto const& [xy, c] : most_specific) {
    auto const base = base_of(tt, xy.from_);
    if (base != base_of(tt, xy.to_)) {
      if (rule_hubs) {
        auto& g = cross[c.rule_idx_];
        g.x_.insert(xy.from_);
        g.y_.insert(xy.to_);
      }
    } else if (durations[c.rule_idx_] > base_time(base)) {
      base_split.mark(xy.from_, xy.to_);
    }
  }
  auto const derivable_at = [&](location_idx_t const x, location_idx_t const y,
                                location_idx_t const base) {
    auto const slow = x != base && tt.locations_.transfer_time_[x] >
                                       tt.locations_.transfer_time_[base];
    return !slow && base_split.derives(x, y);
  };

  auto hub_rules = hash_map<rule_idx_t, hub_split>{};
  auto const emit_hub = [&](std::vector<location_idx_t> const& in,
                            std::vector<location_idx_t> const& out,
                            duration_t const d) {
    if (in.empty() || out.empty()) {
      return;
    }
    tt.locations_.hub_in_[kDefaultProfile].emplace_back(in);
    tt.locations_.hub_out_[kDefaultProfile].emplace_back(out);
    tt.locations_.hub_time_[kDefaultProfile].push_back(d);
  };
  for (auto const& [rule_idx, g] : cross) {
    auto const d = durations[rule_idx];
    if (d == footpath::kMaxDuration) {
      continue;  // a ban is no transfer, a hub would hand it out as one
    }
    if (g.x_.size() * g.y_.size() <= g.x_.size() + g.y_.size()) {
      continue;  // a hub would not even be smaller than the cells
    }

    // Every pair of X x Y has a winner, the rule applies to the whole cross
    // product of its sides. The one pair no rule states is a location with
    // itself, which a same-station rule produces: there the hub would stand
    // for the location's own change time and must not undercut it.
    auto split = hub_split{};
    for (auto const x : g.x_) {
      for (auto const y : g.y_) {
        auto slower = false;
        if (x == y) {
          slower = from_transfer_time(tt.locations_.transfer_time_[x]) > d;
        } else {
          auto const it = most_specific.find(transfer_pair{x, y});
          utl::verify(it != end(most_specific),
                      "rule {}: no winner for a pair of its cross product",
                      to_idx(rule_idx));
          slower = it->second.rule_idx_ != rule_idx &&
                   durations[it->second.rule_idx_] > d;
        }
        if (slower) {
          split.mark(x, y);
        }
      }
    }

    auto xs = std::vector<location_idx_t>{begin(g.x_), end(g.x_)};
    auto ys = std::vector<location_idx_t>{begin(g.y_), end(g.y_)};
    utl::sort(xs);
    utl::sort(ys);
    emit_split_hubs(xs, ys, d, split, emit_hub);
    hub_rules.emplace(rule_idx, std::move(split));
  }

  // Write the most specific transfer per pair.
  for (auto const& [xy, c] : most_specific) {
    auto const d = durations[c.rule_idx_];
    auto const base = base_of(tt, xy.from_);
    if (base == base_of(tt, xy.to_)) {
      if (d == base_time(base) && derivable_at(xy.from_, xy.to_, base)) {
        continue;  // derived by the base's own hubs
      }
    } else if (auto const it = hub_rules.find(c.rule_idx_);
               it != end(hub_rules) && it->second.derives(xy.from_, xy.to_)) {
      continue;  // derived by one of this rule's hubs
    }
    tt.locations_.transfer_rule_fps_[xy.from_].emplace_back(xy.to_, d);
  }

  // Apply the default between all pairs without a rule.
  for (auto virt = first_virt; virt != tt.n_locations(); ++virt) {
    auto const base = tt.locations_.parents_[virt];
    auto const d = base_time(base);
    auto const add_default_rule = [&](location_idx_t const x,
                                      location_idx_t const y) {
      if (!most_specific.contains({x, y}) && !derivable_at(x, y, base)) {
        tt.locations_.transfer_rule_fps_[x].emplace_back(y, d);
      }
    };

    add_default_rule(virt, base);
    add_default_rule(base, virt);
    for (auto const sibling : tt.locations_.children_[base]) {
      if (sibling != virt &&
          tt.locations_.types_[sibling] == location_type::kVirt) {
        add_default_rule(virt, sibling);  // sibling - virt added from sibling
      }
    }
  }
}

}  // namespace nigiri::loader
