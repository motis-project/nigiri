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

  // A transfer x -> y can be left out only if ALL of these hold:
  //   - x and y have the same base, because the derivation works per base
  //   - the transfer itself takes exactly the base's transfer time, because
  //     that is the only value the derivation produces
  //   - x's own transfer time, the one for changing at x, is not longer than
  //     the base's, because the derivation also yields x -> x at the base's
  //     time and would undercut it
  //   - no slower transfer starts at x, OR none leads to y (either one is
  //     enough), because the derivation reaches either every member from x,
  //     or only the members that no slower transfer leads to
  //
  // A pair of one location with itself is left out: the only derivation it
  // could undercut is the one starting at that location, and the check above
  // already bars it - its slow value is its own transfer time.
  //
  // A rule can also state one value for a whole cross product of locations -
  // typically a stop pair whose sides both carry virtual locations. That costs
  // |X| * |Y| transfers, where one hub covers it in |X| + |Y| edges. The cells
  // are grouped by their rule here, the hubs follow below.
  auto slow_from = hash_set<location_idx_t>{};
  auto slow_to = hash_set<location_idx_t>{};
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
      slow_from.insert(xy.from_);
      slow_to.insert(xy.to_);
    }
  }
  auto const derivable_at = [&](location_idx_t const x, location_idx_t const y,
                                location_idx_t const base) {
    return derivable(
        {.slow_ = x != base && tt.locations_.transfer_time_[x] >
                                   tt.locations_.transfer_time_[base],
         .slow_from_ = slow_from.contains(x)},
        {.slow_to_ = slow_to.contains(y)});
  };

  // A hub hands its value to every pair of its two lists, so it may only be
  // built where that value cannot beat what the data says. A pair a more
  // specific rule made *faster* is no obstacle: that cell is written and the
  // routing takes the minimum of the two. A pair it made *slower* is, and the
  // answer is the same as for a base's own hubs - the sources of slower cells
  // move to a second hub that only reaches the targets no slower cell leads
  // to. What neither hub can cover is written.
  struct hub_split {
    hash_set<location_idx_t> slow_from_, slow_to_;
  };
  auto hub_rules = hash_map<rule_idx_t, hub_split>{};
  auto hub_in = std::vector<location_idx_t>{};
  auto hub_out = std::vector<location_idx_t>{};
  auto const emit_hub = [&](std::vector<location_idx_t>& in,
                            std::vector<location_idx_t>& out,
                            duration_t const d) {
    if (in.empty() || out.empty()) {
      return;
    }
    utl::sort(in);
    utl::sort(out);
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
          split.slow_from_.insert(x);
          split.slow_to_.insert(y);
        }
      }
    }

    hub_in.clear();
    for (auto const x : g.x_) {
      if (!split.slow_from_.contains(x)) {
        hub_in.push_back(x);
      }
    }
    hub_out.assign(begin(g.y_), end(g.y_));
    emit_hub(hub_in, hub_out, d);  // unrestricted: reaches all of Y

    if (!split.slow_from_.empty()) {
      hub_in.assign(begin(split.slow_from_), end(split.slow_from_));
      hub_out.clear();
      for (auto const y : g.y_) {
        if (!split.slow_to_.contains(y)) {
          hub_out.push_back(y);
        }
      }
      emit_hub(hub_in, hub_out, d);  // restricted: only the clean targets
    }

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
               it != end(hub_rules) &&
               derivable(
                   {.slow_from_ = it->second.slow_from_.contains(xy.from_)},
                   {.slow_to_ = it->second.slow_to_.contains(xy.to_)})) {
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
