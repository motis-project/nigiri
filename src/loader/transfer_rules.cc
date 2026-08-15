#include "nigiri/loader/transfer_rules.h"

#include <algorithm>
#include <vector>

#include "nigiri/timetable.h"

namespace nigiri::loader {

bool covers(timetable const& tt,
            location_idx_t const rule_stop,
            location_idx_t const l) {
  return rule_stop == l || tt.locations_.parents_[l] == rule_stop;
}

void write_transfer_rules(
    timetable& tt,
    hash_map<transfer_pair, candidate> const& most_specific,
    vector_map<rule_idx_t, duration_t> const& durations,
    location_idx_t const first_virt) {
  auto const base_of = [&](location_idx_t const l) {
    return tt.locations_.types_[l] == location_type::kVirt
               ? tt.locations_.parents_[l]
               : l;
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
  auto slow_from = hash_set<location_idx_t>{};
  auto slow_to = hash_set<location_idx_t>{};
  for (auto const& [xy, c] : most_specific) {
    if (xy.from_ != xy.to_ && base_of(xy.from_) == base_of(xy.to_) &&
        durations[c.rule_idx_] >
            tt.locations_.transfer_time_[base_of(xy.from_)]) {
      slow_from.insert(xy.from_);
      slow_to.insert(xy.to_);
    }
  }

  auto const is_derivable = [&](location_idx_t const x, location_idx_t const y,
                                location_idx_t const base) {
    return derivable(
        {.slow_ = x != base && tt.locations_.transfer_time_[x] >
                                   tt.locations_.transfer_time_[base],
         .slow_from_ = slow_from.contains(x)},
        {.slow_to_ = slow_to.contains(y)});
  };

  // A rule can also state one value for a whole cross product of locations -
  // typically a stop pair whose sides both carry virtual locations. That costs
  // |X| * |Y| transfers, where one hub covers it in |X| + |Y| edges. It is
  // only safe when every pair of the cross product really belongs to this
  // rule: otherwise the hub would deliver its value for a pair that a more
  // specific rule gave a slower one.
  struct cross_rule {
    std::size_t n_cells_{0U};
    hash_set<location_idx_t> x_, y_;
  };
  auto cross = hash_map<rule_idx_t, cross_rule>{};
  for (auto const& [xy, c] : most_specific) {
    if (base_of(xy.from_) == base_of(xy.to_)) {
      continue;  // covered by the base's own hubs
    }
    auto& g = cross[c.rule_idx_];
    ++g.n_cells_;
    g.x_.insert(xy.from_);
    g.y_.insert(xy.to_);
  }

  auto hub_rules = hash_set<rule_idx_t>{};
  auto hub_in = std::vector<location_idx_t>{};
  auto hub_out = std::vector<location_idx_t>{};
  for (auto const& [rule_idx, g] : cross) {
    auto const cells = g.x_.size() * g.y_.size();
    if (g.n_cells_ != cells || cells <= g.x_.size() + g.y_.size()) {
      continue;
    }
    hub_rules.insert(rule_idx);

    hub_in.assign(begin(g.x_), end(g.x_));
    hub_out.assign(begin(g.y_), end(g.y_));
    utl::sort(hub_in);
    utl::sort(hub_out);
    tt.locations_.hub_in_.emplace_back(hub_in);
    tt.locations_.hub_out_.emplace_back(hub_out);
    tt.locations_.hub_time_.push_back(durations[rule_idx]);
  }

  // Write the most specific transfer per pair.
  for (auto const& [xy, c] : most_specific) {
    auto const d = durations[c.rule_idx_];
    auto const base = base_of(xy.from_);
    if (base == base_of(xy.to_)) {
      if (d == tt.locations_.transfer_time_[base] &&
          is_derivable(xy.from_, xy.to_, base)) {
        continue;
      }
    } else if (hub_rules.contains(c.rule_idx_)) {
      continue;  // derived by this rule's hub
    }
    tt.locations_.transfer_rule_fps_[xy.from_].emplace_back(xy.to_, d);
  }

  // Apply the default between all pairs without a rule.
  for (auto virt = first_virt; virt != tt.n_locations(); ++virt) {
    auto const base = tt.locations_.parents_[virt];
    auto const d = duration_t{tt.locations_.transfer_time_[base]};
    auto const add_default_rule = [&](location_idx_t const x,
                                      location_idx_t const y) {
      if (!most_specific.contains({x, y}) && !is_derivable(x, y, base)) {
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
