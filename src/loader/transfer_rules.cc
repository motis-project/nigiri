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
  // |X| * |Y| transfers, where one hub covers it in |X| + |Y| edges.
  //
  // A hub hands its value to every pair of its two lists, so it may only be
  // built where that value cannot beat what the data says. A pair a more
  // specific rule made *faster* is no obstacle: that cell is written and the
  // routing takes the minimum of the two. A pair it made *slower* is, and the
  // answer is the same as for a base's own hubs - the sources of slower cells
  // move to a second hub that only reaches the targets no slower cell leads
  // to. What neither hub can cover is written.
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

  // the slow cells of a hub'd rule, by which side they touch
  struct hub_split {
    hash_set<location_idx_t> slow_from_, slow_to_;
  };
  auto hub_rules = hash_map<rule_idx_t, hub_split>{};
  auto hub_in = std::vector<location_idx_t>{};
  auto hub_out = std::vector<location_idx_t>{};
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
    auto const cells = g.x_.size() * g.y_.size();
    if (cells <= g.x_.size() + g.y_.size()) {
      continue;  // a hub would not even be smaller than the cells
    }

    auto const d = durations[rule_idx];
    auto split = hub_split{};
    auto complete = true;
    for (auto const x : g.x_) {
      for (auto const y : g.y_) {
        auto const it = most_specific.find(transfer_pair{x, y});
        if (it == end(most_specific)) {
          complete = false;  // no rule speaks about this pair: a hub would
          break;  // invent a transfer the data does not state
        }
        if (it->second.rule_idx_ != rule_idx &&
            durations[it->second.rule_idx_] > d) {
          split.slow_from_.insert(x);
          split.slow_to_.insert(y);
        }
      }
      if (!complete) {
        break;
      }
    }
    if (!complete) {
      continue;
    }

    hub_in.clear();
    hub_out.assign(begin(g.y_), end(g.y_));
    for (auto const x : g.x_) {
      if (!split.slow_from_.contains(x)) {
        hub_in.push_back(x);
      }
    }
    utl::sort(hub_in);
    utl::sort(hub_out);
    emit_hub(hub_in, hub_out, d);  // unrestricted: reaches all of Y

    if (!split.slow_from_.empty()) {
      hub_in.assign(begin(split.slow_from_), end(split.slow_from_));
      hub_out.clear();
      for (auto const y : g.y_) {
        if (!split.slow_to_.contains(y)) {
          hub_out.push_back(y);
        }
      }
      utl::sort(hub_in);
      utl::sort(hub_out);
      emit_hub(hub_in, hub_out, d);  // restricted: only the clean targets
    }

    hub_rules.emplace(rule_idx, std::move(split));
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
    } else if (auto const it = hub_rules.find(c.rule_idx_);
               it != end(hub_rules) &&
               derivable(
                   {.slow_from_ = it->second.slow_from_.contains(xy.from_)},
                   {.slow_to_ = it->second.slow_to_.contains(xy.to_)})) {
      continue;  // derived by one of this rule's hubs
    }
    tt.locations_.transfer_rule_fps_[xy.from_].emplace_back(xy.to_, d);
  }

  // Apply the default between all pairs without a rule. Driven by the stop's
  // children, not by the range of virtual locations: one that was merged into
  // another is no longer a child, and nothing should be stated about it.
  for (auto virt = first_virt; virt != tt.n_locations(); ++virt) {
    auto const base = tt.locations_.parents_[virt];
    if (utl::none_of(tt.locations_.children_[base],
                     [&](location_idx_t const c) { return c == virt; })) {
      continue;  // merged into another virtual location of its stop
    }
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
