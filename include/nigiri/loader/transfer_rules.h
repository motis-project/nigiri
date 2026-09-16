#pragma once

#include <compare>
#include <cstdint>
#include <tuple>
#include <vector>

#include "utl/erase_if.h"
#include "utl/helpers/algorithm.h"

#include "cista/reflection/comparable.h"

#include "nigiri/footpath.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::loader {

using rule_idx_t = cista::strong<std::uint32_t, struct rule_idx_>;

// One side (from/to) of one rule, packed as rule << 1 | side. A rule has no
// other sides, so in a sorted, deduplicated signature its from side sits
// directly before its to side - which is what lets the virtual location split
// find the rules that match with both of their sides by looking at adjacent
// entries.
using sided_rule_idx_t = cista::strong<std::uint32_t, struct sided_rule_idx_>;

inline sided_rule_idx_t side_ref(rule_idx_t const rule_idx,
                                 bool const is_from) {
  return sided_rule_idx_t{(to_idx(rule_idx) << 1U) | (is_from ? 0U : 1U)};
}

inline rule_idx_t rule_of(sided_rule_idx_t const s) {
  return rule_idx_t{to_idx(s) >> 1U};
}

// all rule sides that apply to a trip stop
using signature_t = std::vector<sided_rule_idx_t>;

// An ordered pair of locations: a transfer leads from_ -> to_.
struct transfer_pair {
  CISTA_COMPARABLE()
  location_idx_t from_{location_idx_t::invalid()};
  location_idx_t to_{location_idx_t::invalid()};
};

// The rule that won a location pair. rank_ is how specific that rule is, on
// whatever scale the loader ranks its rules by - higher wins, and the shared
// code never interprets it, it only ever reads rule_idx_. Which rule beats
// which is a question of the input format (GTFS states the ladder in its
// reference, other formats qualify their rules by other things), so only the
// comparison lives here, not the meaning.
struct candidate {
  // the index only makes the order total, so that the winner does not depend
  // on hash order; spelled out because a strong index has no operator<=>
  auto operator<=>(candidate const& o) const {
    return std::tuple{rank_, to_idx(rule_idx_)} <=>
           std::tuple{o.rank_, to_idx(o.rule_idx_)};
  }
  bool operator==(candidate const&) const = default;

  std::uint16_t rank_{0U};
  rule_idx_t rule_idx_{0U};
};

// A rule stop covers an event location if they are the same stop or if the
// event location is a child of the rule stop (station level cascade).
inline bool covers(timetable const& tt,
                   location_idx_t const rule_stop,
                   location_idx_t const l) {
  return rule_stop == l || tt.locations_.parents_[l] == rule_stop;
}

// The stop a location's transfers are derived at: for a virtual location the
// stop it was split off, for everything else the location itself.
inline location_idx_t base_of(timetable const& tt, location_idx_t const l) {
  return tt.locations_.types_[l] == location_type::kVirt
             ? tt.locations_.parents_[l]
             : l;
}

// Detects the most common rule between two stops -> removes them and makes
// their min_transfer_time the new default. Only rows that state a time vote:
// a guaranteed connection (type 1 without a time) or a ban says which pairs
// are special, not how long a change at the stop takes, so they stay
// exceptions and never become the default for the pairs nobody named. Works
// on stop pairs and durations only; Rule just has to offer from_stop_,
// to_stop_, is_qualified(), states_time(), duration() and a (from, to,
// duration) constructor for the synthesized unqualified rules.
template <typename Rule>
void fold_pair_defaults(timetable& tt, vector_map<rule_idx_t, Rule>& rules) {
  struct counted_duration {
    duration_t d_;
    unsigned n_{0U};
  };

  auto qualified = hash_map<transfer_pair, std::vector<counted_duration>>{};
  auto pair_default = hash_map<transfer_pair, duration_t>{};
  for (auto const& r : rules) {
    auto const p = transfer_pair{r.from_stop_, r.to_stop_};
    if (!r.is_qualified()) {
      pair_default[p] = r.duration();  // duplicate rows: last one wins
    } else if (r.states_time()) {
      auto& durations = qualified[p];
      auto const it = utl::find_if(durations, [&](counted_duration const& c) {
        return c.d_ == r.duration();
      });
      if (it == end(durations)) {
        durations.push_back({r.duration(), 1U});
      } else {
        ++it->n_;
      }
    }
  }

  auto synthetic = std::vector<Rule>{};
  for (auto const& [p, durations] : qualified) {
    if (pair_default.contains(p)) {
      continue;  // explicit unqualified row
    }

    auto const majority = std::max_element(
        begin(durations), end(durations),
        [](auto const& a, auto const& b) { return a.n_ < b.n_; });
    pair_default.emplace(p, majority->d_);

    if (p.from_ == p.to_) {
      tt.locations_.transfer_time_[p.from_] = to_transfer_time(majority->d_);
    } else {
      // Add an unqualified rule
      // -> applies to all trips
      // -> won't be overwritten by street routing
      synthetic.emplace_back(p.from_, p.to_, majority->d_);
    }
  }

  // Remove all rules that re-state the default derived from the majority.
  utl::erase_if(rules, [&](Rule const& r) {
    if (!r.is_qualified()) {
      return false;
    }
    auto const it = pair_default.find(transfer_pair{r.from_stop_, r.to_stop_});
    return it != end(pair_default) && r.duration() == it->second;
  });

  // Add the new default rules derived from the majority.
  for (auto const& r : synthetic) {
    rules.emplace_back(r);
  }
}

// A hub stands for every pair of two lists at one weight. It may only be
// built where that weight cannot beat what the data says: a pair stated
// slower elsewhere - a rule cell slower than the weight, or a location's own
// change time slower than it - must stay out of its reach, while a pair
// stated faster is no obstacle, its cell wins the minimum the routing takes.
// The split is always the same: the sources of the slower pairs move to a
// second, restricted hub that reaches only the targets no slower pair leads
// to, so every other pair is still derived (emit_split_hubs). What the two
// hubs do not reach the caller states one by one - usually just the slower
// pairs, which are stated anyway.
//
// Who is slow is found differently by each caller - from the rule table, from
// the written cells, from the walks - but build_hubs re-derives the very
// same classification from the finished footpaths, so the sets have to agree.
// If they ever disagree, transfers go missing without a trace.
struct hub_split {
  void mark(location_idx_t const from, location_idx_t const to) {
    slow_from_.insert(from);
    slow_to_.insert(to);
  }
  bool derives(location_idx_t const from, location_idx_t const to) const {
    return !slow_from_.contains(from) || !slow_to_.contains(to);
  }
  hash_set<location_idx_t> slow_from_, slow_to_;
};

// The hubs standing for x times y at weight d under a split: the unrestricted
// one, whose sources start no slower pair and which reaches all of y, and -
// if any source does - the restricted one for the others, reaching only the
// targets no slower pair leads to. Sorted lists stay sorted.
template <typename Emit>
void emit_split_hubs(std::vector<location_idx_t> const& x,
                     std::vector<location_idx_t> const& y,
                     duration_t const d,
                     hub_split const& split,
                     Emit&& emit) {
  auto in = std::vector<location_idx_t>{};
  for (auto const m : x) {
    if (!split.slow_from_.contains(m)) {
      in.push_back(m);
    }
  }
  emit(in, y, d);
  if (in.size() == x.size()) {
    return;
  }
  in.clear();
  auto out = std::vector<location_idx_t>{};
  for (auto const m : x) {
    if (split.slow_from_.contains(m)) {
      in.push_back(m);
    }
  }
  for (auto const t : y) {
    if (!split.slow_to_.contains(t)) {
      out.push_back(t);
    }
  }
  emit(in, out, d);
}

// Turns the resolved rule cells into timetable entries: writes the transfers
// that have to be stored and leaves out every pair a hub derives, emitting the
// hubs for rules that state one value for a whole cross product. Reaches the
// rules only through their duration, so every loader can feed it whatever rule
// representation it likes - it just has to resolve its rules to location pairs
// first (which needs its own notion of route and trip) and split off virtual
// locations for the qualified ones, starting at first_virt. With rule_hubs
// off, every cell is written and no hub is emitted.
void write_transfer_rules(
    timetable&,
    hash_map<transfer_pair, candidate> const& most_specific,
    vector_map<rule_idx_t, duration_t> const& durations,
    location_idx_t first_virt,
    bool rule_hubs);

}  // namespace nigiri::loader
