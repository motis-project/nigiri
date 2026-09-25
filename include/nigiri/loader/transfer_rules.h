#pragma once

#include <compare>
#include <cstdint>
#include <optional>
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

// Whether a side names a trip or a route (its rank) only matters where sides
// of other durations compete for the same partners: same end of the transfer,
// the same other stop or its station. Only there can the ranking pick another
// winner for two trip stops whose rules state the same values - e.g. "trip A
// -> anything: 5 min" beats "route R -> route Q: 10 min", which beats "route R
// -> anything: 5 min". Elsewhere the rank is left out of the key, so trips
// whose rules only agree keep sharing a virtual location. Shared by the
// loader's key and the real-time one (rt_transfer_rules.cc), which resolve()
// compares.
template <typename Side>
void keep_rank_where_it_decides(timetable const& tt, std::vector<Side>& sides) {
  auto const& parents = tt.locations_.parents_;
  auto const same_partners = [&](Side const& a, Side const& b) {
    return a.is_from_ == b.is_from_ &&
           (a.other_stop_ == b.other_stop_ ||
            parents[a.other_stop_] == b.other_stop_ ||
            parents[b.other_stop_] == a.other_stop_);
  };
  auto competes = std::vector<bool>(sides.size());
  for (auto i = 0U; i != sides.size(); ++i) {
    competes[i] = utl::any_of(sides, [&](Side const& b) {
      return b.duration_ != sides[i].duration_ && same_partners(sides[i], b);
    });
  }
  for (auto i = 0U; i != sides.size(); ++i) {
    sides[i].by_trip_ = sides[i].by_trip_ && competes[i];
  }
}

// Detects the most common rule between two stops -> removes them and makes
// their min_transfer_time the new default. Only rows that state a time vote:
// a guaranteed connection (type 1 without a time) or a ban says which pairs
// are special, not how long a change at the stop takes, so they stay
// exceptions and never become the default for the pairs nobody named. The
// fold only fills gaps: a pair an explicit unqualified row covers - stated
// for the pair itself or for its stations - has its default already. Works
// on stop pairs and durations only; Rule just has to offer from_stop_,
// to_stop_, is_qualified(), get_specificity(), states_time(), duration(),
// overlaps(other) (some trip pair both name) and a (from, to, duration)
// constructor for the synthesized unqualified rules.
template <typename Rule>
void fold_pair_defaults(timetable& tt, vector_map<rule_idx_t, Rule>& rules) {
  struct counted_duration {
    duration_t d_;
    unsigned n_{0U};
  };

  auto const& parents = tt.locations_.parents_;
  auto qualified = hash_map<transfer_pair, std::vector<counted_duration>>{};
  auto explicit_default = hash_map<transfer_pair, duration_t>{};
  for (auto const& r : rules) {
    auto const p = transfer_pair{r.from_stop_, r.to_stop_};
    if (!r.is_qualified()) {
      explicit_default[p] = r.duration();  // duplicate rows: last one wins
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

  // The explicit unqualified row for a pair: stated for the pair itself, else
  // for its stations (the most specific first).
  auto const covering_default =
      [&](transfer_pair const p) -> std::optional<duration_t> {
    for (auto const from : {p.from_, parents[p.from_]}) {
      for (auto const to : {p.to_, parents[p.to_]}) {
        if (from == location_idx_t::invalid() ||
            to == location_idx_t::invalid()) {
          continue;
        }
        if (auto const it = explicit_default.find(transfer_pair{from, to});
            it != end(explicit_default)) {
          return it->second;
        }
      }
    }
    return std::nullopt;
  };

  auto pair_default = hash_map<transfer_pair, duration_t>{};
  auto synthetic = std::vector<Rule>{};
  for (auto const& [p, durations] : qualified) {
    if (covering_default(p).has_value()) {
      continue;  // no gap: an explicit unqualified row states the default
    }

    auto const majority = std::max_element(
        begin(durations), end(durations),
        [](auto const& a, auto const& b) { return a.n_ < b.n_; });
    pair_default.emplace(p, majority->d_);

    if (p.from_ == p.to_) {
      tt.locations_.transfer_time_[p.from_] = to_transfer_time(majority->d_);
    }
    if (p.from_ != p.to_ || !tt.locations_.children_[p.from_].empty()) {
      // Add an unqualified rule
      // -> applies to all trips (for a station: to the pairs of its platforms)
      // -> won't be overwritten by street routing
      synthetic.emplace_back(p.from_, p.to_, majority->d_);
    }
  }

  // A rule that re-states the default is only redundant if the default is what
  // its pairs would get without it. It is not if a rule that is at most as
  // specific and applies to some of its pairs says something else - on the
  // same stops, their stations or their child stops (e.g. a trip pair back at
  // the default under a route pair that is faster, or banned, or a station
  // trip rule above a platform route rule) - or if its child stops have a
  // default of their own that differs. A rule for other trips (another trip
  // pair, a route the trips don't belong to) never applies to its pairs, so
  // it keeps nothing: one guarantee does not hold every timed row at its stop.
  auto by_pair = hash_map<transfer_pair, std::vector<rule_idx_t>>{};
  for (auto i = rule_idx_t{0U}; i != rules.size(); ++i) {
    if (rules[i].is_qualified()) {
      by_pair[{rules[i].from_stop_, rules[i].to_stop_}].push_back(i);
    }
  }
  auto const below = [&](location_idx_t const l) {
    auto v = std::vector<location_idx_t>{l};
    for (auto const c : tt.locations_.children_[l]) {
      v.push_back(c);
    }
    return v;
  };
  auto const around = [&](location_idx_t const l) {
    auto v = below(l);
    if (parents[l] != location_idx_t::invalid()) {
      v.push_back(parents[l]);
    }
    return v;
  };
  auto const is_exception = [&](Rule const& r) {
    for (auto const from : around(r.from_stop_)) {
      for (auto const to : around(r.to_stop_)) {
        auto const it = by_pair.find(transfer_pair{from, to});
        if (it != end(by_pair) &&
            utl::any_of(it->second, [&](rule_idx_t const i) {
              return rules[i].duration() != r.duration() &&
                     rules[i].get_specificity() <= r.get_specificity() &&
                     rules[i].overlaps(r);
            })) {
          return true;
        }
      }
    }
    for (auto const from : below(r.from_stop_)) {
      for (auto const to : below(r.to_stop_)) {
        auto const p = transfer_pair{from, to};
        if (p == transfer_pair{r.from_stop_, r.to_stop_}) {
          continue;
        }
        for (auto const* defaults : {&explicit_default, &pair_default}) {
          if (auto const it = defaults->find(p);
              it != end(*defaults) && it->second != r.duration()) {
            return true;
          }
        }
      }
    }
    return false;
  };

  // Remove all rules that re-state the default: the one derived from the
  // majority, or the explicit one that covers their pair.
  auto redundant = std::vector<bool>(rules.size());
  for (auto i = rule_idx_t{0U}; i != rules.size(); ++i) {
    auto const& r = rules[i];
    if (!r.is_qualified()) {
      continue;
    }
    auto const p = transfer_pair{r.from_stop_, r.to_stop_};
    auto const it = pair_default.find(p);
    auto const d = it != end(pair_default) ? std::optional{it->second}
                                           : covering_default(p);
    redundant[to_idx(i)] =
        d.has_value() && r.duration() == *d && !is_exception(r);
  }
  auto i = 0U;
  utl::erase_if(rules, [&](Rule const&) { return redundant[i++]; });

  // Add the new default rules derived from the majority.
  for (auto const& r : synthetic) {
    rules.emplace_back(r);
  }

  // A same-stop default of a station - stated or folded - is the change time
  // of its child stops too, also for a change that stays at one of them (a
  // station row applies to all its child stops), unless a child has a
  // same-stop default of its own. Before the virtual locations exist, which
  // take their own change time from their stop.
  auto const has_own = [&](location_idx_t const l) {
    auto const p = transfer_pair{l, l};
    return explicit_default.contains(p) || pair_default.contains(p);
  };
  for (auto const* defaults : {&explicit_default, &pair_default}) {
    for (auto const& [p, d] : *defaults) {
      if (p.from_ != p.to_) {
        continue;
      }
      for (auto const c : tt.locations_.children_[p.from_]) {
        if (tt.locations_.types_[c] != location_type::kVirt && !has_own(c)) {
          tt.locations_.transfer_time_[c] = to_transfer_time(d);
        }
      }
    }
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
// off, no cross product gets a hub: every cell between two stops is written.
// Cells within one stop that its own hubs derive are left out either way,
// because build_hubs always builds those.
void write_transfer_rules(
    timetable&,
    hash_map<transfer_pair, candidate> const& most_specific,
    vector_map<rule_idx_t, duration_t> const& durations,
    location_idx_t first_virt,
    bool rule_hubs);

}  // namespace nigiri::loader
