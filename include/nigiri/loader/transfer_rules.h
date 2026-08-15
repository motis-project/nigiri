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
bool covers(timetable const&, location_idx_t rule_stop, location_idx_t l);

// Detects the most common rule between two stops -> removes them and makes
// their min_transfer_time the new default. Works on stop pairs and durations
// only; Rule just has to offer from_stop_, to_stop_, is_qualified(),
// duration() and a (from, to, duration) constructor for the synthesized
// unqualified rules.
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
    } else {
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
    if (majority->d_ == footpath::kMaxDuration) {
      continue;
    }
    pair_default.emplace(p, majority->d_);

    if (p.from_ == p.to_) {
      tt.locations_.transfer_time_[p.from_] = majority->d_;
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

// What a member of a base is, as far as the derivation is concerned. The
// loaders find this out differently - one from a rule table, one from a
// resolved matrix - but the rule below has to be the same for all of them,
// because build_hubs re-derives the very same classification from the
// finished footpaths. If the two ever disagree, transfers go missing without
// a trace.
struct member_flags {
  bool slow_{false};  // its own transfer time exceeds the base's
  bool slow_from_{false};  // some slower transfer starts here
  bool slow_to_{false};  // some slower transfer leads here
};

// A transfer at exactly the base's transfer time need not be stored if the
// hubs of that base derive it: a member that is slow itself feeds no hub at
// all, and the rest reach either every member (nothing slow starts at them)
// or the members no slow transfer leads to.
inline bool derivable(member_flags const& from, member_flags const& to) {
  return !from.slow_ && (!from.slow_from_ || !to.slow_to_);
}

// Turns the resolved rule cells into timetable entries: writes the transfers
// that have to be stored and leaves out every pair a hub derives, emitting the
// hubs for rules that state one value for a whole cross product. Reaches the
// rules only through their duration, so every loader can feed it whatever rule
// representation it likes - it just has to resolve its rules to location pairs
// first (which needs its own notion of route and trip) and split off virtual
// locations for the qualified ones, starting at first_virt.
void write_transfer_rules(
    timetable&,
    hash_map<transfer_pair, candidate> const& most_specific,
    vector_map<rule_idx_t, duration_t> const& durations,
    location_idx_t first_virt);

}  // namespace nigiri::loader
