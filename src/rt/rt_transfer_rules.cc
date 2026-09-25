#include "nigiri/rt/rt_transfer_rules.h"

#include <algorithm>
#include <array>
#include <optional>
#include <vector>

#include "utl/erase_duplicates.h"
#include "utl/helpers/algorithm.h"
#include "utl/verify.h"

#include "nigiri/loader/transfer_rules.h"
#include "nigiri/footpath.h"
#include "nigiri/routing/for_each_hub_source.h"

namespace nigiri::rt {

namespace {

using loader::candidate;
using loader::covers;
using loader::rule_idx_t;
using side_t = transfer_rule_side_t;

// What one matched rule side states - the loader's virt_key, side by side.
struct side_value {
  bool is_from_;
  location_idx_t rule_stop_;
  location_idx_t other_stop_;
  source_idx_t src_;
  route_id_idx_t other_route_;
  trip_idx_t other_trip_;
  duration_t duration_;
  bool by_trip_;  // this side names a trip (see keep_rank_where_it_decides)

  // Only what erase_duplicates needs; the anonymous namespace would flag the
  // unused templates of CISTA_COMPARABLE.
  bool operator==(side_value const& o) const {
    return cista::to_tuple(*this) == cista::to_tuple(o);
  }
  bool operator<(side_value const& o) const {
    return cista::to_tuple(*this) < cista::to_tuple(o);
  }
};

std::vector<side_value> values_of(timetable const& tt,
                                  std::vector<side_t> const& sides) {
  auto v = std::vector<side_value>{};
  v.reserve(sides.size());
  for (auto const s : sides) {
    auto const& r = tt.transfer_rules_.rules_[transfer_rules::rule_of(s)];
    auto const is_from = transfer_rules::is_from(s);
    v.push_back({.is_from_ = is_from,
                 .rule_stop_ = is_from ? r.from_stop_ : r.to_stop_,
                 .other_stop_ = is_from ? r.to_stop_ : r.from_stop_,
                 .src_ = r.src_,
                 .other_route_ = is_from ? r.to_route_ : r.from_route_,
                 .other_trip_ = is_from ? r.to_trip_ : r.from_trip_,
                 .duration_ = r.duration_,
                 .by_trip_ = (is_from ? r.from_trip_ : r.to_trip_) !=
                             trip_idx_t::invalid()});
  }
  loader::keep_rank_where_it_decides(tt, v);
  utl::erase_duplicates(v);
  return v;
}

candidate make_candidate(stop_transfer_rule const& r,
                         std::uint32_t const rule,
                         location_idx_t const from_platform,
                         location_idx_t const to_platform) {
  return {.rank_ = transfer_rules::rank(
              r.specificity_,
              static_cast<unsigned>((r.from_stop_ == from_platform) +
                                    (r.to_stop_ == to_platform))),
          .rule_idx_ = rule_idx_t{rule}};
}

// The qualified rule sides that match a trip stop at `platform`, sorted.
std::vector<side_t> get_signature(timetable const& tt,
                                  transport_idx_t const t,
                                  stop_idx_t const stop_idx,
                                  location_idx_t const platform) {
  auto const& tr = tt.transfer_rules_;
  auto sig = std::vector<side_t>{};
  auto const add_if_covered = [&](side_t const s) {
    auto const& r = tr.rules_[transfer_rules::rule_of(s)];
    if (covers(tt, transfer_rules::is_from(s) ? r.from_stop_ : r.to_stop_,
               platform)) {
      sig.push_back(s);
    }
  };

  // the trips the transport arrives as (section stop_idx - 1) and departs as
  // (section stop_idx); one section = the same trips all the way
  auto const sections = tt.transport_to_trip_section_[t];
  auto const n = sections.size();
  auto const from = n == 1U || stop_idx == 0U ? 0U : stop_idx - 1U;
  auto const to = n == 1U ? 1U : std::min<std::size_t>(stop_idx + 1U, n);
  for (auto i = std::size_t{from}; i < to; ++i) {
    for (auto const trip : tt.merged_trips_[sections[i]]) {
      transfer_rules::for_each(tr.trip_sides_, to_idx(trip), add_if_covered);
      // route ids are per source
      auto const src = tt.trip_id_src_[tt.trip_ids_[trip].front()];
      transfer_rules::for_each(
          tr.route_sides_, to_idx(tt.trip_route_id_[trip]),
          [&](side_t const s) {
            if (tr.rules_[transfer_rules::rule_of(s)].src_ == src) {
              add_if_covered(s);
            }
          });
    }
  }
  utl::erase_duplicates(sig);
  return sig;
}

// Rules with both sides on this trip stop state its own change time (same as
// the loader): a qualified side is on it if it is in the sorted signature, an
// unqualified one if its stop covers the platform.
duration_t get_own(timetable const& tt,
                   std::vector<side_t> const& sig,
                   location_idx_t const platform) {
  auto const& rules = tt.transfer_rules_.rules_;
  auto const applies = [&](std::uint32_t const rule, bool const is_from) {
    auto const& r = rules[rule];
    return (is_from ? r.from_qualified() : r.to_qualified())
               ? std::binary_search(begin(sig), end(sig),
                                    transfer_rules::side(rule, is_from))
               : covers(tt, is_from ? r.from_stop_ : r.to_stop_, platform);
  };
  auto best = std::optional<candidate>{};
  for (auto const s : sig) {
    auto const rule = transfer_rules::rule_of(s);
    if (applies(rule, true) && applies(rule, false)) {
      best = std::max(best, std::optional{make_candidate(rules[rule], rule,
                                                         platform, platform)});
    }
  }
  return best.has_value()
             ? rules[to_idx(best->rule_idx_)].duration_
             : from_transfer_time(tt.locations_.transfer_time_[platform]);
}

// Gives a new real-time virtual location its transfers: the platform's own
// transfers are the default, the most specific rule per partner overrides it.
void link(timetable const& tt, rt_timetable& rtt, location_idx_t const v) {
  auto const& tr = tt.transfer_rules_;
  auto const& loc = tt.locations_;
  auto const& virt = rtt.rt_virts_[to_idx(v) - tt.n_locations()];
  auto const p = virt.parent_;
  auto const change_time = from_transfer_time(loc.transfer_time_[p]);

  // [0] = transfers leaving v, [1] = transfers entering v
  struct {
    hash_map<location_idx_t, duration_t> durations_;
    hash_map<location_idx_t, candidate> rules_;
  } edges[2];

  // --- defaults: whatever the platform reaches / is reached from ---
  auto const add_defaults = [&]<direction Dir>(auto& durations) {
    auto const add = [&](footpath const fp) {
      auto const [it, is_new] = durations.emplace(fp.target(), fp.duration());
      it->second = std::min(it->second, fp.duration());
      return true;
    };
    routing::for_each_transfer<Dir>(tt, nullptr, kDefaultProfile, p, add);

    // the platform itself and the other real-time virtual locations, which
    // are reached like their platform (kNoTransfer: no change at this stop)
    durations.erase(p);
    rtt.for_each_rt_virt([&](location_idx_t const w, auto const& other) {
      if (auto const it = durations.find(other.parent_);
          w != v && it != end(durations)) {
        durations.emplace(w, it->second);
      }
    });
    if (change_time != footpath::kMaxDuration) {
      durations.emplace(p, change_time);
      rtt.for_each_rt_virt([&](location_idx_t const w, auto const& other) {
        if (w != v && other.parent_ == p) {
          durations.emplace(w, change_time);
        }
      });
    }
  };
  add_defaults.template operator()<direction::kForward>(edges[0].durations_);
  add_defaults.template operator()<direction::kBackward>(edges[1].durations_);

  // --- rules: the most specific one per partner ---
  auto const for_each_location = [&](std::uint32_t const rule,
                                     bool const is_from, auto&& fn) {
    auto const& r = tr.rules_[rule];
    auto const side = transfer_rules::side(rule, is_from);
    if (is_from ? r.from_qualified() : r.to_qualified()) {
      // the virtual locations split off for this side
      transfer_rules::for_each(
          tr.side_virts_, side,
          [&](std::uint32_t const l) { fn(location_idx_t{l}); });
      rtt.for_each_rt_virt([&](location_idx_t const w, auto const& other) {
        if (std::binary_search(begin(other.sides_), end(other.sides_), side)) {
          fn(w);
        }
      });
    } else {
      // unqualified: the rule stop and everything below it
      auto const stop = is_from ? r.from_stop_ : r.to_stop_;
      fn(stop);
      for (auto const c : loc.children_[stop]) {
        fn(c);
        for (auto const cc : loc.children_[c]) {
          fn(cc);
        }
      }
      rtt.for_each_rt_virt([&](location_idx_t const w, auto const& other) {
        if (covers(tt, stop, other.parent_)) {
          fn(w);
        }
      });
    }
  };

  auto const compete = [&](side_t const mine) {
    auto const rule = transfer_rules::rule_of(mine);
    auto const is_from = transfer_rules::is_from(mine);
    for_each_location(rule, !is_from, [&](location_idx_t const y) {
      if (y != v) {
        auto const y_platform =
            rtt.is_rt_virt(y) ? rtt.physical(y) : tt.locations_.get_base_idx(y);
        auto& best = edges[is_from ? 0U : 1U].rules_[y];
        best = std::max(best, make_candidate(tr.rules_[rule], rule,
                                             is_from ? p : y_platform,
                                             is_from ? y_platform : p));
      }
    });
  };
  for (auto const s : virt.sides_) {
    compete(s);  // qualified sides that match the trip stop
  }
  for (auto const stop : {p, loc.parents_[p]}) {  // unqualified sides
    if (stop != location_idx_t::invalid()) {
      transfer_rules::for_each(tr.stop_sides_, to_idx(stop), compete);
    }
  }

  // --- write, both ways ---
  auto const fps = std::array{&rtt.rt_fps_out_, &rtt.rt_fps_in_};
  for (auto const dir : {0U, 1U}) {
    auto& [durations, rules] = edges[dir];
    for (auto const& [y, c] : rules) {
      if (auto const d = tr.rules_[to_idx(c.rule_idx_)].duration_;
          d == footpath::kMaxDuration) {
        durations.erase(y);  // transfer not possible
      } else {
        durations[y] = d;
      }
    }
    for (auto const& [y, d] : durations) {
      (*fps[dir])[v].emplace_back(y, d);
      (*fps[1U - dir])[y].emplace_back(v, d);
    }
  }
}

// platform / static virtual location / real-time virtual location
location_idx_t resolve(timetable const& tt,
                       rt_timetable& rtt,
                       transport_idx_t const t,
                       stop_idx_t const stop_idx,
                       location_idx_t const platform) {
  if (tt.transfer_rules_.empty() || t == transport_idx_t::invalid()) {
    return platform;
  }

  auto sig = get_signature(tt, t, stop_idx, platform);
  if (sig.empty()) {
    return platform;
  }

  // same key as in the loader: own change time + what the sides state
  auto const own = to_transfer_time(get_own(tt, sig, platform));
  auto const values = values_of(tt, sig);

  for (auto const c : tt.locations_.children_[platform]) {
    if (tt.locations_.types_[c] != location_type::kVirt ||
        tt.locations_.transfer_time_[c] != own) {
      continue;
    }
    auto sides = std::vector<side_t>{};
    transfer_rules::for_each(tt.transfer_rules_.virt_sides_, to_idx(c),
                             [&](side_t const s) { sides.push_back(s); });
    if (values_of(tt, sides) == values) {
      return c;
    }
  }

  auto existing = location_idx_t::invalid();
  rtt.for_each_rt_virt([&](location_idx_t const w, auto const& other) {
    if (other.parent_ == platform && other.transfer_time_ == own &&
        values_of(tt, other.sides_) == values) {
      existing = w;
    }
  });
  if (existing != location_idx_t::invalid()) {
    return existing;
  }

  auto const v = location_idx_t{rtt.n_routing_locations()};
  utl::verify(to_idx(v) < footpath::kMaxTarget, "rt virt index overflow");
  rtt.rt_virts_.push_back(
      {.parent_ = platform, .transfer_time_ = own, .sides_ = std::move(sig)});
  rtt.location_rt_transports_[v];  // make room
  link(tt, rtt, v);
  return v;
}

}  // namespace

location_idx_t route_stop_at(timetable const& tt,
                             rt_timetable& rtt,
                             rt_transport_idx_t const rt_t,
                             stop_idx_t const stop_idx,
                             location_idx_t const l) {
  auto const is_platform = tt.locations_.types_[l] != location_type::kVirt;
  auto const routing =
      is_platform
          ? resolve(tt, rtt, rtt.resolve_static(rt_t).t_idx_, stop_idx, l)
          : l;
  auto const is_rt = rtt.is_rt_virt(routing);

  auto seq = rtt.rt_transport_location_seq_[rt_t];
  auto const s = stop{seq[stop_idx]};
  seq[stop_idx] = s.with_location(is_rt ? l : routing).value();

  if (is_rt || rtt.rt_routing_locations_.contains(rt_t)) {
    auto& locs = rtt.rt_routing_locations_[rt_t];
    locs.resize(seq.size(), location_idx_t::invalid());
    locs[stop_idx] = is_rt ? routing : location_idx_t::invalid();
    if (utl::all_of(locs, [](location_idx_t const x) {
          return x == location_idx_t::invalid();
        })) {
      rtt.rt_routing_locations_.erase(rt_t);
    }
  }

  for (auto const x : {routing, l}) {
    auto transports = rtt.location_rt_transports_[x];
    if (utl::find(transports, rt_t) == end(transports)) {
      transports.push_back(rt_t);
    }
  }
  return routing;
}

}  // namespace nigiri::rt
