#include "nigiri/rt/rt_transfer_rules.h"

#include <cassert>
#include <algorithm>
#include <span>
#include <vector>

#include "utl/get_or_create.h"
#include "utl/helpers/algorithm.h"
#include "utl/lookup.h"
#include "utl/to_vec.h"

#include "nigiri/footpath.h"
#include "nigiri/routing/for_each_hub_source.h"
#include "nigiri/transfer_rule_sides.h"

namespace nigiri::rt {

using side_t = transfer_rule_side_idx;

std::vector<side_t> get_stop_signature(timetable const& tt,
                                       transport_idx_t const t,
                                       stop_idx_t const stop_idx,
                                       location_idx_t const base) {
  auto const sections = tt.transport_to_trip_section_[t];
  auto const n = sections.size();
  auto const from = n == 1U || stop_idx == 0U ? 0U : stop_idx - 1U;
  auto const to = n == 1U ? 1U : std::min<std::size_t>(stop_idx + 1U, n);

  auto const arriving_trips = tt.merged_trips_[sections[from]];
  auto const departing_trips = tt.merged_trips_[sections[to - 1U]];
  auto const is_change = to - from == 2U;
  auto const is_handover =
      is_change && utl::none_of(arriving_trips, [&](trip_idx_t const x) {
        return utl::find(departing_trips, x) != end(departing_trips);
      });

  return get_change_signature(
      tt, {begin(arriving_trips), end(arriving_trips)},
      is_change ? std::span<trip_idx_t const>{begin(departing_trips),
                                              end(departing_trips)}
                : std::span<trip_idx_t const>{},
      is_handover, base);
}

void add_rt_transfers(timetable const& tt,
                      rt_timetable& rtt,
                      location_idx_t const v) {
  auto const& tr = tt.transfer_rules_;
  auto const& loc = tt.locations_;
  auto const p = rtt.base(v);
  auto const change_time = to_fp_duration(loc.transfer_time_[p]);

  auto const for_each_location = [&](side_t const side, auto&& fn) {
    for_each_side_location(tt, side, fn);
    rtt.for_each_rt_virt([&](location_idx_t const w,
                             rt_location_idx_t const i) {
      auto const rules = rtt.rt_virt_rules_[i];
      if (is_applicable(tt, side, {begin(rules), end(rules)}, rtt.base(w))) {
        fn(w);
      }
    });
  };

  for (auto const dir : {direction::kForward, direction::kBackward}) {
    auto const is_from = dir == direction::kForward;

    auto durations = hash_map<location_idx_t, duration_t>{};
    routing::for_each_transfer(
        tt, nullptr, kDefaultProfile, dir, p, [&](footpath const fp) {
          auto const it = durations.emplace(fp.target(), fp.duration()).first;
          it->second = std::min(it->second, fp.duration());
        });
    durations.erase(p);
    if (change_time != footpath::kMaxDuration) {
      durations.emplace(p, change_time);
    }
    rtt.for_each_rt_virt([&](location_idx_t const w, rt_location_idx_t) {
      if (auto const d = utl::lookup(durations, rtt.base(w));
          w != v && d.has_value()) {
        durations.emplace(w, *d);
      }
    });

    auto most_specific = hash_map<location_idx_t, transfer_rule_idx_t>{};
    auto const compete = [&](side_t const s) {
      if (s.is_from() != is_from) {
        return;
      }
      for_each_location(
          side_t{s.rule(), !is_from}, [&](location_idx_t const y) {
            if (y != v) {
              auto const [it, is_new] = most_specific.emplace(y, s.rule());
              if (!is_new) {
                it->second = get_more_specific(tt, it->second, s.rule());
              }
            }
          });
    };
    for (auto const s : rtt.rt_virt_rules_[rtt.to_rt_location(v)]) {
      compete(s);
    }
    for (auto const stop : {p, loc.parents_[p]}) {
      if (stop != location_idx_t::invalid()) {
        for (auto const s : values_of(tr.stop_rules_, stop)) {
          compete(s);
        }
      }
    }

    for (auto const& [y, rule] : most_specific) {
      if (auto const d = tr.rules_[rule].duration_;
          d == footpath::kMaxDuration) {
        durations.erase(y);
      } else {
        durations[y] = d;
      }
    }
    auto& out = is_from ? rtt.rt_footpaths_out_ : rtt.rt_footpaths_in_;
    auto& in = is_from ? rtt.rt_footpaths_in_ : rtt.rt_footpaths_out_;
    for (auto const& [y, d] : durations) {
      out[v].emplace_back(y, d);
      in[y].emplace_back(v, d);
    }
  }
}

location_idx_t get_or_create_location(timetable const& tt,
                                      rt_timetable& rtt,
                                      rt_transport_idx_t const rt_t,
                                      stop_idx_t const stop_idx,
                                      location_idx_t const base) {
  assert(!tt.locations_.is_virt(base));
  auto const t = rtt.resolve_static(rt_t).t_idx_;
  if (tt.transfer_rules_.empty() || t == transport_idx_t::invalid()) {
    return base;
  }

  auto sig = get_stop_signature(tt, t, stop_idx, base);
  if (sig.empty()) {
    return base;
  }

  auto const key = get_virt_key(tt, sig, base);
  auto existing = location_idx_t::invalid();
  tt.locations_.for_each_virt(base, [&](location_idx_t const c) {
    if (existing == location_idx_t::invalid() &&
        tt.locations_.transfer_time_[c] == key.transfer_time_ &&
        get_transfer_rule_sides(
            tt, utl::to_vec(values_of(tt.transfer_rules_.virt_rules_, c)),
            base) == key.sides_) {
      existing = c;
    }
  });
  if (existing != location_idx_t::invalid()) {
    return existing;
  }

  return utl::get_or_create(rtt.rt_virts_, key, [&]() {
    auto const v = rtt.add_rt_location(base, key.transfer_time_, sig);
    add_rt_transfers(tt, rtt, v);
    return v;
  });
}

}  // namespace nigiri::rt
