#include "nigiri/loader/apply_transfer_rules.h"

#include <cassert>
#include <algorithm>
#include <optional>
#include <tuple>
#include <vector>

#include "utl/erase_if.h"
#include "utl/expect.h"
#include "utl/helpers/algorithm.h"

#include "nigiri/loader/import_context.h"
#include "nigiri/loader/register.h"
#include "nigiri/loader/virt/footpaths.h"
#include "nigiri/loader/virt/merge_locations.h"
#include "nigiri/loader/virt/rewrite_routes.h"
#include "nigiri/loader/virt/route_stop_pairs.h"
#include "nigiri/loader/virt/signatures.h"
#include "nigiri/loader/virt/split_routes.h"
#include "nigiri/loader/virt/transfer_durations.h"
#include "nigiri/footpath.h"
#include "nigiri/for_each_meta.h"
#include "nigiri/logging.h"
#include "nigiri/shapes_storage.h"
#include "nigiri/stop.h"
#include "nigiri/timetable.h"

namespace nigiri::loader {

using stop_seq_t = std::vector<stop::value_type>;

// Rules without route/trip replace the walk between their stops and all
// their descendants (forbidden: no walk at all). Applied from least to most
// specific (as get_more_specific): the most specific rule wins.
void apply_unqualified_rules(
    timetable const& tt,
    mutable_fws_multimap<location_idx_t, footpath>& walks) {
  auto const& rules = tt.transfer_rules_.rules_;
  auto order = std::vector<transfer_rule_idx_t>{};
  for (auto r = transfer_rule_idx_t{0U}; r != rules.size(); ++r) {
    if (!rules[r].is_qualified()) {
      order.push_back(r);
    }
  }
  utl::sort(order,
            [&](transfer_rule_idx_t const a, transfer_rule_idx_t const b) {
              return std::tie(rules[a].specificity_, a) <
                     std::tie(rules[b].specificity_, b);
            });
  auto const for_each_stop = [&](location_idx_t const l, auto&& fn) {
    routing::for_each_meta(tt, nullptr,
                           routing::location_match_mode::kOnlyChildren, l, fn);
  };
  for (auto const r : order) {
    for_each_stop(rules[r].from_stop_, [&](location_idx_t const from) {
      for_each_stop(rules[r].to_stop_, [&](location_idx_t const to) {
        if (from == to) {
          return;
        }
        auto bucket = walks[from];
        utl::erase_if(bucket,
                      [&](footpath const fp) { return fp.target() == to; });
        if (!rules[r].is_forbidden()) {
          bucket.emplace_back(to, rules[r].duration_);
        }
      });
    });
  }
}

duration_t to_duration(duration_t const d) {
  return d >= kNever ? footpath::kBan : d;
}

// Location of each route stop (joined ones: of the route stop they joined).
vector_map<route_stop_idx_t, location_idx_t> add_virts(
    timetable& tt,
    import_context& ctx,
    route_stop_pairs const& pairs,
    vector_map<route_stop_idx_t, route_stop_idx_t> const& route_stop_location) {
  auto virts = vector_map<route_stop_idx_t, location_idx_t>(
      pairs.route_stops_.size(), location_idx_t::invalid());
  auto n_virts = std::size_t{0U};
  for (auto s = route_stop_idx_t{0U}; s != pairs.route_stops_.size(); ++s) {
    if (route_stop_location[s] != s) {
      // Route stops only join earlier ones (their base or an earlier virt).
      assert(route_stop_location[s] < s);
      virts[s] = virts[route_stop_location[s]];
      continue;
    }
    auto const base = pairs.route_stops_[s].base_;
    if (pairs.is_base(s)) {
      virts[s] = base;
      continue;
    }
    auto const fallback = get_transfer_time(tt, base);
    auto const self = pairs.pair_of_.find(route_stop_pair{s, s});
    auto d = self == end(pairs.pair_of_)
                 ? std::optional{fallback}
                 : pick_duration(pairs.windows_[self->second], fallback);
    if (!utl_expect(d.has_value(), "loader.virtual_locations",
                    "virtual location without transfer time at {}", base)) {
      d = fallback;
    }
    auto l = location{};
    l.src_ = tt.locations_.src_[base];
    l.pos_ = tt.locations_.coordinates_[base];
    l.type_ = location_type::kVirt;
    l.parent_ = base;
    l.transfer_time_ = to_duration(*d);
    virts[s] = register_location(tt, ctx, l);
    ctx.children_[base].push_back(virts[s]);
    ++n_virts;
  }
  if (n_virts != 0U) {
    tt.locations_.children_ = make_sparse_vecvec(ctx.children_);
  }
  log(log_lvl::info, "loader.virtual_locations", "{} virtual locations",
      n_virts);
  return virts;
}

mutable_fws_multimap<location_idx_t, footpath> get_virt_footpaths(
    timetable const& tt,
    route_stop_pairs const& pairs,
    vector_map<route_stop_idx_t, route_stop_idx_t> const& route_stop_location,
    vector_map<route_stop_idx_t, location_idx_t> const& virts) {
  auto virt_footpaths = mutable_fws_multimap<location_idx_t, footpath>{};
  virt_footpaths[location_idx_t{tt.n_locations() - 1U}];
  auto n_footpaths = std::size_t{0U};
  for (auto p = route_stop_pair_idx_t{0U}; p != pairs.pairs_.size(); ++p) {
    auto const [a, b] = pairs.pairs_[p];
    if (a == b  // self-transfer time is already set on virt locations
        || route_stop_location[a] != a  // a joined another location
        || route_stop_location[b] != b  // b joined another location
        || (pairs.is_base(a) && pairs.is_base(b))) {
      continue;
    }
    auto const fallback = pairs.durations_[p].fallback_;
    auto const d = pick_duration(pairs.windows_[p], fallback);
    if (!utl_expect(d.has_value(), "loader.virtual_locations",
                    "location pair without duration: {} -> {}", virts[a],
                    virts[b]) ||
        *d == fallback) {
      continue;
    }
    virt_footpaths[virts[a]].push_back(footpath{virts[b], to_duration(*d)});
    ++n_footpaths;
  }
  log(log_lvl::info, "loader.virtual_locations", "{} footpaths", n_footpaths);
  return virt_footpaths;
}

void write_routes(timetable& tt,
                  import_context& ctx,
                  signatures const& sigs,
                  route_stop_pairs const& pairs,
                  transport_routes const& routes,
                  vector_map<route_stop_idx_t, location_idx_t> const& virts,
                  shapes_storage* shapes) {
  auto const n_routes = tt.n_routes();

  // Write route -> transport mapping.
  auto route_transports = vecvec<route_idx_t, transport_idx_t>{};
  auto transports = std::vector<transport_idx_t>{};
  for (auto r = route_idx_t{0U}; r != routes.base_route_.size(); ++r) {
    transports.clear();
    for (auto const t : tt.route_transport_ranges_[routes.base_route_[r]]) {
      if (routes.route_[t] == r) {
        transports.push_back(t);
      }
    }
    route_transports.emplace_back(transports);
  }

  // Update stop sequences.
  auto seq = stop_seq_t{};
  auto const get_seq = [&](route_idx_t const r) -> stop_seq_t const& {
    auto const base_r = routes.base_route_[r];
    auto const base_seq = tt.route_location_seq_[base_r];
    seq.assign(base_seq.begin(), base_seq.end());
    for (auto i = stop_idx_t{0U}; i != seq.size(); ++i) {
      auto const s = stop{seq[i]};
      auto const l = tt.locations_.get_base_idx(s.location_idx());
      auto const stop_sigs = sigs.get(base_r, i);
      auto const it = pairs.route_stop_of_.find(
          stop_sigs.size() == 1U  // uniform: shared signature route stop
              ? to_signature_stop(l, stop_sigs.front())
              : route_stop{r, i, l});
      seq[i] = s.with_location(
                    it == end(pairs.route_stop_of_) ? l : virts[it->second])
                   .value();
    }
    return seq;
  };
  for (auto r = route_idx_t{0U}; r != routes.base_route_.size(); ++r) {
    if (r >= n_routes) {
      tt.route_location_seq_.emplace_back(get_seq(r));
    } else if (sigs.has_rules(r)) {
      std::ranges::copy(get_seq(r), tt.route_location_seq_[r].begin());
    }
  }

  // Write new routes and permute transports.
  rewrite_routes(tt, ctx, shapes, routes.base_route_, route_transports);

  log(log_lvl::info, "loader.virtual_locations", "{} routes -> {}", n_routes,
      tt.n_routes());
}

void apply_transfer_rules(
    timetable& tt,
    import_context& ctx,
    shapes_storage* shapes,
    mutable_fws_multimap<location_idx_t, footpath>& walks) {
  auto virt_footpaths = mutable_fws_multimap<location_idx_t, footpath>{};
  if (!tt.transfer_rules_.rules_.empty()) {
    apply_unqualified_rules(tt, walks);

    auto const timer = scoped_timer{"loader.virtual_locations"};
    auto const sigs = get_signatures(tt);
    auto pairs = get_route_stop_pairs(tt, ctx, sigs,
                                      get_transfer_durations(tt, walks, sigs));
    auto const routes = split_routes(tt, pairs);
    auto const route_stop_location = merge_locations(pairs);
    auto const n_locations = tt.n_locations();
    auto const virts = add_virts(tt, ctx, pairs, route_stop_location);
    if (tt.n_locations() != n_locations) {
      write_routes(tt, ctx, sigs, pairs, routes, virts, shapes);
      virt_footpaths =
          get_virt_footpaths(tt, pairs, route_stop_location, virts);
      tt.location_location_groups_.resize(tt.n_locations());
      if (tt.location_areas_.size() != 0U) {
        tt.location_areas_.resize(tt.n_locations());
      }
    }
  }
  write_profiles(tt, walks, virt_footpaths);
}

}  // namespace nigiri::loader
