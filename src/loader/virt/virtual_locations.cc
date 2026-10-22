#include "nigiri/loader/virtual_locations.h"

#include <cstdint>
#include <algorithm>
#include <optional>
#include <utility>
#include <vector>

#include "utl/helpers/algorithm.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "nigiri/loader/register.h"
#include "nigiri/loader/transfer_rules.h"
#include "nigiri/loader/virt/route_stop_pairs.h"
#include "nigiri/loader/virt/signatures.h"
#include "nigiri/loader/virt/transfer_durations.h"
#include "nigiri/common/interval.h"
#include "nigiri/footpath.h"
#include "nigiri/logging.h"
#include "nigiri/shapes_storage.h"
#include "nigiri/stop.h"
#include "nigiri/timetable.h"

namespace nigiri::loader {

using stop_seq_t = std::vector<stop::value_type>;

constexpr auto const kAnyDuration =
    window_t{duration_t::min(), duration_t::max()};

using group_idx_t = cista::strong<std::uint32_t, struct group_>;
using pair_side_idx_t = cista::strong<std::uint32_t, struct pair_side_>;
using split_route_idx_t = cista::strong<std::uint32_t, struct split_route_>;

struct location_pair {
  route_stop_idx_t to_;
  window_t window_;
};

std::optional<duration_t> pick_duration(window_t const& w,
                                        duration_t const fallback) {
  if (w.contains(fallback)) {
    return fallback;
  }
  if (w.contains(kNever)) {
    return kNever;
  }
  return w.from_ < std::min(w.to_, footpath::kBan) ? std::optional{w.from_}
                                                   : std::nullopt;
}

bool is_admissible(window_t const& w) {
  return pick_duration(w, kNever).has_value();
}

template <typename T>
auto* find_pair(T& location_pairs, route_stop_idx_t const to) {
  auto const it =
      std::ranges::lower_bound(location_pairs, to, {}, &location_pair::to_);
  return it != end(location_pairs) && it->to_ == to ? &*it : nullptr;
}

struct transport_routes {
  route_idx_t get_route(timetable const& tt,
                        route_idx_t const r,
                        route_transport_idx_t const t) const {
    return route_[tt.route_transport_ranges_[base_route_[r]].from_ + to_idx(t)];
  }

  vector_map<route_idx_t, route_idx_t> base_route_;
  vector_map<transport_idx_t, route_idx_t> route_;
};

transport_routes split_routes(
    timetable const& tt,
    hash_map<transfer_pair, transfer_durations> const& durations_of,
    route_stop_pairs& pairs) {
  // Initialize:
  // - Map each route to iself.
  // - Map each transport to its base route.
  auto routes = transport_routes{};
  for (auto r = route_idx_t{0U}; r != tt.n_routes(); ++r) {
    routes.base_route_.push_back(r);
  }
  routes.route_ = tt.transport_route_;

  auto const repartition = [&](route_stop_pair_idx_t const p) {
    auto const route_of = [&](route_stop const& s,
                              route_transport_idx_t const t) {
      return !s.has_route() ? s.route_ : routes.get_route(tt, s.route_, t);
    };

    auto const from = pairs.route_stops_[pairs.pairs_[p].first];
    auto const to = pairs.route_stops_[pairs.pairs_[p].second];
    auto const hull =
        durations_of.at(transfer_pair{from.stop_, to.stop_}).hull_;
    auto transfers = std::move(pairs.transfers_[p]);
    auto kept = begin(transfers);
    for (auto const& t : transfers) {
      auto const from_route = route_of(from, t.from_);
      auto const to_route = route_of(to, t.to_);
      if (from_route == from.route_ && to_route == to.route_) {
        *kept++ = t;
      } else {
        auto const new_from = pairs.get_or_create_route_stop(
            route_stop{from_route, from.stop_idx_, from.stop_, from.sig_});
        auto const new_to = pairs.get_or_create_route_stop(
            route_stop{to_route, to.stop_idx_, to.stop_, to.sig_});
        pairs.add_transfer(pairs.get_or_create_pair(new_from, new_to, hull), t);
      }
    }
    if (kept == end(transfers)) {
      pairs.transfers_[p] = std::move(transfers);
      return;
    }
    auto& w = pairs.windows_[p];
    w = hull;
    for (auto it = begin(transfers); it != kept; ++it) {
      narrow(w, it->gap_, it->gap_ >= it->expected_);
    }
    pairs.transfers_[p] = std::vector<transfer>(begin(transfers), kept);
  };

  struct transport_groups {
    vector_map<route_transport_idx_t, group_idx_t> group_;
    std::uint32_t n_groups_{0U};
  };
  auto const get_groups = [&](route_stop_pair_idx_t const p,
                              bool const is_from) -> transport_groups {
    auto const& from = pairs.route_stops_[pairs.pairs_[p].first];
    auto const& to = pairs.route_stops_[pairs.pairs_[p].second];
    auto const& side = is_from ? from : to;
    if (!side.has_route()) {
      return {};
    }
    auto const& pair_durations =
        durations_of.at(transfer_pair{from.stop_, to.stop_});
    auto const n_transports = route_transport_idx_t{
        tt.route_transport_ranges_[routes.base_route_[side.route_]].size()};
    auto w = vector_map<route_transport_idx_t, std::optional<window_t>>(
        to_idx(n_transports));
    for (auto const& t : pairs.transfers_[p]) {
      auto& tw = w[is_from ? t.from_ : t.to_];
      if (!tw.has_value()) {
        tw = pair_durations.hull_;
      }
      narrow(*tw, t.gap_, t.gap_ >= t.expected_);
    }

    auto const items = [&]() {
      auto v = std::vector<std::pair<route_transport_idx_t, window_t>>{};
      for (auto i = route_transport_idx_t{0U}; i != n_transports; ++i) {
        if (w[i].has_value()) {
          v.emplace_back(i, *w[i]);
        }
      }
      utl::sort(v, [](auto const& l, auto const& r) {
        return std::pair{l.second.to_, l.first} <
               std::pair{r.second.to_, r.first};
      });
      return v;
    }();

    auto groups = transport_groups{};
    groups.group_.resize(to_idx(n_transports), group_idx_t::invalid());
    auto window = window_t{};
    for (auto const& [t, tw] : items) {
      auto const both = window.intersect(tw);
      if (groups.n_groups_ != 0U &&
          pick_duration(both, pair_durations.fallback_).has_value()) {
        window = both;
      } else {
        window = tw;
        ++groups.n_groups_;
      }
      groups.group_[t] = group_idx_t{groups.n_groups_ - 1U};
    }
    return groups.n_groups_ < 2U ? transport_groups{} : groups;
  };

  auto pair_groups =
      vector_map<pair_side_idx_t,
                 vector_map<route_transport_idx_t, group_idx_t>>{};
  auto route_groups =
      vector_map<split_route_idx_t, vector_map<pair_side_idx_t, group_idx_t>>{};
  auto new_route_of = vector_map<route_transport_idx_t, split_route_idx_t>{};
  auto route_sizes = vector_map<split_route_idx_t, std::uint32_t>{};

  // Enqueue all pairs that have conflicts (i.e. >= 2 transports that would need
  // different transfer durations of this pair).
  auto q = std::vector<route_stop_pair_idx_t>{};
  for (auto p = route_stop_pair_idx_t{0U}; p != pairs.pairs_.size(); ++p) {
    if (!is_admissible(pairs.windows_[p])) {
      q.push_back(p);
    }
  }

  auto n_split = 0U;
  while (!q.empty()) {
    auto const p = q.back();
    q.pop_back();

    // Conflict got solved by a previous split between insert and pop of p.
    if (is_admissible(pairs.windows_[p])) {
      continue;
    }

    auto const from_groups = get_groups(p, true);
    auto const to_groups = get_groups(p, false);
    utl::verify(from_groups.n_groups_ != 0U || to_groups.n_groups_ != 0U,
                "split_routes: inadmissible pair without groups");

    auto const use_from = from_groups.n_groups_ != 0U &&
                          (to_groups.n_groups_ == 0U ||
                           from_groups.n_groups_ <= to_groups.n_groups_);
    auto const [from, to] = pairs.pairs_[p];
    auto const r = pairs.route_stops_[use_from ? from : to].route_;
    auto const base_route = routes.base_route_[r];
    auto const range = tt.route_transport_ranges_[base_route];
    auto const n_transports = route_transport_idx_t{range.size()};

    pair_groups.clear();
    for (auto const pq : pairs.of_route_[r]) {
      if (is_admissible(pairs.windows_[pq])) {
        continue;
      }
      auto const [pq_from, pq_to] = pairs.pairs_[pq];
      for (auto const is_from : {true, false}) {
        if (pairs.route_stops_[is_from ? pq_from : pq_to].route_ == r) {
          auto g = get_groups(pq, is_from);
          if (g.n_groups_ != 0U) {
            pair_groups.push_back(std::move(g.group_));
          }
        }
      }
    }

    route_groups.clear();
    route_sizes.clear();
    new_route_of.clear();
    new_route_of.resize(to_idx(n_transports), split_route_idx_t::invalid());
    for (auto t = route_transport_idx_t{0U}; t != n_transports; ++t) {
      if (routes.get_route(tt, r, t) != r ||
          utl::all_of(pair_groups, [&](auto const& g) {
            return g[t] == group_idx_t::invalid();
          })) {
        continue;
      }
      auto const fits = [&](auto const& groups) {
        for (auto i = pair_side_idx_t{0U}; i != pair_groups.size(); ++i) {
          auto const g = pair_groups[i][t];
          if (g != group_idx_t::invalid() &&
              groups[i] != group_idx_t::invalid() && groups[i] != g) {
            return false;
          }
        }
        return true;
      };
      auto const it = utl::find_if(route_groups, fits);
      auto const k = split_route_idx_t{it - begin(route_groups)};
      if (it == end(route_groups)) {
        route_groups.emplace_back(pair_groups.size(), group_idx_t::invalid());
        route_sizes.push_back(0U);
      }
      for (auto i = pair_side_idx_t{0U}; i != pair_groups.size(); ++i) {
        if (pair_groups[i][t] != group_idx_t::invalid()) {
          route_groups[k][i] = pair_groups[i][t];
        }
      }
      new_route_of[t] = k;
      ++route_sizes[k];
    }
    auto const kept_route = split_route_idx_t{
        std::ranges::max_element(route_sizes) - begin(route_sizes)};

    auto changed = std::vector{r};
    for (auto k = split_route_idx_t{0U}; k != route_sizes.size(); ++k) {
      if (k == kept_route) {
        continue;
      }
      auto const new_route = route_idx_t{routes.base_route_.size()};
      changed.push_back(new_route);
      routes.base_route_.push_back(base_route);
      pairs.of_route_.emplace_back();
      for (auto t = route_transport_idx_t{0U}; t != n_transports; ++t) {
        if (new_route_of[t] == k) {
          routes.route_[range.from_ + to_idx(t)] = new_route;
        }
      }
    }
    auto const route_pairs = pairs.of_route_[r];
    for (auto const pq : route_pairs) {
      repartition(pq);
    }
    for (auto const c : changed) {
      for (auto const pq : pairs.of_route_[c]) {
        if (!is_admissible(pairs.windows_[pq])) {
          q.push_back(pq);
        }
      }
    }
    q.push_back(p);
    ++n_split;
  }
  log(log_lvl::info, "loader.virtual_locations",
      "{} route stops, {} pairs, {} splits", pairs.route_stops_.size(),
      pairs.pairs_.size(), n_split);
  return routes;
}

struct route_stop_locations {
  vector_map<route_stop_idx_t, route_stop_idx_t> location_;
  vector_map<route_stop_idx_t, std::vector<location_pair>> pairs_;
};

route_stop_locations merge_locations(
    hash_map<transfer_pair, transfer_durations> const& durations_of,
    route_stop_pairs const& pairs) {
  auto locations = route_stop_locations{};
  auto& location = locations.location_;
  auto& lps = locations.pairs_;
  lps.resize(pairs.route_stops_.size());
  for (auto s = route_stop_idx_t{0U}; s != pairs.route_stops_.size(); ++s) {
    location.push_back(s);
  }
  for (auto p = route_stop_pair_idx_t{0U}; p != pairs.pairs_.size(); ++p) {
    auto const [from, to] = pairs.pairs_[p];
    lps[from].push_back({to, pairs.windows_[p]});
  }
  for (auto& v : lps) {
    std::ranges::sort(v, {}, &location_pair::to_);
  }
  auto const get_partners = [&](bool const is_out) {
    auto n = vector_map<route_stop_idx_t, std::uint32_t>(
        pairs.route_stops_.size(), 0U);
    for (auto const [from, to] : pairs.pairs_) {
      ++n[is_out ? from : to];
    }
    auto partners = vecvec<route_stop_idx_t, route_stop_idx_t>{};
    for (auto const k : n) {
      partners.add_back_sized(k);
    }
    std::ranges::fill(n, 0U);
    for (auto const [from, to] : pairs.pairs_) {
      auto const x = is_out ? from : to;
      partners[x][n[x]++] = is_out ? to : from;
    }
    return partners;
  };
  auto const out = get_partners(true);
  auto const in = get_partners(false);
  auto out_partners = std::vector<route_stop_idx_t>{};
  auto in_partners = std::vector<route_stop_idx_t>{};
  auto added = std::vector<location_pair>{};
  auto is_partner = bitvec_map<route_stop_idx_t>{};
  is_partner.resize(pairs.route_stops_.size());
  auto const locations_of = [&](auto const& v,
                                std::vector<route_stop_idx_t>& partners) {
    partners.clear();
    for (auto const x : v) {
      auto const q = location[x];
      if (!is_partner.test(q)) {
        is_partner.set(q);
        partners.push_back(q);
      }
    }
    for (auto const q : partners) {
      is_partner.set(q, false);
    }
  };
  auto const find = [&](route_stop_idx_t const from,
                        route_stop_idx_t const to) {
    return find_pair(lps[from], to);
  };
  auto const get = [&](route_stop_idx_t const from,
                       route_stop_idx_t const to) -> location_pair const& {
    auto const* o = find(from, to);
    utl::verify(o != nullptr, "location pair missing");
    return *o;
  };
  auto const is_ok = [&](route_stop_idx_t const from, route_stop_idx_t const to,
                         window_t const& w) {
    return pairs.is_base(from) && pairs.is_base(to)
               ? w.contains(
                     durations_of
                         .at(transfer_pair{pairs.route_stops_[from].stop_,
                                           pairs.route_stops_[to].stop_})
                         .fallback_)
               : is_admissible(w);
  };
  struct pair_update {
    route_stop_idx_t q_;
    location_pair* pair_;
    window_t window_;
  };
  auto out_updates = std::vector<pair_update>{};
  auto in_updates = std::vector<pair_update>{};
  auto const join = [&](route_stop_idx_t const s, route_stop_idx_t const l) {
    locations_of(out[s], out_partners);
    locations_of(in[s], in_partners);
    auto* const self_pair = find(l, l);
    auto self = self_pair != nullptr ? self_pair->window_ : kAnyDuration;
    auto has_self = false;
    out_updates.clear();
    in_updates.clear();
    for (auto const q : out_partners) {
      auto const& o = get(s, q).window_;
      if (q == s || q == l) {
        has_self = true;
        self = self.intersect(o);
        continue;
      }
      auto* const m = find(l, q);
      auto const w = (m != nullptr ? m->window_ : kAnyDuration).intersect(o);
      if (!is_ok(l, q, w)) {
        return false;
      }
      out_updates.push_back({q, m, w});
    }
    for (auto const q : in_partners) {
      if (q == s) {
        continue;
      }
      auto const& o = get(q, s).window_;
      if (q == l) {
        has_self = true;
        self = self.intersect(o);
        continue;
      }
      auto* const m = find(q, l);
      auto const w = (m != nullptr ? m->window_ : kAnyDuration).intersect(o);
      if (!is_ok(q, l, w)) {
        return false;
      }
      in_updates.push_back({q, m, w});
    }
    if (has_self && !is_ok(l, l, self)) {
      return false;
    }
    added.clear();
    for (auto const& u : out_updates) {
      if (u.pair_ != nullptr) {
        u.pair_->window_ = u.window_;
      } else {
        added.push_back(location_pair{u.q_, u.window_});
      }
    }
    for (auto const& u : in_updates) {
      if (u.pair_ != nullptr) {
        u.pair_->window_ = u.window_;
      } else {
        auto& v = lps[u.q_];
        v.insert(std::ranges::lower_bound(v, l, {}, &location_pair::to_),
                 location_pair{l, u.window_});
      }
    }
    if (has_self) {
      if (self_pair != nullptr) {
        self_pair->window_ = self;
      } else {
        added.push_back(location_pair{l, self});
      }
    }
    if (!added.empty()) {
      std::ranges::sort(added, {}, &location_pair::to_);
      auto& v = lps[l];
      auto const n = static_cast<std::ptrdiff_t>(v.size());
      v.insert(end(v), std::make_move_iterator(begin(added)),
               std::make_move_iterator(end(added)));
      std::ranges::inplace_merge(v, begin(v) + n, {}, &location_pair::to_);
    }
    location[s] = l;
    return true;
  };
  auto at_stop = hash_map<location_idx_t, std::vector<route_stop_idx_t>>{};
  auto n_joined = 0U;
  auto n_locations = 0U;
  for (auto s = route_stop_idx_t{0U}; s != pairs.route_stops_.size(); ++s) {
    if (pairs.is_base(s)) {
      continue;
    }
    auto const st = pairs.route_stops_[s].stop_;
    auto& candidates = at_stop[st];
    if (candidates.empty()) {
      candidates.push_back(pairs.route_stop_of_.at(to_base_stop(st)));
    }
    if (utl::any_of(candidates,
                    [&](route_stop_idx_t const l) { return join(s, l); })) {
      ++n_joined;
    } else {
      candidates.push_back(s);
      ++n_locations;
    }
  }
  log(log_lvl::info, "loader.virtual_locations", "{} joined, {} locations",
      n_joined, n_locations);
  return locations;
}

void rewrite_routes(
    timetable& tt,
    shapes_storage* shapes,
    vector_map<route_idx_t, route_idx_t> const& base_route,
    vecvec<route_idx_t, transport_idx_t> const& route_transports) {
  auto const n_routes = tt.route_transport_ranges_.size();
  auto ranges = vector_map<route_idx_t, interval<transport_idx_t>>{};
  auto time_ranges = vector_map<route_idx_t, interval<std::uint32_t>>{};
  auto times = vector<delta>{};
  auto const& starts = route_transports.bucket_starts_;
  for (auto r = route_idx_t{0U}; r != base_route.size(); ++r) {
    auto const o = base_route[r];
    auto const transports = route_transports[r];
    ranges.emplace_back(transport_idx_t{starts[to_idx(r)]},
                        transport_idx_t{starts[to_idx(r) + 1U]});

    auto const begin_times = times.size();
    auto const n_stops =
        static_cast<stop_idx_t>(tt.route_location_seq_[r].size());
    for (auto i = stop_idx_t{1U}; i != n_stops; ++i) {
      for (auto const t : transports) {
        times.push_back(tt.event_mam(o, t, static_cast<stop_idx_t>(i - 1U),
                                     event_type::kDep));
      }
      for (auto const t : transports) {
        times.push_back(tt.event_mam(o, t, i, event_type::kArr));
      }
    }
    time_ranges.emplace_back(static_cast<std::uint32_t>(begin_times),
                             static_cast<std::uint32_t>(times.size()));

    if (r < n_routes) {
      continue;
    }
    auto const clasz = tt.route_clasz_[o];
    tt.route_clasz_.push_back(clasz);
    auto const section_clasz = utl::to_vec(tt.route_section_clasz_[o]);
    tt.route_section_clasz_.emplace_back(section_clasz);
    for (auto f = 0U; f != kNumRouteFlags; ++f) {
      auto& flags = tt.route_flags_[f];
      flags.resize(flags.size() + 2U);
      flags.set(to_idx(r) * 2U, flags.test(to_idx(o) * 2U));
      flags.set(to_idx(r) * 2U + 1U, flags.test(to_idx(o) * 2U + 1U));
      auto const per_section = utl::to_vec(tt.route_flags_per_section_[f][o]);
      tt.route_flags_per_section_[f].emplace_back(per_section);
    }
    if (shapes != nullptr && shapes->route_bboxes_.size() == to_idx(r)) {
      auto const bbox = shapes->route_bboxes_[o];
      auto const segments = utl::to_vec(shapes->route_segment_bboxes_[o]);
      shapes->route_bboxes_.push_back(bbox);
      shapes->route_segment_bboxes_.emplace_back(segments);
    }
  }
  tt.route_transport_ranges_ = std::move(ranges);
  tt.route_stop_time_ranges_ = std::move(time_ranges);
  tt.route_stop_times_ = std::move(times);

  auto const& order = route_transports.data_;
  auto inverse = vector_map<transport_idx_t, transport_idx_t>{};
  inverse.resize(order.size());
  for (auto t = transport_idx_t{0U}; t != order.size(); ++t) {
    inverse[order[to_idx(t)]] = t;
  }
  auto const permute = [&](auto& v) {
    auto copy = v;
    for (auto t = transport_idx_t{0U}; t != order.size(); ++t) {
      copy[t] = v[order[to_idx(t)]];
    }
    v = std::move(copy);
  };
  auto const permute_buckets = [&](auto& v) {
    auto copy = std::remove_reference_t<decltype(v)>{};
    for (auto const t : order) {
      copy.emplace_back(v[t]);
    }
    v = std::move(copy);
  };
  permute(tt.transport_first_dep_offset_);
  permute(tt.transport_traffic_days_);
  permute_buckets(tt.transport_to_trip_section_);
  permute_buckets(tt.transport_section_attributes_);
  permute_buckets(tt.transport_section_providers_);
  permute_buckets(tt.transport_section_directions_);
  for (auto r = route_idx_t{0U}; r != tt.n_routes(); ++r) {
    for (auto const t : tt.route_transport_ranges_[r]) {
      tt.transport_route_[t] = r;
    }
  }
  for (auto trip = trip_idx_t{0U}; trip != tt.trip_transport_ranges_.size();
       ++trip) {
    for (auto& range : tt.trip_transport_ranges_[trip]) {
      range.first = inverse[range.first];
    }
  }

  auto location_routes = mutable_fws_multimap<location_idx_t, route_idx_t>{};
  for (auto r = route_idx_t{0U}; r != tt.n_routes(); ++r) {
    for (auto const s : tt.route_location_seq_[r]) {
      auto routes = location_routes[stop{s}.location_idx()];
      if (routes.empty() || routes.back() != r) {
        routes.push_back(r);
      }
    }
  }
  tt.location_routes_.clear();
  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    tt.location_routes_.emplace_back(location_routes[l]);
  }
}

duration_t to_duration(duration_t const d) {
  return d >= kNever ? footpath::kBan : d;
}

vector_map<route_stop_idx_t, location_idx_t> add_virts(
    timetable& tt,
    route_stop_pairs const& pairs,
    route_stop_locations const& locations) {
  auto virts = vector_map<route_stop_idx_t, location_idx_t>(
      pairs.route_stops_.size(), location_idx_t::invalid());
  auto n_virts = std::size_t{0U};
  for (auto s = route_stop_idx_t{0U}; s != pairs.route_stops_.size(); ++s) {
    if (locations.location_[s] != s) {
      continue;
    }
    auto const st = pairs.route_stops_[s].stop_;
    if (pairs.is_base(s)) {
      virts[s] = st;
      continue;
    }
    auto const fallback = get_walk(tt, st, st);
    auto const* self = find_pair(locations.pairs_[s], s);
    auto const d = self == nullptr ? std::optional{fallback}
                                   : pick_duration(self->window_, fallback);
    utl::verify(d.has_value(), "virtual location without transfer time");
    auto l = location{};
    l.src_ = tt.locations_.src_[st];
    l.pos_ = tt.locations_.coordinates_[st];
    l.type_ = location_type::kVirt;
    l.parent_ = st;
    l.transfer_time_ = to_duration(*d);
    virts[s] = register_location(tt, l);
    tt.locations_.children_[st].push_back(virts[s]);
    ++n_virts;
  }
  log(log_lvl::info, "loader.virtual_locations", "{} virtual locations",
      n_virts);
  return virts;
}

void add_footpaths(
    timetable const& tt,
    hash_map<transfer_pair, transfer_durations> const& durations_of,
    route_stop_pairs const& pairs,
    route_stop_locations const& locations,
    vector_map<route_stop_idx_t, location_idx_t> const& virts,
    mutable_fws_multimap<location_idx_t, footpath>& durations) {
  durations[location_idx_t{tt.n_locations() - 1U}];
  auto n_footpaths = std::size_t{0U};
  for (auto a = route_stop_idx_t{0U}; a != locations.pairs_.size(); ++a) {
    if (locations.location_[a] != a) {
      continue;
    }
    for (auto const& m : locations.pairs_[a]) {
      auto const c = m.to_;
      if (a == c || locations.location_[c] != c ||
          (pairs.is_base(a) && pairs.is_base(c))) {
        continue;
      }
      auto const fallback = durations_of
                                .at(transfer_pair{pairs.route_stops_[a].stop_,
                                                  pairs.route_stops_[c].stop_})
                                .fallback_;
      auto const d = pick_duration(m.window_, fallback);
      utl::verify(d.has_value(), "location pair without duration");
      if (*d == fallback) {
        continue;
      }
      durations[virts[a]].push_back(footpath{virts[c], to_duration(*d)});
      ++n_footpaths;
    }
  }
  log(log_lvl::info, "loader.virtual_locations", "{} footpaths", n_footpaths);
}

void write_routes(timetable& tt,
                  signatures const& sigs,
                  route_stop_pairs const& pairs,
                  transport_routes const& routes,
                  route_stop_locations const& locations,
                  vector_map<route_stop_idx_t, location_idx_t> const& virts,
                  shapes_storage* shapes) {
  auto const n_routes = tt.n_routes();
  auto route_transports = vecvec<route_idx_t, transport_idx_t>{};
  auto transports = std::vector<transport_idx_t>{};
  for (auto q = route_idx_t{0U}; q != routes.base_route_.size(); ++q) {
    transports.clear();
    for (auto const t : tt.route_transport_ranges_[routes.base_route_[q]]) {
      if (routes.route_[t] == q) {
        transports.push_back(t);
      }
    }
    route_transports.emplace_back(transports);
  }
  auto seq = stop_seq_t{};
  auto const get_seq = [&](route_idx_t const q) -> stop_seq_t const& {
    auto const r = routes.base_route_[q];
    auto const base_seq = tt.route_location_seq_[r];
    seq.assign(base_seq.begin(), base_seq.end());
    for (auto i = stop_idx_t{0U}; i != seq.size(); ++i) {
      auto const l = stop{seq[i]}.location_idx();
      auto s = pairs.route_stop_of_.find(route_stop{q, i, l});
      if (s == end(pairs.route_stop_of_) && q != r) {
        auto const b = pairs.route_stop_of_.find(route_stop{r, i, l});
        if (b != end(pairs.route_stop_of_) &&
            !pairs.route_stops_[b->second].has_route()) {
          s = b;
        }
      }
      if (s != end(pairs.route_stop_of_)) {
        seq[i] = stop{seq[i]}
                     .with_location(virts[locations.location_[s->second]])
                     .value();
      }
    }
    return seq;
  };
  for (auto q = route_idx_t{n_routes}; q != routes.base_route_.size(); ++q) {
    tt.route_location_seq_.emplace_back(get_seq(q));
  }
  for (auto r = route_idx_t{0U}; r != n_routes; ++r) {
    if (pairs.of_route_[r].empty() && !sigs.has_rules(r)) {
      continue;
    }
    std::ranges::copy(get_seq(r), tt.route_location_seq_[r].begin());
  }
  rewrite_routes(tt, shapes, routes.base_route_, route_transports);
  log(log_lvl::info, "loader.virtual_locations", "{} routes -> {}", n_routes,
      tt.n_routes());
}

bool add_virtual_locations(
    timetable& tt,
    shapes_storage* shapes,
    mutable_fws_multimap<location_idx_t, footpath>& durations) {
  if (tt.transfer_rules_.rules_.empty()) {
    return false;
  }
  auto const timer = scoped_timer{"loader.virtual_locations"};
  auto const sigs = get_signatures(tt);
  auto const durations_of = get_transfer_durations(tt, sigs);
  auto pairs = get_route_stop_pairs(tt, sigs, durations_of);
  auto const routes = split_routes(tt, durations_of, pairs);
  auto const locations = merge_locations(durations_of, pairs);
  auto const n_locations = tt.n_locations();
  auto const virts = add_virts(tt, pairs, locations);
  if (tt.n_locations() == n_locations) {
    return false;
  }
  add_footpaths(tt, durations_of, pairs, locations, virts, durations);
  write_routes(tt, sigs, pairs, routes, locations, virts, shapes);
  return true;
}

}  // namespace nigiri::loader
