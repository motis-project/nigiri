#include "nigiri/loader/virt/merge_locations.h"

#include <numeric>
#include <ranges>
#include <vector>

#include "utl/helpers/algorithm.h"

namespace nigiri::loader {

constexpr auto const kAnyDuration =
    window_t{duration_t::min(), duration_t::max()};

vector_map<route_stop_idx_t, route_stop_idx_t> merge_locations(
    hash_map<transfer_pair, transfer_durations> const& durations_of,
    route_stop_pairs& pairs) {
  auto rs_pairs =
      vector_map<route_stop_idx_t, std::vector<route_stop_pair_idx_t>>(
          pairs.route_stops_.size());
  for (auto p = route_stop_pair_idx_t{0U}; p != pairs.pairs_.size(); ++p) {
    auto const [from, to] = pairs.pairs_[p];
    rs_pairs[from].push_back(p);
    if (from != to) {
      rs_pairs[to].push_back(p);
    }
  }

  // route_stop -> joined route_stop (itself if not joined)
  // Start with identity x->x.
  auto location =
      vector_map<route_stop_idx_t, route_stop_idx_t>(pairs.route_stops_.size());
  std::iota(begin(location), end(location), route_stop_idx_t{0U});

  struct pair_window {
    route_stop_pair pair_;
    window_t window_;
  };
  auto src_pairs = std::vector<pair_window>{};
  auto updates = std::vector<pair_window>{};
  auto const join = [&](route_stop_idx_t const src,
                        route_stop_idx_t const dst) {
    updates.clear();
    src_pairs.clear();

    // Pretend src == dst (src joined dst).
    // -> src_pairs are updated accordingly
    // (considering all joined route stops)
    auto const location_of = [&](route_stop_idx_t const x) {
      return x == src ? dst : location[x];
    };
    for (auto const p : rs_pairs[src]) {
      auto const [from, to] = pairs.pairs_[p];
      src_pairs.push_back({route_stop_pair{location_of(from), location_of(to)},
                           pairs.windows_[p]});
    }

    // For each pair, intersect their intervals.
    utl::sort(src_pairs, [](auto&& a, auto&& b) { return a.pair_ < b.pair_; });
    for (auto const run :
         src_pairs | std::views::chunk_by([](auto&& a, auto&& b) {
           return a.pair_ == b.pair_;
         })) {
      auto const x = run.front().pair_;
      auto const existing = pairs.pair_of_.find(x);
      auto w = existing == end(pairs.pair_of_)
                   ? kAnyDuration
                   : pairs.windows_[existing->second];
      for (auto const& u : run) {
        w = w.intersect(u.window_);
      }
      auto const is_ok =
          pairs.is_base(x.from_) && pairs.is_base(x.to_)
              ? w.contains(
                    durations_of
                        .at(transfer_pair{pairs.route_stops_[x.from_].base_,
                                          pairs.route_stops_[x.to_].base_})
                        .fallback_)
              : is_admissible(w);
      if (!is_ok) {
        return false;
      }
      updates.push_back({x, w});
    }

    // Join successful.
    // -> write updated windows + assign route stop src to dst.
    for (auto const& [x, w] : updates) {
      pairs.windows_[pairs.get_or_create_pair(x.from_, x.to_, w)] = w;
    }
    location[src] = dst;

    return true;
  };

  auto base_virts = hash_map<location_idx_t, std::vector<route_stop_idx_t>>{};
  for (auto s = route_stop_idx_t{0U}; s != pairs.route_stops_.size(); ++s) {
    if (pairs.is_base(s)) {
      continue;
    }

    // Try to join base stops first, then any existing virt.
    auto const base = pairs.route_stops_[s].base_;
    auto& virts = base_virts[base];
    if (!join(s, pairs.route_stop_of_.at(to_base_stop(base))) &&
        utl::none_of(virts,
                     [&](route_stop_idx_t const v) { return join(s, v); })) {
      virts.push_back(s);
    }
  }

  return location;
}

}  // namespace nigiri::loader
