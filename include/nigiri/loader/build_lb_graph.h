#pragma once

#include "utl/pairwise.h"

#include "nigiri/logging.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"
#include <algorithm>
#include <vector>

namespace nigiri::loader {

static constexpr auto const kEnableCh = true;

struct departure {
  bool operator<(departure const& o) const { return dep_ < o.dep_; }
  location_idx_t to_;
  delta dep_;
  delta arr_;
  route_idx_t r_;
};

struct arrival {
  duration_t min_;
  duration_t max_;
  delta last_dep_;
  hash_set<route_idx_t> routes_;
};


template <direction SearchDir>
void build_lb_graph(timetable& tt, profile_idx_t const prf_idx) {
  hash_map<location_idx_t, duration_t> weights;
  hash_map<location_idx_t, arrival> arrivals;
  std::vector<departure> departures;
  hash_map<std::pair<location_idx_t, location_idx_t>, size_t> edges_map;
  paged_vecvec<size_t, route_idx_t> routes;

  auto const update_weight = [&](location_idx_t const target,
                                 duration_t const d) {
    if (auto const it = weights.find(target); it != end(weights)) {
      it->second = std::min(it->second, d);
    } else {
      weights.emplace_hint(it, target, d);
    }
  };

  auto const compute_ch_edges = [&](location_idx_t const from_l) {
    std::sort(begin(departures), end(departures));
    for (auto& dep : departures) {
      auto const dur = (dep.arr_ - dep.dep_).as_duration();
      if (auto const it = arrivals.find(dep.to_); it != end(arrivals)) {
        it->second.min_ = std::min(it->second.min_, dur);
        it->second.max_ = std::max(
            it->second.max_, (dep.arr_ - it->second.last_dep_).as_duration());
        it->second.last_dep_ = dep.dep_;  // TODO overtaking connections
        it->second.routes_.emplace(dep.r_);
      } else {
        arrivals.emplace_hint(it, dur, dep.dep_, dep.r_);
      }
    }
    departures.clear();
    for (auto const& entry : arrivals) {
      edges_map.emplace(from_l, entry.first,
                        tt.fwd_search_ch_graph_[prf_idx].at(from_l).size());
                        auto const list_idx = tt.ch_graph_transfers_[prf_idx].size();
      tt.fwd_search_ch_graph_[prf_idx].at(from_l).push_back(
          {entry.first, entry.second.min_, entry.second.max_,
           list_idx});
      tt.bwd_search_ch_graph_[prf_idx]
          .at(entry.first)
          .push_back({from_l, entry.second.min_, entry.second.max_,
            list_idx});
      tt.ch_graph_transfers_[prf_idx].emplace_back_empty();
      routes.emplace_back_empty();
      for (auto r : entry.second.routes_) {
        routes.at(list_idx).push_back(r); // TODO direct insertion?
      }
    }
  };

  auto const create_ch_edge = [&](nigiri::timetable::ch_edge const& dep, nigiri::timetable::ch_edge const& arr) {
    auto const routes_intersection = std::set_intersection(routes[dep.transfer_list_idx_], routes[arr.transfer_list_idx_])

  };

  auto const compute_ch =
      [&]() {
        auto location_ids = std::vector<location_idx_t>{100};
        std::iota(begin(location_ids), end(location_ids), location_idx_t{0});
        std::sort(begin(location_ids), end(location_ids), [&](auto a, auto b) {
          return tt.location_routes_[a].size() < tt.location_routes_[b].size();
        });
        auto location_levels = std::vector<size_t>{location_ids.size()};
        for (auto location_id : location_ids) {
          auto const& deps = tt.fwd_search_ch_graph_[prf_idx].at(location_id);
          auto const& arrs = tt.bwd_search_ch_graph_[prf_idx].at(location_id);
          for (auto dep : deps) {
            auto const to = dep.target_;
            if (location_levels[to.v_] > 0U) {
              continue;
            }
            for (auto arr : arrs) {
              auto const from = arr.target_;
              if (location_levels[from.v_] > 0U || from == to) {
                continue;
              }
              if (auto const it = edges_map.find({from, to});
                  it != end(edges_map)) {
                auto const& ch_edge =
                    tt.fwd_search_ch_graph_[prf_idx].at(from).at(it->second);
                auto const min_dur = dep.min_dur_ + arr.min_dur_;
                auto const max_dur = dep.max_dur_ + arr.max_dur_;
                if (max_dur < ch_edge.min_dur_) {
                  // replace

                } else if (min_dur < ch_edge.max_dur_) {
                  // update
                }
              }
            }
          }
        }
      };

  auto const add_edges = [&](location_idx_t const l) {
    auto const parent_l = tt.locations_.get_root_idx(l);

    auto const& footpaths = SearchDir == direction::kForward
                                ? tt.locations_.footpaths_in_[prf_idx][l]
                                : tt.locations_.footpaths_out_[prf_idx][l];
    for (auto const& fp : footpaths) {
      auto const target = tt.locations_.get_root_idx(fp.target());
      if (target != parent_l) {
        update_weight(target, fp.duration());
      }
    }

    for (auto const& r : tt.location_routes_[l]) {
      if ((prf_idx == kCarProfile && !tt.has_car_transport(r)) ||
          (prf_idx == kBikeProfile && !tt.has_bike_transport(r))) {
        continue;
      }

      auto const location_seq = tt.route_location_seq_[r];
      for (auto const [from, to] : utl::pairwise(interval{
               stop_idx_t{0U}, static_cast<stop_idx_t>(location_seq.size())})) {
        auto const from_l = stop{location_seq[from]}.location_idx();
        auto const to_l = stop{location_seq[to]}.location_idx();

        if ((SearchDir == direction::kForward ? to_l : from_l) != l) {
          continue;
        }

        auto const target_l =
            (SearchDir == direction::kForward ? from_l : to_l);
        auto const target = tt.locations_.get_root_idx(target_l);
        if (target == parent_l) {
          continue;
        }

        auto min = duration_t{std::numeric_limits<duration_t::rep>::max()};
        for (auto const t : tt.route_transport_ranges_[r]) {
          auto const from_time = tt.event_mam(t, from, event_type::kDep);
          auto const to_time = tt.event_mam(t, to, event_type::kArr);
          if (kEnableCh) {
            departures.emplace_back(to_l, from_time, to_time, r);
          }
          min = std::min((to_time - from_time).as_duration(), min);
        }
        update_weight(target, min);
      }
    }
    compute_ch_edges(l);
  };

  auto const timer = scoped_timer{"nigiri.loader.lb"};
  std::vector<footpath> footpaths;
  auto& lb_graph = SearchDir == direction::kForward
                       ? tt.fwd_search_lb_graph_[prf_idx]
                       : tt.bwd_search_lb_graph_[prf_idx];
  for (auto i = location_idx_t{0U}; i != tt.locations_.ids_.size(); ++i) {
    if (tt.locations_.parents_[i] != location_idx_t::invalid()) {
      lb_graph.emplace_back(std::vector<footpath>{});
      continue;
    }

    for (auto const& c : tt.locations_.children_[i]) {
      add_edges(c);
      for (auto const& cc : tt.locations_.children_[c]) {
        add_edges(cc);
      }
    }
    add_edges(i);

    for (auto const& [target, duration] : weights) {
      footpaths.emplace_back(footpath{target, duration});
    }

    lb_graph.emplace_back(footpaths);

    footpaths.clear();
    weights.clear();

    compute_ch();
  }
}

}  // namespace nigiri::loader
