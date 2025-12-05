#pragma once

#include "nigiri/loader/dir.h"
#include "utl/helpers/algorithm.h"
#include "utl/insert_sorted.h"
#include "utl/pairwise.h"

#include "nigiri/common/dial.h"
#include "nigiri/for_each_meta.h"
#include "nigiri/logging.h"
#include "nigiri/routing/limits.h"
#include "nigiri/td_footpath.h"

#include "nigiri/timetable.h"
#include "nigiri/types.h"
#include <algorithm>
#include <vector>

namespace nigiri::loader {

static constexpr auto const kChMaxTravelTime =
    routing::kMaxTravelTime * 5;  // TODO

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

struct ch_stats {
  size_t inserts_{0};
  size_t replacements_{0};
  size_t updates_{0};
};

template <direction SearchDir>
void build_lb_graph(timetable& tt, profile_idx_t const prf_idx) {
  hash_map<location_idx_t, duration_t> weights;
  hash_map<location_idx_t, arrival> arrivals;
  std::vector<departure> departures;
  hash_map<std::pair<location_idx_t, location_idx_t>, ch_edge_idx_t> edges_map;
  vector_map<ch_edge_idx_t, std::vector<route_idx_t>> routes;
  vector_map<ch_edge_idx_t, std::vector<location_idx_t>> transfers;
  vector_map<location_idx_t, std::vector<ch_edge_idx_t>> fwd_search_ch_graph;
  vector_map<location_idx_t, std::vector<ch_edge_idx_t>> bwd_search_ch_graph;
  ch_stats stats;

  if (SearchDir == direction::kBackward && kEnableCh) {
    fwd_search_ch_graph.resize(tt.n_locations());
    bwd_search_ch_graph.resize(tt.n_locations());
  }

  auto const update_weight = [&](location_idx_t const target,
                                 duration_t const d) {
    if (auto const it = weights.find(target); it != end(weights)) {
      it->second = std::min(it->second, d);
    } else {
      weights.emplace_hint(it, target, d);
    }
  };

  auto const insert_ch_edge = [&](location_idx_t const from_l,
                                  location_idx_t const to_l,
                                  duration_t const min, duration_t const max,
                                  bool const defer_graph_insertion = false) {
    auto const edge_idx = ch_edge_idx_t{tt.ch_graph_edges_[prf_idx].size()};
    edges_map.emplace(std::pair{from_l, to_l}, edge_idx);
    tt.ch_graph_edges_[prf_idx].push_back({from_l, to_l, min, max});
    if (!defer_graph_insertion) {
      fwd_search_ch_graph.at(from_l).push_back(edge_idx);
      bwd_search_ch_graph.at(to_l).push_back(edge_idx);
    }
    transfers.emplace_back();
    routes.emplace_back();
    return edge_idx;
  };

  auto const upsert_ch_footpath_edge =
      [&](location_idx_t const from_l, location_idx_t const to_l,
          duration_t const min_dur, duration_t const max_dur,
          hash_set<route_idx_t> const& new_routes) {
        if (auto const it = edges_map.find({from_l, to_l});
            it != end(edges_map)) {
          auto& shortcut = tt.ch_graph_edges_[prf_idx].at(it->second);
          if (max_dur < shortcut.min_dur_) {
            // replace
            shortcut.min_dur_ = min_dur;
            shortcut.max_dur_ = max_dur;
            // TODO clear routes?
            routes.at(it->second).clear();
            for (auto r : new_routes) {
              routes.at(it->second).push_back(r);  // TODO direct insertion?
            }
            std::sort(begin(routes.at(it->second)), end(routes.at(it->second)));
            stats.replacements_++;
          } else if (min_dur <= shortcut.max_dur_) {
            // update
            shortcut.min_dur_ = std::min(shortcut.min_dur_, min_dur);
            shortcut.max_dur_ = std::min(shortcut.max_dur_, max_dur);

            for (auto r : new_routes) {
              utl::insert_sorted(routes.at(it->second),
                                 r);  // TODO direct insertion?
            }
            stats.updates_++;
          }
        } else {
          // insert
          auto const edge_idx = insert_ch_edge(from_l, to_l, min_dur, max_dur);
          for (auto r : new_routes) {
            routes.at(edge_idx).push_back(r);  // TODO direct insertion?
          }
          std::sort(begin(routes.at(edge_idx)), end(routes.at(edge_idx)));
          stats.inserts_++;
        }
      };

  auto const compute_initial_ch_edges = [&](location_idx_t const from_l) {
    std::sort(begin(departures), end(departures));
    for (auto& dep : departures) {
      auto const dur = (dep.arr_ - dep.dep_).as_duration();
      if (auto const it = arrivals.find(dep.to_); it != end(arrivals)) {
        it->second.min_ = std::min(it->second.min_, dur);
        it->second.max_ = std::max(
            it->second.max_,
            (dep.arr_.as_duration() - it->second.last_dep_.as_duration()));
        it->second.last_dep_ = dep.dep_;  // TODO overtaking connections
        it->second.routes_.emplace(dep.r_);
      } else {
        arrivals.emplace_hint(it, dep.to_,
                              arrival{dur, dur, dep.dep_, {dep.r_}});
      }
    }
    departures.clear();
    for (auto const& entry : arrivals) {
      upsert_ch_footpath_edge(from_l, entry.first, entry.second.min_,
                              entry.second.max_, entry.second.routes_);
      for (auto fp : tt.locations_.footpaths_out_[prf_idx].at(entry.first)) {
        upsert_ch_footpath_edge(
            from_l, fp.target(), entry.second.min_ + fp.duration(),
            entry.second.max_ + fp.duration(), entry.second.routes_);
      }
    }
    arrivals.clear();
  };

  auto const update_ch_shortcut = [&](location_idx_t contracted,
                                      ch_edge_idx_t const dep_idx,
                                      ch_edge_idx_t const arr_idx,
                                      ch_edge_idx_t shortcut_idx) {
    auto const& dep = tt.ch_graph_edges_[prf_idx].at(dep_idx);
    auto const& arr = tt.ch_graph_edges_[prf_idx].at(arr_idx);
    auto& shortcut = tt.ch_graph_edges_[prf_idx].at(shortcut_idx);

    std::vector<route_idx_t> routes_merged;
    std::set_intersection(begin(routes[dep_idx]), end(routes[dep_idx]),
                          begin(routes[arr_idx]), end(routes[arr_idx]),
                          std::back_inserter(routes_merged));

    std::vector<location_idx_t> transfers_union;
    std::set_union(begin(transfers[dep_idx]), end(transfers[dep_idx]),
                   begin(transfers[arr_idx]), end(transfers[arr_idx]),
                   std::back_inserter(transfers_union));

    auto max_dur =
        dep.min_dur_ +
        arr.max_dur_;  // TODO dwell time, slow/fast cross, one route set larger
    if (routes_merged.size() != routes[dep_idx].size() &&  // TODO ||
        routes_merged.size() != routes[arr_idx].size()) {
      utl::insert_sorted(transfers_union,
                         contracted);  // TODO use level because it will
                                       // automatically be sorted?
      routes_merged.clear();
      std::set_union(begin(routes[dep_idx]), end(routes[dep_idx]),
                     begin(routes[arr_idx]), end(routes[arr_idx]),
                     std::back_inserter(routes_merged));
      max_dur = dep.max_dur_ + arr.max_dur_;
    }
    shortcut.min_dur_ =
        std::min(dep.min_dur_ + arr.min_dur_, shortcut.min_dur_);
    shortcut.max_dur_ = std::min(max_dur, shortcut.max_dur_);
    /*utl::verify(shortcut.min_dur_.count() > 0, "weird 0 min {} {} {} {}",
                dep.min_dur_, arr.min_dur_, shortcut.min_dur_, shortcut_idx);*/
    utl::verify(shortcut.min_dur_.count() < kChMaxTravelTime.count(),
                "overfl 0 min {} {} {} {}", dep.min_dur_, arr.min_dur_,
                shortcut.min_dur_, shortcut_idx);
    /*utl::verify(shortcut.max_dur_.count() > 0, "weird 0 max {} {} {} {}",
                dep.max_dur_, arr.max_dur_, shortcut.max_dur_, shortcut_idx);*/
    utl::verify(shortcut.max_dur_.count() < kChMaxTravelTime.count(),
                "overfl 0 max {} {} {} {}", dep.max_dur_, arr.max_dur_,
                shortcut.max_dur_, shortcut_idx);

    std::vector<route_idx_t> routes_new;
    std::set_union(begin(routes_merged), end(routes_merged),
                   begin(routes[shortcut_idx]), end(routes[shortcut_idx]),
                   std::back_inserter(routes_new));  // TODO inplace
    routes[shortcut_idx] = std::move(routes_new);

    std::vector<location_idx_t> transfers_new;
    std::set_union(begin(transfers_union), end(transfers_union),
                   begin(transfers[shortcut_idx]), end(transfers[shortcut_idx]),
                   std::back_inserter(transfers_new));  // TODO inplace
    transfers[shortcut_idx] = std::move(transfers_new);
  };

  auto const compute_ch = [&]() {
    std::cout << "original inserts: " << stats.inserts_
              << " replacements: " << stats.replacements_
              << " updates: " << stats.updates_ << std::endl;
    auto location_ids = std::vector<location_idx_t>{tt.n_locations()};
    std::iota(begin(location_ids), end(location_ids), location_idx_t{0});
    std::sort(begin(location_ids), end(location_ids), [&](auto a, auto b) {
      return fwd_search_ch_graph[a].size() < fwd_search_ch_graph[b].size();
    });
    tt.ch_levels_.resize(static_cast<unsigned>(location_ids.size()));
    auto level = 0U;
    auto write_ahead_edges = std::vector<ch_edge_idx_t>{};
    for (auto location_id : location_ids) {
      if (level > 0.8 * tt.n_locations()) {  // TODO
        tt.ch_levels_.at(location_id) = level;
        continue;
      }
      ++level;
      auto const& deps = fwd_search_ch_graph.at(location_id);
      auto const& arrs = bwd_search_ch_graph.at(location_id);
      for (auto dep_idx : deps) {
        auto const to = tt.ch_graph_edges_[prf_idx].at(dep_idx).to_;
        if (tt.ch_levels_.at(to) > 0U) {
          continue;
        }
        for (auto arr_idx : arrs) {
          auto const& dep = tt.ch_graph_edges_[prf_idx].at(dep_idx);
          auto const& arr = tt.ch_graph_edges_[prf_idx].at(arr_idx);
          auto const from = arr.from_;
          if (tt.ch_levels_.at(from) > 0U || from == to) {
            continue;
          }
          if (auto const it = edges_map.find({from, to});
              it != end(edges_map)) {
            auto& shortcut = tt.ch_graph_edges_[prf_idx].at(it->second);
            auto const min_dur = dep.min_dur_ + arr.min_dur_;
            auto const max_dur = dep.max_dur_ + arr.max_dur_;
            if (max_dur < shortcut.min_dur_) {
              // replace
              shortcut.min_dur_ = duration_t::max();
              shortcut.max_dur_ = duration_t::max();
              routes.at(it->second).clear();
              transfers.at(it->second).clear();
              update_ch_shortcut(location_id, dep_idx, arr_idx, it->second);
              stats.replacements_++;
            } else if (min_dur <= shortcut.max_dur_) {
              // update
              update_ch_shortcut(location_id, dep_idx, arr_idx, it->second);
              stats.updates_++;
            }
          } else {
            // insert
            auto const edge_idx = insert_ch_edge(from, to, duration_t::max(),
                                                 duration_t::max(), true);
            update_ch_shortcut(location_id, dep_idx, arr_idx, edge_idx);
            write_ahead_edges.push_back(edge_idx);
            stats.inserts_++;
          }
        }
      }
      tt.ch_levels_.at(location_id) = level;
      if (level % 100 == 0) {
        std::cout << level << " " << deps.size() << " " << arrs.size() << " "
                  << location_id << " "
                  << tt.locations_.names_[location_id].view() << std::endl;
        std::cout << "inserts: " << stats.inserts_
                  << " replacements: " << stats.replacements_
                  << " updates: " << stats.updates_ << std::endl;
        std::cout << "edges: " << tt.ch_graph_edges_[prf_idx].size()
                  << " stations: " << tt.n_locations()
                  << " transfers size: " << std::accumulate(std::next(transfers.begin()), transfers.end(),
                  0, 
                  [](auto const& a, auto const& b)
                  {
                      return a + b.size();
                  }) << std::endl;
      }
      for (auto const e_idx : write_ahead_edges) {
        fwd_search_ch_graph.at(tt.ch_graph_edges_[prf_idx].at(e_idx).from_)
            .push_back(e_idx);
        bwd_search_ch_graph.at(tt.ch_graph_edges_[prf_idx].at(e_idx).to_)
            .push_back(e_idx);
      }
      write_ahead_edges.clear();
    }
    std::cout << "persisting..." << std::endl;
    auto transfer_count = 0U;
    for (auto t : transfers) {
      transfer_count += t.size();
      tt.ch_graph_transfers_[prf_idx].emplace_back(std::move(t));
    }
    for (auto i = location_idx_t{0U}; i != tt.locations_.ids_.size(); ++i) {
      tt.fwd_search_ch_graph_[prf_idx].emplace_back(
          std::move(fwd_search_ch_graph[i]));
      tt.bwd_search_ch_graph_[prf_idx].emplace_back(
          std::move(bwd_search_ch_graph[i]));
    }
    fwd_search_ch_graph.clear();
    bwd_search_ch_graph.clear();
    std::cout << "inserts: " << stats.inserts_
              << " replacements: " << stats.replacements_
              << " updates: " << stats.updates_ << std::endl;
    std::cout << "edges: " << tt.ch_graph_edges_[prf_idx].size()
              << " stations: " << tt.n_locations()
              << " transfers: " << transfer_count << std::endl;
    std::cout << "num edges after contr: " << tt.ch_graph_edges_[prf_idx].size()
              << std::endl;
    if (tt.ch_graph_edges_[prf_idx].size() > 0) {
      std::cout
          << "max max: "
          << utl::max_element(tt.ch_graph_edges_[prf_idx],
                              [](auto const& a, auto const& b) {
                                return a.max_dur_ < b.max_dur_;
                              })
                 ->max_dur_
          << " max min: "
          << utl::max_element(tt.ch_graph_edges_[prf_idx],
                              [](auto const& a, auto const& b) {
                                return a.min_dur_ < b.min_dur_;
                              })
                 ->min_dur_
          << " Lorem ipsum dolor sit amet, \n consetetur sadipscing elitr, \n "
             "sed diam nonumy eirmod tempor invidunt \n ut labore et dolore "
             "magna aliquyam erat, \n sed diam voluptua. At vero eos\n  et "
             "accusam et justo duo dolores et ea rebum. Stet clita kasd\n  "
             "gubergren, no sea takimata sanctus est Lorem ipsum dolor sit "
             "amet. Lorem ipsum dolor\n  sit amet, consetetur sadipscing "
             "elitr, sed \n diam nonumy eirmod tempor invidunt ut labore et "
             "dolore magna aliquyam erat, sed diam voluptua. At vero eos et "
             "accusam et justo duo dolores et ea rebum. Stet clita kasd "
             "gubergren, no sea takimata sanctus est Lorem ipsum dolor sit "
             "amet."
          << std::endl;
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
        if (target == parent_l &&
            (SearchDir != direction::kBackward || !kEnableCh)) {
          continue;
        }

        auto min = duration_t::max();
        for (auto const t : tt.route_transport_ranges_[r]) {
          auto const from_time = tt.event_mam(t, from, event_type::kDep);
          auto const to_time = tt.event_mam(t, to, event_type::kArr);
          if (SearchDir == direction::kBackward && kEnableCh) {
            departures.emplace_back(to_l, from_time, to_time, r);
          }
          min = std::min((to_time - from_time).as_duration(), min);
        }
        if (target != parent_l) {
          update_weight(target, min);
        }
      }
    }
    if (SearchDir == direction::kBackward && kEnableCh) {
      compute_initial_ch_edges(l);
    }
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
  }
  if (SearchDir == direction::kBackward && kEnableCh) {
    std::cout << "num edges: " << tt.ch_graph_edges_[prf_idx].size()
              << std::endl;
    if (tt.ch_graph_edges_[prf_idx].size() > 0) {
      std::cout << "max max: "
                << utl::max_element(tt.ch_graph_edges_[prf_idx],
                                    [](auto const& a, auto const& b) {
                                      return a.max_dur_ < b.max_dur_;
                                    })
                       ->max_dur_
                << " max min: "
                << utl::max_element(tt.ch_graph_edges_[prf_idx],
                                    [](auto const& a, auto const& b) {
                                      return a.min_dur_ < b.min_dur_;
                                    })
                       ->min_dur_
                << std::endl;
    }
    std::cout << "lb done." << std::endl;
    compute_ch();

    /*tt.ch_levels_.resize(static_cast<unsigned>(tt.n_locations()));
    for (auto t : transfers) {
      tt.ch_graph_transfers_[prf_idx].emplace_back(std::move(t));
    }*/
    std::cout << "ch done." << std::endl;
  }
}

}  // namespace nigiri::loader
