#pragma once

#include <cstdint>
#include <algorithm>
#include <limits>
#include <vector>

#include "nigiri/loader/dir.h"
#include "utl/helpers/algorithm.h"
#include "utl/insert_sorted.h"
#include "utl/pairwise.h"
#include "utl/pipes/avg.h"

#include "nigiri/common/dial.h"
#include "nigiri/for_each_meta.h"
#include "nigiri/logging.h"
#include "nigiri/routing/dijkstra.h"
#include "nigiri/routing/limits.h"
#include "nigiri/td_footpath.h"

#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::loader {

static constexpr auto const kChMaxTravelTime =
    routing::kMaxTravelTime * 5;  // TODO

static constexpr auto const kEnableCh = true;

struct tooth {
  std::int16_t mam_;
  u16_minutes travel_dur_;
};

struct saw {
  std::vector<tooth> saw_;
  u16_minutes max() {
    auto max = saw_.front().mam_ + 24 * 60 - saw_.back().mam_ +
               saw_.front().travel_dur_.count();
    for (auto i = 1U; i < saw_.size(); ++i) {
      max = std::max(
          max, saw_[i].mam_ - saw_[i - 1].mam_ + saw_[i].travel_dur_.count());
    }
    return u16_minutes{max};
  }
  void simplify() {  // TODO insitu
    auto tmp = std::vector<tooth>{};
    auto last_duration_of_day = saw_.back().travel_dur_;
    for (auto i = 0U; i < saw_.size(); ++i) {
      if (saw_.back().mam_ + saw_.back().travel_dur_.count() - 24 * 60 <
          saw_[i].mam_) {
        break;
      }
      last_duration_of_day =
          std::min(last_duration_of_day,
                   u16_minutes{saw_[i].travel_dur_.count() + 24 * 60 -
                               saw_.back().mam_ + saw_[i].mam_});
    }
    tmp.push_back({saw_.back().mam_, last_duration_of_day});

    for (auto i = 1U; i < saw_.size(); ++i) {
      auto j = saw_.size() - i - 1U;
      if (tmp.back().mam_ - saw_[j].mam_ + tmp.back().travel_dur_.count() >
          saw_[j].travel_dur_.count()) {
        tmp.push_back(saw_[j]);
      }
    }
    std::reverse(begin(tmp), end(tmp));
    saw_ = std::move(tmp);
  }
};

struct departure {
  bool operator<(departure const& o) const { return dep_.mam_ < o.dep_.mam_; }
  location_idx_t to_;
  delta dep_;
  delta arr_;
  route_idx_t r_;
  transport_idx_t t_;
};

struct arrival {
  u16_minutes min_;
  std::vector<delta> deps_;
  std::vector<u16_minutes> travel_durs_;
  std::vector<bitfield_idx_t> traffic_days_;
  hash_set<route_idx_t> routes_;
};

struct ch_stats {
  std::int64_t inserts_{0};
  std::int64_t direct_inserts_{0};
  std::int64_t replacements_{0};
  std::int64_t good_updates_{0};
  std::int64_t bad_updates_{0};
  std::int64_t skips_{0};
  std::int64_t contracted_neighbors_{0};
};

template <direction SearchDir>
void build_lb_graph(timetable& tt, profile_idx_t const prf_idx) {
  hash_map<location_idx_t, duration_t> weights;
  hash_map<location_idx_t, arrival> arrivals;
  std::vector<departure> departures;
  hash_map<std::pair<location_idx_t, location_idx_t>, ch_edge_idx_t> edges_map;
  vector_map<ch_edge_idx_t, std::vector<route_idx_t>> routes;
  vector_map<ch_edge_idx_t,
             std::vector<std::pair<ch_edge_idx_t, ch_edge_idx_t>>>
      unpack;
  vector_map<ch_edge_idx_t, std::vector<location_idx_t>>
      transfers;  // TODO use bool and explicitly insert low level footpaths?
  vector_map<location_idx_t, std::vector<ch_edge_idx_t>> fwd_search_ch_graph;
  vector_map<location_idx_t, std::vector<ch_edge_idx_t>> bwd_search_ch_graph;
  ch_stats stats;
  auto ignore_timetable_offset_mask = ~bitfield{};
  for (auto i = 0U; i < kTimetableOffset.count(); ++i) {
    ignore_timetable_offset_mask.set(i, false);
  }
  auto to_routes = bitvec{};

  if (SearchDir == direction::kBackward && kEnableCh) {
    fwd_search_ch_graph.resize(tt.n_locations());
    bwd_search_ch_graph.resize(tt.n_locations());
    to_routes.resize((tt.n_routes()));
  }

  auto const update_weight = [&](location_idx_t const target,
                                 duration_t const d) {
    if (auto const it = weights.find(target); it != end(weights)) {
      it->second = std::min(it->second, d);
    } else {
      weights.emplace_hint(it, target, d);
    }
  };

  auto const find_direct_trips = [&](location_idx_t const a,
                                     location_idx_t const b) {
    // TODO parent mode
    for (auto const& r : tt.location_routes_[b]) {
      to_routes.set(r.v_);
    }

    for (auto const& r : tt.location_routes_[a]) {
      if ((prf_idx == kCarProfile && !tt.has_car_transport(r)) ||
          (prf_idx == kBikeProfile && !tt.has_bike_transport(r))) {
        continue;
      }
      if (!to_routes.test(r.v_)) {
        continue;
      }

      auto const location_seq = tt.route_location_seq_[r];
      auto from_stop_idx = std::numeric_limits<stop_idx_t>::max();
      for (auto const [from, to] : utl::pairwise(interval{
               stop_idx_t{0U}, static_cast<stop_idx_t>(location_seq.size())})) {
        auto const from_l = stop{location_seq[from]}.location_idx();
        auto const to_l = stop{location_seq[to]}.location_idx();

        if (from_stop_idx == std::numeric_limits<stop_idx_t>::max() &&
            from_l != a) {
          continue;
        }
        if (from_l == a) {
          from_stop_idx = from;
        }
        if (to_l != b) {
          continue;
        }

        for (auto const t : tt.route_transport_ranges_[r]) {
          auto const from_time =
              tt.event_mam(t, from_stop_idx, event_type::kDep);
          auto const to_time = tt.event_mam(t, to, event_type::kArr);
          departures.emplace_back(to_l, from_time, to_time, r, t);
        }

        from_stop_idx = std::numeric_limits<stop_idx_t>::max();
      }
    }
    to_routes.zero_out();
  };

  auto const insert_ch_edge = [&](location_idx_t const from_l,
                                  location_idx_t const to_l,
                                  u16_minutes const min, u16_minutes const max,
                                  bool const defer_graph_insertion = false) {
    auto const edge_idx = ch_edge_idx_t{tt.ch_graph_edges_[prf_idx].size()};
    edges_map.emplace(std::pair{from_l, to_l}, edge_idx);
    tt.ch_graph_edges_[prf_idx].push_back({from_l, to_l, min, max});
    if (!defer_graph_insertion) {
      fwd_search_ch_graph.at(from_l).push_back(edge_idx);
      bwd_search_ch_graph.at(to_l).push_back(edge_idx);
    }
    unpack.emplace_back();
    transfers.emplace_back();
    routes.emplace_back();
    return edge_idx;
  };

  auto const upsert_ch_footpath_edge =
      [&](location_idx_t const from_l, location_idx_t const to_l,
          u16_minutes const min_dur, u16_minutes const max_dur,
          hash_set<route_idx_t> const& new_routes, location_idx_t transfer) {
        if (auto const it = edges_map.find({from_l, to_l});
            it != end(edges_map)) {
          auto& shortcut = tt.ch_graph_edges_[prf_idx].at(it->second);
          if (max_dur < shortcut.min_dur_) {
            // replace
            shortcut.min_dur_ = min_dur;
            shortcut.max_dur_ = max_dur;
            utl::verify(shortcut.max_dur_.count() > 2, "weird 0 max {} {} {}",
                        max_dur, shortcut.max_dur_, it->second);
            utl::verify(shortcut.max_dur_.count() < kChMaxTravelTime.count(),
                        "overfl 0 max {} {} {}", max_dur, shortcut.max_dur_,
                        it->second);
            // TODO clear routes?
            routes.at(it->second).clear();
            unpack.at(it->second).clear();
            transfers.at(it->second).clear();
            for (auto r : new_routes) {
              routes.at(it->second).push_back(r);  // TODO direct insertion?
            }
            std::sort(begin(routes.at(it->second)), end(routes.at(it->second)));
            unpack.at(it->second)
                .push_back(
                    {ch_edge_idx_t::invalid(), ch_edge_idx_t::invalid()});
            transfers.at(it->second).push_back(transfer);
            stats.replacements_++;
          } else if (min_dur <= shortcut.max_dur_) {
            // update
            shortcut.min_dur_ = std::min(shortcut.min_dur_, min_dur);
            shortcut.max_dur_ = std::min(shortcut.max_dur_, max_dur);
            utl::verify(shortcut.max_dur_.count() > 2, "weird 0 max {} {} {}",
                        max_dur, shortcut.max_dur_, it->second);
            utl::verify(shortcut.max_dur_.count() < kChMaxTravelTime.count(),
                        "overfl 0 max {} {} {}", max_dur, shortcut.max_dur_,
                        it->second);

            for (auto r : new_routes) {
              utl::insert_sorted(routes.at(it->second),
                                 r);  // TODO direct insertion?
            }
            unpack.at(it->second)
                .push_back(
                    {ch_edge_idx_t::invalid(), ch_edge_idx_t::invalid()});
            transfers.at(it->second).push_back(transfer);
            if (max_dur < shortcut.max_dur_) {
              ++stats.good_updates_;
            } else {
              ++stats.bad_updates_;
            }
          } else {
            stats.skips_++;
          }
        } else {
          // insert
          auto const edge_idx = insert_ch_edge(from_l, to_l, min_dur, max_dur);
          for (auto r : new_routes) {
            routes.at(edge_idx).push_back(r);  // TODO direct insertion?
          }
          std::sort(begin(routes.at(edge_idx)), end(routes.at(edge_idx)));
          unpack.at(edge_idx).push_back(
              {ch_edge_idx_t::invalid(), ch_edge_idx_t::invalid()});
          transfers.at(edge_idx).push_back(transfer);
          stats.inserts_++;
        }
      };

  auto const assign_departures_to_arrivals = [&]() {
    std::sort(begin(departures), end(departures));
    for (auto const& dep : departures) {
      auto const dur = (dep.arr_ - dep.dep_).as_duration();
      if (auto const it = arrivals.find(dep.to_); it != end(arrivals)) {
        it->second.min_ =
            std::min(it->second.min_, static_cast<u16_minutes>(dur));
        it->second.deps_.push_back(dep.dep_);
        it->second.travel_durs_.push_back(dur);
        it->second.traffic_days_.push_back(tt.transport_traffic_days_[dep.t_]);
        it->second.routes_.emplace(dep.r_);
      } else {
        arrivals.emplace_hint(
            it, dep.to_,
            arrival{dur,
                    {dep.dep_},
                    {dur},
                    {tt.transport_traffic_days_[dep.t_]},
                    {dep.r_}});  // TODO overnight waiting time to arr
      }
    }
    departures.clear();
  };

  auto const arrivals_to_saw = [&](arrival const& e, saw& max_saw) {
    for (auto i = 0UL; i < e.deps_.size(); ++i) {
      max_saw.saw_.push_back(
          {static_cast<std::int16_t>(e.deps_[i].mam_), e.travel_durs_[i]});
    }
    for (auto i = 0UL; i < e.deps_.size(); ++i) {

      auto remaining_traffic_days = tt.bitfields_.at(e.traffic_days_[i])
                                    << static_cast<unsigned>(e.deps_[i].days());
      remaining_traffic_days &= ignore_timetable_offset_mask;
      auto day_offset = 0;
      auto j = i;
      while (true) {
        if (j == 0) {
          j = e.deps_.size() - 1;
          ++day_offset;
          if (day_offset > routing::kMaxTravelTime / 1_days) {
            break;
          }
          remaining_traffic_days >>= 1U;
          remaining_traffic_days.set(kTimetableOffset.count() - 1, false);
        } else {
          --j;
        }

        remaining_traffic_days &= ~tt.bitfields_.at(e.traffic_days_[j])
                                  << static_cast<unsigned>(e.deps_[j].days());
        if (remaining_traffic_days.none()) {
          break;
        }

        auto const mam_diff =
            e.deps_[i].mam_ - e.deps_[j].mam_ + day_offset * 24 * 60;
        max_saw.saw_[j].travel_dur_ =
            std::max(max_saw.saw_[j].travel_dur_,
                     u16_minutes{e.travel_durs_[i].count() + mam_diff});
      }
    }
    max_saw.simplify();
  };

  auto const compute_initial_ch_edges = [&](location_idx_t const from_l) {
    assign_departures_to_arrivals();

    auto max_saw = saw{};
    for (auto& entry : arrivals) {
      auto const& e = entry.second;
      arrivals_to_saw(e, max_saw);
      auto max = max_saw.max();
      // if (i % 50 == 0) std::cout << "init max saw " << max << std::endl;
      upsert_ch_footpath_edge(from_l, entry.first, e.min_, max, e.routes_,
                              location_idx_t::invalid());
      for (auto fp : tt.locations_.footpaths_out_[prf_idx].at(entry.first)) {
        upsert_ch_footpath_edge(  // TODO transfer?
            from_l, fp.target(), e.min_ + fp.duration(), max + fp.duration(),
            e.routes_, entry.first);
      }
      max_saw.saw_.clear();
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

    utl::verify(shortcut.max_dur_.count() > 2, "pre weird 0 max {} {} {} {}",
                dep.max_dur_, arr.max_dur_, shortcut.max_dur_, shortcut_idx);

    shortcut.min_dur_ =
        std::min(dep.min_dur_ + arr.min_dur_, shortcut.min_dur_);
    shortcut.max_dur_ =
        std::min(dep.max_dur_ + arr.max_dur_, shortcut.max_dur_);
    /*utl::verify(shortcut.min_dur_.count() >= 0, "weird 0 min {} {} {} {}",
                dep.min_dur_, arr.min_dur_, shortcut.min_dur_, shortcut_idx);*/
    utl::verify(shortcut.min_dur_.count() < kChMaxTravelTime.count(),
                "overfl 0 min {} {} {} {}", dep.min_dur_, arr.min_dur_,
                shortcut.min_dur_, shortcut_idx);
    utl::verify(shortcut.max_dur_.count() > 2, "weird 0 max {} {} {} {}",
                dep.max_dur_, arr.max_dur_, shortcut.max_dur_, shortcut_idx);
    utl::verify(shortcut.max_dur_.count() < kChMaxTravelTime.count(),
                "overfl 0 max {} {} {} {}", dep.max_dur_, arr.max_dur_,
                shortcut.max_dur_, shortcut_idx);

    unpack[shortcut_idx].push_back({arr_idx, dep_idx});
    transfers[shortcut_idx].push_back(contracted);
  };

  auto const contract_ch_node =
      [&](location_idx_t const location_id,
          std::vector<ch_edge_idx_t>& write_ahead_edges,
          ch_stats& contract_stats, bool const dry_run = false) {
        auto const& deps = fwd_search_ch_graph.at(location_id);
        auto const& arrs = bwd_search_ch_graph.at(location_id);
        for (auto dep_idx : deps) {
          auto const to = tt.ch_graph_edges_[prf_idx].at(dep_idx).to_;
          if (tt.ch_levels_.at(to) > 0U) {
            ++contract_stats.contracted_neighbors_;
            continue;
          }
          for (auto arr_idx : arrs) {
            auto const from = tt.ch_graph_edges_[prf_idx].at(arr_idx).from_;
            if (tt.ch_levels_.at(from) > 0U) {
              ++contract_stats.contracted_neighbors_;
              continue;
            }
            if (from == to) {
              continue;
            }
            auto edge_idx = ch_edge_idx_t::invalid();
            if (auto const it = edges_map.find({from, to});
                it != end(edges_map)) {
              edge_idx = it->second;
            } else {
              // insert TODO weigh differently in ordering?
              ++contract_stats.inserts_;
              if (dry_run) {
                continue;
              }
              find_direct_trips(from, to);
              if (!departures.empty()) {
                ++contract_stats.direct_inserts_;
              }
              assign_departures_to_arrivals();
              auto max_saw = saw{};
              utl::verify(
                  arrivals.size() <= 1,
                  "more than one direct relation found between a and b");

              if (arrivals.size() == 1) {
                auto const& e = begin(arrivals)->second;
                arrivals_to_saw(e, max_saw);
                auto max = max_saw.max();
                std::cout << "max saw " << max << std::endl;
                max_saw.saw_.clear();
                arrivals.clear();
                edge_idx = insert_ch_edge(from, to, e.min_, max, true);
                unpack.at(edge_idx).push_back(
                    {ch_edge_idx_t::invalid(), ch_edge_idx_t::invalid()});
                transfers.at(edge_idx).push_back(location_idx_t::invalid());
              } else {
                arrivals.clear();
                edge_idx = insert_ch_edge(from, to, u16_minutes::max(),
                                          u16_minutes::max(), true);
              }
              write_ahead_edges.push_back(edge_idx);
            }
            auto& shortcut = tt.ch_graph_edges_[prf_idx].at(edge_idx);
            auto const& dep = tt.ch_graph_edges_[prf_idx].at(dep_idx);
            auto const& arr = tt.ch_graph_edges_[prf_idx].at(arr_idx);
            auto const min_dur = dep.min_dur_ + arr.min_dur_;
            auto const max_dur = dep.max_dur_ + arr.max_dur_;
            if (max_dur < shortcut.min_dur_) {
              // replace
              ++contract_stats.replacements_;
              if (dry_run) {
                continue;
              }
              shortcut.min_dur_ = u16_minutes::max();
              shortcut.max_dur_ = u16_minutes::max();
              routes.at(edge_idx).clear();
              unpack.at(edge_idx).clear();
              transfers.at(edge_idx).clear();
              update_ch_shortcut(location_id, dep_idx, arr_idx, edge_idx);
            } else if (min_dur <= shortcut.max_dur_) {
              // update
              if (max_dur < shortcut.max_dur_) {
                ++contract_stats.good_updates_;
              } else {
                ++contract_stats.bad_updates_;
              }
              if (dry_run) {
                continue;
              }
              update_ch_shortcut(location_id, dep_idx, arr_idx, edge_idx);
            } else {
              ++contract_stats.skips_;
            }
          }
        }
      };

  auto const update_node_order =
      [&](location_idx_t l, std::vector<ch_edge_idx_t>& write_ahead_edges,
          dial<routing::label, routing::get_bucket>& pq,
          vector_map<location_idx_t, routing::label::dist_t>& current_order) {
        auto contract_stats = ch_stats{};
        contract_ch_node(l, write_ahead_edges, contract_stats, true);
        auto const edges =
            static_cast<std::int64_t>(fwd_search_ch_graph.at(l).size() +
                                      bwd_search_ch_graph.at(l).size());
        auto const order = static_cast<routing::label::dist_t>(std::clamp(
            10000L + contract_stats.contracted_neighbors_ +
                contract_stats.inserts_ + contract_stats.bad_updates_ - edges -
                contract_stats.good_updates_ - contract_stats.replacements_ -
                contract_stats.skips_,
            0L, 20000L));  // TODO include direct inserts, max dur and/or
                           // transfers because depending on order, shortcuts
                           // will be replaced?
        if (current_order.at(l) != order) {
          pq.push(routing::label{l, order});
          current_order.at(l) = order;
        }
      };

  auto const update_neighbours_node_order =
      [&](location_idx_t l, std::vector<ch_edge_idx_t>& write_ahead_edges,
          dial<routing::label, routing::get_bucket>& pq,
          vector_map<location_idx_t, routing::label::dist_t>& current_order) {
        auto const& deps = fwd_search_ch_graph.at(l);
        for (auto dep_idx : deps) {
          auto const to = tt.ch_graph_edges_[prf_idx].at(dep_idx).to_;
          if (tt.ch_levels_.at(to) > 0U) {
            continue;
          }
          update_node_order(to, write_ahead_edges, pq, current_order);
        }
        auto const& arrs = bwd_search_ch_graph.at(l);
        for (auto arr_idx : arrs) {
          auto const from = tt.ch_graph_edges_[prf_idx].at(arr_idx).from_;
          if (tt.ch_levels_.at(from) > 0U) {
            continue;
          }
          update_node_order(from, write_ahead_edges, pq, current_order);
        }
      };

  auto const print_stats = [&]() {
    std::cout << "inserts: " << stats.inserts_
              << " direct inserts: " << stats.direct_inserts_
              << " replacements: " << stats.replacements_
              << " skips: " << stats.skips_
              << " good updates: " << stats.good_updates_
              << " bad updates: " << stats.bad_updates_ << std::endl;
  };
  auto const print_edge_stats = [&]() {
    std::cout << "num edges: " << tt.ch_graph_edges_[prf_idx].size()
              << std::endl;
    if (tt.ch_graph_edges_[prf_idx].size() > 0) {
      auto min_max = u16_minutes::max();
      auto max_max = u16_minutes::min();
      auto sum_max = 0;
      auto min_min = u16_minutes::max();
      auto max_min = u16_minutes::min();
      auto sum_min = 0;
      for (auto const& e : tt.ch_graph_edges_[prf_idx]) {
        if (e.max_dur_ > max_max) {
          max_max = e.max_dur_;
        }
        if (e.max_dur_ < min_max) {
          min_max = e.max_dur_;
        }
        sum_max += e.max_dur_.count();
        if (e.min_dur_ > max_min) {
          max_min = e.min_dur_;
        }
        if (e.min_dur_ < min_min) {
          min_min = e.min_dur_;
        }
        sum_min += e.min_dur_.count();
      }
      std::cout << " min max: " << min_max << " max max: " << max_max
                << " avg max: "
                << static_cast<unsigned>(sum_max) /
                       tt.ch_graph_edges_[prf_idx].size()
                << "\n"
                << " min min: " << min_min << " max min: " << max_min
                << " avg min: "
                << static_cast<unsigned>(sum_min) /
                       tt.ch_graph_edges_[prf_idx].size()
                << std::endl;
    }
  };

  auto const compute_ch = [&]() {
    print_stats();
    tt.ch_levels_.resize(static_cast<unsigned>(tt.n_locations()));
    auto pq = dial<routing::label, routing::get_bucket>{20001};
    auto current_order = vector_map<location_idx_t, routing::label::dist_t>{};
    current_order.resize(tt.n_locations());
    auto write_ahead_edges = std::vector<ch_edge_idx_t>{};
    std::cout << "initial node ordering..." << std::endl;
    for (auto l = location_idx_t{0}; l < tt.n_locations(); ++l) {
      update_node_order(l, write_ahead_edges, pq, current_order);
    }
    std::cout << "initial node ordering done" << std::endl;
    write_ahead_edges.clear();
    auto level = 0U;
    while (!pq.empty()) {
      auto const& label = pq.top();
      auto const location_id = label.l_;
      auto const order = label.d_;
      pq.pop();
      if (current_order.at(location_id) != order ||
          tt.ch_levels_.at(location_id) > 0U) {
        continue;
      }
      if (level > 0.98 * tt.n_locations()) {  // TODO
        tt.ch_levels_.at(location_id) = level;
        continue;
      }
      ++level;
      contract_ch_node(location_id, write_ahead_edges, stats);
      tt.ch_levels_.at(location_id) = level;
      if (level % 100 == 0) {
        std::cout << level << " " << fwd_search_ch_graph.at(location_id).size()
                  << " " << bwd_search_ch_graph.at(location_id).size() << " "
                  << location_id << " "
                  << tt.locations_.names_[location_id].view() << " " << order
                  << std::endl;
        print_stats();
        std::cout << "edges: " << tt.ch_graph_edges_[prf_idx].size()
                  << " stations: " << tt.n_locations() << " transfers size: "
                  << std::accumulate(std::next(transfers.begin()),
                                     transfers.end(), 0U,
                                     [](auto const& a, auto const& b) {
                                       return a + b.size();
                                     })
                  << std::endl;
      }
      for (auto const e_idx : write_ahead_edges) {
        fwd_search_ch_graph.at(tt.ch_graph_edges_[prf_idx].at(e_idx).from_)
            .push_back(e_idx);
        bwd_search_ch_graph.at(tt.ch_graph_edges_[prf_idx].at(e_idx).to_)
            .push_back(e_idx);
      }
      write_ahead_edges.clear();
      update_neighbours_node_order(location_id, write_ahead_edges, pq,
                                   current_order);
    }
    std::cout << "persisting..." << std::endl;
    for (auto const& u : unpack) {
      tt.ch_graph_unpack_[prf_idx].emplace_back(std::move(u));
    }
    unpack.clear();
    auto transfer_count = 0U;
    for (auto const& t : transfers) {
      transfer_count += t.size();
      tt.ch_graph_transfers_[prf_idx].emplace_back(std::move(t));
    }
    transfers.clear();
    for (auto i = location_idx_t{0U}; i != tt.locations_.ids_.size(); ++i) {
      tt.fwd_search_ch_graph_[prf_idx].emplace_back(
          std::move(fwd_search_ch_graph[i]));
      tt.bwd_search_ch_graph_[prf_idx].emplace_back(
          std::move(bwd_search_ch_graph[i]));
    }
    fwd_search_ch_graph.clear();
    bwd_search_ch_graph.clear();
    print_stats();
    std::cout << "edges: " << tt.ch_graph_edges_[prf_idx].size()
              << " stations: " << tt.n_locations()
              << " transfers: " << transfer_count << std::endl;
    std::cout << "num edges after contr: " << tt.ch_graph_edges_[prf_idx].size()
              << std::endl;
    print_edge_stats();

    std::cout
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
            departures.emplace_back(to_l, from_time, to_time, r, t);
          }
          min = std::min(((to_time - from_time).as_duration()), min);
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
    print_edge_stats();
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
