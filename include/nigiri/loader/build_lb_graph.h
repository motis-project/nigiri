#pragma once

#include <cstdint>
#include <algorithm>
#include <limits>
#include <vector>

#include "nigiri/loader/dir.h"
#include "utl/get_or_create.h"
#include "utl/helpers/algorithm.h"
#include "utl/insert_sorted.h"
#include "utl/pairwise.h"
#include "utl/pipes/avg.h"

#include "nigiri/common/dial.h"
#include "nigiri/for_each_meta.h"
#include "nigiri/logging.h"
#include "nigiri/routing/ch/ch_data.h"
#include "nigiri/routing/ch/ch_query.h"
#include "nigiri/routing/ch/saw.h"
#include "nigiri/routing/dijkstra.h"
#include "nigiri/routing/limits.h"
#include "nigiri/td_footpath.h"

#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::loader {

using namespace nigiri::routing;

static constexpr auto const kChMaxTravelTime =
    routing::kMaxTravelTime * 5;  // TODO

static constexpr auto const kEnableCh = true;
static constexpr auto const kChGroupParents = true;
static constexpr auto const kChAtomicFootpaths = true;
static constexpr auto const kChMaxLevelFraction = 1.0;
static constexpr auto const kChMaxNodeOrderUpdateFraction = 0.99;

struct departure {
  bool operator<(departure const& o) const {
    if (dep_.mam_ == o.dep_.mam_) {
      return arr_ - dep_ < o.arr_ - o.dep_;
    }
    return dep_.mam_ > o.dep_.mam_;
  }
  location_idx_t to_;
  delta dep_;
  delta arr_;
  route_idx_t r_;
  transport_idx_t t_;
};

struct arrival {
  std::vector<delta> deps_;
  std::vector<u16_minutes> travel_durs_;
  std::vector<transport_idx_t> transports_;
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
  std::int64_t min_max_diff_sum{0};
  std::int64_t min_max_diff_count{0};
};

template <direction SearchDir>
void build_lb_graph(timetable& tt, profile_idx_t const prf_idx) {
  hash_map<location_idx_t, duration_t> weights;
  hash_map<location_idx_t, std::pair<duration_t, duration_t>> arrival_fps;
  hash_map<location_idx_t, arrival> arrivals;
  std::vector<departure> departures;
  hash_map<std::pair<location_idx_t, location_idx_t>, ch_edge_idx_t> edges_map;
  vector_map<ch_edge_idx_t, std::vector<route_idx_t>> routes;
  vector_map<ch_edge_idx_t,
             std::vector<std::pair<ch_edge_idx_t, ch_edge_idx_t>>>
      unpack;
  vector_map<ch_edge_idx_t, std::vector<location_idx_t>>
      transfers;  // TODO use bool and explicitly insert low level footpaths?
  vector_map<ch_edge_idx_t, std::vector<tooth>> edge_min;
  vector_map<ch_edge_idx_t, std::vector<tooth>> edge_max;
  traffic_days traffic_days;
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
                                     location_idx_t const b,
                                     bool const dry_run = false) {
    utl::verify(departures.empty(), "departures not empty");
    for (auto const& r : tt.location_routes_[b]) {
      to_routes.set(r.v_);
    }
    if (kChGroupParents) {
      for (auto const& c : tt.locations_.children_[b]) {
        for (auto const& r : tt.location_routes_[c]) {
          to_routes.set(r.v_);
        }
        for (auto const& cc : tt.locations_.children_[c]) {
          for (auto const& r : tt.location_routes_[cc]) {
            to_routes.set(r.v_);
          }
        }
      }
    }
    auto found_direct = false;
    auto const check_routes = [&](location_idx_t const aa) {
      for (auto const& r : tt.location_routes_[aa]) {
        if ((prf_idx == kCarProfile && !tt.has_car_transport(r)) ||
            (prf_idx == kBikeProfile && !tt.has_bike_transport(r))) {
          continue;
        }
        if (!to_routes.test(r.v_)) {
          continue;
        }
        found_direct = true;
        if (dry_run) {
          return;
        }

        auto const location_seq = tt.route_location_seq_[r];
        auto from_stop_idx = std::numeric_limits<stop_idx_t>::max();
        for (auto const [from, to] : utl::pairwise(
                 interval{stop_idx_t{0U},
                          static_cast<stop_idx_t>(location_seq.size())})) {
          auto const from_l = stop{location_seq[from]}.location_idx();
          auto const to_l = stop{location_seq[to]}.location_idx();
          auto const to_parent = tt.locations_.get_root_idx(to_l);

          if (from_stop_idx == std::numeric_limits<stop_idx_t>::max() &&
              from_l != aa) {
            continue;
          }
          if (from_l == aa) {
            from_stop_idx = from;
          }
          if (kChGroupParents ? (to_parent != b) : (to_l != b)) {
            continue;
          }

          for (auto const t : tt.route_transport_ranges_[r]) {
            auto const from_time =
                tt.event_mam(t, from_stop_idx, event_type::kDep);
            auto const to_time = tt.event_mam(t, to, event_type::kArr);
            departures.emplace_back(kChGroupParents ? to_parent : to_l,
                                    from_time, to_time, r, t);
          }

          from_stop_idx = std::numeric_limits<stop_idx_t>::max();
        }
      }
    };
    check_routes(a);
    if (kChGroupParents) {
      for (auto const& c : tt.locations_.children_[a]) {
        if (dry_run && found_direct) {
          break;
        }
        check_routes(c);
        for (auto const& cc : tt.locations_.children_[c]) {
          if (dry_run && found_direct) {
            break;
          }
          check_routes(cc);
        }
      }
    }
    to_routes.zero_out();
    return found_direct;
  };

  auto const insert_ch_edge =
      [&](location_idx_t const from_l, location_idx_t const to_l,
          std::vector<tooth> const min, std::vector<tooth> const max,
          bool const defer_graph_insertion = false) {
        auto const edge_idx = ch_edge_idx_t{tt.ch_graph_edges_[prf_idx].size()};
        edges_map.emplace(std::pair{from_l, to_l}, edge_idx);
        tt.ch_graph_edges_[prf_idx].push_back({from_l, to_l});
        if (!defer_graph_insertion) {
          fwd_search_ch_graph.at(from_l).push_back(edge_idx);
          bwd_search_ch_graph.at(to_l).push_back(edge_idx);
        }
        unpack.emplace_back();
        transfers.emplace_back();
        routes.emplace_back();
        edge_min.emplace_back(std::move(min));
        edge_max.emplace_back(std::move(max));
        return edge_idx;
      };

  auto const update_ch_shortcut =
      [&](std::vector<tooth> const min_dur, std::vector<tooth> const max_dur,
          ch_edge_idx_t const shortcut_idx, ch_stats& contract_stats,
          bool const dry_run) {
        /*{
          auto const const_max =
              saw<kChSawType>{edge_max.at(shortcut_idx), traffic_days}.max();
          utl::verify(const_max.count() >= 2, "pre weird 0 max {} {}",
                      const_max, shortcut_idx);
        }*/

        if (saw<kChSawType>{max_dur, traffic_days}.less(
                saw<kChSawType>{edge_min.at(shortcut_idx), traffic_days},
                true)) {  // TODO kMaxTravelDays case?, cheat exact_true=true so
                          // that direct can dominate
          // replace
          //std::cout << "repl" << std::endl;
          ++contract_stats.replacements_;
          if (dry_run) {
            return false;
          }
          edge_min.at(shortcut_idx) = std::move(min_dur);
          edge_max.at(shortcut_idx) = std::move(max_dur);
          routes.at(shortcut_idx).clear();
          unpack.at(shortcut_idx).clear();
          transfers.at(shortcut_idx).clear();
        } else if (saw<kChSawType>{min_dur, traffic_days}.leq(
                       saw<kChSawType>{edge_max.at(shortcut_idx), traffic_days},
                       true)) {  // TODO cheat (?) exact_false

          //std::cout << "upd" << std::endl;
          // std::cout << "max prev " <<
          // saw<kChSawType>{edge_max.at(shortcut_idx), traffic_days} <<
          // std::endl; std::cout << "min tent " << saw<kChSawType>{min_dur,
          // traffic_days} << std::endl;
          //  update
          /*if (saw<kChSawType>{max_dur, traffic_days} <
              saw<kChSawType>{edge_max.at(shortcut_idx),
                              traffic_days}) {  // TODO too expensive?
            ++contract_stats.good_updates_;
          } else {
            ++contract_stats.bad_updates_;
          }*/
          ++contract_stats.bad_updates_;
          if (dry_run) {
            return false;
          }

          auto new_min_dur = std::vector<tooth>{};
          saw<kChSawType>{min_dur, traffic_days}.simplify(
              saw<kChSawType>{edge_min.at(shortcut_idx), traffic_days}, false,
              new_min_dur);
          auto new_max_dur = std::vector<tooth>{};
          saw<kChSawType>{max_dur, traffic_days}.simplify(
              saw<kChSawType>{edge_max.at(shortcut_idx), traffic_days}, true,
              new_max_dur);
          edge_min.at(shortcut_idx) = std::move(new_min_dur);
          edge_max.at(shortcut_idx) = std::move(new_max_dur);
        } else {
          //std::cout << "skip" << std::endl;
          ++contract_stats.skips_;
          return false;
        }

        /*utl::verify(shortcut.min_dur_.count() >= 0, "weird 0 min {} {} {} {}",
        dep.min_dur_, arr.min_dur_, shortcut.min_dur_,
        shortcut_idx);*/
        /*auto const const_min =
            saw<kChSawType>{edge_min.at(shortcut_idx), traffic_days}.min();
        if (const_min.count() >= kChMaxTravelTime.count()) {
          std::cout << saw<kChSawType>{edge_min.at(shortcut_idx), traffic_days}
                    << std::endl;
        }
        utl::verify(const_min.count() < kChMaxTravelTime.count(),
                    "overfl 0 min {} {} {} {}", const_min, shortcut_idx,
                    min_dur.size(), edge_min.at(shortcut_idx).size());
        auto const const_max =
            saw<kChSawType>{edge_max.at(shortcut_idx), traffic_days}.max();
        if (const_max.count() < 2) {
          std::cout << saw<kChSawType>{edge_max.at(shortcut_idx), traffic_days}
                    << std::endl;
        }
        utl::verify(const_max.count() >= 2, "weird 0 max {} {}", const_max,
                    shortcut_idx);
        if (const_max.count() >= kChMaxTravelTime.count()) {
          std::cout << saw<kChSawType>{edge_max.at(shortcut_idx), traffic_days}
                    << std::endl;
        }
        utl::verify(const_max.count() < kChMaxTravelTime.count(),
                    "overfl 0 max {} {}", const_max, shortcut_idx);*/
        return true;
      };

  auto const upsert_ch_footpath_edge =
      [&](location_idx_t const from_l, location_idx_t const to_l,
          std::vector<tooth> const min_dur, std::vector<tooth> const max_dur,
          hash_set<route_idx_t> const& new_routes, location_idx_t transfer) {
        if (auto const it = edges_map.find({from_l, to_l});
            it != end(edges_map)) {
          if (update_ch_shortcut(std::move(min_dur), std::move(max_dur),
                                 it->second, stats, false) &&
              (transfer != location_idx_t::invalid() ||
               unpack.at(it->second).empty())) {
            unpack.at(it->second)
                .push_back(
                    {ch_edge_idx_t::invalid(), ch_edge_idx_t::invalid()});
            transfers.at(it->second).push_back(transfer);
          }
          return it->second;
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
          return edge_idx;
        }
      };

  auto const assign_departures_to_arrivals = [&]() {
    utl::verify(arrivals.empty(), "arrivals not empty");
    std::sort(begin(departures), end(departures));
    for (auto const& dep : departures) {
      auto const dur = (dep.arr_ - dep.dep_).as_duration();
      if (auto const it = arrivals.find(dep.to_); it != end(arrivals)) {
        it->second.deps_.push_back(dep.dep_);
        it->second.travel_durs_.push_back(dur);
        it->second.transports_.push_back(dep.t_);
        it->second.routes_.emplace(dep.r_);
      } else {
        arrivals.emplace_hint(it, dep.to_,
                              arrival{{dep.dep_}, {dur}, {dep.t_}, {dep.r_}});
      }
    }
    departures.clear();
  };

  auto const arrivals_to_saw = [&](arrival const& e,
                                   std::vector<tooth>& min_saw,
                                   std::vector<tooth>& max_saw) {
    auto min_saw_tmp = std::vector<tooth>{};
    auto max_saw_tmp = std::vector<tooth>{};
    saw<kChSawType>{min_saw_tmp, traffic_days}.init_metadata(min_saw_tmp, 0);
    saw<kChSawType>{max_saw_tmp, traffic_days}.init_metadata(max_saw_tmp, 0);

    auto lsb = 0;
    for (auto i = 0UL; i < e.deps_.size(); ++i) {
      auto remaining_traffic_days =
          tt.bitfields_.at(tt.transport_traffic_days_[e.transports_[i]])
          << static_cast<unsigned>(e.deps_[i].days());
      remaining_traffic_days &= ignore_timetable_offset_mask;
      auto last_set_bit = saw<kChSawType>::last_set_bit(remaining_traffic_days);
      for (auto j = 0U; j < static_cast<unsigned>(e.deps_[i].days()); ++j) {
        remaining_traffic_days.set(
            last_set_bit - j, false);  // TODO do in one go with first 5 days?
      }
      lsb = std::max(lsb, last_set_bit - e.deps_[i].days());
      auto const traffic_days_idx = traffic_days.get_or_create(
          remaining_traffic_days,
          static_cast<std::uint16_t>(last_set_bit - e.deps_[i].days()));
      auto const new_tooth = tooth{static_cast<std::int16_t>(e.deps_[i].mam_),
                                   e.travel_durs_[i],
                                   traffic_days_idx,
                                   ch_edge_idx_t::invalid(),
                                   e.transports_[i],
                                   ch_edge_idx_t::invalid(),
                                   e.transports_[i]};
      min_saw_tmp.push_back(new_tooth);
      max_saw_tmp.push_back(new_tooth);
    }

    if constexpr (kChSawType == saw_type::kDay ||
                  kChSawType == saw_type::kConstant) {
      auto const s = saw<saw_type::kTrafficDays>{max_saw_tmp, traffic_days};
      for (auto b_it = s.begin(); b_it != s.end(); ++b_it) {
        auto remaining_traffic_days =
            traffic_days.bitfields_.at(b_it->traffic_days_).first;
        auto day_offset = 0;
        auto a_it = b_it;
        while (true) {
          ++a_it;

          if (a_it.day_offset_ != day_offset) {
            if (a_it.day_offset_ * -1 > routing::kMaxTravelTime / 1_days) {
              break;
            }
            remaining_traffic_days >>= 1U;
            remaining_traffic_days.set(kTimetableOffset.count() - 1, false);
            day_offset = a_it.day_offset_;
          }

          remaining_traffic_days &=
              ~traffic_days.bitfields_.at(a_it->traffic_days_).first;
          if (remaining_traffic_days.none()) {
            break;
          }
          auto const mam_diff =
              b_it->mam_ - a_it->mam_ + a_it.day_offset_ * -24 * 60;
          max_saw_tmp[a_it.pos_].travel_dur_ = std::min(
              std::max(
                  max_saw_tmp[a_it.pos_].travel_dur_,
                  u16_minutes{e.travel_durs_[b_it.pos_].count() + mam_diff}),
              kChMaxEdgeTime);
        }
      }
      saw<saw_type::kDay>{min_saw_tmp, traffic_days}.min(
          min_saw, kChSawType);  // TODO refactor trafficdays simplify

      saw<saw_type::kDay>{max_saw_tmp, traffic_days}.max(max_saw, kChSawType);
    } else {

      auto const s_min = saw<kChSawType>{min_saw_tmp, traffic_days};
      s_min.set_last_set_bit(min_saw_tmp, static_cast<std::uint16_t>(lsb));
      s_min.min(min_saw, kChSawType);  // TODO refactor trafficdays simplify

      auto const s_max = saw<kChSawType>{max_saw_tmp, traffic_days};
      s_min.set_last_set_bit(max_saw_tmp, static_cast<std::uint16_t>(lsb));
      s_max.max(max_saw, kChSawType);
    }
  };

  auto const update_arrival_fps = [&](location_idx_t const via) {
    for (auto fp : tt.locations_.footpaths_out_[prf_idx].at(via)) {
      auto const target = kChGroupParents
                              ? tt.locations_.get_root_idx(fp.target())
                              : fp.target();

      if (auto const it = arrival_fps.find(target); it != end(arrival_fps)) {
        it->second.first = std::min(it->second.first, fp.duration());
        it->second.second = std::max(it->second.second, fp.duration());
      } else {
        arrival_fps.emplace_hint(it, target,
                                 std::pair{fp.duration(), fp.duration()});
      }
    }
  };

  auto const compute_initial_ch_edges = [&](location_idx_t const from_l) {
    assign_departures_to_arrivals();

    auto max_saw = std::vector<tooth>{};
    auto min_saw = std::vector<tooth>{};
    for (auto& entry : arrivals) {
      auto const& e = entry.second;
      arrivals_to_saw(e, min_saw, max_saw);
      // if (i % 50 == 0) std::cout << "init max saw " << max << std::endl;

      arrival_fps.emplace(entry.first, std::pair{0_minutes, 2_minutes});
      update_arrival_fps(entry.first);
      if (kChGroupParents) {  // TODO for atomic & grouped, add fps once at
                              // from_l
        for (auto const& c : tt.locations_.children_[entry.first]) {
          update_arrival_fps(c);
          for (auto const& cc : tt.locations_.children_[c]) {
            update_arrival_fps(cc);
          }
        }
      }
      auto const upsert_edge_with_footpath =
          [&](location_idx_t const to_l,
              std::pair<u16_minutes, u16_minutes> const& minmax_duration,
              ch_edge_idx_t pt_edge) {
            auto min_saw_transfer = std::vector<tooth>{};
            auto max_saw_transfer = std::vector<tooth>{};
            saw<kChSawType>{min_saw, traffic_days}.concat_const(
                kForward,
                saw<saw_type::kConstant>{
                    saw<saw_type::kConstant>::of(minmax_duration.first),
                    traffic_days},
                pt_edge, ch_edge_idx_t::invalid(), min_saw_transfer);
            saw<kChSawType>{max_saw, traffic_days}.concat_const(
                kForward,
                saw<saw_type::kConstant>{
                    saw<saw_type::kConstant>::of(minmax_duration.second),
                    traffic_days},
                pt_edge, ch_edge_idx_t::invalid(), max_saw_transfer);
            return upsert_ch_footpath_edge(
                from_l, to_l, std::move(min_saw_transfer),
                std::move(max_saw_transfer), e.routes_,
                to_l == entry.first ? location_idx_t::invalid() : entry.first);
          };
      auto const pt_edge = upsert_edge_with_footpath(
          entry.first, arrival_fps.at(entry.first), ch_edge_idx_t::invalid());

      // std::cout << "arr fps " << arrival_fps.size() << std::endl;
      // TODO take into account PT directs?
      for (auto const& [to_l, minmax_duration] : arrival_fps) {
        /*std::cout << "fp mm fr: " << entry.first << " to: " << to_l << " "
                  << minmax_duration.first << " " << minmax_duration.second
                  << std::endl;*/
        if (to_l == entry.first) {
          continue;
        }
        if (kChAtomicFootpaths) {
          upsert_ch_footpath_edge(
              entry.first, to_l,
              saw<saw_type::kConstant>::of(minmax_duration.first),
              saw<saw_type::kConstant>::of(minmax_duration.second), {},
              location_idx_t::invalid());
        } else {  // TODO set edge_idx for PT edge so that transfer is taken
                  // into account
          upsert_edge_with_footpath(to_l, minmax_duration, pt_edge);
        }
      }

      max_saw.clear();
      min_saw.clear();
      arrival_fps.clear();
    }
    arrivals.clear();
  };

  auto const contract_ch_node =
      [&](location_idx_t const location_id,
          std::vector<ch_edge_idx_t>& write_ahead_edges,
          ch_stats& contract_stats, bool const dry_run = false) {
        auto const& deps = fwd_search_ch_graph.at(location_id);
        auto const& arrs = bwd_search_ch_graph.at(location_id);
        for (auto dep_idx : deps) {
          auto const to = tt.ch_graph_edges_[prf_idx].at(dep_idx).to_;
          if (tt.ch_levels_[prf_idx].at(to) > 0U) {
            ++contract_stats.contracted_neighbors_;
            continue;
          }
          auto const dep_max =
              saw<kChSawType>{edge_max.at(dep_idx), traffic_days}.max();
          auto const dep_min =
              saw<kChSawType>{edge_min.at(dep_idx), traffic_days}.min();
          stats.min_max_diff_sum += (dep_max - dep_min).count();
          ++stats.min_max_diff_count;

          for (auto arr_idx : arrs) {
            auto const from = tt.ch_graph_edges_[prf_idx].at(arr_idx).from_;
            if (tt.ch_levels_[prf_idx].at(from) > 0U) {
              ++contract_stats.contracted_neighbors_;
              continue;
            }
            if (from == to) {
              continue;
            }
            auto const arr_max =
                saw<kChSawType>{edge_max.at(arr_idx), traffic_days}.max();
            auto const arr_min =
                saw<kChSawType>{edge_min.at(arr_idx), traffic_days}.min();
            stats.min_max_diff_sum += (arr_max - arr_min).count();
            ++stats.min_max_diff_count;

            auto edge_idx = ch_edge_idx_t::invalid();
            if (auto const it = edges_map.find({from, to});
                it != end(edges_map)) {
              edge_idx = it->second;
            } else {
              // insert TODO weigh differently in ordering?
              ++contract_stats.inserts_;
              auto found_direct = find_direct_trips(from, to, dry_run);
              if (found_direct) {
                ++contract_stats.direct_inserts_;
              }
              if (dry_run) {
                departures.clear();
                continue;
              }
              auto min_saw = std::vector<tooth>{};  // TODO reuse alloc
              auto max_saw = std::vector<tooth>{};
              assign_departures_to_arrivals();
              if (!arrivals.empty()) {
                utl::verify(
                    arrivals.size() == 1,
                    "more than one direct relation found between a and b");
                auto const& e = begin(arrivals)->second;
                arrivals_to_saw(e, min_saw, max_saw);
                arrivals.clear();
                /*std::cout << "max saw "
                          << saw<kChSawType>{max_saw, traffic_days}.max()
                          << std::endl;*/
              }
              edge_idx = insert_ch_edge(from, to, std::move(min_saw),
                                        std::move(max_saw), true);
              if (found_direct) {
                unpack.at(edge_idx).push_back(
                    {ch_edge_idx_t::invalid(), ch_edge_idx_t::invalid()});
                transfers.at(edge_idx).push_back(location_idx_t::invalid());
              }
              write_ahead_edges.push_back(edge_idx);
            }

            auto min_dur = std::vector<tooth>{};
            saw<kChSawType>{edge_min.at(arr_idx), traffic_days}.concat(
                saw<kChSawType>{edge_min.at(dep_idx), traffic_days}, arr_idx,
                dep_idx, false, min_dur);
            auto max_dur = std::vector<tooth>{};
            saw<kChSawType>{edge_max.at(arr_idx), traffic_days}.concat(
                saw<kChSawType>{edge_max.at(dep_idx), traffic_days}, arr_idx,
                dep_idx, true, max_dur);
            if (!dry_run && false) {
              std::cout
                  << "eidx " << edge_idx << " " << edge_max.at(edge_idx).size()
                  << " "
                  << saw<kChSawType>{edge_min.at(edge_idx), traffic_days}.max()
                  << " "
                  << saw<kChSawType>{edge_max.at(edge_idx), traffic_days}.max()
                  << "  " << arr_idx << " " << edge_max.at(arr_idx).size()
                  << " " << arr_max << " aah " << dep_idx << " "
                  << edge_max.at(dep_idx).size() << " " << dep_max << " aah md "
                  << max_dur.size() << " "
                  << saw<kChSawType>{max_dur, traffic_days}.max() << std::endl;
            }
            if (!dry_run && false) {
              std::cout
                  << "via: " << location_id << " "
                  << tt.get_default_translation(
                         tt.locations_.names_.at(location_id))
                  << " to: " << to << " "
                  << tt.get_default_translation(tt.locations_.names_.at(to))
                  << " from: " << from << " "
                  << tt.get_default_translation(tt.locations_.names_.at(from))
                  << " pre min: "
                  << saw<kChSawType>{edge_min.at(edge_idx), traffic_days}.min()
                  << " pre max: "
                  << saw<kChSawType>{edge_max.at(edge_idx), traffic_days}.max()
                  << " tent min: "
                  << saw<kChSawType>{min_dur, traffic_days}.min()
                  << " tent max: "
                  << saw<kChSawType>{max_dur, traffic_days}.max() << std::endl;
            }
            if (update_ch_shortcut(min_dur, max_dur, edge_idx, contract_stats,
                                   dry_run)) {
              unpack[edge_idx].push_back({arr_idx, dep_idx});
              transfers[edge_idx].push_back(location_id);
            }
            if (!dry_run && false) {
              std::cout
                  << "eidx " << edge_idx << " " << edge_max.at(edge_idx).size()
                  << " "
                  << saw<kChSawType>{edge_min.at(edge_idx), traffic_days}.max()
                  << " "
                  << saw<kChSawType>{edge_max.at(edge_idx), traffic_days}.max()
                  << "  " << arr_idx << " " << edge_max.at(arr_idx).size()
                  << " aah " << dep_idx << " " << edge_max.at(dep_idx).size()
                  << std::endl;
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
        if (edges == 0) {
          return false;
        }
        auto const order = static_cast<routing::label::dist_t>(std::clamp(
            10000L + contract_stats.contracted_neighbors_ +
                contract_stats.inserts_ + contract_stats.bad_updates_ - edges -
                contract_stats.good_updates_ - contract_stats.replacements_ -
                contract_stats.skips_ -
                contract_stats.direct_inserts_ /
                    std::max(contract_stats.direct_inserts_, 1L) -
                contract_stats.min_max_diff_sum /
                    std::max(contract_stats.min_max_diff_count, 1L) / 60 / 6,
            0L, 20000L));  // TODO include direct inserts, max dur
                           // and/or transfers because depending on
                           // order, shortcuts will be replaced?
                           // do not subtract direct_inserts because big
                           // junctions will win
                           // special weight for stops without transfers?

        if (current_order.at(l) != order) {
          pq.push(routing::label{l, order});
          current_order.at(l) = order;
        }
        return true;
      };

  auto const update_neighbours_node_order =
      [&](location_idx_t l, std::vector<ch_edge_idx_t>& write_ahead_edges,
          dial<routing::label, routing::get_bucket>& pq,
          vector_map<location_idx_t, routing::label::dist_t>& current_order) {
        auto const& deps = fwd_search_ch_graph.at(l);
        for (auto dep_idx : deps) {
          auto const to = tt.ch_graph_edges_[prf_idx].at(dep_idx).to_;
          if (tt.ch_levels_[prf_idx].at(to) > 0U) {
            continue;
          }
          update_node_order(to, write_ahead_edges, pq, current_order);
        }
        auto const& arrs = bwd_search_ch_graph.at(l);
        for (auto arr_idx : arrs) {
          auto const from = tt.ch_graph_edges_[prf_idx].at(arr_idx).from_;
          if (tt.ch_levels_[prf_idx].at(from) > 0U) {
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
              << " bad updates: " << stats.bad_updates_
              << " traffic bitfields: " << traffic_days.bitfields_.size() << "/"
              << tt.bitfields_.size() << std::endl;
  };
  auto const print_edge_stats = [&]() {
    std::cout << "num edges: " << tt.ch_graph_edges_[prf_idx].size()
              << std::endl;
    if (tt.ch_graph_edges_[prf_idx].size() > 0) {
      auto min_max = u16_minutes::max();
      auto max_max = u16_minutes::min();
      auto sum_max = 0;
      auto sum_max_len = 0;
      auto min_min = u16_minutes::max();
      auto max_min = u16_minutes::min();
      auto sum_min = 0;
      auto sum_min_len = 0;
      for (auto i = ch_edge_idx_t{0}; i < tt.ch_graph_edges_[prf_idx].size();
           ++i) {
        auto tmp = saw<kChSawType>{edge_max.at(i), traffic_days}.max();
        if (tmp > max_max) {
          max_max = tmp;
        }
        if (tmp < min_max) {
          min_max = tmp;
        }
        sum_max += tmp.count();
        sum_max_len += saw<kChSawType>{edge_max.at(i), traffic_days}.size();
        tmp = saw<kChSawType>{edge_min.at(i), traffic_days}.min();
        if (tmp > max_min) {
          max_min = tmp;
        }
        if (tmp < min_min) {
          min_min = tmp;
        }
        sum_min += tmp.count();
        sum_min_len += saw<kChSawType>{edge_min.at(i), traffic_days}.size();
      }
      std::cout << " min max: " << min_max << " max max: " << max_max
                << " avg max: "
                << static_cast<unsigned>(sum_max) /
                       tt.ch_graph_edges_[prf_idx].size()
                << " total max tooths: " << static_cast<unsigned>(sum_max_len)
                << " avg max tooths: "
                << static_cast<unsigned>(sum_max_len) /
                       tt.ch_graph_edges_[prf_idx].size()
                << "\n"
                << " min min: " << min_min << " max min: " << max_min
                << " avg min: "
                << static_cast<unsigned>(sum_min) /
                       tt.ch_graph_edges_[prf_idx].size()
                << " total min tooths: " << static_cast<unsigned>(sum_min_len)
                << " avg min tooths: "
                << static_cast<unsigned>(sum_min_len) /
                       tt.ch_graph_edges_[prf_idx].size()
                << std::endl;
    }
  };

  auto const compute_ch = [&]() {
    print_stats();
    tt.ch_levels_[prf_idx].resize(static_cast<unsigned>(tt.n_locations()));
    auto pq = dial<routing::label, routing::get_bucket>{20001};
    auto current_order = vector_map<location_idx_t, routing::label::dist_t>{};
    current_order.resize(tt.n_locations());
    auto write_ahead_edges = std::vector<ch_edge_idx_t>{};
    std::cout << "initial node ordering..." << std::endl;
    auto empty_stops = 0U;
    for (auto l = location_idx_t{0}; l < tt.n_locations(); ++l) {
      if (!update_node_order(l, write_ahead_edges, pq, current_order)) {
        ++empty_stops;
      }
    }
    std::cout << "initial node ordering done" << std::endl;
    write_ahead_edges.clear();
    auto level = empty_stops;
    while (!pq.empty()) {
      auto const& label = pq.top();
      auto const location_id = label.l_;
      auto const order = label.d_;
      pq.pop();
      if (current_order.at(location_id) != order ||
          tt.ch_levels_[prf_idx].at(location_id) > 0U) {
        continue;
      }
      if (level > kChMaxLevelFraction * tt.n_locations()) {  // TODO
        tt.ch_levels_[prf_idx].at(location_id) = level;
        continue;
      }
      ++level;
      contract_ch_node(location_id, write_ahead_edges, stats);
      tt.ch_levels_[prf_idx].at(location_id) = level;
      for (auto const e_idx : write_ahead_edges) {
        fwd_search_ch_graph.at(tt.ch_graph_edges_[prf_idx].at(e_idx).from_)
            .push_back(e_idx);
        bwd_search_ch_graph.at(tt.ch_graph_edges_[prf_idx].at(e_idx).to_)
            .push_back(e_idx);
      }
      write_ahead_edges.clear();
      std::cout << tt.get_default_translation(
                       tt.locations_.names_.at(location_id))
                << std::endl;
      if (level % 100 == 0) {
        std::cout
            << level << " "
            << utl::count_if(
                   fwd_search_ch_graph.at(location_id),
                   [&](auto const& e) {
                     return tt.ch_levels_[prf_idx].at(
                                tt.ch_graph_edges_[prf_idx].at(e).to_) == 0;
                   })
            << "/" << fwd_search_ch_graph.at(location_id).size() << " "
            << utl::count_if(
                   bwd_search_ch_graph.at(location_id),
                   [&](auto const& e) {
                     return tt.ch_levels_[prf_idx].at(
                                tt.ch_graph_edges_[prf_idx].at(e).from_) == 0;
                   })
            << "/" << bwd_search_ch_graph.at(location_id).size() << " "
            << location_id << " "
            << tt.get_default_translation(tt.locations_.names_.at(location_id))
            << " " << order << std::endl;
        print_stats();
        if (level % 1000 == 0) {
          print_edge_stats();
        }
        std::cout << " empty stops: " << empty_stops
                  << " edges: " << tt.ch_graph_edges_[prf_idx].size()
                  << " stations: " << tt.n_locations() << " transfers size: "
                  << std::accumulate(std::next(transfers.begin()),
                                     transfers.end(), 0U,
                                     [](auto const& a, auto const& b) {
                                       return a + b.size();
                                     })
                  << std::endl;
      }
      if (level <= kChMaxNodeOrderUpdateFraction * tt.n_locations()) {
        update_neighbours_node_order(location_id, write_ahead_edges, pq,
                                     current_order);
      }
    }
    std::cout << "persisting..." << std::endl;
    /*auto j = 0;
    for (auto const& b : traffic_days.bitfields_) {
      std::cout << j << " " << b.first << std::endl;
      ++j;
    }*/
    print_edge_stats();

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
    for (auto const& e : edge_min) {
      tt.ch_graph_min_[prf_idx].emplace_back(std::move(e));
    }
    edge_min.clear();
    for (auto const& e : edge_max) {
      tt.ch_graph_max_[prf_idx].emplace_back(std::move(e));
    }
    edge_max.clear();
    for (auto const& e : traffic_days.bitfields_) {
      tt.ch_traffic_days_[prf_idx].emplace_back(std::move(e));
    }
    traffic_days.bitfields_.clear();
    traffic_days.bitfield_indices_.clear();
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
        if (target == parent_l &&  // TODO multiple same-parent calls for max?
            (SearchDir != direction::kBackward || !kEnableCh ||
             kChGroupParents)) {
          continue;
        }

        auto min = duration_t::max();
        for (auto const t : tt.route_transport_ranges_[r]) {
          auto const from_time = tt.event_mam(t, from, event_type::kDep);
          auto const to_time = tt.event_mam(t, to, event_type::kArr);
          if (SearchDir == direction::kBackward && kEnableCh) {
            departures.emplace_back(kChGroupParents ? target : to_l, from_time,
                                    to_time, r, t);
          }
          min = std::min(((to_time - from_time).as_duration()), min);
        }
        if (target != parent_l) {
          update_weight(target, min);
        }
      }
    }
    if (SearchDir == direction::kBackward && kEnableCh && !kChGroupParents) {
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

    if (SearchDir == direction::kBackward && kEnableCh && kChGroupParents) {
      compute_initial_ch_edges(i);
    }

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
