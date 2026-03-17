#include "utl/helpers/algorithm.h"
#include "utl/insert_sorted.h"
#include "utl/pairwise.h"
#include "utl/verify.h"
#include "utl/zip.h"

#include <cstdint>
#include "nigiri/loader/build_lb_graph.h"
#include "nigiri/common/dial.h"
#include "nigiri/for_each_meta.h"
#include "nigiri/logging.h"
#include "nigiri/routing/ch/ch_data.h"
#include "nigiri/routing/ch/ch_query.h"
#include "nigiri/routing/ch/saw.h"
#include "nigiri/routing/limits.h"
#include "nigiri/td_footpath.h"

#include "nigiri/timetable.h"
#include "nigiri/types.h"
#include <algorithm>
#include <limits>
#include <queue>
#include <vector>

namespace nigiri::routing {

static constexpr auto const kChMaxTravelTime = kMaxTravelTime * 5;  // TODO
static constexpr auto const kToothUnpackMode = false;
static constexpr auto const kDirectUnpackMode = false;
// static constexpr auto const kReconstructMode = true;

void obtain_relevant_stops(timetable const& tt,
                           routing::query const& q,
                           profile_idx_t const prf_idx,
                           bitvec& relevant_stops) {

  auto marked_stations = 0;
  vector_map<ch_edge_idx_t, std::vector<tooth>> edge_min;
  vector_map<ch_edge_idx_t, std::vector<tooth>> edge_max;
  vector_map<ch_edge_idx_t, timetable::ch_edge> graph_edges;
  auto const tmp_edge_offset = tt.ch_graph_max_[prf_idx].size();

  if (tt.fwd_search_ch_graph_[prf_idx].size() != tt.n_locations()) {
    std::cout << "no ch for profile, skipping" << std::endl;
    relevant_stops.one_out();
    return;
  }

  std::cout << "upsearch" << std::endl;

  if (kChSawType == saw_type::kTrafficDaysPower) {
    auto bf = bitvec{};
    bf.resize(tt.ch_traffic_days_[prf_idx].size());
    for (auto const e : tt.ch_graph_min_[prf_idx]) {
      for (auto const& t : e) {
        if (t.traffic_days_ != bitfield_idx_t::invalid()) {

          bf.set(t.traffic_days_.v_);
        }
      }
    }
    std::cout << bf.count() << "/" << bf.size() << std::endl;
    for (auto const e : tt.ch_graph_max_[prf_idx]) {
      for (auto const& t : e) {
        if (t.traffic_days_ != bitfield_idx_t::invalid()) {
          bf.set(t.traffic_days_.v_);
        }
      }
    }
    std::cout << bf.count() << "/" << bf.size() << std::endl;
  }

  std::array<vector_map<location_idx_t, ch_edge_idx_t>, 2> dists;
  dists[0].resize(tt.n_locations());
  dists[1].resize(tt.n_locations());
  auto nonce_map = vector_map<location_idx_t, std::uint32_t>{};
  nonce_map.resize(tt.n_locations());

  auto pq = dial<ch_label, ch_get_bucket>{
      (loader::kEnableDgp ? kDistanceGroups + 1U
                          : std::numeric_limits<std::uint16_t>::max())};
  auto ch_traffic_days =
      traffic_days{tt.ch_traffic_days_[prf_idx], {}};  // TODO avoid copy

  auto const distance_group = [&](location_idx_t const l) {
    if constexpr (!loader::kEnableDgp) {
      return static_cast<std::uint16_t>(
          tt.ch_levels_[prf_idx].at(l) *
          std::numeric_limits<std::uint32_t>::max() /
          tt.ch_levels_[prf_idx].size());  // TODO hack
    }
    for (auto i = static_cast<std::uint16_t>(0U); i < kDistanceGroups - 1U;
         ++i) {
      if (tt.ch_levels_[prf_idx].at(l) < tt.ch_distance_groups_[prf_idx][i]) {
        return i;
      }
    }
    return static_cast<std::uint16_t>(kDistanceGroups - 1U);
  };

  auto const mark_relevant_stop = [&](location_idx_t const parent) {
    if (loader::kChGroupParents && !relevant_stops.test(parent.v_)) {
      ++marked_stations;
      for (auto const& c : tt.locations_.children_[parent]) {
        relevant_stops.set(c.v_);
        for (auto const& cc : tt.locations_.children_[c]) {
          relevant_stops.set(cc.v_);
        }
      }
    }
    relevant_stops.set(parent.v_);
  };

  auto new_max_dist = std::vector<tooth>{};
  auto new_min_dist = std::vector<tooth>{};
  auto tmp_saw = std::vector<tooth>{};
  // auto mode = kMin;

  auto const min_max_dist_idx = ch_edge_idx_t{edge_max.size()};
  graph_edges.push_back({location_idx_t{0U}, location_idx_t{1U}});
  edge_max.push_back({});
  edge_min.push_back({});
  auto min_max_dist = std::vector<tooth>{};
  auto meetpoints = hash_set<location_idx_t>{};  // TODO other way of dedup?
  auto counter = 0;

  auto const mark_mp = [&](location_idx_t const l, unsigned const l_dir) {
    auto const other_dir = l_dir ^ 1U;

    if (dists[other_dir][l] != ch_edge_idx_t::invalid() &&
        !edge_max.at(dists[other_dir][l]).empty()) {
      auto max_concat =
          saw<kChSawType>{edge_max.at(dists[l_dir].at(l)), ch_traffic_days}
              .concat(l_dir,
                      saw<kChSawType>{edge_max.at(dists[other_dir][l]),
                                      ch_traffic_days},
                      dists[l_dir].at(l) + tmp_edge_offset,
                      dists[other_dir][l] + tmp_edge_offset, true, tmp_saw);
      max_concat.simplify(saw<kChSawType>{min_max_dist, ch_traffic_days}, true,
                          new_max_dist);
      if (saw<kChSawType>{new_max_dist, ch_traffic_days} !=
          saw<kChSawType>{min_max_dist, ch_traffic_days}) {
        std::swap(min_max_dist, new_max_dist);
        meetpoints.emplace(l);
      } else if (tmp_saw.clear();

                 saw<kChSawType>{edge_min.at(dists[l_dir].at(l)),
                                 ch_traffic_days}
                     .concat(l_dir,
                             saw<kChSawType>{edge_min.at(dists[other_dir][l]),
                                             ch_traffic_days},
                             ch_edge_idx_t::invalid(), ch_edge_idx_t::invalid(),
                             false, tmp_saw) <=
                 saw<kChSawType>{min_max_dist, ch_traffic_days}) {
        meetpoints.emplace(l);
      }
      std::cout << "mp found "
                << saw<kChSawType>{min_max_dist, ch_traffic_days}.max()
                << std::endl;

      new_max_dist.clear();
      tmp_saw.clear();
    }
  };

  auto const follow_edges = [&](location_idx_t const l, unsigned const l_dir,
                                u16_minutes const const_dist,
                                std::array<ch_label::dist_t, 2> const&) {
    auto const other_dir = l_dir ^ 1U;
    auto const& graph = l_dir == kForward ? tt.fwd_search_ch_graph_[prf_idx]
                                          : tt.bwd_search_ch_graph_[prf_idx];

    // std::cout << "weird" << graph.at(l).size() << std::endl;
    for (auto const& e_idx : graph.at(l)) {
      auto const e = tt.ch_graph_edges_[prf_idx].at(e_idx);
      auto const edge_target = l_dir == kForward ? e.to_ : e.from_;
      // std::cout << "const_dist1" << const_dist << std::endl;
      if (tt.ch_levels_[prf_idx].at(l) >
          tt.ch_levels_[prf_idx].at(edge_target)) {
        continue;
      }
      auto const l_e_idx = dists[l_dir].at(l);

      if (const_dist != u16_minutes::max()) {
        // TODO dead code (?)
        std::cout << "const_dist2" << const_dist << std::endl;
        auto const d = owning_saw<saw_type::kConstant>{
            saw<saw_type::kConstant>::of(const_dist), {}};

        saw<kChSawType>{tt.ch_graph_max_[prf_idx].at(e_idx), ch_traffic_days}
            .concat_const(other_dir, d.to_saw(ch_traffic_days),
                          ch_edge_idx_t::invalid(), ch_edge_idx_t::invalid(),
                          new_max_dist);

        saw<kChSawType>{tt.ch_graph_min_[prf_idx].at(e_idx), ch_traffic_days}
            .concat_const(other_dir, d.to_saw(ch_traffic_days),
                          ch_edge_idx_t::invalid(), ch_edge_idx_t::invalid(),
                          new_min_dist);
      } else {
        saw<kChSawType>{edge_max.at(l_e_idx), ch_traffic_days}.concat(
            l_dir,
            saw<kChSawType>{tt.ch_graph_max_[prf_idx].at(e_idx),
                            ch_traffic_days},
            l_e_idx + tmp_edge_offset, e_idx, true, new_max_dist);

        saw<kChSawType>{edge_min.at(l_e_idx), ch_traffic_days}.concat(
            l_dir,
            saw<kChSawType>{tt.ch_graph_min_[prf_idx].at(e_idx),
                            ch_traffic_days},
            l_e_idx + tmp_edge_offset, e_idx, false, new_min_dist);
      }
      // std::cout << "tar" << edge_target << " " << new_max_dist << " ld " <<
      //  l.d_[kMax] << " em " << e.max_dur_.count() << " " << new_min_dist <<
      //  std::endl;

      /*auto const max_true = saw<kChSawType>{new_max_dist,
      ch_traffic_days}.leq(
          dists[l_dir].at(edge_target).d_[kMax].to_saw(ch_traffic_days), true);
      auto const min_true = saw<kChSawType>{new_min_dist, ch_traffic_days}.leq(
          dists[l_dir].at(edge_target).d_[kMin].to_saw(ch_traffic_days), true);
      if (max_true || min_true) {*/
      if (dists[l_dir][edge_target] == 0U) {
        dists[l_dir][edge_target] = ch_edge_idx_t{edge_max.size()};
        graph_edges.push_back({graph_edges.at(l_e_idx).from_, edge_target});
        edge_max.push_back({});
        edge_min.push_back({});
      }

      saw<kChSawType>{new_max_dist, ch_traffic_days}.simplify(
          saw<kChSawType>{edge_max.at(dists[l_dir][edge_target]),
                          ch_traffic_days},
          true,
          tmp_saw);  // TODO move to leq again? or detect within simplify if not
      // equal (new bitfields created etc)
      /*std::cout << "push pq" << edge_target << " " << 0 << " " << 0
                << " new: " << saw<kChSawType>{new_max_dist, ch_traffic_days}
                << std::endl
                << " old: "
                << dists[l_dir].at(edge_target).d_[kMax].to_saw(ch_traffic_days)
                << std::endl
                << "simpl:" << saw<kChSawType>{tmp_saw, ch_traffic_days}
                << std::endl;*/
      auto max_leq = false;
      auto min_leq = false;
      if (saw<kChSawType>{edge_max.at(dists[l_dir].at(edge_target)),
                          ch_traffic_days} !=
          saw<kChSawType>{tmp_saw, ch_traffic_days}) {
        std::swap(edge_max.at(dists[l_dir].at(edge_target)), tmp_saw);
        max_leq = true;
      }
      new_max_dist.clear();
      tmp_saw.clear();
      saw<kChSawType>{new_min_dist, ch_traffic_days}.simplify(
          saw<kChSawType>{edge_min.at(dists[l_dir].at(edge_target)),
                          ch_traffic_days},
          false, tmp_saw);
      if (saw<kChSawType>{edge_min.at(dists[l_dir].at(edge_target)),
                          ch_traffic_days} !=
          saw<kChSawType>{tmp_saw, ch_traffic_days}) {

        std::swap(edge_min.at(dists[l_dir][edge_target]), tmp_saw);
        min_leq = true;
      }
      new_min_dist.clear();
      tmp_saw.clear();
      if (!min_leq && !max_leq) {
        /*std::cout << "skip " << e.from_ << " " << e.to_ << " " << edge_target
                  << std::endl;*/
        continue;
      }
      mark_mp(edge_target, l_dir);
      if (kEnableDgp && distance_group(l) == distance_group(edge_target)) {
        continue;
      }
      auto const const_max =
          saw<kChSawType>{edge_max.at(dists[l_dir][edge_target]),
                          ch_traffic_days}
              .max();
      auto const const_min =
          saw<kChSawType>{edge_min.at(dists[l_dir][edge_target]),
                          ch_traffic_days}
              .min();
      utl::verify(const_max.count() < kChMaxTravelTime.count() &&
                      const_min.count() < kChMaxTravelTime.count(),
                  "extra weird {} {}", const_max, const_min);
      /*std::cout << "push " << e.from_ << " " << e.to_ << " " << edge_target <<
         " " << tt.get_default_translation(tt.locations_.names_.at(edge_target))
                << " minmay " << const_min << " " << const_max << std::endl;*/

      /*auto const const_min_edge =
          saw<kChSawType>{tt.ch_graph_min_[prf_idx].at(e_idx), ch_traffic_days}
              .min();
      if (dist[kMax] + const_min_edge.count() >= kChMaxTravelTime.count()) {

        std::cout << "extra extra weird " << dist[kMax] << " "
                  << const_min_edge.count() << std::endl;

        throw utl::fail("extra extra weird");
      }*/
      pq.push(ch_label{
          edge_target,
          {distance_group(edge_target),
           static_cast<ch_label::dist_t>(nonce_map.at(edge_target) + 1)},
          static_cast<std::uint8_t>(l_dir)});
      //}
    }
  };

  for (auto i = location_idx_t{0U}; i < tt.n_locations(); ++i) {
    std::cout << i << " "
              << tt.get_default_translation(tt.locations_.names_.at(i))
              << " l:" << tt.ch_levels_[prf_idx].at(i) << std::endl;
  }

  auto const init = [&](std::vector<routing::offset> offsets,
                        std::uint8_t dir) {
    for (auto const& start : offsets) {  // TODO correct offsets
      for_each_meta(
          tt, dir == kForward ? q.start_match_mode_ : q.dest_match_mode_,
          start.target_, [&](location_idx_t const x) {
            dists[dir].at(x) = ch_edge_idx_t{edge_min.size()};
            graph_edges.push_back({location_idx_t{dir}, x});
            edge_max.push_back(saw<saw_type::kConstant>::of(start.duration()));
            edge_min.push_back(saw<saw_type::kConstant>::of(start.duration()));
            mark_mp(x, dir);
            pq.push(ch_label{x,
                             {distance_group(x), static_cast<ch_label::dist_t>(
                                                     nonce_map.at(x) + 1)},
                             dir});
            std::cout << "input" << x << " " << start.duration() << " "
                      << (dir == kForward ? "fw " : "bw ")
                      << tt.get_default_translation(tt.locations_.names_.at(x))
                      << " t:" << relevant_stops.test(x.v_) << std::endl;
          });
    }
  };

  init(q.start_, kForward);
  init(q.destination_, kReverse);

  while (!pq.empty()) {
    ++counter;
    auto l = pq.top();
    pq.pop();
    auto const l_dir = l.dir_ % kModeOffset;
    auto const other_dir = l_dir ^ 1U;

    if (l.d_[kMin] <= nonce_map.at(l.l_)) {
      //
      // continue; TODO
    }
    nonce_map.at(l.l_) = l.d_[kMin];

    /*if (dists[l_dir].at(l.l_).d_[kMax].to_saw(ch_traffic_days).max().count() <
            l.d_[kMax] &&  // TODO extrema
        dists[l_dir].at(l.l_).d_[kMin].to_saw(ch_traffic_days).min().count() <
            l.d_[kMin]) {  // TODO nonce?
      continue;
    }*/
    std::cout << "steop " << l.l_ << " "
              << tt.get_default_translation(tt.locations_.names_.at(l.l_))
              << " nonce: " << l.d_[kMin] << " " << " max: " << l.d_[kMax]
              << " other:"
              << saw<kChSawType>{edge_max.at(dists[other_dir][l.l_]),
                                 ch_traffic_days}
                     .max()
              << " " << l_dir << " l:" << tt.ch_levels_[prf_idx].at(l.l_)
              << std::endl;
    /*std::cout << "max " << dists[l_dir][l.l_].d_[kMax].to_saw(ch_traffic_days)
              << std::endl;
    std::cout << "max " << dists[l_dir][l.l_].d_[kMin].to_saw(ch_traffic_days)
              << std::endl;*/
    // mark_mp(l.l_, l_dir);
    //  std::cout << "mmd" << min_max_dist << std::endl;
    if (saw<kChSawType>{edge_min.at(dists[l_dir].at(l.l_)), ch_traffic_days} >
        saw<kChSawType>{min_max_dist, ch_traffic_days}) {
      /*if (mode == kMax) {
        auto buffer = std::vector<ch_label>{};
        while (!pq.empty()) {
          auto b = pq.top();
          //b.dir_ += kModeOffset;
          b.d_[kMax] =
      dists[l_dir].at(l.l_).d_[kMin].to_saw(ch_traffic_days).min().count();
          buffer.emplace_back(b);
          pq.pop();
        }
        l.dir_ += kModeOffset;
        pq.push(l);
        for (auto const& b : buffer) {
          pq.push(b);
        }
        std::cout << "switching to min mode " << counter << " "
                  << "minmax: "
                  << saw<kChSawType>{min_max_dist, ch_traffic_days}.max()
                  << " infty: " << std::numeric_limits<ch_label::dist_t>::max()
                  << std::endl;
        mode = kMin;
        continue;
      } else {
        std::cout << "reached ḿax with min " << counter << std::endl;
        break;
      }*/
      std::cout << "reached ḿax with min " << counter << std::endl;
      // break;
    }

    //    std::cout << "follow " << mode << l.l_ << std::endl;
    follow_edges(l.l_, l_dir, u16_minutes::max(), l.d_);
  }
  pq.clear();

  auto const invert = [&](ch_label::dist_t d) {
    return static_cast<ch_label::dist_t>(
        std::numeric_limits<std::uint16_t>::max() - d);
  };

  auto const const_min_max_dist = static_cast<int>(
      saw<kChSawType>{min_max_dist, ch_traffic_days}.max().count());
  edge_max.at(min_max_dist_idx) = min_max_dist;  // TODO avoid copy
  std::cout << "downsearch " << counter << " " << meetpoints.size()
            << std::endl;

  auto queue =
      std::queue<std::tuple<ch_edge_idx_t, ch_label::dist_t, bool, bool,
                            std::vector<tooth>, std::vector<tooth>,
                            std::vector<tooth>, std::vector<tooth>>>{};
  auto visited = vector_map<ch_edge_idx_t, ch_label::dist_t>{};
  visited.resize(tmp_edge_offset + edge_max.size());

  auto const unpack_children = [&](int const l_d_max) {
    while (!queue.empty()) {
      auto [child_edge_idx, child_max_dur, child_max, child_end,
            child_max_dur_saw, total_max_dur_saw, left, right] =
          queue.front();  // TODO avoid copy
      std::cout << "stack " << child_edge_idx << " cmd: " << child_max_dur
                << std::endl;

      queue.pop();
      if (!kToothUnpackMode &&
          visited.at(child_edge_idx) >=
              child_max_dur) {  // TODO use pq ordered by child_max_dur?
        continue;
      }
      visited[child_edge_idx] = child_max_dur;

      if (kToothUnpackMode) {
        auto const tooth_idx = child_max_dur;  // TODO cleanup

        std::cout << "tooth" << tooth_idx << std::endl;

        auto const tooth =
            child_max ? (child_edge_idx >= tmp_edge_offset
                             ? edge_max.at(child_edge_idx - tmp_edge_offset)
                                   .at(tooth_idx)
                             : tt.ch_graph_max_[prf_idx]
                                   .at(child_edge_idx)
                                   .at(tooth_idx))
                      : (child_edge_idx >= tmp_edge_offset
                             ? edge_min.at(child_edge_idx - tmp_edge_offset)
                                   .at(tooth_idx)
                             : tt.ch_graph_min_[prf_idx]
                                   .at(child_edge_idx)
                                   .at(tooth_idx));  // TODO avoid copy
        auto const& parent_edge =
            child_edge_idx >= tmp_edge_offset
                ? graph_edges.at(child_edge_idx - tmp_edge_offset)
                : tt.ch_graph_edges_[prf_idx].at(child_edge_idx);

        std::cout << "ft ldmax" << l_d_max << " " << parent_edge.from_
                  << " l:" << tt.ch_levels_[prf_idx].at(parent_edge.from_)
                  << " "
                  << tt.get_default_translation(
                         tt.locations_.names_.at(parent_edge.from_))
                  << " -> " << parent_edge.to_
                  << " l:" << tt.ch_levels_[prf_idx].at(parent_edge.to_) << " "
                  << tt.get_default_translation(
                         tt.locations_.names_.at(parent_edge.to_))
                  << std::endl;

        if (tooth.start_ != ch_edge_idx_t::invalid()) {
          auto const& start_edge =
              tooth.start_ >= tmp_edge_offset
                  ? graph_edges.at(tooth.start_ - tmp_edge_offset)
                  : tt.ch_graph_edges_[prf_idx].at(tooth.start_);

          mark_relevant_stop(start_edge.to_);
          std::cout << " transfer a " << start_edge.to_
                    << " l:" << tt.ch_levels_[prf_idx].at(start_edge.to_) << " "
                    << tt.get_default_translation(
                           tt.locations_.names_.at(start_edge.to_))
                    << std::endl;

          std::cout << "stack push a" << tooth.start_ << " " << tooth.start_idx_
                    << std::endl;
          queue.push({tooth.start_,
                      tooth.start_idx_,
                      child_max,
                      false,
                      {},
                      {},
                      {},
                      {}});
        }
        if (tooth.end_ != ch_edge_idx_t::invalid()) {
          auto const& end_edge =  // TODO might be invalid?
              tooth.end_ >= tmp_edge_offset
                  ? graph_edges.at(tooth.end_ - tmp_edge_offset)
                  : tt.ch_graph_edges_[prf_idx].at(tooth.end_);

          if (tooth.start_ == ch_edge_idx_t::invalid()) {
            mark_relevant_stop(end_edge.from_);
          }
          std::cout << " transfer b " << end_edge.from_
                    << " l:" << tt.ch_levels_[prf_idx].at(end_edge.from_) << " "
                    << tt.get_default_translation(
                           tt.locations_.names_.at(end_edge.from_))
                    << std::endl;
          std::cout << "stack push b" << " " << tooth.end_ << " "
                    << tooth.end_idx_ << std::endl;

          queue.push(
              {tooth.end_, tooth.end_idx_, child_max, true, {}, {}, {}, {}});
        }

      } else {
        for (auto const [unpack, transfer] :
             utl::zip(tt.ch_graph_unpack_[prf_idx].at(child_edge_idx),
                      tt.ch_graph_transfers_[prf_idx].at(child_edge_idx))) {
          if (unpack.second == ch_edge_idx_t::invalid()) {
            if (transfer != location_idx_t::invalid()) {
              mark_relevant_stop(transfer);
            }
            continue;
          }
          auto const arr_min_saw = saw<kChSawType>{
              tt.ch_graph_min_[prf_idx].at(unpack.first), ch_traffic_days};
          auto const dep_min_saw = saw<kChSawType>{
              tt.ch_graph_min_[prf_idx].at(unpack.second), ch_traffic_days};
          auto const arr_min = arr_min_saw.min();
          auto const dep_min = dep_min_saw.min();

          auto arr_max_saw = std::vector<tooth>{};
          auto dep_max_saw = std::vector<tooth>{};

          saw<kChSawType>{child_max_dur_saw, ch_traffic_days}.concat_const(
              kForward,
              saw<saw_type::kConstant>{saw<saw_type::kConstant>::of(dep_min),
                                       ch_traffic_days},
              ch_edge_idx_t::invalid(), ch_edge_idx_t::invalid(), tmp_saw,
              true);
          saw<kChSawType>{tmp_saw, ch_traffic_days}.simplify(
              saw<kChSawType>{tt.ch_graph_max_[prf_idx].at(unpack.first),
                              ch_traffic_days},
              true, arr_max_saw);
          tmp_saw.clear();

          saw<kChSawType>{child_max_dur_saw, ch_traffic_days}.concat_const(
              kReverse,
              saw<saw_type::kConstant>{saw<saw_type::kConstant>::of(arr_min),
                                       ch_traffic_days},
              ch_edge_idx_t::invalid(), ch_edge_idx_t::invalid(), tmp_saw,
              true);
          saw<kChSawType>{tmp_saw, ch_traffic_days}.simplify(
              saw<kChSawType>{tt.ch_graph_max_[prf_idx].at(unpack.second),
                              ch_traffic_days},
              true, dep_max_saw);

          tmp_saw.clear();

          if (transfer != location_idx_t::invalid()) {
            std::cout << "ft ldmax" << l_d_max << " "
                      << tt.get_default_translation(tt.locations_.names_.at(
                             tt.ch_graph_edges_[prf_idx][child_edge_idx].from_))
                      << " -> "
                      << tt.get_default_translation(tt.locations_.names_.at(
                             tt.ch_graph_edges_[prf_idx][child_edge_idx].to_))
                      << " transfer "
                      << tt.get_default_translation(
                             tt.locations_.names_.at(transfer))
                      << " arr: " << arr_min << " "
                      << saw<kChSawType>{arr_max_saw, ch_traffic_days}.max()
                      << " "
                      << " dep: " << dep_min << " "
                      << saw<kChSawType>{dep_max_saw, ch_traffic_days}.max()
                      << std::endl;
          }
          if (arr_min_saw > saw<kChSawType>{arr_max_saw, ch_traffic_days} ||
              dep_min_saw > saw<kChSawType>{dep_max_saw, ch_traffic_days}) {
            std::cout << "skip" << std::endl;
            continue;  // TODO count occurs
          }
          auto left_next = std::vector<tooth>{};
          auto right_next = std::vector<tooth>{};

          auto pushdown_left = std::vector<tooth>{};
          auto pushdown_right = std::vector<tooth>{};

          saw<kChSawType>{left, ch_traffic_days}.concat(
              kForward, arr_min_saw, ch_edge_idx_t::invalid(),
              ch_edge_idx_t::invalid(), false, left_next);

          saw<kChSawType>{right, ch_traffic_days}.concat(
              kReverse, dep_min_saw, ch_edge_idx_t::invalid(),
              ch_edge_idx_t::invalid(), false, right_next);

          saw<kChSawType>{left, ch_traffic_days}
              .concat(
                  kForward,
                  saw<kChSawType>{tt.ch_graph_max_[prf_idx].at(unpack.first),
                                  ch_traffic_days},
                  ch_edge_idx_t::invalid(), ch_edge_idx_t::invalid(), true,
                  tmp_saw)
              .concat(kForward, saw<kChSawType>{right_next, ch_traffic_days},
                      ch_edge_idx_t::invalid(), ch_edge_idx_t::invalid(), true,
                      new_min_dist)
              .simplify(saw<kChSawType>{total_max_dur_saw, ch_traffic_days},
                        true, pushdown_left);

          tmp_saw.clear();
          new_min_dist.clear();

          if (saw<kChSawType>{left, ch_traffic_days}
                  .concat(kForward,
                          saw<kChSawType>{
                              tt.ch_graph_min_[prf_idx].at(unpack.first),
                              ch_traffic_days},
                          ch_edge_idx_t::invalid(), ch_edge_idx_t::invalid(),
                          true, tmp_saw)
                  .concat(kForward,
                          saw<kChSawType>{right_next, ch_traffic_days},
                          ch_edge_idx_t::invalid(), ch_edge_idx_t::invalid(),
                          true, new_min_dist) >
              saw<kChSawType>{pushdown_left, ch_traffic_days}) {
            std::cout << "skip pushdown l" << std::endl;
            tmp_saw.clear();
            new_min_dist.clear();
            continue;
          }

          tmp_saw.clear();
          new_min_dist.clear();

          saw<kChSawType>{left_next, ch_traffic_days}
              .concat(
                  kForward,
                  saw<kChSawType>{tt.ch_graph_max_[prf_idx].at(unpack.second),
                                  ch_traffic_days},
                  ch_edge_idx_t::invalid(), ch_edge_idx_t::invalid(), true,
                  tmp_saw)
              .concat(kForward, saw<kChSawType>{right, ch_traffic_days},
                      ch_edge_idx_t::invalid(), ch_edge_idx_t::invalid(), true,
                      new_min_dist)
              .simplify(saw<kChSawType>{total_max_dur_saw, ch_traffic_days},
                        true, pushdown_right);

          tmp_saw.clear();
          new_min_dist.clear();
          if (saw<kChSawType>{left_next, ch_traffic_days}
                  .concat(kForward,
                          saw<kChSawType>{
                              tt.ch_graph_min_[prf_idx].at(unpack.second),
                              ch_traffic_days},
                          ch_edge_idx_t::invalid(), ch_edge_idx_t::invalid(),
                          true, tmp_saw)
                  .concat(kForward, saw<kChSawType>{right, ch_traffic_days},
                          ch_edge_idx_t::invalid(), ch_edge_idx_t::invalid(),
                          true, new_min_dist) >
              saw<kChSawType>{pushdown_right, ch_traffic_days}) {
            std::cout << "skip pushdown r" << std::endl;
            tmp_saw.clear();
            new_min_dist.clear();
            continue;
          }

          tmp_saw.clear();
          new_min_dist.clear();
          if (transfer != location_idx_t::invalid()) {
            mark_relevant_stop(transfer);
          }
          queue.push(
              {unpack.first,
               saw<kChSawType>{arr_max_saw, ch_traffic_days}.max().count(),
               false, false, std::move(arr_max_saw), std::move(pushdown_left),
               left, std::move(right_next)});
          queue.push(
              {unpack.second,
               saw<kChSawType>{dep_max_saw, ch_traffic_days}.max().count(),
               false, false, std::move(dep_max_saw), std::move(pushdown_right),
               std::move(left_next), right});
          std::cout << "stack push " << unpack.first << " " << unpack.second
                    << " qs:" << queue.size() << std::endl;
        }
      }
    }
  };

  if constexpr (kDirectUnpackMode) {
    utl::verify(kToothUnpackMode, "needs kToothUnpackMode");
    auto const s = saw<kChSawType>{min_max_dist, ch_traffic_days};
    for (auto it = s.begin(); it != s.end(); ++it) {
      std::cout << "min max dist idx " << min_max_dist_idx + tmp_edge_offset
                << " " << it.pos_ << std::endl;
      queue.push({min_max_dist_idx + tmp_edge_offset,
                  static_cast<std::uint16_t>(it.pos_),
                  true,
                  false,
                  {},
                  {},
                  {},
                  {}});
    }
    unpack_children(const_min_max_dist);
    std::cout << "directly marked stops: " << relevant_stops.count() << "/"
              << relevant_stops.size() << std::endl;
    std::cout << "bitfields: " << "/" << ch_traffic_days.bitfields_.size()
              << std::endl;
    return;
  }

  for (auto const m : meetpoints) {
    saw<kChSawType>{edge_min.at(dists[kForward][m]), ch_traffic_days}.concat(
        saw<kChSawType>{edge_min.at(dists[kReverse][m]), ch_traffic_days},
        ch_edge_idx_t::invalid(), ch_edge_idx_t::invalid(), false, tmp_saw);
    std::cout << "mp filter " << m << " "
              << tt.get_default_translation(tt.locations_.names_.at(m)) << " "
              << saw<kChSawType>{tmp_saw, ch_traffic_days}.min() << " "
              << saw<kChSawType>{min_max_dist, ch_traffic_days}.max()
              << std::endl;
    if (saw<kChSawType>{tmp_saw, ch_traffic_days} >
            saw<kChSawType>{min_max_dist, ch_traffic_days} ||
        saw<kChSawType>{tmp_saw, ch_traffic_days}.min() >
            saw<kChSawType>{min_max_dist, ch_traffic_days}
                .max()) {  // TODO werid
      tmp_saw.clear();
      continue;
    }
    std::cout << "taken" << std::endl;
    std::cout << "concat " << saw<kChSawType>{tmp_saw, ch_traffic_days}
              << std::endl;
    std::cout << "min_max" << saw<kChSawType>{min_max_dist, ch_traffic_days}
              << std::endl;

    tmp_saw.clear();
    for (auto const dir : {kForward, kReverse}) {
      auto const other_dir = dir ^ 1U;

      saw<kChSawType>{min_max_dist, ch_traffic_days}.concat_const(
          dir,
          saw<saw_type::kConstant>{
              saw<saw_type::kConstant>::of(saw<kChSawType>{
                  edge_min.at(dists[other_dir][m]), ch_traffic_days}
                                               .min()),
              ch_traffic_days},
          ch_edge_idx_t::invalid(), ch_edge_idx_t::invalid(), new_max_dist,
          true);
      saw<kChSawType>{new_max_dist, ch_traffic_days}.simplify(
          saw<kChSawType>{edge_max.at(dists[dir][m]), ch_traffic_days}, true,
          tmp_saw);
      // std::swap(edge_max.at(dists[dir][m]), tmp_saw);
      auto const d = static_cast<ch_label::dist_t>(
          saw<kChSawType>{tmp_saw, ch_traffic_days}.max().count());
      tmp_saw.clear();
      new_max_dist.clear();
      pq.push(ch_label{
          m,
          {invert(kEnableDgp ? kDistanceGroups
                             : std::numeric_limits<std::uint16_t>::max()),
           static_cast<ch_label::dist_t>(nonce_map.at(m) + 1)},
          static_cast<std::uint8_t>(dir)});
      std::cout << "added mp " << d << std::endl;
    }
  }

  while (!pq.empty()) {
    auto l = pq.top();
    auto const l_d_max = static_cast<int>(invert(l.d_[kMax]));
    auto const other_dir = l.dir_ ^ 1U;
    pq.pop();

    if (l.d_[kMin] <= nonce_map.at(l.l_)) {
      // continue;
    }
    std::cout << "down " << l.l_ << " "
              << tt.get_default_translation(tt.locations_.names_.at(l.l_))
              // << " min: " << dists[l.dir_][l.l_].d_[kMin] << " "
              << " max: " << l_d_max << " nonce: " << l.d_[kMin]
              << " dir:" << (l.dir_ == kForward ? "fwd" : "bwd")
              << "| l:" << tt.ch_levels_[prf_idx].at(l.l_) << std::endl;
    nonce_map.at(l.l_) = l.d_[kMin];

    auto edge_max_dist = std::vector<tooth>{};
    saw<kChSawType>{min_max_dist, ch_traffic_days}.concat_const(
        l.dir_,
        saw<saw_type::kConstant>{
            saw<saw_type::kConstant>::of(saw<kChSawType>{
                edge_min.at(dists[other_dir][l.l_]), ch_traffic_days}
                                             .min()),
            ch_traffic_days},
        ch_edge_idx_t::invalid(), ch_edge_idx_t::invalid(), tmp_saw, true);

    saw<kChSawType>{tmp_saw, ch_traffic_days}.simplify(
        saw<kChSawType>{edge_max.at(dists[l.dir_][l.l_]), ch_traffic_days},
        true, edge_max_dist);

    tmp_saw.clear();

    saw<kChSawType>{edge_max.at(dists[l.dir_][l.l_]), ch_traffic_days}.concat(
        l.dir_,
        saw<kChSawType>{edge_min.at(dists[other_dir][l.l_]), ch_traffic_days},
        ch_edge_idx_t::invalid(), ch_edge_idx_t::invalid(), true, tmp_saw);

    saw<kChSawType>{tmp_saw, ch_traffic_days}.simplify(
        saw<kChSawType>{min_max_dist, ch_traffic_days}, true, new_max_dist);
    /*if (saw<kChSawType>{tmp_saw, ch_traffic_days} ==
        dists[l.dir_][l.l_].d_[kMax].to_saw(
            ch_traffic_days)) {  // TODO improve leq pre-pq-push?
      mark_relevant_stop(l.l_);
      tmp_saw.clear();
      continue;
    }*/
    /*std::swap(edge_max.at(dists[l.dir_][l.l_]),
              tmp_saw); */  // TODO min with l_d_max-e.min_dur
    tmp_saw.clear();
    mark_relevant_stop(l.l_);
    auto const& graph = l.dir_ == kReverse ? tt.fwd_search_ch_graph_[prf_idx]
                                           : tt.bwd_search_ch_graph_[prf_idx];

    for (auto const& e_idx : graph[l.l_]) {
      // std::cout << "edge" << e_idx << std::endl;
      auto const e = tt.ch_graph_edges_[prf_idx][e_idx];
      auto const edge_target = l.dir_ == kReverse ? e.to_ : e.from_;
      if (tt.ch_levels_[prf_idx][l.l_] < tt.ch_levels_[prf_idx][edge_target]) {
        continue;
      }
      if (kEnableDgp && l_d_max == distance_group(edge_target)) {
        continue;
      }
      auto const& prev_label = dists[l.dir_][edge_target];
      if (prev_label == ch_edge_idx_t::invalid()) {
        continue;
      }

      saw<kChSawType>{edge_min.at(prev_label), ch_traffic_days}.concat(
          l.dir_,
          saw<kChSawType>{tt.ch_graph_min_[prf_idx].at(e_idx), ch_traffic_days},
          ch_edge_idx_t::invalid(), ch_edge_idx_t::invalid(), false, tmp_saw);
      auto min_dist_via_prev = saw<kChSawType>{tmp_saw, ch_traffic_days};
      // auto min_dist_via_prev_const = min_dist_via_prev.min().count();

      min_dist_via_prev.concat(
          l.dir_,
          saw<kChSawType>{edge_min.at(dists[other_dir][l.l_]), ch_traffic_days},
          ch_edge_idx_t::invalid(), ch_edge_idx_t::invalid(), false,
          new_min_dist);

      // TODO l_d_max cheat, stopping criterion, cutoff?
      if (/*min_dist_via_prev_const > l_d_max ||*/
          !min_dist_via_prev.leq(
              saw<kChSawType>{edge_max_dist, ch_traffic_days}, true) ||
          saw<kChSawType>{new_max_dist, ch_traffic_days}.less(
              saw<kChSawType>{new_min_dist, ch_traffic_days},
              true)) {  // TODO exact_true correct?
        tmp_saw.clear();
        new_min_dist.clear();
        continue;
      }
      std::cout << saw<kChSawType>{new_min_dist, ch_traffic_days} << std::endl;
      tmp_saw.clear();
      new_min_dist.clear();

      std::cout << "down edge " << l_d_max << " "
                << tt.get_default_translation(tt.locations_.names_.at(
                       tt.ch_graph_edges_[prf_idx][e_idx].from_))
                << " -> "
                << tt.get_default_translation(tt.locations_.names_.at(
                       tt.ch_graph_edges_[prf_idx][e_idx].to_))
                << std::endl;

      /*if (min_dist_via_prev.max().count() >=
          kChMaxTravelTime.count()) {  // TODO expensive
        std::cout << "weird" << min_dist_via_prev.max().count() << " "
                  << l_d_max << " " << std::endl;
        tmp_saw.clear();
        continue;
      }*/

      // TODO move to pq pop?

      if (kToothUnpackMode) {
        auto max = false;
        for (auto const& saw :
             {saw<kChSawType>{tt.ch_graph_min_[prf_idx].at(e_idx),
                              ch_traffic_days},
              saw<kChSawType>{tt.ch_graph_max_[prf_idx].at(e_idx),
                              ch_traffic_days}}) {
          for (auto it = saw.begin(); it != saw.end(); ++it) {
            queue.push({e_idx,
                        static_cast<std::uint16_t>(it.pos_),
                        max,
                        false,
                        {},
                        {},
                        {},
                        {}});
          }
          max = true;
        }
      } else {
        auto pushdown_edge_max_dist = std::vector<tooth>{};
        saw<kChSawType>{edge_max_dist, ch_traffic_days}.concat_const(
            other_dir,
            saw<saw_type::kConstant>{
                saw<saw_type::kConstant>::of(
                    saw<kChSawType>{edge_min.at(prev_label), ch_traffic_days}
                        .min()),
                ch_traffic_days},
            ch_edge_idx_t::invalid(), ch_edge_idx_t::invalid(), new_min_dist,
            true);
        saw<kChSawType>{new_min_dist, ch_traffic_days}.simplify(
            saw<kChSawType>{tt.ch_graph_max_[prf_idx].at(e_idx),
                            ch_traffic_days},
            true, pushdown_edge_max_dist);

        new_min_dist.clear();

        auto pushdown_max_dist = std::vector<tooth>{};

        saw<kChSawType>{edge_min.at(prev_label), ch_traffic_days}
            .concat(l.dir_,
                    saw<kChSawType>{tt.ch_graph_max_[prf_idx].at(e_idx),
                                    ch_traffic_days},
                    ch_edge_idx_t::invalid(), ch_edge_idx_t::invalid(), true,
                    tmp_saw)
            .concat(l.dir_,
                    saw<kChSawType>{edge_min.at(dists[other_dir][l.l_]),
                                    ch_traffic_days},
                    ch_edge_idx_t::invalid(), ch_edge_idx_t::invalid(), true,
                    new_min_dist)
            .simplify(saw<kChSawType>{min_max_dist, ch_traffic_days}, true,
                      pushdown_max_dist);

        queue.push({e_idx,
                    static_cast<ch_label::dist_t>(
                        saw<kChSawType>{pushdown_edge_max_dist, ch_traffic_days}
                            .max()
                            .count()),
                    false, false, std::move(pushdown_edge_max_dist),
                    std::move(pushdown_edge_max_dist),
                    saw<saw_type::kConstant>::of(duration_t{0U}),
                    saw<saw_type::kConstant>::of(duration_t{0U})});
        /*std::move(pushdown_max_dist), // appears to be slightly worse
        l.dir_ == kReverse ? edge_min.at(dists[other_dir][l.l_])
                           : edge_min.at(prev_label),
        l.dir_ == kForward ? edge_min.at(dists[other_dir][l.l_])
                           : edge_min.at(prev_label)});*/
        tmp_saw.clear();
        new_min_dist.clear();
      }

      unpack_children(l_d_max);

      if (dists[other_dir][edge_target] == 0U) {
        dists[other_dir][edge_target] = ch_edge_idx_t{edge_max.size()};
        graph_edges.push_back(
            {location_idx_t::invalid(), location_idx_t::invalid()});
        edge_max.push_back({});
        edge_min.push_back({});
      }
      saw<kChSawType>{edge_min.at(dists[other_dir][l.l_]), ch_traffic_days}
          .concat(other_dir,
                  saw<kChSawType>{tt.ch_graph_min_[prf_idx].at(e_idx),
                                  ch_traffic_days},
                  ch_edge_idx_t::invalid(), ch_edge_idx_t::invalid(), false,
                  tmp_saw);
      saw<kChSawType>{tmp_saw, ch_traffic_days}.simplify(
          saw<kChSawType>{edge_min.at(dists[other_dir][edge_target]),
                          ch_traffic_days},
          false, new_min_dist);

      if (saw<kChSawType>{new_min_dist, ch_traffic_days} ==
          saw<kChSawType>{edge_min.at(dists[other_dir][edge_target]),
                          ch_traffic_days}) {
        tmp_saw.clear();
        new_min_dist.clear();
        continue;
      }
      std::swap(edge_min.at(dists[other_dir][edge_target]), new_min_dist);
      tmp_saw.clear();
      new_min_dist.clear();

      auto const x = saw<kChSawType>{tt.ch_graph_min_[prf_idx].at(e_idx),
                                     ch_traffic_days}  // TODO deconcat?
                         .min();
      utl::verify(x != u16_minutes::max(), "min is infty");
      std::cout << "diff " << x << " ld " << l_d_max << " em "
                << saw<kChSawType>{edge_max.at(dists[l.dir_][edge_target]),
                                   ch_traffic_days}
                       .max()
                       .count()
                << std::endl;
      pq.push(
          ch_label{edge_target,
                   {invert(distance_group(edge_target)),
                    static_cast<ch_label::dist_t>(nonce_map[edge_target] +
                                                  1)},  // TODO is this correct?
                   static_cast<std::uint8_t>(l.dir_)});

      tmp_saw.clear();
    }
    new_max_dist.clear();
  }
  // relevant_stops.one_out();
  /*relevant_stops.zero_out();
  for (auto l : {66733, 66707,
    66707, 14037,
    14037, 24022,
    24022, 44946,
    44946, 60390,
    66733, 66707,
    66707, 14037,
    14037, 17360,
    17360, 17346,
    17346, 23630,
    23630, 23631,
    23631, 24051,
    24051, 45331,
    45331, 60392,
    66731, 67390,
    67390, 67386,
    67386, 66705,
    66705, 14037,
    14037, 24022,
    24022, 44946,
    44946, 60390,
    66731, 67390,
    67390, 67386,
    67386, 66705,
    66705, 14037,
    14037, 17360,
    17360, 17346,
    17346, 23630,
    23630, 23631,
    23631, 24051,
    24051, 45331,
    45331, 60392,
    66733, 66707,
    66707, 14038,
    14038, 24022,
    24022, 44946,
    44946, 60390,
    66733, 66707,
    66707, 14038,
    14038, 17360,
    17360, 17370,
    17370, 23630,
    23630, 23631,
    23631, 24051,
    24051, 45331,
    45331, 60392}) {
    relevant_stops.set(static_cast<unsigned>(l));
  }*/
  init(q.start_, kForward);
  init(q.destination_, kReverse);
  std::cout << "marked stops: " << relevant_stops.count() << "/"
            << relevant_stops.size() << std::endl;
  std::cout << "marked stations: " << marked_stations << std::endl;
  std::cout << "bitfields: " << "/" << ch_traffic_days.bitfields_.size()
            << std::endl;
}

}  // namespace nigiri::routing