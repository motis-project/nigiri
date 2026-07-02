#pragma once

#include "cooperative_groups.h"

#include "nigiri/common/delta_t.h"
#include "nigiri/common/it_range.h"
#include "nigiri/common/linear_lower_bound.h"
#include "nigiri/routing/clasz_mask.h"
#include "nigiri/routing/limits.h"
#include "nigiri/types.h"

#include "nigiri/routing/gpu/device_bitvec.cuh"
#include "nigiri/routing/gpu/device_times.h"
#include "nigiri/routing/gpu/device_timetable.cuh"
#include "nigiri/routing/gpu/journey_pod.h"
#include "nigiri/routing/gpu/stride.cuh"
#include "nigiri/routing/gpu/types.cuh"

namespace nigiri::routing::gpu {

#ifndef NIGIRI_CUDA_DEBUG
#define debug(...)
#else
#define debug(...) printf(__VA_ARGS__)
#endif

#define start_timing() (void)0
#define debug_timing(...)

// #define start_timing() auto const round_start = clock64()
// #define debug_timing(str, ...)                        \
//  if (global_t_id == 0) {                             \
//    printf(str ": %" PRIi64 " cycles\n", __VA_ARGS__, \
//           clock64() - round_start);                  \
//  }

#define kInvalid (kInvalidDelta<SearchDir>)
#define kFwd (SearchDir == direction::kForward)
#define kBwd (SearchDir == direction::kBackward)
#define kUnreachable (std::numeric_limits<std::uint16_t>::max())
#define kIntermodalTarget (get_special_station(special_station::kEnd))

__device__ char const* b2s(bool const b) { return b ? "true" : "false"; }

template <direction SearchDir, bool Rt>
struct raptor_impl {
  static constexpr via_offset_t Vias = 0U;

  __device__ __forceinline__ bool is_better(auto a, auto b) {
    return kFwd ? a < b : a > b;
  }
  __device__ __forceinline__ bool is_better_or_eq(auto a, auto b) {
    return kFwd ? a <= b : a >= b;
  }
  __device__ __forceinline__ auto get_best(auto a, auto b) {
    return is_better(a, b) ? a : b;
  }
  __device__ __forceinline__ auto get_best(auto x, auto... y) {
    ((x = get_best(x, y)), ...);
    return x;
  }
  __device__ __forceinline__ auto min(auto x, auto y) { return x <= y ? x : y; }
  __device__ __forceinline__ auto dir(auto a) { return (kFwd ? 1 : -1) * a; }

  __device__ void execute(unixtime_t const start_time,
                          std::uint8_t const max_transfers,
                          unixtime_t const worst_time_at_dest) {
    auto const global_t_id = get_global_thread_id();
    auto const global_stride = get_global_stride();

    for (auto i = global_t_id; i < dist_to_end_.size(); i += global_stride) {
      if (dist_to_end_[i] != kUnreachable) {
        end_reachable_.mark(i);
      }
    }

    for (auto i = global_t_id; i < starts_.size(); i += global_stride) {
      auto const l = starts_[i].first;
      auto const t = unix_to_delta(base(), starts_[i].second);
      auto const v = via_offset_t{0};
      best_.update_min(l, v, t);
      round_times_.update_min(0U, l, v, t, make_start_bc());
      station_mark_.mark(to_idx(l));
    }

    auto const d_worst_at_dest = unix_to_delta(base(), worst_time_at_dest);
    for (auto i = global_t_id; i < kMaxTransfers + 2U; i += global_stride) {
      time_at_dest_.update_min(i, d_worst_at_dest);
    }

    sync();

    auto const end_k = min(max_transfers, kMaxTransfers) + 2U;
    for (auto k = 1U; k != end_k; ++k) {
      start_timing();

      // Reuse best time from previous time at start (for range queries).
      for (auto i = global_t_id; i < n_locations_; i += global_stride) {
        debug("round %u: location %d / %u\n", k, i, n_locations_);
        auto const l = location_idx_t{i};
        for (auto v = 0U; v != Vias + 1; ++v) {
          best_.update_min(l, v, round_times_.get(k, l, v));
        }
        if (is_dest_[i]) {
          update_time_at_dest(k, best_.get(l, Vias));
        }
      }

      if (global_t_id == 0) {
        *any_marked_ = 0U;
      }

      sync();
      debug_timing("round %u: after reuse times from prev start", k);

      for (auto i = global_t_id; i < tt_.n_locations_; i += global_stride) {
        if (station_mark_[i]) {
          if (!tt_.location_routes_[location_idx_t{i}].empty() &&
              !*any_marked_) {
            atomicOr(any_marked_, 1U);
          }
          for (auto r : tt_.location_routes_[location_idx_t{i}]) {
            debug("round %u: marking route %u\n", k, to_idx(r));
            route_mark_.mark(to_idx(r));
          }
        }
      }

      sync();
      debug_timing("round %u: after station flags -> route flags", k);

      if (!*any_marked_) {
        debug("round %d: no route marked -> break;\n", k);
        debug_timing("round %u: TOTAL", k);
        break;
      }

      prev_station_mark_.swap_reset(station_mark_);
      if (global_t_id == 0) {
        *any_marked_ = false;
      }
      sync();

      compute_et(k);
      sync();

      (allowed_claszes_ == all_clasz_allowed())
          ? (require_bike_transport_ ? loop_routes<false, true>(k)
                                     : loop_routes<false, false>(k))
          : (require_bike_transport_ ? loop_routes<true, true>(k)
                                     : loop_routes<true, false>(k));

      sync();
      debug_timing("round %u: after visit routes", k);

      if (!*any_marked_) {
        debug("round %d: no location marked after loop_routes -> break;\n", k);
        debug_timing("round %u: TOTAL", k);
        break;
      }
      prev_station_mark_.swap_reset(station_mark_);
      sync();
      debug_timing("round %u: after swap station marks", k);

      update_transfers(k);
      update_intermodal_footpaths(k);
      update_footpaths(k);

      route_mark_.reset();
      sync();
      debug_timing("round %u: TOTAL", k);
    }
  }

  __device__ __forceinline__ void sync() const {
    cooperative_groups::this_grid().sync();
  }

  __device__ void reconstruct_journey(location_idx_t const dest,
                                      unsigned const K,
                                      gpu_journey* out) {
    out->valid_ = 0U;
    auto cur_v = static_cast<via_offset_t>(Vias);
    auto const dest_time =
        round_times_.get(static_cast<std::uint8_t>(K), dest, cur_v);
    if (dest_time == kInvalid) {
      return;
    }
    out->dest_l_ = dest.v_;
    out->dest_time_ = dest_time;
    out->transfers_ = static_cast<std::uint8_t>(K - 1U);

    auto const ev_arr_type = kFwd ? event_type::kArr : event_type::kDep;
    auto const ev_dep_type = kFwd ? event_type::kDep : event_type::kArr;

    auto cur_l = dest;
    auto cur_k = K;
    auto n = 0U;
    auto egress_src = false;
    while (cur_k >= 1U) {
      auto const bc =
          round_times_.get_bc(static_cast<std::uint8_t>(cur_k), cur_l, cur_v);

      if (is_intermodal_dest() && cur_l == kIntermodalTarget) {
        auto const src = location_idx_t{bc_transport(bc)};
        out->dest_l_ = src.v_;
        cur_l = src;
        egress_src = true;
        continue;
      }
      if (bc_is_start(bc)) {
        break;
      }

      auto const t_idx = transport_idx_t{bc_transport(bc)};
      auto const board = static_cast<stop_idx_t>(bc_board(bc));
      auto const alight = static_cast<stop_idx_t>(bc_alight(bc));
      auto const r = tt_.transport_route_[t_idx];
      auto const arr_at_cur =
          round_times_.get(static_cast<std::uint8_t>(cur_k), cur_l, cur_v);

      // recover the traffic (first-departure) day of the transport
      auto const event_mam_full =
          tt_.event_mam(r, t_idx, alight, ev_arr_type).count();
      auto const split_arr = split(arr_at_cur);
      auto found_day = false;
      auto day = day_idx_t{0U};
      auto train_arr = kInvalid;
      for (auto off = 0; off != 2; ++off) {
        auto const cand = as_int(split_arr.first) - event_mam_full / 1440 -
                          (kFwd ? off : -off);
        if (cand < 0) {
          continue;
        }
        if (!is_transport_active(t_idx, static_cast<std::size_t>(cand))) {
          continue;
        }
        auto const tr =
            transport{t_idx, day_idx_t{static_cast<day_idx_t::value_t>(cand)}};
        auto const ev = time_at_stop(r, tr, alight, ev_arr_type);
        if (is_better_or_eq(ev, arr_at_cur)) {
          day = day_idx_t{static_cast<day_idx_t::value_t>(cand)};
          train_arr = ev;
          found_day = true;
          break;
        }
      }
      if (!found_day) {
        return;  // could not recover -> leave invalid
      }

      // at the egress stop, use the raw train arrival (drop the folded-in
      // transfer_time) so no self-footpath is emitted and the egress time is
      // anchored to the train.
      auto const eff_arr = egress_src ? train_arr : arr_at_cur;
      egress_src = false;

      auto const tr = transport{t_idx, day};
      auto const dep_at_board = time_at_stop(r, tr, board, ev_dep_type);
      auto const stop_seq = tt_.route_location_seq_[r];
      auto const board_loc = stop{stop_seq[board]}.location_idx();
      auto const alight_loc = stop{stop_seq[alight]}.location_idx();

      // footpath/transfer leg from the train's alighting stop to cur_l. nigiri
      // represents the same-station transfer between two trains as a self
      // footpath (B->B) with the transfer time, so we emit it whenever there is
      // a non-zero cost (a real footpath, or a same-station transfer), and skip
      // it only for a zero-cost direct arrival (e.g. at the destination).
      if (alight_loc != cur_l || train_arr != eff_arr) {
        if (n >= kMaxRecLegs) {
          return;
        }
        auto& lg = out->legs_[n++];
        lg.is_footpath_ = 1U;
        lg.from_l_ = alight_loc.v_;
        lg.to_l_ = cur_l.v_;
        lg.dep_ = train_arr;
        lg.arr_ = eff_arr;
        lg.fp_duration_ = static_cast<std::uint16_t>(
            kFwd ? (eff_arr - train_arr) : (train_arr - eff_arr));
      }

      // transport leg board_loc -> alight_loc
      if (n >= kMaxRecLegs) {
        return;
      }
      auto& lg = out->legs_[n++];
      lg.is_footpath_ = 0U;
      lg.from_l_ = board_loc.v_;
      lg.to_l_ = alight_loc.v_;
      lg.dep_ = dep_at_board;
      lg.arr_ = train_arr;
      lg.transport_ = t_idx.v_;
      lg.day_ = static_cast<std::uint16_t>(day.v_);
      lg.enter_stop_ = board;
      lg.exit_stop_ = alight;

      cur_l = board_loc;
      cur_k -= 1U;
    }

    out->start_l_ = cur_l.v_;
    out->n_legs_ = static_cast<std::uint8_t>(n);
    out->valid_ = (n != 0U) ? 1U : 0U;  // should not happen, safe guard
  }

  __device__ date::sys_days base() const {
    return tt_.internal_interval_days().from_ + as_int(base_) * date::days{1};
  }

  template <bool WithClaszFilter, bool WithBikeFilter>
  __device__ void loop_routes(unsigned const k) {
    // one warp per route (lanes cooperate on the route's stops via
    // update_route_warp); all 32 lanes share warp_id so they stay converged for
    // the warp shuffles / __any_sync.
    auto const full = 0xFFFFFFFFU;
    auto const lane = get_global_thread_id() % 32U;
    auto const warp_id = get_global_thread_id() / 32U;
    auto const n_warps = get_global_stride() / 32U;

    for (auto i = warp_id; i < tt_.n_routes_; i += n_warps) {
      if (!route_mark_.test(i)) {
        continue;
      }

      auto const r = route_idx_t{i};
      if constexpr (WithClaszFilter) {
        if (!is_allowed(allowed_claszes_, tt_.route_clasz_[r])) {
          continue;
        }
      }

      auto section_bike_filter = false;
      if constexpr (WithBikeFilter) {
        auto const bikes_allowed_on_all_sections =
            tt_.route_bikes_allowed_.test(i * 2);
        if (!bikes_allowed_on_all_sections) {
          auto const bikes_allowed_on_some_sections =
              tt_.route_bikes_allowed_.test(i * 2 + 1);
          if (!bikes_allowed_on_some_sections) {
            continue;
          }
          section_bike_filter = true;
        }
      }

      bool route_any_marked;
      if constexpr (Vias == 0) {
        if (!section_bike_filter) {
          route_any_marked = update_route_warp(k, r, lane);
        } else {
          // rare path: section bike filter -> sequential on lane 0
          auto const m = (lane == 0U) ? update_route<true>(k, r) : false;
          route_any_marked = __any_sync(full, m);
        }
      } else {
        // Vias>0 is unused on the GPU (kGpuViaSlots==1); sequential on lane 0
        auto const m = (lane == 0U)
                           ? (section_bike_filter ? update_route<true>(k, r)
                                                  : update_route<false>(k, r))
                           : false;
        route_any_marked = __any_sync(full, m);
      }
      if (route_any_marked && lane == 0U && !*any_marked_) {
        atomicOr(any_marked_, 1U);
      }
    }
  }

  __device__ void update_transfers(unsigned const k) {
    auto const global_t_id = get_global_thread_id();
    auto const global_stride = get_global_stride();
    for (auto i = global_t_id; i < n_locations_; i += global_stride) {
      if (!prev_station_mark_.test(i)) {
        continue;
      }

      auto const l = location_idx_t{i};

      for (auto v = 0U; v != Vias + 1; ++v) {
        auto const tmp_time = tmp_.get(l, v);
        if (tmp_time == kInvalid) {
          continue;
        }

        auto const target_v = v;
        auto const is_dest = is_dest_[i];

        auto const transfer_time =
            (!is_intermodal_dest() && is_dest)
                ? 0
                : dir(adjusted_transfer_time(transfer_time_settings_,
                                             tt_.transfer_time_[l].count()));
        auto const fp_target_time =
            static_cast<delta_t>(tmp_time + transfer_time);

        if (!is_better(fp_target_time, time_at_dest_.get(k)) ||
            lb_[i] == kUnreachable ||
            !is_better(fp_target_time + dir(lb_[i]), time_at_dest_.get(k))) {
          continue;
        }

        if (is_better(fp_target_time, best_.get(l, target_v))) {
          round_times_.update_min(k, l, target_v, fp_target_time,
                                  tmp_.get_bc(0U, l, v));
          best_.update_min(l, target_v, fp_target_time);
          station_mark_.mark(i);
          if (is_dest) {
            update_time_at_dest(k, fp_target_time);
          }
        }
      }
    }
  }

  __device__ void update_footpaths(unsigned const k) {
    auto const global_t_id = get_global_thread_id();
    auto const global_stride = get_global_stride();
    for (auto i = global_t_id; i < n_locations_; i += global_stride) {
      if (!prev_station_mark_.test(i)) {
        continue;
      }

      auto const l = location_idx_t{i};
      auto const& fps = kFwd ? tt_.footpaths_out_[l] : tt_.footpaths_in_[l];

      for (auto const& fp : fps) {
        auto const target = to_idx(fp.target());

        for (auto v = 0U; v != Vias + 1; ++v) {
          auto const tmp_time = tmp_.get(l, v);
          if (tmp_time == kInvalid) {
            continue;
          }

          auto const target_v = v;

          auto const fp_target_time = clamp(
              tmp_time + dir(adjusted_transfer_time(transfer_time_settings_,
                                                    fp.duration().count())));

          // Admissible pruning.
          if (!is_better(fp_target_time, time_at_dest_.get(k)) ||
              lb_[target] == kUnreachable ||
              !is_better(fp_target_time + dir(lb_[target]),
                         time_at_dest_.get(k))) {
            continue;
          }
          // Match the CPU (raptor.h update_footpaths): write round_times + best
          // and mark together, only on a genuine improvement over best_.
          if (is_better(fp_target_time, best_.get(fp.target(), target_v))) {
            round_times_.update_min(k, fp.target(), target_v, fp_target_time,
                                    tmp_.get_bc(0U, l, v));
            best_.update_min(fp.target(), target_v, fp_target_time);
            station_mark_.mark(target);
            if (target_v == Vias && is_dest_[target]) {
              update_time_at_dest(k, fp_target_time);
            }
          }
        }
      }
    }
  }

  __device__ void update_intermodal_footpaths(unsigned const k) {
    if (dist_to_end_.empty()) {
      return;
    }

    auto const global_t_id = get_global_thread_id();
    auto const global_stride = get_global_stride();
    for (auto i = global_t_id; i < n_locations_; i += global_stride) {
      if (!prev_station_mark_.test(i)) {
        continue;
      }

      auto const l = location_idx_t{i};
      if (prev_station_mark_[i] || station_mark_[i]) {
        if (dist_to_end_[i] != std::numeric_limits<std::uint16_t>::max()) {
          // Source the egress from tmp_ (this round's raw transport arrival),
          // like update_transfers / update_footpaths do -- NOT best_. best_
          // persists across rounds and range-query start-times, so keying off
          // it recorded a target arrival whose source has no valid
          // round_times[k][src] (unreconstructable -> dropped as a 0-leg
          // journey), and it also folds in no transfer, which the egress must
          // not add (RAPTOR does not chain transfer->egress). Reading tmp_
          // makes the egress round-consistent and independent of the
          // transfer/footpath passes (so no barrier is needed between them).
          auto const src_arr = tmp_.get(l, Vias);
          if (src_arr == kInvalid) {
            continue;
          }
          auto const end_time = clamp(src_arr + dir(dist_to_end_[i]));

          if (is_better(end_time, best_.get(kIntermodalTarget, Vias))) {
            // Make the egress source (l) reconstructable. Even with option A,
            // update_transfers may not have recorded round_times[k][l]: its
            // admissible prune drops l when l's arrival + transfer_time + lb[l]
            // can't beat time_at_dest, but the egress from l's RAW arrival +
            // dist_to_end[l] (a direct walk, typically shorter than the lb
            // estimate) still can. Reconstruction reads round_times[k][src], so
            // record l's transport arrival (+ its transfer time, matching
            // update_transfers) with tmp_'s transport breadcrumb. Don't touch
            // best_/station_mark_: this arrival is not better than best_, so it
            // must not re-propagate.
            round_times_.update_min(
                k, l, Vias,
                clamp(src_arr + dir(adjusted_transfer_time(
                                    transfer_time_settings_,
                                    tt_.transfer_time_[l].count()))),
                tmp_.get_bc(0U, l, Vias));
            round_times_.update_min(k, kIntermodalTarget, Vias, end_time,
                                    make_egress_bc(i));
            best_.update_min(kIntermodalTarget, Vias, end_time);
            update_time_at_dest(k, end_time);
          }
        }
      }
    }
  }

  // Warp-cooperative route scan (Vias==0, no section-bike-filter): one warp per
  // route, lane = stop in scan order. The boarding earliest-transport per stop
  // is precomputed by compute_et into et_result_ (packed (day<<32)|t_idx, which
  // is the per-route total order -> lower = earlier). A warp prefix-min of that
  // value gives, for every stop, the earliest transport boardable at any
  // earlier stop (carrying its board stop); each lane then propagates that
  // transport's arrival to its stop. This replaces the per-route sequential
  // scan (huge per-route work variance -> warps stalling at the round barrier).
  __device__ bool update_route_warp(unsigned const k,
                                    route_idx_t const r,
                                    unsigned const lane) {
    auto const stop_seq = tt_.route_location_seq_[r];
    auto const n = static_cast<unsigned>(stop_seq.size());
    auto const base_flat = tt_.route_stop_offset_[to_idx(r)];
    auto const full = 0xFFFFFFFFU;
    auto local_marked = false;

    std::uint64_t run_et = kEtInvalid;  // prefix-min et across chunks
    unsigned run_board_i = full;  // scan position where run_et was boarded

    for (auto chunk = 0U; chunk < n; chunk += 32U) {
      auto const i = chunk + lane;
      auto stop_idx = stop_idx_t{};
      std::uint64_t my_et = kEtInvalid;
      if (i < n) {
        stop_idx = static_cast<stop_idx_t>(kFwd ? i : n - 1U - i);
        my_et =
            et_result_[base_flat + stop_idx];  // gated boardable by compute_et
      }
      auto my_board_i = (my_et != kEtInvalid) ? i : full;

      // within-chunk inclusive prefix-min of (et value, board scan-pos)
      auto incl_et = my_et;
      auto incl_bi = my_board_i;
      for (auto off = 1U; off < 32U; off <<= 1) {
        auto const o_et = __shfl_up_sync(full, incl_et, off);
        auto const o_bi = __shfl_up_sync(full, incl_bi, off);
        if (lane >= off && (et_is_better(o_et, incl_et) ||
                            (!et_is_better(incl_et, o_et) && o_bi < incl_bi))) {
          incl_et = o_et;
          incl_bi = o_bi;
        }
      }
      // EXCLUSIVE prefix (stops strictly before this one) = inclusive of the
      // previous lane, combined with the carry from earlier chunks. The alight
      // must use a transport boarded *before* this stop -- if we boarded a
      // better transport *at* this stop, we still alight here on the previous
      // one (and the new one serves later stops).
      auto act_et = __shfl_up_sync(full, incl_et, 1);
      auto act_board_i = __shfl_up_sync(full, incl_bi, 1);
      if (lane == 0U) {
        act_et = kEtInvalid;
        act_board_i = full;
      }
      if (et_is_better(run_et, act_et) ||
          (!et_is_better(act_et, run_et) && run_board_i < act_board_i)) {
        act_et = run_et;
        act_board_i = run_board_i;
      }

      // propagate: alight here on the transport boarded at an earlier stop
      if (i < n && act_et != kEtInvalid && act_board_i < i) {
        auto const stp = stop{stop_seq[stop_idx]};
        if (stp.can_finish<SearchDir>(is_wheelchair_)) {
          auto const l = stp.location_idx();
          auto const l_idx = cista::to_idx(l);
          auto const et = unpack_et(act_et);
          auto const by_transport = time_at_stop(
              r, et, stop_idx, kFwd ? event_type::kArr : event_type::kDep);
          // Match the CPU (raptor.h update_route): write tmp_ and mark the
          // station based ONLY on the admissible time_at_dest+lb prune, NOT on
          // whether the arrival beats the station's best. A later same-round
          // TRANSPORT arrival that is worse than an earlier FOOTPATH arrival at
          // the same station is still essential: with non-transitive footpaths
          // (A-B and B-C exist but not A-C) an arrival at B that can't improve
          // best[B] is still needed, because the label that reached B via A->B
          // cannot label C, while a genuine arrival at B can now walk B->C. So
          // tmp_ (this round's transport arrivals) must record every arrival
          // passing the prune; round_times/best writes stay improvement-gated
          // (see update_transfers/update_footpaths).
          if (is_better(by_transport, time_at_dest_.get(k)) &&
              lb_[l_idx] != kUnreachable &&
              is_better(by_transport + dir(lb_[l_idx]), time_at_dest_.get(k))) {
            auto const board_stop = static_cast<stop_idx_t>(
                kFwd ? act_board_i : n - 1U - act_board_i);
            tmp_.update_min(
                l, 0U, by_transport,
                make_transport_payload(et.t_idx_.v_, board_stop, stop_idx));
            station_mark_.mark(l_idx);
            local_marked = true;
          }
        }
      }

      // carry forward the INCLUSIVE prefix (min over earlier chunks + this
      // one); lane 31's inclusive value is the chunk-wide min.
      auto new_run_et = incl_et;
      auto new_run_bi = incl_bi;
      if (et_is_better(run_et, new_run_et) ||
          (!et_is_better(new_run_et, run_et) && run_board_i < new_run_bi)) {
        new_run_et = run_et;
        new_run_bi = run_board_i;
      }
      run_et = __shfl_sync(full, new_run_et, 31);
      run_board_i = __shfl_sync(full, new_run_bi, 31);
    }

    return __any_sync(full, local_marked);
  }

  template <bool WithSectionBikeFilter>
  __device__ bool update_route(unsigned const k, route_idx_t const r) {
    auto const stop_seq = tt_.route_location_seq_[r];
    bool any_marked = false;

    auto et = std::array<transport, Vias + 1>{};
    auto v_offset = std::array<std::size_t, Vias + 1>{};
    // boarding stop_idx of the transport currently in et[v] (for breadcrumbs)
    auto et_board_stop = cuda::std::array<stop_idx_t, Vias + 1>{};

    for (auto i = 0U; i != stop_seq.size(); ++i) {
      auto const stop_idx =
          static_cast<stop_idx_t>(kFwd ? i : stop_seq.size() - i - 1U);
      auto const stp = stop{stop_seq[stop_idx]};
      auto const l = stp.location_idx();
      auto const l_idx = cista::to_idx(l);
      auto const is_first = i == 0U;
      auto const is_last = i == stop_seq.size() - 1U;

      auto current_best = cuda::std::array<delta_t, Vias + 1>{};
      current_best.fill(kInvalid);

      // v = via state when entering the transport
      // v + v_offset = via state at the current stop after entering the
      // transport (v_offset > 0 if the transport passes via stops)
      for (auto j = 0U; j != Vias + 1; ++j) {
        auto const v = Vias - j;
        if (!et[v].is_valid() && !prev_station_mark_[l_idx]) {
          continue;
        }

        if constexpr (WithSectionBikeFilter) {
          if (!is_first &&
              !const_cast<device_timetable const&>(tt_)
                   .route_bikes_allowed_per_section_[r][kFwd ? stop_idx - 1
                                                             : stop_idx]) {
            et[v] = {};
            v_offset[v] = 0;
          }
        }

        auto target_v = v + v_offset[v];

        if (et[v].is_valid() && stp.can_finish<SearchDir>(is_wheelchair_)) {
          auto const by_transport = time_at_stop(
              r, et[v], stop_idx, kFwd ? event_type::kArr : event_type::kDep);

          current_best[v] =
              get_best(round_times_.get(k - 1, l, target_v),
                       tmp_.get(l, target_v), best_.get(l, target_v));

          // See update_route_warp: mark on the admissible prune alone, not on
          // beating current_best (which folds in best_).
          if (is_better(by_transport, time_at_dest_.get(k))) {
            tmp_.update_min(l, target_v, by_transport,
                            make_transport_payload(et[v].t_idx_.v_,
                                                   et_board_stop[v], stop_idx));
            station_mark_.mark(l_idx);
            current_best[v] = by_transport;
            any_marked = true;
            debug("round %u: route=%u, marking l=%u\n", k, to_idx(r), l_idx);
          }
        }
      }

      if (is_last || !stp.can_start<SearchDir>(is_wheelchair_) ||
          !prev_station_mark_[l_idx]) {
        debug(
            "round %u: route=%u, stop=%u [l=%u] -> "
            "is_last=%s, can_start=%s, prev_station_mark=%s   => no new et\n",
            k, r.v_, stop_idx, l_idx, b2s(is_last),
            b2s(stp.can_start<SearchDir>(is_wheelchair_)),
            b2s(prev_station_mark_[l_idx]));
        continue;
      }

      for (auto v = 0U; v != Vias + 1; ++v) {
        if (!et[v].is_valid() && !prev_station_mark_[l_idx]) {
          debug(
              "round %u: route=%u, stop=%u [l=%u] -> et_valid=%s, "
              "prev_station_mark=%s   => no new et\n",
              k, r.v_, stop_idx, l_idx, b2s(et[v].is_valid()),
              b2s(prev_station_mark_[l_idx]));
          continue;
        }

        auto const target_v = v + v_offset[v];
        auto const et_time_at_stop =
            et[v].is_valid()
                ? time_at_stop(r, et[v], stop_idx,
                               kFwd ? event_type::kDep : event_type::kArr)
                : kInvalid;
        auto const prev_round_time = round_times_.get(k - 1, l, target_v);
        if (prev_round_time != kInvalid &&
            is_better_or_eq(prev_round_time, et_time_at_stop)) {
          transport new_et{};
          if constexpr (Vias == 0) {
            // boarding et precomputed by compute_et (load-balanced pre-pass)
            new_et = unpack_et(
                et_result_[tt_.route_stop_offset_[to_idx(r)] + stop_idx]);
          } else {
            auto const [day, mam] = split(prev_round_time);
            new_et = get_earliest_transport(k, r, stop_idx, day, mam,
                                            stp.location_idx());
          }
          current_best[v] = get_best(current_best[v], best_.get(l, target_v),
                                     tmp_.get(l, target_v));
          if (new_et.is_valid() &&
              (current_best[v] == kInvalid ||
               is_better_or_eq(
                   time_at_stop(r, new_et, stop_idx,
                                kFwd ? event_type::kDep : event_type::kArr),
                   et_time_at_stop))) {
            et[v] = new_et;
            v_offset[v] = 0;
            et_board_stop[v] = stop_idx;
            debug("round %u: route=%u, stop=%u [l=%u] -> transport=%u\n", k,
                  r.v_, stop_idx, l_idx, new_et.t_idx_.v_);
          }
        } else {
          debug(
              "round %u: route=%u, stop=%u [l=%u] -> prev_round_time=%d, "
              "time_at_stop=%d => no new et\n",
              k, r.v_, stop_idx, l_idx, prev_round_time, et_time_at_stop);
        }
      }
    }
    return any_marked;
  }

  __device__ transport
  get_earliest_transport(unsigned const k,
                         route_idx_t const r,
                         stop_idx_t const stop_idx,
                         day_idx_t const day_at_stop,
                         minutes_after_midnight_t const mam_at_stop,
                         location_idx_t const l) {
    auto const event_times = tt_.event_times_at_stop(
        r, stop_idx, kFwd ? event_type::kDep : event_type::kArr);

    auto const seek_first_day = [&]() {
      return linear_lb(get_begin_it(event_times), get_end_it(event_times),
                       mam_at_stop,
                       [&](delta const a, minutes_after_midnight_t const b) {
                         return is_better(a.mam(), b.count());
                       });
    };

    // Must match the CPU raptor: look up to kMaxTravelTime days ahead for the
    // next departure. The previous value (2) silently missed connections when
    // the next active service was >2 days out (sparse/rural service, or reduced
    // service near the timetable-start dates) -> GPU returned empty where CPU
    // found journeys.
    constexpr auto const kNDaysToIterate = static_cast<day_idx_t::value_t>(
        kMaxTravelTime / std::chrono::days{1} + 1U);
    for (auto i = day_idx_t::value_t{0U}; i != kNDaysToIterate; ++i) {
      auto const day = kFwd ? day_at_stop + i : day_at_stop - i;
      // Skip days on which this route runs no service before iterating its
      // events (matches the CPU raptor). Without this the wider day window
      // (kNDaysToIterate) would scan every event on every inactive day.
      if (!is_route_active(r, day)) {
        continue;
      }

      auto const ev_time_range =
          it_range{i == 0U ? seek_first_day() : get_begin_it(event_times),
                   get_end_it(event_times)};
      if (ev_time_range.empty()) {
        continue;
      }

      for (auto it = begin(ev_time_range); it != end(ev_time_range); ++it) {
        auto const t_offset =
            static_cast<std::size_t>(&*it - event_times.data());
        auto const ev = *it;
        auto const ev_mam = ev.mam();

        if (is_better_or_eq(time_at_dest_.get(k),
                            to_delta(day, ev_mam) + dir(lb_[to_idx(l)]))) {
          return {transport_idx_t::invalid(), day_idx_t::invalid()};
        }

        auto const t = tt_.route_transport_ranges_[r][t_offset];
        if (i == 0U && !is_better_or_eq(mam_at_stop.count(), ev_mam)) {
          continue;
        }

        auto const ev_day_offset = ev.days();
        auto const start_day =
            static_cast<std::size_t>(as_int(day) - ev_day_offset);
        if (!is_transport_active(t, start_day)) {
          continue;
        }
        return {t, static_cast<day_idx_t>(as_int(day) - ev_day_offset)};
      }
    }
    return {};
  }

  __device__ __forceinline__ bool is_transport_active(
      transport_idx_t const t, std::size_t const day) const {
    return tt_.bitfields_[tt_.transport_traffic_days_[t]].test(day);
  }

  __device__ __forceinline__ bool is_route_active(route_idx_t const r,
                                                  day_idx_t const day) const {
    return as_int(day) >= 0 && tt_.bitfields_[tt_.route_traffic_days_[r]].test(
                                   static_cast<std::size_t>(as_int(day)));
  }

  static constexpr std::uint64_t kEtInvalid = ~std::uint64_t{0};

  __device__ __forceinline__ std::uint64_t pack_et(transport const t) const {
    return t.is_valid() ? ((static_cast<std::uint64_t>(to_idx(t.day_)) << 32) |
                           static_cast<std::uint64_t>(to_idx(t.t_idx_)))
                        : kEtInvalid;
  }

  // Is packed et `a` a better boarding than `b`? Forward search wants the
  // earliest transport (min (day,t_idx)); backward wants the latest (max).
  // Invalid (kEtInvalid) is always worse.
  __device__ __forceinline__ bool et_is_better(std::uint64_t const a,
                                               std::uint64_t const b) const {
    if (a == kEtInvalid) {
      return false;
    }
    if (b == kEtInvalid) {
      return true;
    }
    return kFwd ? (a < b) : (a > b);
  }

  __device__ __forceinline__ transport unpack_et(std::uint64_t const p) const {
    return p == kEtInvalid
               ? transport{}
               : transport{transport_idx_t{
                               static_cast<std::uint32_t>(p & 0xFFFFFFFFU)},
                           day_idx_t{static_cast<std::uint16_t>(p >> 32)}};
  }

  __device__ void compute_et(unsigned const k) {
    if constexpr (Vias == 0) {
      auto const gid = get_global_thread_id();
      auto const stride = get_global_stride();
      auto const total = tt_.route_stop_offset_[tt_.n_routes_];

      // phase 1a: compact candidate tasks (default non-candidates to invalid).
      if (gid == 0U) {
        *et_task_count_ = 0U;
      }
      sync();
      for (auto flat = gid; flat < total; flat += stride) {
        auto const r = route_idx_t{tt_.route_of_stop_[flat]};
        if (!route_mark_.test(to_idx(r))) {
          continue;
        }
        et_result_[flat] = kEtInvalid;
        auto const stop_seq = tt_.route_location_seq_[r];
        auto const stop_idx =
            static_cast<stop_idx_t>(flat - tt_.route_stop_offset_[to_idx(r)]);
        auto const is_dir_last =
            kFwd ? (stop_idx + 1U == stop_seq.size()) : (stop_idx == 0U);
        if (is_dir_last) {
          continue;
        }
        auto const stp = stop{stop_seq[stop_idx]};
        auto const l = stp.location_idx();
        if (!prev_station_mark_[cista::to_idx(l)] ||
            !stp.can_start<SearchDir>(is_wheelchair_) ||
            round_times_.get(k - 1, l, 0U) == kInvalid) {
          continue;
        }
        et_task_list_[atomicAdd(et_task_count_, 1U)] = flat;
      }
      sync();

      // phase 1b: one earliest-transport lookup per task (no route divergence).
      auto const n_tasks = *et_task_count_;
      for (auto i = gid; i < n_tasks; i += stride) {
        auto const flat = et_task_list_[i];
        auto const r = route_idx_t{tt_.route_of_stop_[flat]};
        auto const stop_idx =
            static_cast<stop_idx_t>(flat - tt_.route_stop_offset_[to_idx(r)]);
        auto const stop_seq = tt_.route_location_seq_[r];
        auto const stp = stop{stop_seq[stop_idx]};
        auto const l = stp.location_idx();
        auto const [day, mam] = split(round_times_.get(k - 1, l, 0U));
        et_result_[flat] =
            pack_et(get_earliest_transport(k, r, stop_idx, day, mam, l));
      }
    }
  }

  __device__ delta_t time_at_stop(route_idx_t const r,
                                  transport const t,
                                  stop_idx_t const stop_idx,
                                  event_type const ev_type) {
    return to_delta(t.day_,
                    tt_.event_mam(r, t.t_idx_, stop_idx, ev_type).count());
  }

  __device__ delta_t to_delta(day_idx_t const day, std::int16_t const mam) {
    return clamp((as_int(day) - as_int(base_)) * 1440 + mam);
  }

  __device__ unixtime_t to_unix(delta_t const t) {
    return delta_to_unix(base(), t);
  }

  __device__ std::pair<day_idx_t, minutes_after_midnight_t> split(
      delta_t const x) {
    return split_day_mam(base_, x);
  }

  __device__ __forceinline__ bool is_intermodal_dest() const {
    return !dist_to_end_.empty();
  }

  __device__ void update_time_at_dest(unsigned const k, delta_t const t) {
    for (auto i = k; i != max_transfers_ + 1U; ++i) {
      time_at_dest_.update_min(i, t);
    }
  }

  __device__ __forceinline__ int as_int(day_idx_t const d) const {
    return static_cast<int>(d.v_);
  }

  template <typename T>
  __device__ __forceinline__ auto get_begin_it(T const& t) {
    if constexpr (kFwd) {
      return t.begin();
    } else {
      return t.rbegin();
    }
  }

  template <typename T>
  __device__ __forceinline__ auto get_end_it(T const& t) {
    if constexpr (kFwd) {
      return t.end();
    } else {
      return t.rend();
    }
  }

  std::uint32_t* any_marked_;
  device_timetable tt_;
  std::uint32_t n_locations_, n_routes_, n_rt_transports_;
  transfer_time_settings transfer_time_settings_;
  std::uint8_t max_transfers_;
  clasz_mask_t allowed_claszes_;
  bool require_bike_transport_;
  day_idx_t base_;
  unixtime_t worst_time_at_dest_;
  bool is_intermodal_dest_;
  bool is_wheelchair_;
  cuda::std::span<std::pair<location_idx_t, unixtime_t> const> starts_;
  device_bitvec<std::uint64_t const> is_dest_;
  device_bitvec<std::uint32_t> end_reachable_;
  cuda::std::span<std::uint16_t const> dist_to_end_;
  cuda::std::span<std::uint16_t const> lb_;  // per-location lower bound to dest
  device_times<SearchDir, Vias + 1> round_times_;
  device_times<SearchDir, Vias + 1> best_;
  device_times<SearchDir, Vias + 1> tmp_;
  device_times<SearchDir, 1U> time_at_dest_;
  device_bitvec<std::uint32_t> station_mark_;
  device_bitvec<std::uint32_t> prev_station_mark_;
  device_bitvec<std::uint32_t> route_mark_;
  cuda::std::span<std::uint64_t> et_result_;  // packed et per flat (route,stop)
  cuda::std::span<std::uint32_t>
      et_task_list_;  // compacted boarding candidates
  std::uint32_t* et_task_count_;  // number of tasks this round
};

}  // namespace nigiri::routing::gpu