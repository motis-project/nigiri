#pragma once

#include <cassert>
#include <span>
#include <type_traits>

#include "nigiri/common/delta_t.h"
#include "nigiri/common/linear_lower_bound.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/routing/raptor/debug.h"
#include "nigiri/routing/raptor/raptor_state.h"
#include "nigiri/routing/raptor/raptor_stats.h"
#include "nigiri/routing/raptor/reconstruct.h"
#include "nigiri/routing/raptor/schedrt_counters.h"
#include "nigiri/routing/raptor/via_criterion.h"
#include "nigiri/routing/transfer_time_settings.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/special_stations.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::routing {

enum class search_mode { kOneToOne, kOneToAll };

template <direction SearchDir,
          bool Rt,
          typename Crit,
          search_mode SearchMode>
struct basic_raptor {
  using algo_state_t = raptor_state;
  using algo_stats_t = raptor_stats;
  using bag_t = typename Crit::bag_t;

  // number of label slots per stop + the slot journeys are read from
  static constexpr auto const kN = Crit::kN;
  static constexpr auto const kFinalSlot = Crit::kFinalSlot;
  // copy-on-diverge: scheduled base plane + sparse rt overlay
  static constexpr auto const kCoD = Crit::kCopyOnDiverge;
  // reset_slots_ tag distinguishing the CoD plane layout from slot layouts
  static constexpr auto const kCoDResetTag = std::uint8_t{0xC0U};
  // raptor_state is allocated kMaxVias + 1 slots wide
  static constexpr auto const kLastSlot = static_cast<via_offset_t>(kN - 1U);
  static_assert(kN <= kMaxVias + 1U);
  static_assert(std::is_same_v<bag_t, std::array<delta_t, kN>>);

  static constexpr bool kUseLowerBounds = true;
  static constexpr auto const kFwd = (SearchDir == direction::kForward);
  static constexpr auto const kBwd = (SearchDir == direction::kBackward);
  static constexpr auto const kInvalid = kInvalidDelta<SearchDir>;
  static constexpr auto const kUnreachable =
      std::numeric_limits<std::uint16_t>::max();
  static constexpr auto const kIntermodalTarget =
      to_idx(get_special_station(special_station::kEnd));
  static constexpr auto const kInvalidArray = []() {
    auto a = bag_t{};
    a.fill(kInvalid);
    return a;
  }();
  // one time-at-dest bound per criterion dest bound (per world for sched/rt)
  using dest_bounds_t = std::array<delta_t, Crit::kNDestBounds>;
  static constexpr auto const kInvalidBounds = []() {
    auto a = dest_bounds_t{};
    a.fill(kInvalid);
    return a;
  }();

  // below this size, full sequential sweeps beat touched-set iteration
  // (bit-scan + strided row writes): ger @ 500k loses ~8%, EU @ 6M wins ~20%
  static constexpr auto const kSparseStateThreshold = 1'000'000U;
  bool use_sparse() const { return n_locations_ >= kSparseStateThreshold; }

  static bool is_better(auto a, auto b) { return kFwd ? a < b : a > b; }
  static bool is_better_or_eq(auto a, auto b) { return kFwd ? a <= b : a >= b; }
  static auto get_best(auto a, auto b) { return is_better(a, b) ? a : b; }
  static auto get_best(auto x, auto... y) {
    ((x = get_best(x, y)), ...);
    return x;
  }
  static auto dir(auto a) { return (kFwd ? 1 : -1) * a; }

  basic_raptor(
      timetable const& tt,
      rt_timetable const* rtt,
      raptor_state& state,
      bitvec& is_dest,
      std::array<bitvec, kMaxVias>& is_via,
      std::vector<std::uint16_t>& dist_to_dest,
      hash_map<location_idx_t, std::vector<td_offset>> const& td_dist_to_dest,
      std::vector<std::uint16_t>& lb,
      std::vector<via_stop> const& via_stops,
      day_idx_t const base,
      clasz_mask_t const allowed_claszes,
      bool const require_bike_transport,
      bool const require_car_transport,
      bool const is_wheelchair,
      bool const no_compulsory_reservation,
      transfer_time_settings const& tts,
      profile_idx_t const prf_idx)
      : tt_{tt},
        rtt_{rtt},
        n_days_{tt_.internal_interval_days().size().count()},
        n_locations_{tt_.n_locations()},
        n_routes_{tt.n_routes()},
        n_rt_transports_{Rt ? rtt->n_rt_transports() : 0U},
        state_{state.resize(n_locations_, n_routes_, n_rt_transports_)},
        tmp_{state_.get_tmp<kLastSlot>()},
        best_{state_.get_best<kLastSlot>()},
        round_times_{state.get_round_times<kLastSlot>()},
        tmp0_{state_.get_tmp_plane(0U)},
        tmp1_{state_.get_tmp_plane(1U)},
        best0_{state_.get_best_plane(0U)},
        best1_{state_.get_best_plane(1U)},
        rt0_{state_.get_round_times_plane(0U)},
        rt1_{state_.get_round_times_plane(1U)},
        is_dest_{is_dest},
        crit_{is_via, via_stops},
        dist_to_end_{dist_to_dest},
        td_dist_to_end_{td_dist_to_dest},
        lb_{lb},
        base_{base},
        bounds_{std::as_const(state_).template get_bounds<kLastSlot>()},
        prf_idx_{prf_idx},
        allowed_claszes_{allowed_claszes},
        require_bike_transport_{require_bike_transport},
        require_car_transport_{require_car_transport},
        no_compulsory_reservation_{no_compulsory_reservation},
        is_wheelchair_{is_wheelchair},
        transfer_time_settings_{tts} {
    reset_arrivals();
    if constexpr (kCoD) {
      auto const version = rtt_ == nullptr ? 0U : rtt_->bitfields_.size();
      if (state_.route_has_rt_src_ != rtt_ ||
          state_.route_has_rt_version_ != version) {
        state_.route_has_rt_src_ = rtt_;
        state_.route_has_rt_version_ = version;
        state_.route_has_rt_.resize(tt_.n_routes());
        state_.route_has_rt_.zero_out();
        if (rtt_ != nullptr) {
          auto const n_transports = tt_.transport_route_.size();
          for (auto i = 0U; i != n_transports; ++i) {
            auto const t = transport_idx_t{i};
            if (rtt_->has_rt_traffic_days(t)) {
              state_.route_has_rt_.set(to_idx(tt_.transport_route_[t]), true);
            }
          }
        }
      }
    }
    if (!dist_to_end_.empty()) {
      // only used for intermodal queries (dist_to_dest != empty)
      end_reachable_.resize(n_locations_);
      for (auto i = 0U; i != dist_to_end_.size(); ++i) {
        if (dist_to_end_[i] != kUnreachable) {
          end_reachable_.set(i, true);
        }
      }
      for (auto const& [l, _] : td_dist_to_end_) {
        end_reachable_.set(to_idx(l), true);
      }
    }
  }

  algo_stats_t get_stats() const { return stats_; }

  void fill_bounds(std::size_t const n_rows) {
    auto& s = state_;
    auto const n = static_cast<std::size_t>(s.n_locations_);

    auto const td_stops = rtt_ != nullptr && prf_idx_ != 0U
                              ? &(kFwd ? rtt_->has_td_footpaths_out_
                                       : rtt_->has_td_footpaths_in_)[prf_idx_]
                              : nullptr;

    auto const src = std::as_const(s).template get_round_times<kLastSlot>();
    auto dst = s.template get_bounds<kLastSlot>();

    if constexpr (kCoD) {
      // materialize per-world bounds from the base plane + overlay
      for (auto x = std::size_t{0U}; x != n; ++x) {
        dst[0U][x] = bag_t{rt0_[0U][x][0], rt_read1(0U, x)};
      }
      for (auto k = std::size_t{1U}; k < n_rows; ++k) {
        for (auto x = std::size_t{0U}; x != n; ++x) {
          auto const s0 = rt0_[k][x][0];
          auto const s1 = rt_read1(static_cast<unsigned>(k), x);
          dst[k][x][0] = kFwd ? std::min(s0, dst[k - 1U][x][0])
                              : std::max(s0, dst[k - 1U][x][0]);
          dst[k][x][1] = kFwd ? std::min(s1, dst[k - 1U][x][1])
                              : std::max(s1, dst[k - 1U][x][1]);
        }
      }
      if (td_stops != nullptr) {
        constexpr auto const kPassAll =
            kFwd ? std::numeric_limits<delta_t>::min()
                 : std::numeric_limits<delta_t>::max();
        td_stops->for_each_set_bit([&](location_idx_t const x) {
          for (auto k = std::size_t{0U}; k != n_rows; ++k) {
            for (auto v = std::size_t{0U}; v != kN; ++v) {
              dst[k][to_idx(x)][v] = kPassAll;
            }
          }
        });
      }
      return;
    }

    // Copy k=0 verbatim (rest is folded from here).
    for (auto x = std::size_t{0U}; x != n; ++x) {
      dst[0U][x] = src[0U][x];
    }

    // Fill gaps from lower rounds to higher rounds.
    for (auto k = std::size_t{1U}; k < n_rows; ++k) {
      for (auto x = std::size_t{0U}; x != n; ++x) {
        for (auto v = std::size_t{0U}; v != kN; ++v) {
          dst[k][x][v] = kFwd ? std::min(src[k][x][v], dst[k - 1U][x][v])
                              : std::max(src[k][x][v], dst[k - 1U][x][v]);
        }
      }
    }

    if constexpr (kN > 1U) {
      // Fold each slot with the slots dominating it.
      for (auto k = std::size_t{0U}; k != n_rows; ++k) {
        for (auto x = std::size_t{0U}; x != n; ++x) {
          crit_.fold_bounds(dst[k][x]);
        }
      }
    }

    if (td_stops != nullptr) {
      // td_footpaths have no upper bound -> disable pruning
      constexpr auto const kPassAll = kFwd
                                          ? std::numeric_limits<delta_t>::min()
                                          : std::numeric_limits<delta_t>::max();
      td_stops->for_each_set_bit([&](location_idx_t const x) {
        for (auto k = std::size_t{0U}; k != n_rows; ++k) {
          for (auto v = std::size_t{0U}; v != kN; ++v) {
            dst[k][to_idx(x)][v] = kPassAll;
          }
        }
      });
    }
  }

  void set_bounds(unsigned const last_round) { bounds_last_k_ = last_round; }

  void reset_arrivals() {
    if constexpr (kCoD) {
      touch_marked();
      utl::fill(time_at_dest_, kInvalidBounds);
      constexpr auto const kInv1 = std::array<delta_t, 1>{kInvalid};
      if (!use_sparse() || state_.needs_full_reset_ ||
          state_.reset_slots_ != kCoDResetTag || state_.reset_fwd_ != kFwd) {
        rt0_.reset(kInv1);
        utl::fill(tmp0_, kInvalid);
        utl::fill(best0_, kInvalid);
        utl::fill(state_.diverged_.blocks_, 0U);
        utl::fill(state_.diverged_best_.blocks_, 0U);
        state_.needs_full_reset_ = false;
        state_.reset_slots_ = kCoDResetTag;
        state_.reset_fwd_ = kFwd;
      } else {
        state_.touched_.for_each_set_bit([&](std::uint64_t const i) {
          for (auto k = 0U; k != kMaxTransfers + 2U; ++k) {
            rt0_[k][i][0] = kInvalid;
          }
          best0_[i] = kInvalid;
          tmp0_[i] = kInvalid;
        });
        for (auto const i : state_.diverged_list_) {
          state_.diverged_.set(static_cast<bitvec::size_type>(i), false);
        }
        for (auto const l : state_.diverged_best_list_) {
          state_.diverged_best_.set(l, false);
        }
      }
      state_.diverged_list_.clear();
      state_.diverged_best_list_.clear();
      utl::fill(state_.touched_.blocks_, 0U);
      if (is_intermodal_dest()) {
        state_.touched_.set(kIntermodalTarget, true);
      }
      return;
    }
    touch_marked();  // fold pending marks: execute() may break with marks set
    if constexpr (Crit::kFusedEt && NIGIRI_SCHEDRT_COUNTERS != 0) {
      // divergence instrumentation: how many touched cells differ per world
      auto& c = get_schedrt_divergence_counters();
      auto eq = std::uint64_t{0U}, ne = std::uint64_t{0U};
      state_.touched_.for_each_set_bit([&](std::uint64_t const i) {
        for (auto k = 0U; k != kMaxTransfers + 2U; ++k) {
          auto const& cell = round_times_[k][i];
          if (cell[0] == kInvalid && cell[1] == kInvalid) {
            continue;
          }
          (cell[0] == cell[1] ? eq : ne) += 1U;
        }
      });
      c.cells_equal_ += eq;
      c.cells_diverged_ += ne;
    }
    utl::fill(time_at_dest_, kInvalidBounds);
    if (!use_sparse() || state_.needs_full_reset_ ||
        state_.reset_slots_ != kN || state_.reset_fwd_ != kFwd) {
      round_times_.reset(kInvalidArray);
      utl::fill(best_, kInvalidArray);
      utl::fill(tmp_, kInvalidArray);
      state_.needs_full_reset_ = false;
      state_.reset_slots_ = static_cast<std::uint8_t>(kN);
      state_.reset_fwd_ = kFwd;
    } else {
      // rows outside touched_ still hold kInvalid from the previous reset
      state_.touched_.for_each_set_bit([&](std::uint64_t const i) {
        for (auto k = 0U; k != kMaxTransfers + 2U; ++k) {
          round_times_[k][i] = kInvalidArray;
        }
        best_[i] = kInvalidArray;
        tmp_[i] = kInvalidArray;
      });
    }
    utl::fill(state_.touched_.blocks_, 0U);
    if (is_intermodal_dest()) {
      // the intermodal target is written without a station mark
      state_.touched_.set(kIntermodalTarget, true);
    }
  }

  void next_start_time() {
    if constexpr (kCoD) {
      touch_marked();
      if (use_sparse()) {
        state_.touched_.for_each_set_bit([&](std::uint64_t const i) {
          best0_[i] = kInvalid;
          tmp0_[i] = kInvalid;
        });
      } else {
        utl::fill(best0_, kInvalid);
        utl::fill(tmp0_, kInvalid);
      }
      for (auto const l : state_.diverged_best_list_) {
        state_.diverged_best_.set(l, false);
      }
      state_.diverged_best_list_.clear();
      utl::fill(state_.prev_station_mark_.blocks_, 0U);
      utl::fill(state_.station_mark_.blocks_, 0U);
      utl::fill(state_.route_mark_.blocks_, 0U);
      if constexpr (Rt) {
        utl::fill(state_.rt_transport_mark_.blocks_, 0U);
      }
      return;
    }
    touch_marked();  // fold pending marks: execute() may break with marks set
    if (use_sparse()) {
      // best_/tmp_ writes always come with a station mark, and marks are
      // folded into touched_ before they are cleared
      state_.touched_.for_each_set_bit([&](std::uint64_t const i) {
        best_[i] = kInvalidArray;
        tmp_[i] = kInvalidArray;
      });
    } else {
      utl::fill(best_, kInvalidArray);
      utl::fill(tmp_, kInvalidArray);
    }
    utl::fill(state_.prev_station_mark_.blocks_, 0U);
    utl::fill(state_.station_mark_.blocks_, 0U);
    utl::fill(state_.route_mark_.blocks_, 0U);
    if constexpr (Rt) {
      utl::fill(state_.rt_transport_mark_.blocks_, 0U);
    }
  }

  void touch_marked() {
    auto& t = state_.touched_.blocks_;
    auto const& m = state_.station_mark_.blocks_;
    for (auto j = std::size_t{0U}; j != t.size(); ++j) {
      t[j] |= m[j];
    }
  }

  // ==== copy-on-diverge helpers ====
  // rt-world reads fall through to the scheduled base unless diverged;
  // divergence materializes the overlay from the base BEFORE any write
  bool rt_diverged(unsigned const k, std::size_t const l) const {
    return state_.diverged_.test(
        static_cast<bitvec::size_type>(k * n_locations_ + l));
  }

  delta_t rt_read1(unsigned const k, std::size_t const l) const {
    return rt_diverged(k, l) ? rt1_[k][l][0] : rt0_[k][l][0];
  }

  void rt_diverge(unsigned const k, std::size_t const l) {
    auto const i = static_cast<bitvec::size_type>(k * n_locations_ + l);
    if (!state_.diverged_.test(i)) {
      rt1_[k][l][0] = rt0_[k][l][0];
      state_.diverged_.set(i, true);
      state_.diverged_list_.push_back(i);
    }
  }

  bool bt_diverged(std::size_t const l) const {
    return state_.diverged_best_.test(static_cast<bitvec::size_type>(l));
  }

  delta_t best_read1(std::size_t const l) const {
    return bt_diverged(l) ? best1_[l] : best0_[l];
  }

  delta_t tmp_read1(std::size_t const l) const {
    return bt_diverged(l) ? tmp1_[l] : tmp0_[l];
  }

  void bt_diverge(std::size_t const l) {
    if (!bt_diverged(l)) {
      best1_[l] = best0_[l];
      tmp1_[l] = tmp0_[l];
      state_.diverged_best_.set(static_cast<bitvec::size_type>(l), true);
      state_.diverged_best_list_.push_back(static_cast<std::uint32_t>(l));
    }
  }

  // monotone min-merge of per-world values into a round_times cell;
  // a single-world or unequal write forces the cell to diverge first
  void cod_cell_min(unsigned const k,
                    std::size_t const l,
                    bool const w0,
                    bool const w1,
                    delta_t const v0,
                    delta_t const v1) {
    if (w0 && w1 && v0 == v1 && !rt_diverged(k, l)) {
      rt0_[k][l][0] = get_best(v0, rt0_[k][l][0]);
      return;
    }
    if (w0 != w1 || v0 != v1) {
      rt_diverge(k, l);
    }
    if (w0) {
      rt0_[k][l][0] = get_best(v0, rt0_[k][l][0]);
    }
    if (w1) {
      if (rt_diverged(k, l)) {
        rt1_[k][l][0] = get_best(v1, rt1_[k][l][0]);
      } else {
        rt0_[k][l][0] = get_best(v1, rt0_[k][l][0]);
      }
    }
  }

  void cod_best_min(std::size_t const l,
                    bool const w0,
                    bool const w1,
                    delta_t const v0,
                    delta_t const v1) {
    if (w0 && w1 && v0 == v1 && !bt_diverged(l)) {
      best0_[l] = get_best(v0, best0_[l]);
      return;
    }
    if (w0 != w1 || v0 != v1) {
      bt_diverge(l);
    }
    if (w0) {
      best0_[l] = get_best(v0, best0_[l]);
    }
    if (w1) {
      if (bt_diverged(l)) {
        best1_[l] = get_best(v1, best1_[l]);
      } else {
        best0_[l] = get_best(v1, best0_[l]);
      }
    }
  }

  void cod_tmp_min(std::size_t const l,
                   bool const w0,
                   bool const w1,
                   delta_t const v0,
                   delta_t const v1) {
    if (w0 && w1 && v0 == v1 && !bt_diverged(l)) {
      tmp0_[l] = get_best(v0, tmp0_[l]);
      return;
    }
    if (w0 != w1 || v0 != v1) {
      bt_diverge(l);
    }
    if (w0) {
      tmp0_[l] = get_best(v0, tmp0_[l]);
    }
    if (w1) {
      if (bt_diverged(l)) {
        tmp1_[l] = get_best(v1, tmp1_[l]);
      } else {
        tmp0_[l] = get_best(v1, tmp0_[l]);
      }
    }
  }

  void add_start(location_idx_t const l, unixtime_t const t) {
    if constexpr (kCoD) {
      // both worlds share every start label: base plane only, no divergence
      auto const d = unix_to_delta(base(), t);
      best0_[to_idx(l)] = get_best(d, best0_[to_idx(l)]);
      rt0_[0U][to_idx(l)][0] = get_best(d, rt0_[0U][to_idx(l)][0]);
      state_.station_mark_.set(to_idx(l), true);
      return;
    }
    crit_.for_each_start_slot(to_idx(l), [&](auto const v) {
      trace_upd(
          "adding start [fwd={}] {}: {}, v={} [current: best={}, round={} => "
          "best={}]\n",
          kFwd, loc{tt_, l}, t, v, to_unix(best_[to_idx(l)][v]),
          to_unix(round_times_[0U][to_idx(l)][v]),
          get_best(t, to_unix(best_[to_idx(l)][v])));
      best_[to_idx(l)][v] =
          get_best(unix_to_delta(base(), t), best_[to_idx(l)][v]);
      round_times_[0U][to_idx(l)][v] =
          get_best(unix_to_delta(base(), t), round_times_[0U][to_idx(l)][v]);
    });
    state_.station_mark_.set(to_idx(l), true);
  }

  void execute(unixtime_t const start_time,
               std::uint8_t const max_transfers,
               unixtime_t const worst_time_at_dest,
               pareto_set<journey>& results) {
    auto const end_k = std::min(max_transfers, kMaxTransfers) + 2U;

    auto const d_worst_at_dest = unix_to_delta(base(), worst_time_at_dest);
    for (auto& bounds : time_at_dest_) {
      for (auto& b : bounds) {
        b = get_best(d_worst_at_dest, b);
      }
    }

    trace_print_init_state();

    touch_marked();

    for (auto k = 1U; k != end_k; ++k) {
      // fold this round's times into best_: rows outside touched_ are
      // all-kInvalid, so only touched rows can contribute
      if constexpr (kCoD) {
        state_.touched_.for_each_set_bit([&](std::uint64_t const i) {
          if (rt_diverged(k, i) || bt_diverged(i)) {
            bt_diverge(i);
            best1_[i] = get_best(rt_read1(k, i), best1_[i]);
          }
          best0_[i] = get_best(rt0_[k][i][0], best0_[i]);
        });
      } else if (use_sparse()) {
        state_.touched_.for_each_set_bit([&](std::uint64_t const i) {
          for (auto v = std::size_t{0U}; v != kN; ++v) {
            best_[i][v] = get_best(round_times_[k][i][v], best_[i][v]);
          }
        });
      } else {
        for (auto i = 0U; i != n_locations_; ++i) {
          for (auto v = std::size_t{0U}; v != kN; ++v) {
            best_[i][v] = get_best(round_times_[k][i][v], best_[i][v]);
          }
        }
      }
      is_dest_.for_each_set_bit([&](std::uint64_t const i) {
        if constexpr (kCoD) {
          update_time_at_dest(k, 0U, best0_[i]);
          update_time_at_dest(k, 1U, best_read1(i));
        } else {
          Crit::for_each_dest_slot(
              [&](auto const v) { update_time_at_dest(k, v, best_[i][v]); });
        }
      });

      auto any_marked = false;
      state_.station_mark_.for_each_set_bit([&](std::uint64_t const i) {
        for (auto const& r : tt_.location_routes_[location_idx_t{i}]) {
          any_marked = true;
          state_.route_mark_.set(to_idx(r), true);
        }
        if constexpr (Rt) {
          for (auto const& rt_t :
               rtt_->location_rt_transports_[location_idx_t{i}]) {
            any_marked = true;
            state_.rt_transport_mark_.set(to_idx(rt_t), true);
          }
        }
      });

      if (!any_marked) {
        trace_print_state_after_round();
        break;
      }

      touch_marked();
      std::swap(state_.prev_station_mark_, state_.station_mark_);
      utl::fill(state_.station_mark_.blocks_, 0U);

      bool const clasz_filter = allowed_claszes_ != all_clasz_allowed();
      uint8_t const filters =
          static_cast<uint8_t>(clasz_filter << 4) |
          static_cast<uint8_t>(require_bike_transport_ << 3) |
          static_cast<uint8_t>(require_car_transport_ << 2) |
          static_cast<uint8_t>(is_wheelchair_ << 1) |
          static_cast<uint8_t>(no_compulsory_reservation_ << 0);

      any_marked |= [&]() {
        switch (filters) {
          case 0b00000:
            return loop_routes<false, false, false, false, false>(k);
          case 0b00001: return loop_routes<false, false, false, false, true>(k);
          case 0b00010: return loop_routes<false, false, false, true, false>(k);
          case 0b00011: return loop_routes<false, false, false, true, true>(k);
          case 0b00100: return loop_routes<false, false, true, false, false>(k);
          case 0b00101: return loop_routes<false, false, true, false, true>(k);
          case 0b00110: return loop_routes<false, false, true, true, false>(k);
          case 0b00111: return loop_routes<false, false, true, true, true>(k);
          case 0b01000: return loop_routes<false, true, false, false, false>(k);
          case 0b01001: return loop_routes<false, true, false, false, true>(k);
          case 0b01010: return loop_routes<false, true, false, true, false>(k);
          case 0b01011: return loop_routes<false, true, false, true, true>(k);
          case 0b01100: return loop_routes<false, true, true, false, false>(k);
          case 0b01101: return loop_routes<false, true, true, false, true>(k);
          case 0b01110: return loop_routes<false, true, true, true, false>(k);
          case 0b01111: return loop_routes<false, true, true, true, true>(k);
          case 0b10000: return loop_routes<true, false, false, false, false>(k);
          case 0b10001: return loop_routes<true, false, false, false, true>(k);
          case 0b10010: return loop_routes<true, false, false, true, false>(k);
          case 0b10011: return loop_routes<true, false, false, true, true>(k);
          case 0b10100: return loop_routes<true, false, true, false, false>(k);
          case 0b10101: return loop_routes<true, false, true, false, true>(k);
          case 0b10110: return loop_routes<true, false, true, true, false>(k);
          case 0b10111: return loop_routes<true, false, true, true, true>(k);
          case 0b11000: return loop_routes<true, true, false, false, false>(k);
          case 0b11001: return loop_routes<true, true, false, false, true>(k);
          case 0b11010: return loop_routes<true, true, false, true, false>(k);
          case 0b11011: return loop_routes<true, true, false, true, true>(k);
          case 0b11100: return loop_routes<true, true, true, false, false>(k);
          case 0b11101: return loop_routes<true, true, true, false, true>(k);
          case 0b11110: return loop_routes<true, true, true, true, false>(k);
          case 0b11111: return loop_routes<true, true, true, true, true>(k);
          default: std::unreachable();
        }
      }();

      if constexpr (Rt) {
        any_marked |= [&]() {
          switch (filters) {
            case 0b00000:
              return loop_rt_routes<false, false, false, false, false>(k);
            case 0b00001:
              return loop_rt_routes<false, false, false, false, true>(k);
            case 0b00010:
              return loop_rt_routes<false, false, false, true, false>(k);
            case 0b00011:
              return loop_rt_routes<false, false, false, true, true>(k);
            case 0b00100:
              return loop_rt_routes<false, false, true, false, false>(k);
            case 0b00101:
              return loop_rt_routes<false, false, true, false, true>(k);
            case 0b00110:
              return loop_rt_routes<false, false, true, true, false>(k);
            case 0b00111:
              return loop_rt_routes<false, false, true, true, true>(k);
            case 0b01000:
              return loop_rt_routes<false, true, false, false, false>(k);
            case 0b01001:
              return loop_rt_routes<false, true, false, false, true>(k);
            case 0b01010:
              return loop_rt_routes<false, true, false, true, false>(k);
            case 0b01011:
              return loop_rt_routes<false, true, false, true, true>(k);
            case 0b01100:
              return loop_rt_routes<false, true, true, false, false>(k);
            case 0b01101:
              return loop_rt_routes<false, true, true, false, true>(k);
            case 0b01110:
              return loop_rt_routes<false, true, true, true, false>(k);
            case 0b01111:
              return loop_rt_routes<false, true, true, true, true>(k);
            case 0b10000:
              return loop_rt_routes<true, false, false, false, false>(k);
            case 0b10001:
              return loop_rt_routes<true, false, false, false, true>(k);
            case 0b10010:
              return loop_rt_routes<true, false, false, true, false>(k);
            case 0b10011:
              return loop_rt_routes<true, false, false, true, true>(k);
            case 0b10100:
              return loop_rt_routes<true, false, true, false, false>(k);
            case 0b10101:
              return loop_rt_routes<true, false, true, false, true>(k);
            case 0b10110:
              return loop_rt_routes<true, false, true, true, false>(k);
            case 0b10111:
              return loop_rt_routes<true, false, true, true, true>(k);
            case 0b11000:
              return loop_rt_routes<true, true, false, false, false>(k);
            case 0b11001:
              return loop_rt_routes<true, true, false, false, true>(k);
            case 0b11010:
              return loop_rt_routes<true, true, false, true, false>(k);
            case 0b11011:
              return loop_rt_routes<true, true, false, true, true>(k);
            case 0b11100:
              return loop_rt_routes<true, true, true, false, false>(k);
            case 0b11101:
              return loop_rt_routes<true, true, true, false, true>(k);
            case 0b11110:
              return loop_rt_routes<true, true, true, true, false>(k);
            case 0b11111:
              return loop_rt_routes<true, true, true, true, true>(k);
            default: std::unreachable();
          }
        }();
      }

      if (!any_marked) {
        trace_print_state_after_round();
        break;
      }

      utl::fill(state_.route_mark_.blocks_, 0U);
      utl::fill(state_.rt_transport_mark_.blocks_, 0U);

      touch_marked();
      std::swap(state_.prev_station_mark_, state_.station_mark_);
      utl::fill(state_.station_mark_.blocks_, 0U);

      update_transfers(k);
      update_intermodal_footpaths(k);
      update_footpaths(k);
      update_td_offsets(k);

      trace_print_state_after_round();
    }

    if constexpr (SearchMode == search_mode::kOneToAll) {
      return;
    }

    is_dest_.for_each_set_bit([&](auto const i) {
      for (auto k = 1U; k != end_k; ++k) {
        Crit::for_each_dest_slot([&](auto const v) {
          auto const dest_time = [&] {
            if constexpr (kCoD) {
              return v == 0U ? rt0_[k][i][0] : rt_read1(k, i);
            } else {
              return round_times_[k][i][v];
            }
          }();
          if (dest_time != kInvalid) {
            trace("ADDING JOURNEY: start={}, dest={} @ {}, transfers={}\n",
                  start_time, delta_to_unix(base(), round_times_[k][i][v]),
                  loc{tt_, location_idx_t{i}}, k - 1);
            auto const [optimal, it, dominated_by] = results.add(
                journey{.legs_ = {},
                        .start_time_ = start_time,
                        .dest_time_ = delta_to_unix(base(), dest_time),
                        .dest_ = location_idx_t{i},
                        .transfers_ = static_cast<std::uint8_t>(k - 1),
                        .slot_ = Crit::journey_slot(v)});
            if (!optimal) {
              trace("  DOMINATED BY: start={}, dest={} @ {}, transfers={}\n",
                    dominated_by->start_time_, dominated_by->dest_time_,
                    loc{tt_, dominated_by->dest_}, dominated_by->transfers_);
            }
          }
        });
      }
    });
  }

  void reconstruct(query const& q, journey& j) {
    if constexpr (SearchMode == search_mode::kOneToAll) {
      return;
    }
    trace("reconstruct({} - {}, {} transfers", j.departure_time(),
          j.arrival_time(), j.transfers_);
    if constexpr (kCoD) {
      if (j.slot_ == 0U) {
        // base plane == the plain single-slot layout
        reconstruct_journey<SearchDir>(tt_, nullptr, q, state_, j, base(),
                                       base_);
      } else {
        reconstruct_journey_schedrt_cod<SearchDir>(tt_, rtt_, q, state_, j,
                                                   base(), base_);
      }
    } else if constexpr (Crit::kN == 2U && !Crit::kAllSlotsRt) {
      reconstruct_journey_schedrt<SearchDir>(tt_, rtt_, q, state_, j, base(),
                                             base_);
    } else {
      reconstruct_journey<SearchDir>(tt_, rtt_, q, state_, j, base(), base_);
    }
  }

private:
  date::sys_days base() const {
    return tt_.internal_interval_days().from_ + as_int(base_) * date::days{1};
  }

  std::uint16_t get_lb(std::uint32_t const i) const {
    if constexpr (kUseLowerBounds) {
      assert(i < lb_.size());
      return lb_[i];
    } else {
      return 0U;
    }
  }

  bool lb_reachable(std::uint32_t const i) const {
    if constexpr (kUseLowerBounds) {
      assert(i < lb_.size());
      return lb_[i] != kUnreachable;
    } else {
      return true;
    }
  }

  // For ping: comparison has to be <= instead of < so cells with equal arrival
  // times are still populated. Otherwise, the ping bounds are too strict for
  // pong to still find all optimal journeys.
  //
  // Loose pruning with <= instead of < is always enabled on the CPU.
  // Measured to have ~ the same performance as strict pruning.
  bool is_better_loose(auto const a, auto const b) const {
    return is_better_or_eq(a, b);
  }

  bool within_bounds(unsigned const k,
                     std::size_t const l,
                     delta_t const t,
                     std::size_t const v) const {
    if (bounds_last_k_ == 0U) {
      return true;
    }

    assert(k <= bounds_last_k_);
    assert(v < kN);

    auto const row = bounds_[bounds_last_k_ - k];
    auto const slot = Crit::bound_slot(v);
    auto const stays_l = crit_.bound_slack(l);
    auto const transfer = dir(adjusted_transfer_time(
        transfer_time_settings_,
        static_cast<int>(
            tt_.locations_.transfer_time_[location_idx_t{l}].count())));
    return is_better_or_eq(t, row[l][slot] + transfer + dir(stays_l));
  }

  template <bool WithClaszFilter,
            bool WithBikeFilter,
            bool WithCarFilter,
            bool WithWheelchairFilter,
            bool WithReservationNotRequiredFilter>
  bool loop_routes(unsigned const k) {
    auto any_marked = false;
    state_.route_mark_.for_each_set_bit([&](auto const r_idx) {
      auto const r = route_idx_t{r_idx};

      if constexpr (WithClaszFilter) {
        if (!is_allowed(allowed_claszes_, tt_.route_clasz_[r])) {
          return;
        }
      }

      auto filters = static_cast<uint8_t>(0);

      auto const apply_filter = [&](route_flag const f) {
        auto const flag_set_on_all_sections =
            tt_.route_flags_[f].test(r_idx * 2);
        if (!flag_set_on_all_sections) {
          auto const flag_set_on_some_sections =
              tt_.route_flags_[f].test(r_idx * 2 + 1);
          if (!flag_set_on_some_sections) {
            return false;
          }
          filters |=
              static_cast<uint8_t>(1 << (route_flag::kNumRouteFlags - f - 1));
          return true;
        }
        return true;
      };

      if constexpr (WithBikeFilter) {
        if (!apply_filter(route_flag::kBikesAllowed)) {
          return;
        }
      }

      if constexpr (WithCarFilter) {
        if (!apply_filter(route_flag::kCarsAllowed)) {
          return;
        }
      }

      if constexpr (WithWheelchairFilter) {
        if (!apply_filter(route_flag::kWheelchairAccessible)) {
          return;
        }
      }

      if constexpr (WithReservationNotRequiredFilter) {
        if (!apply_filter(route_flag::kReservationNotRequired)) {
          return;
        }
      }

      ++stats_.n_routes_visited_;
      trace("┊ ├k={} updating route {}\n", k, r);

      any_marked |= [&]() {
        switch (filters) {
          case 0b0000: return update_route<false, false, false, false>(k, r);
          case 0b0001: return update_route<false, false, false, true>(k, r);
          case 0b0010: return update_route<false, false, true, false>(k, r);
          case 0b0011: return update_route<false, false, true, true>(k, r);
          case 0b0100: return update_route<false, true, false, false>(k, r);
          case 0b0101: return update_route<false, true, false, true>(k, r);
          case 0b0110: return update_route<false, true, true, false>(k, r);
          case 0b0111: return update_route<false, true, true, true>(k, r);
          case 0b1000: return update_route<true, false, false, false>(k, r);
          case 0b1001: return update_route<true, false, false, true>(k, r);
          case 0b1010: return update_route<true, false, true, false>(k, r);
          case 0b1011: return update_route<true, false, true, true>(k, r);
          case 0b1100: return update_route<true, true, false, false>(k, r);
          case 0b1101: return update_route<true, true, false, true>(k, r);
          case 0b1110: return update_route<true, true, true, false>(k, r);
          case 0b1111: return update_route<true, true, true, true>(k, r);
          default: std::unreachable();
        }
      }();
    });
    return any_marked;
  }

  template <bool WithClaszFilter,
            bool WithBikeFilter,
            bool WithCarFilter,
            bool WithWheelchairFilter,
            bool WithReservationNotRequiredFilter>
  bool loop_rt_routes(unsigned const k) {
    auto any_marked = false;
    state_.rt_transport_mark_.for_each_set_bit([&](auto const rt_t_idx) {
      auto const rt_t = rt_transport_idx_t{rt_t_idx};

      if constexpr (WithClaszFilter) {
        if (!is_allowed(allowed_claszes_,
                        rtt_->rt_transport_section_clasz_[rt_t][0])) {
          return;
        }
      }

      auto filters = static_cast<uint8_t>(0);

      auto const apply_filter = [&](route_flag const f) {
        auto const flag_set_on_all_sections =
            rtt_->rt_transport_flags_[f].test(rt_t_idx * 2);
        if (!flag_set_on_all_sections) {
          auto const flag_set_on_some_sections =
              rtt_->rt_transport_flags_[f].test(rt_t_idx * 2 + 1);
          if (!flag_set_on_some_sections) {
            return false;
          }
          filters |=
              static_cast<uint8_t>(1 << (route_flag::kNumRouteFlags - f - 1));
          return true;
        }
        return true;
      };

      if constexpr (WithBikeFilter) {
        if (!apply_filter(route_flag::kBikesAllowed)) {
          return;
        }
      }

      if constexpr (WithCarFilter) {
        if (!apply_filter(route_flag::kCarsAllowed)) {
          return;
        }
      }

      if constexpr (WithWheelchairFilter) {
        if (!apply_filter(route_flag::kWheelchairAccessible)) {
          return;
        }
      }

      if constexpr (WithReservationNotRequiredFilter) {
        if (!apply_filter(route_flag::kReservationNotRequired)) {
          return;
        }
      }

      ++stats_.n_routes_visited_;
      trace("┊ ├k={} updating rt transport {}\n", k, rt_t);

      any_marked |= [&]() {
        switch (filters) {
          case 0b0000:
            return update_rt_transport<false, false, false, false>(k, rt_t);
          case 0b0001:
            return update_rt_transport<false, false, false, true>(k, rt_t);
          case 0b0010:
            return update_rt_transport<false, false, true, false>(k, rt_t);
          case 0b0011:
            return update_rt_transport<false, false, true, true>(k, rt_t);
          case 0b0100:
            return update_rt_transport<false, true, false, false>(k, rt_t);
          case 0b0101:
            return update_rt_transport<false, true, false, true>(k, rt_t);
          case 0b0110:
            return update_rt_transport<false, true, true, false>(k, rt_t);
          case 0b0111:
            return update_rt_transport<false, true, true, true>(k, rt_t);
          case 0b1000:
            return update_rt_transport<true, false, false, false>(k, rt_t);
          case 0b1001:
            return update_rt_transport<true, false, false, true>(k, rt_t);
          case 0b1010:
            return update_rt_transport<true, false, true, false>(k, rt_t);
          case 0b1011:
            return update_rt_transport<true, false, true, true>(k, rt_t);
          case 0b1100:
            return update_rt_transport<true, true, false, false>(k, rt_t);
          case 0b1101:
            return update_rt_transport<true, true, false, true>(k, rt_t);
          case 0b1110:
            return update_rt_transport<true, true, true, false>(k, rt_t);
          case 0b1111:
            return update_rt_transport<true, true, true, true>(k, rt_t);
          default: std::unreachable();
        }
      }();
    });
    return any_marked;
  }

  void update_transfers(unsigned const k) {
    if constexpr (kCoD) {
      state_.prev_station_mark_.for_each_set_bit([&](auto&& i) {
        auto const s0 = tmp0_[i];
        auto const s1 = tmp_read1(i);
        if (s0 == kInvalid && s1 == kInvalid) {
          return;
        }
        auto const is_dest = is_dest_[i];
        auto const transfer_time =
            (!is_intermodal_dest() && is_dest)
                ? 0
                : dir(adjusted_transfer_time(
                      transfer_time_settings_,
                      tt_.locations_.transfer_time_[location_idx_t{i}]
                          .count()));
        auto const lb_ok = lb_reachable(i);
        auto const lb_v = dir(get_lb(i));
        auto const eval = [&](delta_t const s, delta_t const best_w,
                              delta_t const db,
                              std::size_t const v) -> std::pair<delta_t, bool> {
          if (s == kInvalid) {
            return {kInvalid, false};
          }
          auto const ft = clamp(s + transfer_time);
          auto const pass = is_better(ft, best_w) && is_better_loose(ft, db) &&
                            lb_ok && is_better_loose(ft + lb_v, db) &&
                            within_bounds(k, i, ft, v);
          return {ft, pass};
        };
        auto const [f0, p0] = eval(s0, best0_[i], time_at_dest_[k][0], 0U);
        auto const [f1, p1] = eval(s1, best_read1(i), time_at_dest_[k][1], 1U);
        if (bounds_last_k_ == 0U) {
          auto const l0 = f0 != kInvalid && is_better(f0, best0_[i]);
          auto const l1 = f1 != kInvalid && is_better(f1, best_read1(i));
          if (l0 || l1) {
            cod_cell_min(k, i, l0, l1, f0, f1);
            state_.touched_.set(i, true);
          }
        }
        if (p0 || p1) {
          ++stats_.n_earliest_arrival_updated_by_footpath_;
          cod_cell_min(k, i, p0, p1, f0, f1);
          cod_best_min(i, p0, p1, f0, f1);
          state_.station_mark_.set(i, true);
          if (is_dest) {
            if (p0) {
              update_time_at_dest(k, 0U, f0);
            }
            if (p1) {
              update_time_at_dest(k, 1U, f1);
            }
          }
        }
      });
      return;
    }
    state_.prev_station_mark_.for_each_set_bit([&](auto&& i) {
      for (auto v = std::size_t{0U}; v != kN; ++v) {
        auto const tmp_time = tmp_[i][v];
        if (tmp_time == kInvalid) {
          continue;
        }

        auto const [target_v, stay] = crit_.stop_transition(v, i);
        auto const is_dest = Crit::is_dest_slot(target_v) && is_dest_[i];

        trace(
            "  loc={}, v={}, tmp={}, is_dest={}, target_v={}, "
            "stay={}\n",
            loc{tt_, location_idx_t{i}}, v, to_unix(tmp_time), is_dest,
            target_v, stay);

        auto const transfer_time =
            (!is_intermodal_dest() && is_dest)
                ? 0
                : dir(adjusted_transfer_time(
                      transfer_time_settings_,
                      tt_.locations_.transfer_time_[location_idx_t{i}]
                          .count()));
        auto const fp_target_time =
            clamp(tmp_time + transfer_time + dir(stay.count()));

        if (bounds_last_k_ == 0U &&
            is_better(fp_target_time, best_[i][target_v])) {
          round_times_[k][i][target_v] =
              get_best(fp_target_time, round_times_[k][i][target_v]);
          state_.touched_.set(i, true);
        }

        auto const dest_bound = time_at_dest_[k][Crit::dest_bound_of(target_v)];
        trace(
            "    transfer_time={}, fp_target_time={}, best@target={}, "
            "dest={}\n",
            transfer_time, to_unix(fp_target_time), to_unix(best_[i][target_v]),
            to_unix(dest_bound));

        if (is_better(fp_target_time, best_[i][target_v]) &&
            is_better_loose(fp_target_time, dest_bound)) {
          if (!lb_reachable(i) ||
              !is_better_loose(fp_target_time + dir(get_lb(i)), dest_bound)) {
            ++stats_.fp_update_prevented_by_lower_bound_;
            continue;
          }
          if (!within_bounds(k, i, fp_target_time, target_v)) {
            continue;
          }

          ++stats_.n_earliest_arrival_updated_by_footpath_;
          round_times_[k][i][target_v] = fp_target_time;
          best_[i][target_v] = fp_target_time;
          state_.station_mark_.set(i, true);
          if (is_dest) {
            update_time_at_dest(k, target_v, fp_target_time);
          }
        }
      }
    });
  }

  void update_footpaths(unsigned const k) {
    if constexpr (kCoD) {
      state_.prev_station_mark_.for_each_set_bit([&](std::uint64_t const i) {
        auto const l_idx = location_idx_t{i};
        auto const has_td = [&]() {
          if constexpr (Rt) {
            return prf_idx_ != 0U &&
                   (kFwd ? rtt_->has_td_footpaths_out_
                         : rtt_->has_td_footpaths_in_)[prf_idx_]
                       .test(l_idx);
          } else {
            return false;
          }
        }();
        auto const s0 = tmp0_[i];
        auto const src_clean = !has_td && !bt_diverged(i);
        // the rt world uses the td footpaths at td locations instead
        auto const s1 = has_td ? kInvalid : tmp_read1(i);
        if (s0 == kInvalid && s1 == kInvalid) {
          return;
        }
        auto const& fps = kFwd ? tt_.locations_.footpaths_out_[prf_idx_][l_idx]
                               : tt_.locations_.footpaths_in_[prf_idx_][l_idx];
        for (auto const& fp : fps) {
          ++stats_.n_footpaths_visited_;
          auto const target = to_idx(fp.target());
          auto const dur = dir(adjusted_transfer_time(transfer_time_settings_,
                                                      fp.duration().count()));
          auto const lb_ok = lb_reachable(target);
          auto const lb_v = dir(get_lb(target));
          if (src_clean && !bt_diverged(target) && !rt_diverged(k, target)) {
            // clean source + clean target: one eval serves both worlds and
            // the write stays shared (either world's bound admits - bounds
            // only prune, so the wider write is result-neutral)
            auto const f = s0 == kInvalid ? kInvalid : clamp(s0 + dur);
            if (f == kInvalid) {
              continue;
            }
            auto const improves = is_better(f, best0_[target]);
            if (bounds_last_k_ == 0U && improves) {
              rt0_[k][target][0] = get_best(f, rt0_[k][target][0]);
              state_.touched_.set(target, true);
            }
            auto const in_b = [&](std::size_t const v) {
              return is_better_loose(f, time_at_dest_[k][v]) &&
                     is_better_loose(f + lb_v, time_at_dest_[k][v]) &&
                     within_bounds(k, target, f, v);
            };
            if (improves && lb_ok && (in_b(0U) || in_b(1U))) {
              ++stats_.n_earliest_arrival_updated_by_footpath_;
              rt0_[k][target][0] = get_best(f, rt0_[k][target][0]);
              best0_[target] = get_best(f, best0_[target]);
              state_.station_mark_.set(target, true);
              if (is_dest_[target]) {
                update_time_at_dest(k, 0U, f);
                update_time_at_dest(k, 1U, f);
              }
            }
            continue;
          }
          auto const f0 = s0 == kInvalid ? kInvalid : clamp(s0 + dur);
          auto const f1 = s1 == kInvalid ? kInvalid : clamp(s1 + dur);
          auto const pass = [&](delta_t const f, delta_t const best_w,
                                delta_t const db, std::size_t const v) {
            return f != kInvalid && is_better(f, best_w) &&
                   is_better_loose(f, db) && lb_ok &&
                   is_better_loose(f + lb_v, db) &&
                   within_bounds(k, target, f, v);
          };
          auto const b1 = best_read1(target);
          auto const p0 = pass(f0, best0_[target], time_at_dest_[k][0], 0U);
          auto const p1 = pass(f1, b1, time_at_dest_[k][1], 1U);
          if (bounds_last_k_ == 0U) {
            auto const l0 = f0 != kInvalid && is_better(f0, best0_[target]);
            auto const l1 = f1 != kInvalid && is_better(f1, b1);
            if (l0 || l1) {
              cod_cell_min(k, target, l0, l1, f0, f1);
              state_.touched_.set(target, true);
            }
          }
          if (p0 || p1) {
            ++stats_.n_earliest_arrival_updated_by_footpath_;
            cod_cell_min(k, target, p0, p1, f0, f1);
            cod_best_min(target, p0, p1, f0, f1);
            state_.station_mark_.set(target, true);
            if (is_dest_[target]) {
              if (p0) {
                update_time_at_dest(k, 0U, f0);
              }
              if (p1) {
                update_time_at_dest(k, 1U, f1);
              }
            }
          }
        }
      });
      return;
    }
    state_.prev_station_mark_.for_each_set_bit([&](std::uint64_t const i) {
      auto const l_idx = location_idx_t{i};
      auto const has_td = [&]() {
        if constexpr (Rt) {
          return prf_idx_ != 0U && (kFwd ? rtt_->has_td_footpaths_out_
                                         : rtt_->has_td_footpaths_in_)[prf_idx_]
                                       .test(l_idx);
        } else {
          return false;
        }
      }();
      if constexpr (Crit::kAllSlotsRt) {
        if (has_td) {
          // handled by update_td_offsets
          return;
        }
      }

      auto const& fps = kFwd ? tt_.locations_.footpaths_out_[prf_idx_][l_idx]
                             : tt_.locations_.footpaths_in_[prf_idx_][l_idx];

      for (auto const& fp : fps) {
        ++stats_.n_footpaths_visited_;

        auto const target = to_idx(fp.target());

        for (auto v = std::size_t{0U}; v != kN; ++v) {
          if constexpr (!Crit::kAllSlotsRt) {
            // rt-world slots use the td footpaths at td locations
            if (has_td && Crit::slot_uses_rt(v)) {
              continue;
            }
          }
          auto const tmp_time = tmp_[i][v];
          if (tmp_time == kInvalid) {
            continue;
          }

          auto const [start_v, start_stay] = crit_.stop_transition(v, i);
          auto const [target_v, target_stay] =
              crit_.stop_transition(start_v, target);
          auto const stay = start_stay + target_stay;

          auto const fp_target_time = clamp(
              tmp_time + dir(adjusted_transfer_time(transfer_time_settings_,
                                                    fp.duration().count()) +
                             stay.count()));

          if (bounds_last_k_ == 0U &&
              is_better(fp_target_time, best_[target][target_v])) {
            round_times_[k][target][target_v] =
                get_best(fp_target_time, round_times_[k][target][target_v]);
            state_.touched_.set(target, true);
          }

          auto const dest_bound =
              time_at_dest_[k][Crit::dest_bound_of(target_v)];
          if (is_better(fp_target_time, best_[target][target_v]) &&
              is_better_loose(fp_target_time, dest_bound)) {
            if (!lb_reachable(target) ||
                !is_better_loose(fp_target_time + dir(get_lb(target)),
                                 dest_bound)) {
              ++stats_.fp_update_prevented_by_lower_bound_;
              trace_upd(
                  "┊ ├k={} *** LB NO UPD: (from={}, tmp={}) --{}--> (to={}, "
                  "best={}) --> update => {}, LB={}, LB_AT_DEST={}, DEST={}\n",
                  k, loc{tt_, l_idx}, to_unix(tmp_[to_idx(l_idx)][v]),
                  adjusted_transfer_time(transfer_time_settings_,
                                         fp.duration()),
                  loc{tt_, fp.target()}, best_[target][target_v],
                  to_unix(fp_target_time), get_lb(target),
                  to_unix(clamp(fp_target_time + dir(get_lb(target)))),
                  to_unix(dest_bound));
              continue;
            }
            if (!within_bounds(k, target, fp_target_time, target_v)) {
              continue;
            }

            trace_upd(
                "┊ ├k={}   footpath: ({}, tmp={}) --{}--> ({}, best={}) --> "
                "update => {}, v={}->{}, stay={}\n",
                k, loc{tt_, l_idx}, to_unix(tmp_[to_idx(l_idx)][v]),
                adjusted_transfer_time(transfer_time_settings_, fp.duration()),
                loc{tt_, fp.target()}, to_unix(best_[target][target_v]),
                to_unix(fp_target_time), v, target_v, stay);

            ++stats_.n_earliest_arrival_updated_by_footpath_;
            round_times_[k][target][target_v] = fp_target_time;
            best_[target][target_v] = fp_target_time;
            state_.station_mark_.set(target, true);
            if (Crit::is_dest_slot(target_v) && is_dest_[target]) {
              update_time_at_dest(k, target_v, fp_target_time);
            }
          } else {
            trace(
                "┊ ├k={}   NO FP UPDATE: {} [best={}] --{}--> {} "
                "[best={}, time_at_dest={}]\n",
                k, loc{tt_, l_idx}, to_unix(best_[to_idx(l_idx)][target_v]),
                adjusted_transfer_time(transfer_time_settings_, fp.duration()),
                loc{tt_, fp.target()}, to_unix(best_[target][target_v]),
                to_unix(dest_bound));
          }
        }
      }
    });
  }

  void update_td_offsets(unsigned const k) {
    if constexpr (!Rt) {
      return;
    }

    if (prf_idx_ == 0U) {
      return;
    }

    if constexpr (kCoD) {
      // td footpaths only exist in the rt world
      state_.prev_station_mark_.for_each_set_bit([&](std::uint64_t const i) {
        auto const l_idx = location_idx_t{i};
        if (!(kFwd ? rtt_->has_td_footpaths_out_
                   : rtt_->has_td_footpaths_in_)[prf_idx_]
                 .test(l_idx)) {
          return;
        }
        auto const s1 = tmp_read1(i);
        if (s1 == kInvalid) {
          return;
        }
        auto const& fps = kFwd ? rtt_->td_footpaths_out_[prf_idx_][l_idx]
                               : rtt_->td_footpaths_in_[prf_idx_][l_idx];
        for_each_footpath<SearchDir>(fps, to_unix(s1), [&](footpath const fp) {
          ++stats_.n_footpaths_visited_;
          auto const target = to_idx(fp.target());
          auto const f1 = clamp(s1 + dir(fp.duration().count()));
          auto const b1 = best_read1(target);
          if (bounds_last_k_ == 0U && is_better(f1, b1)) {
            cod_cell_min(k, target, false, true, kInvalid, f1);
            state_.touched_.set(target, true);
          }
          auto const db1 = time_at_dest_[k][1];
          if (is_better(f1, b1) && is_better_loose(f1, db1) &&
              lb_reachable(target) &&
              is_better_loose(f1 + dir(get_lb(target)), db1) &&
              within_bounds(k, target, f1, 1U)) {
            ++stats_.n_earliest_arrival_updated_by_footpath_;
            cod_cell_min(k, target, false, true, kInvalid, f1);
            cod_best_min(target, false, true, kInvalid, f1);
            state_.station_mark_.set(target, true);
            if (is_dest_[target]) {
              update_time_at_dest(k, 1U, f1);
            }
          }
          return utl::cflow::kContinue;
        });
      });
      return;
    }

    state_.prev_station_mark_.for_each_set_bit([&](std::uint64_t const i) {
      auto const l_idx = location_idx_t{i};
      if (!(kFwd ? rtt_->has_td_footpaths_out_
                 : rtt_->has_td_footpaths_in_)[prf_idx_]
               .test(l_idx)) {
        return;
      }

      auto const& fps = kFwd ? rtt_->td_footpaths_out_[prf_idx_][l_idx]
                             : rtt_->td_footpaths_in_[prf_idx_][l_idx];

      for (auto v = std::size_t{0U}; v != kN; ++v) {
        if (!Crit::slot_uses_rt(v)) {
          // scheduled-world slots only use the static footpaths
          continue;
        }
        auto const tmp_time = tmp_[i][v];
        if (tmp_time == kInvalid) {
          continue;
        }
        for_each_footpath<
            SearchDir>(fps, to_unix(tmp_time), [&](footpath const fp) {
          ++stats_.n_footpaths_visited_;

          auto const target = to_idx(fp.target());

          auto const [start_v, start_stay] = crit_.stop_transition(v, i);
          auto const [target_v, target_stay] =
              crit_.stop_transition(start_v, target);
          auto const stay = start_stay + target_stay;

          auto const fp_target_time =
              clamp(tmp_time + dir(fp.duration().count() + stay.count()));

          if (bounds_last_k_ == 0U &&
              is_better(fp_target_time, best_[target][target_v])) {
            round_times_[k][target][target_v] =
                get_best(fp_target_time, round_times_[k][target][target_v]);
            state_.touched_.set(target, true);
          }

          auto const dest_bound =
              time_at_dest_[k][Crit::dest_bound_of(target_v)];
          if (is_better(fp_target_time, best_[target][target_v]) &&
              is_better_loose(fp_target_time, dest_bound)) {
            if (!lb_reachable(target) ||
                !is_better_loose(fp_target_time + dir(get_lb(target)),
                                 dest_bound)) {
              ++stats_.fp_update_prevented_by_lower_bound_;
              trace_upd(
                  "┊ ├k={} *** LB NO TD FP UPD: (from={}, tmp={}) --{}--> "
                  "(to={}, best={}) --> update => {}, LB={}, LB_AT_DEST={}, "
                  "DEST={}\n",
                  k, loc{tt_, l_idx}, to_unix(tmp_[to_idx(l_idx)][v]),
                  fp.duration(), loc{tt_, fp.target()}, best_[target][target_v],
                  fp_target_time, get_lb(target),
                  to_unix(clamp(fp_target_time + dir(get_lb(target)))),
                  to_unix(dest_bound));
              return utl::cflow::kContinue;
            }
            if (!within_bounds(k, target, fp_target_time, target_v)) {
              return utl::cflow::kContinue;
            }

            trace_upd(
                "┊ ├k={}   td footpath: ({}, tmp={}) --{}--> ({}, best={}) --> "
                "update => {}, v={}->{}, stay={}\n",
                k, loc{tt_, l_idx}, to_unix(tmp_[to_idx(l_idx)][v]),
                fp.duration(), loc{tt_, fp.target()},
                to_unix(best_[target][target_v]), to_unix(fp_target_time), v,
                target_v, stay);

            ++stats_.n_earliest_arrival_updated_by_footpath_;
            round_times_[k][target][target_v] = fp_target_time;
            best_[target][target_v] = fp_target_time;
            state_.station_mark_.set(target, true);
            if (is_dest_[target]) {
              update_time_at_dest(k, target_v, fp_target_time);
            }
          } else {
            trace(
                "┊ ├k={}   NO TD FP UPDATE: {} [best={}] --{}--> {} "
                "[best={}, time_at_dest={}]\n",
                k, loc{tt_, l_idx}, best_[to_idx(l_idx)][v],
                adjusted_transfer_time(transfer_time_settings_, fp.duration()),
                loc{tt_, fp.target()}, best_[target][v],
                to_unix(dest_bound));
          }

          return utl::cflow::kContinue;
        });
      }
    });
  }

  void update_intermodal_footpaths(unsigned const k) {
    if (dist_to_end_.empty()) {
      return;
    }

    if constexpr (kCoD) {
      state_.prev_station_mark_.for_each_set_bit([&](auto const i) {
        if (!end_reachable_.test(i)) {
          return;
        }
        auto const l = location_idx_t{i};
        auto const s0 = tmp0_[i];
        auto const s1 = tmp_read1(i);
        auto const apply = [&](delta_t const e0, delta_t const e1) {
          auto const p0 =
              e0 != kInvalid && is_better_loose(e0, time_at_dest_[k][0]);
          auto const p1 =
              e1 != kInvalid && is_better_loose(e1, time_at_dest_[k][1]);
          if (p0 || p1) {
            cod_cell_min(k, kIntermodalTarget, p0, p1, e0, e1);
            cod_best_min(kIntermodalTarget, p0, p1, e0, e1);
            if (p0) {
              update_time_at_dest(k, 0U, e0);
            }
            if (p1) {
              update_time_at_dest(k, 1U, e1);
            }
          }
        };
        if (dist_to_end_[i] != std::numeric_limits<std::uint16_t>::max()) {
          apply(s0 == kInvalid ? kInvalid : clamp(s0 + dir(dist_to_end_[i])),
                s1 == kInvalid ? kInvalid : clamp(s1 + dir(dist_to_end_[i])));
        }
        if (auto const it = td_dist_to_end_.find(l);
            it != end(td_dist_to_end_)) {
          [[unlikely]];
          auto const td_end = [&](delta_t const s) {
            if (s == kInvalid) {
              return kInvalid;
            }
            auto const fp = get_td_duration<SearchDir>(it->second, to_unix(s));
            return fp.has_value() ? clamp(s + dir(fp->first.count()))
                                  : kInvalid;
          };
          auto const e0 = td_end(s0);
          auto const e1 = td_end(s1);
          // mirror the non-td guard: only better-than-best updates
          apply(e0 != kInvalid && is_better(e0, best0_[kIntermodalTarget])
                    ? e0
                    : kInvalid,
                e1 != kInvalid && is_better(e1, best_read1(kIntermodalTarget))
                    ? e1
                    : kInvalid);
        }
      });
      return;
    }

    state_.prev_station_mark_.for_each_set_bit([&](auto const i) {
      if (!end_reachable_.test(i)) {
        trace_upd("┊ ├k={}   no end_reachable: {}\n", k,
                  loc{tt_, location_idx_t{i}});
        [[likely]];
        return;
      }

      trace_upd("┊ ├k={}   end_reachable: {}\n", k,
                loc{tt_, location_idx_t{i}});

      auto const l = location_idx_t{i};
      if (dist_to_end_[i] != std::numeric_limits<std::uint16_t>::max()) {
        [[likely]];

        // Case 1: l transitions into the final slot -> add dwell
        crit_.for_each_final_stop_transition(
            i, [&](auto const v, duration_t const stay) {
              if (tmp_[i][v] != kInvalid) {
                auto const end_time = clamp(tmp_[i][v]  //
                                            + dir(stay.count())  //
                                            + dir(dist_to_end_[i]));

                trace_upd(
                    "┊ ├k={}, INTERMODAL FOOTPATH FROM LAST VIA: ({}, tmp={}) "
                    "--({} +stay={})--> "
                    "({}, best={})",
                    k, loc{tt_, l}, to_unix(tmp_[to_idx(l)][v]),
                    dist_to_end_[i], stay,
                    loc{tt_, location_idx_t{kIntermodalTarget}},
                    to_unix(best_[kIntermodalTarget][kFinalSlot]),
                    to_unix(end_time));

                if (is_better_loose(
                        end_time,
                        time_at_dest_[k][Crit::dest_bound_of(kFinalSlot)])) {
                  round_times_[k][kIntermodalTarget][kFinalSlot] = end_time;
                  best_[kIntermodalTarget][kFinalSlot] = end_time;
                  update_time_at_dest(k, kFinalSlot, end_time);
                  trace_upd(" -> update\n");
                } else {
                  trace_upd(" -> no update\n");
                }
              }
            });

        // Case 2: l is already in a dest slot -> no dwell
        Crit::for_each_dest_slot([&](auto const v) {
          auto const tmp_time = tmp_[i][v];
          if (tmp_time == kInvalid) {
            trace_upd("┊ ├k={}, loc={} NOT REACHED\n", k, loc{tt_, l});
            return;
          }

          auto const end_time = clamp(tmp_time + dir(dist_to_end_[i]));

          trace_upd(
              "┊ ├k={}, INTERMODAL FOOTPATH: ({}, tmp={}) --{}--> "
              "({}, best={})",
              k, loc{tt_, l}, to_unix(tmp_[to_idx(l)][v]), dist_to_end_[i],
              loc{tt_, location_idx_t{kIntermodalTarget}},
              to_unix(best_[kIntermodalTarget][v]), to_unix(end_time));

          if (is_better_loose(end_time,
                              time_at_dest_[k][Crit::dest_bound_of(v)])) {
            round_times_[k][kIntermodalTarget][v] = end_time;
            best_[kIntermodalTarget][v] = end_time;
            update_time_at_dest(k, v, end_time);
            trace_upd(" -> update\n");
          } else {
            trace_upd(" -> no update\n");
          }
        });
      }

      if (auto const it = td_dist_to_end_.find(l); it != end(td_dist_to_end_)) {
        [[unlikely]];

        Crit::for_each_dest_slot([&](auto const v) {
          auto const fp_start_time = tmp_[i][v];
          if (fp_start_time == kInvalid) {
            return;
          }
          auto const fp =
              get_td_duration<SearchDir>(it->second, to_unix(fp_start_time));
          if (fp.has_value()) {
            auto const& [duration, _] = *fp;
            auto const end_time = clamp(fp_start_time + dir(duration.count()));

            if (is_better_loose(end_time,
                                time_at_dest_[k][Crit::dest_bound_of(v)]) &&
                is_better(end_time, best_[kIntermodalTarget][v])) {
              round_times_[k][kIntermodalTarget][v] = end_time;
              best_[kIntermodalTarget][v] = end_time;
              update_time_at_dest(k, v, end_time);

              trace(
                  "┊ │k={}  TD INTERMODAL FOOTPATH: location={}, "
                  "start_time={}, dist_to_end={} --> update to {}\n",
                  k, loc{tt_, l}, to_unix(fp_start_time), duration,
                  to_unix(end_time));
            } else {
              trace(
                  "┊ │k={}  TD INTERMODAL FOOTPATH: location={}, "
                  "start_time={}, dist_to_end={} --> NO update to {} best={}\n",
                  k, loc{tt_, l}, to_unix(fp_start_time), duration,
                  to_unix(end_time), best_[kIntermodalTarget][v]);
            }
          }
        });
      }
    });
  }

  template <bool WithSectionBikeFilter,
            bool WithSectionCarFilter,
            bool WithSectionWheelchairFilter,
            bool WithSectionReservationNotRequiredFilter>
  bool update_rt_transport(unsigned const k, rt_transport_idx_t const rt_t) {
    auto const stop_seq = rtt_->rt_transport_location_seq_[rt_t];
    auto ride = typename Crit::rt_ride{crit_};
    auto any_marked = false;
    auto const n_stops = stop_seq.size();

    for (auto i = 0U; i != n_stops; ++i) {
      auto const stop_idx =
          static_cast<stop_idx_t>(kFwd ? i : n_stops - i - 1U);
      auto const stp = stop{stop_seq[stop_idx]};
      auto const l_idx = cista::to_idx(stp.location_idx());
      auto const is_first = i == 0U;
      auto const is_last = i == n_stops - 1U;

      auto const apply_filter = [&](route_flag const f) {
        if (!is_first &&
            !rtt_->rt_flags_per_section_[f][rt_t]
                                        [kFwd ? stop_idx - 1 : stop_idx]) {
          ride.reset();
        }
      };

      if constexpr (WithSectionBikeFilter) {
        apply_filter(route_flag::kBikesAllowed);
      }

      if constexpr (WithSectionCarFilter) {
        apply_filter(route_flag::kCarsAllowed);
      }

      if constexpr (WithSectionWheelchairFilter) {
        apply_filter(route_flag::kWheelchairAccessible);
      }

      if constexpr (WithSectionReservationNotRequiredFilter) {
        apply_filter(route_flag::kReservationNotRequired);
      }

      if ((kFwd && stop_idx != 0U) || (kBwd && stop_idx != n_stops - 1U)) {
        ride.advance(l_idx);

        auto const by_transport = rt_time_at_stop(
            rt_t, stop_idx, kFwd ? event_type::kArr : event_type::kDep);
        if constexpr (kCoD) {
          // rt transports only ever serve the rt world
          ride.for_each_arrival([&](auto const) {
            if (stp.can_finish<SearchDir>(is_wheelchair_)) {
              auto const db1 = time_at_dest_[k][1];
              if (is_better_loose(by_transport, db1) && lb_reachable(l_idx) &&
                  is_better_loose(by_transport + dir(get_lb(l_idx)), db1) &&
                  within_bounds(k, l_idx, by_transport, 1U)) {
                ++stats_.n_earliest_arrival_updated_by_route_;
                cod_tmp_min(l_idx, false, true, kInvalid, by_transport);
                state_.station_mark_.set(l_idx, true);
                any_marked = true;
              }
            }
          });
        } else {
        ride.for_each_arrival([&](auto const v) {
          if (stp.can_finish<SearchDir>(is_wheelchair_)) {
            auto current_best = get_best(round_times_[k - 1][l_idx][v],
                                         tmp_[l_idx][v], best_[l_idx][v]);
            auto const dest_bound = time_at_dest_[k][Crit::dest_bound_of(v)];

            if (is_better_loose(by_transport, dest_bound) &&
                lb_reachable(l_idx) &&
                is_better_loose(by_transport + dir(get_lb(l_idx)),
                                dest_bound) &&
                within_bounds(k, l_idx, by_transport, v)) {
              trace_upd(
                  "┊ │k={}    RT | name={}, dbg={}, "
                  "time_by_transport={}, "
                  "BETTER THAN current_best={} => update, {} marking station "
                  "{}!\n",
                  k, rtt_->default_trip_short_name(tt_, rt_t),
                  rtt_->dbg(tt_, rt_t), to_unix(by_transport),
                  to_unix(current_best),
                  !is_better(by_transport, current_best) ? "NOT" : "",
                  loc{tt_, stp.location_idx()});

              ++stats_.n_earliest_arrival_updated_by_route_;
              tmp_[l_idx][v] = get_best(by_transport, tmp_[l_idx][v]);
              state_.station_mark_.set(l_idx, true);
              if (is_better(by_transport, current_best)) {
                current_best = by_transport;
              }
              any_marked = true;
            }
          }
        });
        }
      }

      if (!lb_reachable(l_idx)) {
        break;
      }

      if (is_last || !(stp.can_start<SearchDir>(is_wheelchair_)) ||
          !state_.prev_station_mark_[l_idx]) {
        continue;
      }

      auto const by_transport = rt_time_at_stop(
          rt_t, stop_idx, kFwd ? event_type::kDep : event_type::kArr);
      if constexpr (kCoD) {
        if (is_better_or_eq(rt_read1(k - 1, l_idx), by_transport)) {
          ride.board(1U);
        }
      } else {
        for (auto v = std::size_t{0U}; v != kN; ++v) {
          if (!Crit::slot_boards_rt(v)) {
            continue;
          }
          auto const prev_round_time = round_times_[k - 1][l_idx][v];
          if (is_better_or_eq(prev_round_time, by_transport)) {
            ride.board(v);
          }
        }
      }
    }
    return any_marked;
  }

  template <bool WithSectionBikeFilter,
            bool WithSectionCarFilter,
            bool WithSectionWheelchairFilter,
            bool WithSectionReservationNotRequiredFilter>
  bool update_route(unsigned const k, route_idx_t const r) {
    if constexpr (kCoD) {
      // clean-route fast lane: no rt-re-pointed transport and no diverged
      // input cell => both worlds compute identical labels in one pass
      if (!state_.route_has_rt_.test(to_idx(r))) {
        auto clean = true;
        for (auto const s : tt_.route_location_seq_[r]) {
          auto const l = to_idx(stop{s}.location_idx());
          if (rt_diverged(k - 1U, l) || bt_diverged(l)) {
            clean = false;
            break;
          }
        }
        if (clean) {
          return update_route_impl<WithSectionBikeFilter, WithSectionCarFilter,
                                   WithSectionWheelchairFilter,
                                   WithSectionReservationNotRequiredFilter,
                                   true>(k, r);
        }
      }
    }
    return update_route_impl<WithSectionBikeFilter, WithSectionCarFilter,
                             WithSectionWheelchairFilter,
                             WithSectionReservationNotRequiredFilter, false>(
        k, r);
  }

  template <bool WithSectionBikeFilter,
            bool WithSectionCarFilter,
            bool WithSectionWheelchairFilter,
            bool WithSectionReservationNotRequiredFilter,
            bool Clean>
  bool update_route_impl(unsigned const k, route_idx_t const r) {
    auto const stop_seq = tt_.route_location_seq_[r];
    bool any_marked = false;

    auto ride = typename Crit::route_ride{crit_};
    auto const n_stops = stop_seq.size();

    for (auto i = 0U; i != n_stops; ++i) {
      auto const stop_idx =
          static_cast<stop_idx_t>(kFwd ? i : n_stops - i - 1U);
      auto const stp = stop{stop_seq[stop_idx]};
      auto const l_idx = cista::to_idx(stp.location_idx());
      auto const is_first = i == 0U;
      auto const is_last = i == n_stops - 1U;

      auto const apply_filter = [&](route_flag const f) {
        if (!is_first && !tt_.route_flags_per_section_[f][r][kFwd ? stop_idx - 1
                                                                  : stop_idx]) {
          ride.reset();
        }
      };

      if constexpr (WithSectionBikeFilter) {
        apply_filter(route_flag::kBikesAllowed);
      }

      if constexpr (WithSectionCarFilter) {
        apply_filter(route_flag::kCarsAllowed);
      }

      if constexpr (WithSectionWheelchairFilter) {
        apply_filter(route_flag::kWheelchairAccessible);
      }

      if constexpr (WithSectionReservationNotRequiredFilter) {
        apply_filter(route_flag::kReservationNotRequired);
      }

      ride.advance(l_idx, [&](transport const t) {
        return time_at_stop(r, t, stop_idx,
                            kFwd ? event_type::kArr : event_type::kDep);
      });

      auto current_best = bag_t{};
      current_best.fill(kInvalid);

      if constexpr (kCoD) {
        if (stp.can_finish<SearchDir>(is_wheelchair_)) {
          auto const arr_ev = kFwd ? event_type::kArr : event_type::kDep;
          auto const lb_ok = lb_reachable(l_idx);
          auto const lb_v = dir(get_lb(l_idx));
          auto const arr_pass = [&](delta_t const by, std::size_t const v) {
            auto const db = time_at_dest_[k][v];
            return is_better_loose(by, db) && lb_ok &&
                   is_better_loose(by + lb_v, db) &&
                   within_bounds(k, l_idx, by, v);
          };
          auto const& e0 = ride.fresh(0U);
          auto const& e1 = ride.fresh(1U);
          if constexpr (Clean) {
            // both worlds ride the same transport with undiverged state:
            // one eval, one shared write (either world's bound admits -
            // bounds only prune, so the wider write stays result-neutral)
            if (e0.is_valid()) {
              auto const by = time_at_stop(r, e0, stop_idx, arr_ev);
              if (arr_pass(by, 0U) || arr_pass(by, 1U)) {
                ++stats_.n_earliest_arrival_updated_by_route_;
                tmp0_[l_idx] = get_best(by, tmp0_[l_idx]);
                state_.station_mark_.set(l_idx, true);
                any_marked = true;
              }
            }
          } else if (e0.is_valid() && e0.t_idx_ == e1.t_idx_ &&
                     e0.day_ == e1.day_) {
            auto const by = time_at_stop(r, e0, stop_idx, arr_ev);
            auto const p0 = arr_pass(by, 0U);
            auto const p1 = arr_pass(by, 1U);
            if (p0 || p1) {
              ++stats_.n_earliest_arrival_updated_by_route_;
              cod_tmp_min(l_idx, p0, p1, by, by);
              state_.station_mark_.set(l_idx, true);
              any_marked = true;
            }
          } else {
            auto const one = [&](transport const& t, std::size_t const v) {
              if (!t.is_valid()) {
                return;
              }
              auto const by = time_at_stop(r, t, stop_idx, arr_ev);
              if (arr_pass(by, v)) {
                ++stats_.n_earliest_arrival_updated_by_route_;
                cod_tmp_min(l_idx, v == 0U, v == 1U, by, by);
                state_.station_mark_.set(l_idx, true);
                any_marked = true;
              }
            };
            one(e1, 1U);
            one(e0, 0U);
          }
        }
      } else {

      ride.for_each_arrival([&](auto const cs, transport const& t) {
        if (!stp.can_finish<SearchDir>(is_wheelchair_)) {
          trace(
              "┊ │k={} cs={}    *** NO UPD: in_allowed={}, "
              "out_allowed={}, label_allowed={}\n",
              k, cs, stp.in_allowed(), stp.out_allowed(),
              (kFwd ? stp.out_allowed() : stp.in_allowed()));
          return;
        }
        auto const by_transport = time_at_stop(
            r, t, stop_idx, kFwd ? event_type::kArr : event_type::kDep);

        if (current_best[cs] == kInvalid) {
          current_best[cs] = get_best(round_times_[k - 1][l_idx][cs],
                                      tmp_[l_idx][cs], best_[l_idx][cs]);
        }

        auto const dest_bound = time_at_dest_[k][Crit::dest_bound_of(cs)];
        assert(by_transport != std::numeric_limits<delta_t>::min() &&
               by_transport != std::numeric_limits<delta_t>::max());
        if (is_better_loose(by_transport, dest_bound) && lb_reachable(l_idx) &&
            is_better_loose(by_transport + dir(get_lb(l_idx)), dest_bound) &&
            within_bounds(k, l_idx, by_transport, cs)) {
          trace_upd(
              "┊ │k={} cs={}    name={}, dbg={}, "
              "time_by_transport={}, "
              "BETTER THAN current_best={} => update, {} marking station "
              "{}!\n",
              k, cs, tt_.transport_name(t.t_idx_), tt_.dbg(t.t_idx_),
              to_unix(by_transport), to_unix(current_best[cs]),
              !is_better(by_transport, current_best[cs]) ? "NOT" : "",
              loc{tt_, stp.location_idx()});

          ++stats_.n_earliest_arrival_updated_by_route_;
          tmp_[l_idx][cs] = get_best(by_transport, tmp_[l_idx][cs]);
          state_.station_mark_.set(l_idx, true);
          if (is_better(by_transport, current_best[cs])) {
            current_best[cs] = by_transport;
          }
          any_marked = true;
        } else {
          trace(
              "┊ │k={} cs={}    *** NO UPD: at={}, name={}, "
              "dbg={}, "
              "time_by_transport={}, current_best={}\n",
              k, cs, loc{tt_, location_idx_t{l_idx}},
              tt_.transport_name(t.t_idx_), tt_.dbg(t.t_idx_),
              to_unix(by_transport), to_unix(current_best[cs]));
        }
      });

      }

      if (is_last || !stp.can_start<SearchDir>(is_wheelchair_) ||
          !state_.prev_station_mark_[l_idx]) {
        continue;
      }

      if (!lb_reachable(l_idx)) {
        break;
      }

      if constexpr (kCoD) {
        auto const dep_ev = kFwd ? event_type::kDep : event_type::kArr;
        auto const et_time = [&](std::size_t const v) {
          return ride.fresh(v).is_valid()
                     ? time_at_stop(r, ride.fresh(v), stop_idx, dep_ev)
                     : kInvalid;
        };
        auto const accept = [&](std::size_t const v, transport const new_et,
                                delta_t const et_t, delta_t const prev,
                                delta_t const best_w, delta_t const tmp_w) {
          auto const cb = get_best(prev, best_w, tmp_w);
          if (new_et.is_valid() &&
              (cb == kInvalid ||
               is_better_or_eq(time_at_stop(r, new_et, stop_idx, dep_ev),
                               et_t))) {
            ride.fresh(v) = new_et;
          }
        };
        if constexpr (Clean) {
          // undiverged label + identical activity: world 0's walk serves both
          auto const p0 = rt0_[k - 1][l_idx][0];
          auto const t0 = et_time(0U);
          if (p0 != kInvalid && is_better_or_eq(p0, t0)) {
            auto const [day, mam] = split(p0);
            accept(0U,
                   get_earliest_transport(k, 0U, r, stop_idx, day, mam,
                                          stp.location_idx()),
                   t0, p0, best0_[l_idx], tmp0_[l_idx]);
          }
          continue;
        }
        auto const p0 = rt0_[k - 1][l_idx][0];
        auto const p1 = rt_read1(k - 1, l_idx);
        auto const t0 = et_time(0U);
        auto const t1 = et_time(1U);
        auto const trig0 = p0 != kInvalid && is_better_or_eq(p0, t0);
        auto const trig1 = p1 != kInvalid && is_better_or_eq(p1, t1);
        if (trig0 && trig1 && p0 == p1) {
          auto const [day, mam] = split(p0);
          auto const ets = get_earliest_transport_dual(k, r, stop_idx, day,
                                                       mam, stp.location_idx());
          accept(0U, ets[0], t0, p0, best0_[l_idx], tmp0_[l_idx]);
          accept(1U, ets[1], t1, p1, best_read1(l_idx), tmp_read1(l_idx));
        } else {
          if (trig0) {
            auto const [day, mam] = split(p0);
            accept(0U,
                   get_earliest_transport(k, 0U, r, stop_idx, day, mam,
                                          stp.location_idx()),
                   t0, p0, best0_[l_idx], tmp0_[l_idx]);
          }
          if (trig1) {
            auto const [day, mam] = split(p1);
            accept(1U,
                   get_earliest_transport(k, 1U, r, stop_idx, day, mam,
                                          stp.location_idx()),
                   t1, p1, best_read1(l_idx), tmp_read1(l_idx));
          }
        }
      } else if constexpr (Crit::kFusedEt) {
        // one event walk serves both worlds when they board from the same
        // label time (the common case with sparse rt deviations)
        static_assert(kN == 2U && Rt);
        static_assert(!Crit::slot_uses_rt(0U) && Crit::slot_uses_rt(1U));

        auto const et_time = [&](std::size_t const v) {
          return ride.fresh(v).is_valid()
                     ? time_at_stop(r, ride.fresh(v), stop_idx,
                                    kFwd ? event_type::kDep : event_type::kArr)
                     : kInvalid;
        };
        auto const accept = [&](std::size_t const v, transport const new_et,
                                delta_t const et_time_at_stop) {
          current_best[v] =
              get_best(current_best[v], best_[l_idx][v], tmp_[l_idx][v]);
          if (new_et.is_valid() &&
              (current_best[v] == kInvalid ||
               is_better_or_eq(
                   time_at_stop(r, new_et, stop_idx,
                                kFwd ? event_type::kDep : event_type::kArr),
                   et_time_at_stop))) {
            ride.fresh(v) = new_et;
          }
        };

        auto const et0 = et_time(0U);
        auto const et1 = et_time(1U);
        auto const prev0 = round_times_[k - 1][l_idx][0];
        auto const prev1 = round_times_[k - 1][l_idx][1];
        auto const trig0 = prev0 != kInvalid && is_better_or_eq(prev0, et0);
        auto const trig1 = prev1 != kInvalid && is_better_or_eq(prev1, et1);
        if (trig0 && trig1 && prev0 == prev1) {
          auto const [day, mam] = split(prev0);
          auto const ets = get_earliest_transport_dual(k, r, stop_idx, day,
                                                       mam, stp.location_idx());
          if constexpr (NIGIRI_SCHEDRT_COUNTERS != 0) {
            ++get_schedrt_divergence_counters().board_fused_;
            if (ets[0].t_idx_ != ets[1].t_idx_ ||
                ets[0].day_ != ets[1].day_) {
              ++get_schedrt_divergence_counters().et_dual_diverged_;
            }
          }
          accept(0U, ets[0], et0);
          accept(1U, ets[1], et1);
        } else {
          if constexpr (NIGIRI_SCHEDRT_COUNTERS != 0) {
            if (trig0 || trig1) {
              ++get_schedrt_divergence_counters().board_split_;
            }
          }
          if (trig0) {
            auto const [day, mam] = split(prev0);
            accept(0U,
                   get_earliest_transport(k, 0U, r, stop_idx, day, mam,
                                          stp.location_idx()),
                   et0);
          }
          if (trig1) {
            auto const [day, mam] = split(prev1);
            accept(1U,
                   get_earliest_transport(k, 1U, r, stop_idx, day, mam,
                                          stp.location_idx()),
                   et1);
          }
        }
      } else {
        for (auto v = std::size_t{0U}; v != kN; ++v) {
          auto& fresh = ride.fresh(v);
          auto const et_time_at_stop =
              fresh.is_valid()
                  ? time_at_stop(r, fresh, stop_idx,
                                 kFwd ? event_type::kDep : event_type::kArr)
                  : kInvalid;
          auto const prev_round_time = round_times_[k - 1][l_idx][v];
          if (prev_round_time != kInvalid &&
              is_better_or_eq(prev_round_time, et_time_at_stop)) {
            auto const [day, mam] = split(prev_round_time);
            auto const new_et = get_earliest_transport(
                k, v, r, stop_idx, day, mam, stp.location_idx());
            current_best[v] =
                get_best(current_best[v], best_[l_idx][v], tmp_[l_idx][v]);
            if (new_et.is_valid() &&
                (current_best[v] == kInvalid ||
                 is_better_or_eq(
                     time_at_stop(r, new_et, stop_idx,
                                  kFwd ? event_type::kDep : event_type::kArr),
                     et_time_at_stop))) {
              fresh = new_et;
              trace("┊ │k={} v={}    update et: time_at_stop={}\n", k, v,
                    to_unix(et_time_at_stop));
            } else if (new_et.is_valid()) {
              trace(
                  "┊ │k={} v={}    update et: no update "
                  "time_at_stop={}\n",
                  k, v, to_unix(et_time_at_stop));
            }
          }
        }
      }
    }
    return any_marked;
  }

  transport get_earliest_transport(unsigned const k,
                                   std::size_t const v,
                                   route_idx_t const r,
                                   stop_idx_t const stop_idx,
                                   day_idx_t const day_at_stop,
                                   minutes_after_midnight_t const mam_at_stop,
                                   location_idx_t const l) {
    ++stats_.n_earliest_trip_calls_;

    auto const dest_bound = time_at_dest_[k][Crit::dest_bound_of(v)];

    auto const event_times = tt_.event_times_at_stop(
        r, stop_idx, kFwd ? event_type::kDep : event_type::kArr);

    auto const seek_first_day = [&]() {
      return linear_lb(get_begin_it(event_times), get_end_it(event_times),
                       mam_at_stop,
                       [&](delta const a, minutes_after_midnight_t const b) {
                         return is_better(a.mam(), b.count());
                       });
    };

    trace("┊ │k={}    et: current_best_at_stop={}, stop_idx={}, location={}\n",
          k, tt_.to_unixtime(day_at_stop, mam_at_stop), stop_idx,
          loc{tt_, stop{tt_.route_location_seq_[r][stop_idx]}.location_idx()});

    auto const n_days_to_iterate = kMaxTravelTime / std::chrono::days{1} + 1U;
    for (auto i = day_idx_t::value_t{0U}; i != n_days_to_iterate; ++i) {
      auto const day = kFwd ? day_at_stop + i : day_at_stop - i;

      if (!tt_.is_route_active(r, day)) {
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

        if (!is_better_loose(to_delta(day, ev_mam) + dir(get_lb(to_idx(l))),
                             dest_bound)) {
          trace(
              "┊ │k={}      => name={}, dbg={}, day={}={}, best_mam={}, "
              "transport_mam={}, transport_time={} => TIME AT DEST {} IS "
              "BETTER!\n",
              k, tt_.transport_name(tt_.route_transport_ranges_[r][t_offset]),
              tt_.dbg(tt_.route_transport_ranges_[r][t_offset]), day,
              tt_.to_unixtime(day, 0_minutes), mam_at_stop, ev_mam,
              tt_.to_unixtime(day, duration_t{ev_mam}), to_unix(dest_bound));
          return {transport_idx_t::invalid(), day_idx_t::invalid()};
        }

        auto const t = tt_.route_transport_ranges_[r][t_offset];
        if (i == 0U && !is_better_or_eq(mam_at_stop.count(), ev_mam)) {
          trace(
              "┊ │k={}      => transport={}, name={}, dbg={}, day={}/{}, "
              "best_mam={}, "
              "transport_mam={}, transport_time={} => NO REACH!\n",
              k, t, tt_.transport_name(t), tt_.dbg(t), i, day, mam_at_stop,
              ev_mam, ev);
          continue;
        }

        auto const ev_day_offset = ev.days();
        auto const start_day =
            static_cast<day_idx_t>(as_int(day) - ev_day_offset);
        if (!is_transport_active(v, t, start_day)) {
          trace(
              "┊ │k={}      => transport={}, name={}, dbg={}, day={}/{}, "
              "ev_day_offset={}, "
              "best_mam={}, "
              "transport_mam={}, transport_time={} => NO TRAFFIC!\n",
              k, t, tt_.transport_name(t), tt_.dbg(t), i, day, ev_day_offset,
              mam_at_stop, ev_mam, ev);
          continue;
        }

        trace(
            "┊ │k={}      => ET FOUND: name={}, dbg={}, at day {} "
            "(day_offset={}) - ev_mam={}, ev_time={}, ev={}\n",
            k, tt_.transport_name(t), tt_.dbg(t), day, ev_day_offset, ev_mam,
            ev, tt_.to_unixtime(day, duration_t{ev_mam}));
        return {t, static_cast<day_idx_t>(as_int(day) - ev_day_offset)};
      }
    }
    return {};
  }

  // fused earliest-transport walk for the scheduled+rt criterion:
  // one pass over the events, each slot takes its first active transport
  // (slot 0: scheduled activity, slot 1: rt activity). For transports the
  // rt updates never touched, the rt check reuses the scheduled result.
  std::array<transport, 2U> get_earliest_transport_dual(
      unsigned const k,
      route_idx_t const r,
      stop_idx_t const stop_idx,
      day_idx_t const day_at_stop,
      minutes_after_midnight_t const mam_at_stop,
      location_idx_t const l) {
    ++stats_.n_earliest_trip_calls_;

    auto const b0 = time_at_dest_[k][Crit::dest_bound_of(0U)];
    auto const b1 = time_at_dest_[k][Crit::dest_bound_of(1U)];

    auto const event_times = tt_.event_times_at_stop(
        r, stop_idx, kFwd ? event_type::kDep : event_type::kArr);

    auto const seek_first_day = [&]() {
      return linear_lb(get_begin_it(event_times), get_end_it(event_times),
                       mam_at_stop,
                       [&](delta const a, minutes_after_midnight_t const b) {
                         return is_better(a.mam(), b.count());
                       });
    };

    auto out = std::array<transport, 2U>{};
    auto done0 = false, done1 = false;

    auto const n_days_to_iterate = kMaxTravelTime / std::chrono::days{1} + 1U;
    for (auto i = day_idx_t::value_t{0U}; i != n_days_to_iterate; ++i) {
      auto const day = kFwd ? day_at_stop + i : day_at_stop - i;

      if (!tt_.is_route_active(r, day)) {
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

        auto const lb_time = to_delta(day, ev_mam) + dir(get_lb(to_idx(l)));
        done0 = done0 || !is_better_loose(lb_time, b0);
        done1 = done1 || !is_better_loose(lb_time, b1);
        if (done0 && done1) {
          return out;
        }

        if (i == 0U && !is_better_or_eq(mam_at_stop.count(), ev_mam)) {
          continue;
        }

        auto const t = tt_.route_transport_ranges_[r][t_offset];
        auto const ev_day_offset = ev.days();
        auto const start_day =
            static_cast<day_idx_t>(as_int(day) - ev_day_offset);

        auto const sched_active = tt_.is_transport_active(t, start_day);
        if (!done0 && sched_active) {
          out[0] = {t, start_day};
          done0 = true;
        }
        if (!done1) {
          auto const rt_active = rtt_->has_rt_traffic_days(t)
                                     ? rtt_->is_transport_active(t, start_day)
                                     : sched_active;
          if (rt_active) {
            out[1] = {t, start_day};
            done1 = true;
          }
        }
        if (done0 && done1) {
          return out;
        }
      }
    }
    return out;
  }

  bool is_transport_active(std::size_t const v,
                           transport_idx_t const t,
                           day_idx_t const day) const {
    if constexpr (Rt) {
      return Crit::slot_uses_rt(v) ? rtt_->is_transport_active(t, day)
                                   : tt_.is_transport_active(t, day);
    } else {
      return tt_.is_transport_active(t, day);
    }
  }

  delta_t time_at_stop(route_idx_t const r,
                       transport const t,
                       stop_idx_t const stop_idx,
                       event_type const ev_type) {
    return to_delta(t.day_,
                    tt_.event_mam(r, t.t_idx_, stop_idx, ev_type).count());
  }

  delta_t rt_time_at_stop(rt_transport_idx_t const rt_t,
                          stop_idx_t const stop_idx,
                          event_type const ev_type) {
    return to_delta(rtt_->base_day_idx_,
                    rtt_->event_time(rt_t, stop_idx, ev_type));
  }

  delta_t to_delta(day_idx_t const day, std::int16_t const mam) {
    return clamp((as_int(day) - as_int(base_)) * 1440 + mam);
  }

  unixtime_t to_unix(delta_t const t) { return delta_to_unix(base(), t); }

  std::pair<day_idx_t, minutes_after_midnight_t> split(delta_t const x) {
    return split_day_mam(base_, x);
  }

  bool is_intermodal_dest() const { return !dist_to_end_.empty(); }

  void update_time_at_dest(unsigned const k,
                           std::size_t const v,
                           delta_t const t) {
    if constexpr (SearchMode == search_mode::kOneToAll) {
      return;
    }
    auto const b = Crit::dest_bound_of(v);
    for (auto i = k; i != time_at_dest_.size(); ++i) {
      time_at_dest_[i][b] = get_best(time_at_dest_[i][b], t);
    }
  }

  int as_int(day_idx_t const d) const { return static_cast<int>(d.v_); }

  template <typename T>
  auto get_begin_it(T const& t) {
    if constexpr (kFwd) {
      return t.begin();
    } else {
      return t.rbegin();
    }
  }

  template <typename T>
  auto get_end_it(T const& t) {
    if constexpr (kFwd) {
      return t.end();
    } else {
      return t.rend();
    }
  }

  timetable const& tt_;
  rt_timetable const* rtt_{nullptr};
  int n_days_;
  std::uint32_t n_locations_, n_routes_, n_rt_transports_;
  raptor_state& state_;
  bitvec end_reachable_;
  std::span<bag_t> tmp_;
  std::span<bag_t> best_;
  flat_matrix_view<bag_t> round_times_;
  // copy-on-diverge planes (aliases into the same flat storage)
  std::span<delta_t> tmp0_, tmp1_, best0_, best1_;
  flat_matrix_view<std::array<delta_t, 1>> rt0_, rt1_;
  bitvec const& is_dest_;
  Crit crit_;
  std::vector<std::uint16_t> const& dist_to_end_;
  hash_map<location_idx_t, std::vector<td_offset>> const& td_dist_to_end_;
  std::vector<std::uint16_t> const& lb_;
  std::array<dest_bounds_t, kMaxTransfers + 2> time_at_dest_;
  day_idx_t base_;
  raptor_stats stats_;
  flat_matrix_view<bag_t const> bounds_;
  unsigned bounds_last_k_{0U};
  profile_idx_t prf_idx_{0U};
  clasz_mask_t allowed_claszes_;
  bool require_bike_transport_;
  bool require_car_transport_;
  bool no_compulsory_reservation_;
  bool is_wheelchair_;
  transfer_time_settings transfer_time_settings_;
};

template <direction SearchDir,
          bool Rt,
          via_offset_t Vias,
          search_mode SearchMode>
using raptor =
    basic_raptor<SearchDir, Rt, via_criterion<SearchDir, Vias>, SearchMode>;

}  // namespace nigiri::routing
