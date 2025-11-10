#include "nigiri/routing/tb/preprocess.h"

#include "nigiri/for_each_meta.h"

#include "utl/enumerate.h"
#include "utl/get_or_create.h"
#include "utl/parallel_for.h"
#include "utl/pool.h"
#include "utl/progress_tracker.h"

#include "nigiri/common/day_list.h"
#include "nigiri/common/linear_lower_bound.h"
#include "nigiri/constants.h"
#include "nigiri/timetable.h"

namespace nigiri::routing::tb {

struct reached_line_based {
  struct rlb_entry {
    std::int8_t shift_amount_;
    std::uint16_t start_time_;
    bitfield bf_;
  };

  void update(stop_idx_t const j,
              std::int8_t const shift_amount_new,
              std::uint16_t const start_time_new,
              bitfield& bf_new) {
    for (auto& entry : transports_[j]) {
      if (bf_new.none()) {
        return;
      }

      if (entry.bf_.none()) {
        continue;
      }

      if (std::tie(shift_amount_new, start_time_new) <
          std::tie(entry.shift_amount_, entry.start_time_)) {
        entry.bf_ &= ~bf_new;
      } else {
        bf_new &= ~entry.bf_;
      }
    }
  }

  void reset(std::size_t const num_stops) {
    transports_.clear();
    for (stop_idx_t j = 0; j < num_stops; ++j) {
      transports_[j].emplace_back(std::numeric_limits<std::int8_t>::max(),
                                  std::numeric_limits<std::uint16_t>::max(),
                                  bitfield::max());
    }
  }

  mutable_fws_multimap<std::uint32_t, rlb_entry> transports_;
};

struct stats {
  std::atomic_uint32_t n_uturn_transfers_;
  std::atomic_uint32_t n_transfers_initial_;
  std::atomic_uint32_t n_transfers_reduced_;
};

struct expanded_transfer {
  void print(std::ostream& out, timetable const& tt) const {
    out << tt.dbg(transport_idx_to_)
        << ", name=" << tt.transport_name(transport_idx_to_) << ", to="
        << location{tt, stop{tt.route_location_seq_
                                 [tt.transport_route_[transport_idx_to_]]
                                 [stop_idx_to_]}
                            .location_idx()}
        << ", days=" << tt.days(bf_) << ", day_offset=" << day_offset_;
  }

  bitfield bf_;
  transport_idx_t transport_idx_to_;
  stop_idx_t stop_idx_to_;
  std::int8_t day_offset_;
};

// [transport_idx - route_first_transport_idx][stop_idx][transfer]
using transfers_t = std::vector<std::vector<std::vector<expanded_transfer>>>;

struct route_transfer {
  stop_idx_t stop_idx_from_;
  route_idx_t route_idx_to_;
  stop_idx_t stop_idx_to_;
  duration_t footpath_length_;
};

struct reached_reduction {
  struct rr_entry {
    std::uint16_t time_;
    bitfield bf_;
  };

  void update(location_idx_t const location,
              std::uint16_t time_new,
              bitfield const& bf,
              bitfield* impr = nullptr) {
    // bitfield is manipulated during update process
    auto bf_new = bf;

    // position of entry with an equal time
    auto same_time_spot = std::optional<std::uint32_t>{};

    // position of entry with no active days
    auto overwrite_spot = std::optional<std::uint32_t>{};

    // compare to existing entries of this location
    for (auto i{0U}; i != times_[location].size(); ++i) {
      if (bf_new.none()) {
        // all bits of new entry were set to zero, new entry does not improve
        // upon any times
        return;
      }

      if (time_new < times_[location][i].time_) {
        // new time is better than existing time, update bit set of existing
        // time
        times_[location][i].bf_ &= ~bf_new;
      } else {
        // new time is greater or equal
        // remove active days from new time that are already active in the
        // existing entry
        bf_new &= ~times_[location][i].bf_;
        if (time_new == times_[location][i].time_) {
          // remember this position to add active days of new time after
          // comparison to existing entries
          same_time_spot = i;
        }
      }

      if (times_[location][i].bf_.none()) {
        // existing entry has no active days left -> remember as overwrite
        // spot
        overwrite_spot = i;
      }
    }

    // after comparison to existing entries
    if (bf_new.any()) {
      // new time has at least one active day after comparison
      if (same_time_spot.has_value()) {
        // entry for this time already exists -> add active days of new time to
        // it
        times_[location][same_time_spot.value()].bf_ |= bf_new;
      } else if (overwrite_spot.has_value()) {
        // overwrite spot was found -> use for new entry
        times_[location][overwrite_spot.value()].time_ = time_new;
        times_[location][overwrite_spot.value()].bf_ = bf_new;
      } else {
        // add new entry
        times_[location].emplace_back(time_new, bf_new);
      }

      // add improvements to impr
      if (impr != nullptr) {
        *impr |= bf_new;
      }
    }
  }

  void reset() noexcept { times_.clear(); }

  mutable_fws_multimap<location_idx_t, rr_entry> times_;
};

struct state {
  std::vector<route_transfer> neighborhood_;
  reached_line_based reached_;
  reached_reduction rr_arr_;
  reached_reduction rr_ch_;
};

void add_non_uturn_transfers(timetable const& tt,
                             route_idx_t const route_from,
                             stop_idx_t const from_stop_idx,
                             footpath const fp,
                             std::vector<route_transfer>& neighborhood,
                             stats& stats) {
  auto const prev_src_stop =
      stop{tt.route_location_seq_[route_from][from_stop_idx - 1]};
  for (auto const route_to : tt.location_routes_[fp.target()]) {
    auto const stop_seq_to = tt.route_location_seq_[route_to];
    for (auto j = 0U; j < stop_seq_to.size() - 1; ++j) {
      auto const target_stop = stop{stop_seq_to[j]};
      if (target_stop.location_idx() != fp.target() ||
          !target_stop.in_allowed()) {
        continue;
      }

      auto const next_tgt_stop = stop{stop_seq_to[j + 1]};

      auto const is_uturn_target_route_terminates =
          j + 1 == stop_seq_to.size() - 1 &&
          prev_src_stop.location_idx() == next_tgt_stop.location_idx() &&
          prev_src_stop.out_allowed();

      auto const is_uturn =
          prev_src_stop.location_idx() == next_tgt_stop.location_idx() &&
          prev_src_stop.out_allowed() && next_tgt_stop.in_allowed() &&
          tt.locations_.transfer_time_[prev_src_stop.location_idx()] <=
              fp.duration();

      if (!is_uturn && !is_uturn_target_route_terminates) {
        neighborhood.emplace_back(from_stop_idx, route_to, j, fp.duration());
      } else {
        ++stats.n_uturn_transfers_;
      }
    }
  }
}

void get_route_neighborhood(timetable const& tt,
                            route_idx_t const route_from,
                            profile_idx_t const prf_idx,
                            std::vector<route_transfer>& neighborhood,
                            stats& stats) {
  neighborhood.clear();

  // Examine stops of the line in reverse order, skip first stop.
  auto const stop_seq = tt.route_location_seq_[route_from];
  for (auto i = stop_seq.size() - 1U; i >= 1; --i) {
    auto const from_stop_idx = static_cast<stop_idx_t>(i);
    if (!stop{stop_seq[from_stop_idx]}.out_allowed()) {
      continue;  // Skip stop if exiting is not allowed.
    }

    // Location from which we transfer
    auto const from = stop{stop_seq[i]}.location_idx();

    // Transfer: reflexive footpath
    add_non_uturn_transfers(tt, route_from, from_stop_idx,
                            footpath{from, tt.locations_.transfer_time_[from]},
                            neighborhood, stats);

    // Outgoing footpaths
    for (auto const& fp : tt.locations_.footpaths_out_[prf_idx][from]) {
      add_non_uturn_transfers(tt, route_from, from_stop_idx, fp, neighborhood,
                              stats);
    }
  }

  utl::sort(neighborhood, [](route_transfer const& a, route_transfer const& b) {
    return a.route_idx_to_ < b.route_idx_to_ ||
           (a.route_idx_to_ == b.route_idx_to_ &&
            a.stop_idx_from_ > b.stop_idx_from_) ||
           (a.route_idx_to_ == b.route_idx_to_ &&
            a.stop_idx_from_ == b.stop_idx_from_ &&
            a.stop_idx_to_ < b.stop_idx_to_);
  });
}

void preprocess_transport(timetable const& tt,
                          state& s,
                          transport_idx_t const t,
                          profile_idx_t const prf_idx,
                          transfers_t::value_type& segment_transfers,
                          stats& stats) {
  // the bitfield of the transport we are transferring from
  auto const& traffic_days = tt.bitfields_[tt.transport_traffic_days_[t]];

  // the previous target route
  auto route_to_prev = s.neighborhood_[0].route_idx_to_;

  // stop sequence of the route we are transferring to
  auto stop_seq_to = tt.route_location_seq_[route_to_prev];

  // initial reset of earliest transport
  s.reached_.reset(stop_seq_to.size());

  // ==========
  // Line Based
  // ----------
  for (auto const& neighbor : s.neighborhood_) {
    // handle change of target line
    if (route_to_prev != neighbor.route_idx_to_) {
      route_to_prev = neighbor.route_idx_to_;
      stop_seq_to = tt.route_location_seq_[route_to_prev];
      s.reached_.reset(stop_seq_to.size());
    }

    auto const t_arr =
        tt.event_mam(t, neighbor.stop_idx_from_, event_type::kArr);
    auto const t_arr_mam = t_arr.mam();
    auto const t_arr_day_offset = static_cast<std::int8_t>(t_arr.days());
    auto const fp_arr = static_cast<std::uint16_t>(
        t_arr_mam + neighbor.footpath_length_.count());
    auto const fp_arr_mam = fp_arr % 1440;

    // departure times of transports of target route at stop j
    auto const event_times = tt.event_times_at_stop(
        neighbor.route_idx_to_, neighbor.stop_idx_to_, event_type::kDep);

    // find first departure at or after a
    // departure time of current transport_to
    auto earliest_dep =
        linear_lb(event_times.begin(), event_times.end(), fp_arr_mam,
                  [&](auto&& x, auto&& y) { return x.mam_ < y; });

    // no departure on this day at or after a
    auto transfer_day_offset = static_cast<std::int8_t>(fp_arr / 1440);
    if (earliest_dep == event_times.end()) {
      ++transfer_day_offset;  // start looking on the following day
      earliest_dep = event_times.begin();  // with the earliest transport
    }

    // days that still require earliest connecting transport
    auto remaining_traffic_days = traffic_days;
    while (remaining_traffic_days.any()) {
      // departure time of current transport in relation to arrival day
      auto const dep = static_cast<std::uint16_t>(transfer_day_offset * 1440 +
                                                  earliest_dep->mam_);

      // check if max transfer time is exceeded
      if (dep - t_arr_mam > kMaxTransferTime) {
        break;
      }

      // offset from begin of tp_to interval
      auto const k = static_cast<std::size_t>(
          std::distance(event_times.begin(), earliest_dep));

      // transport index of transport that we transfer to
      auto const u = tt.route_transport_ranges_[neighbor.route_idx_to_][k];

      // shift amount due to number of times transport u passed midnight
      auto const u_dep_day_offset =
          static_cast<std::int8_t>(earliest_dep->days());

      // total shift amount relative to from transport traffic day
      auto const total_day_offset = static_cast<std::int8_t>(
          u_dep_day_offset - t_arr_day_offset - transfer_day_offset);

      // bitfield transport to
      auto const& neighbor_traffic_days =
          tt.bitfields_[tt.transport_traffic_days_[u]];

      // init theta
      auto const u_shifted_traffic_days =
          (total_day_offset < 0)
              ? (neighbor_traffic_days >>
                 static_cast<unsigned>(-1 * total_day_offset))
              : (neighbor_traffic_days
                 << static_cast<unsigned>(total_day_offset));
      auto common_traffic_days =
          remaining_traffic_days & u_shifted_traffic_days;

      // check for match
      if (common_traffic_days.any()) {
        // remove days that are covered by this transport from omega
        remaining_traffic_days &= ~common_traffic_days;

        // update earliest transport data structure
        auto const day_offset = static_cast<std::int8_t>(
            t_arr_day_offset + transfer_day_offset - u_dep_day_offset);
        s.reached_.update(neighbor.stop_idx_to_, day_offset,
                          tt.event_mam(u, 0, event_type::kDep).mam_,
                          common_traffic_days);

        // recheck common days
        if (common_traffic_days.any()) {
          // add transfer to set
          segment_transfers[neighbor.stop_idx_from_ - 1U].push_back(
              expanded_transfer{common_traffic_days, u, neighbor.stop_idx_to_,
                                day_offset});

          ++stats.n_transfers_initial_;

          // add earliest transport entry
          s.reached_.transports_[neighbor.stop_idx_to_].emplace_back(
              static_cast<std::int8_t>(t_arr_day_offset + transfer_day_offset -
                                       u_dep_day_offset),
              tt.event_mam(u, 0, event_type::kDep).mam_, common_traffic_days);

          // update subsequent stops
          for (stop_idx_t j_prime = neighbor.stop_idx_to_ + 1U;
               j_prime < stop_seq_to.size(); ++j_prime) {
            auto improvement = common_traffic_days;
            s.reached_.update(
                j_prime,
                t_arr_day_offset + transfer_day_offset - u_dep_day_offset,
                tt.event_mam(u, 0, event_type::kDep).mam_, improvement);
            if (improvement.any()) {
              s.reached_.transports_[j_prime].emplace_back(
                  static_cast<std::int8_t>(t_arr_day_offset +
                                           transfer_day_offset -
                                           u_dep_day_offset),
                  tt.event_mam(u, 0, event_type::kDep).mam_, improvement);
            }
          }
        }
      }

      // prep next iteration
      // is this the last transport of the day?
      if (std::next(earliest_dep) == event_times.end()) {
        // passing midnight
        ++transfer_day_offset;

        // start with the earliest transport on the next day
        earliest_dep = event_times.begin();
      } else {
        ++earliest_dep;
      }
    }
  }

  // clear reached reduction
  s.rr_arr_.reset();
  s.rr_ch_.reset();

  // ==================
  // Transfer Reduction
  // ------------------
  auto const stop_seq_from = tt.route_location_seq_[tt.transport_route_[t]];
  for (auto j = 0U; j != stop_seq_from.size() - 1U; ++j) {
    auto const from_stop_idx =
        static_cast<stop_idx_t>(stop_seq_from.size() - j - 1);

    // skip stop if exiting is not allowed
    if (!stop{stop_seq_from[from_stop_idx]}.out_allowed()) {
      continue;
    }

    // the location index from which we are transferring
    auto const from_stop = stop{stop_seq_from[from_stop_idx]}.location_idx();

    // tau_arr(t,i)
    auto const arr = tt.event_mam(t, from_stop_idx, event_type::kArr);
    auto const t_arr = static_cast<std::uint16_t>(arr.count());

    // init the reached reduction data structure
    s.rr_arr_.update(from_stop, t_arr, traffic_days);
    s.rr_ch_.update(from_stop,
                    t_arr + tt.locations_.transfer_time_[from_stop].count(),
                    traffic_days);
    for (auto const& fp : tt.locations_.footpaths_out_[prf_idx][from_stop]) {
      s.rr_arr_.update(fp.target(), t_arr + fp.duration_, traffic_days);
      s.rr_ch_.update(fp.target(), t_arr + fp.duration_, traffic_days);
    }

    // iterate transfers found by line-based pruning
    for (auto transfer = segment_transfers[from_stop_idx - 1U].begin();
         transfer != segment_transfers[from_stop_idx - 1U].end();) {
      auto improvement = bitfield{};

      // update subsequent stops of route we transfer to (transfer + travel)
      auto const route_u = tt.transport_route_[transfer->transport_idx_to_];
      for (auto stop_idx = static_cast<stop_idx_t>(transfer->stop_idx_to_ + 1U);
           stop_idx != tt.route_location_seq_[route_u].size(); ++stop_idx) {
        auto const u_arr_rel_t_first_dep =
            static_cast<std::uint16_t>(transfer->day_offset_ * 1440 +
                                       tt.event_mam(transfer->transport_idx_to_,
                                                    stop_idx, event_type::kArr)
                                           .count());

        // locations after p_u_j
        auto const u_stp =
            stop{tt.route_location_seq_[route_u][stop_idx]}.location_idx();

        s.rr_arr_.update(u_stp, u_arr_rel_t_first_dep, transfer->bf_,
                         &improvement);
        s.rr_ch_.update(
            u_stp,
            u_arr_rel_t_first_dep + tt.locations_.transfer_time_[u_stp].count(),
            transfer->bf_, &improvement);

        for (auto const& fp_r :
             tt.locations_.footpaths_out_[profile_idx_t{0U}][u_stp]) {
          auto const eta = static_cast<std::uint16_t>(u_arr_rel_t_first_dep +
                                                      fp_r.duration_);
          s.rr_arr_.update(fp_r.target(), eta, transfer->bf_, &improvement);
          s.rr_ch_.update(fp_r.target(), eta, transfer->bf_, &improvement);
        }
      }

      transfer->bf_ = improvement;

      // if the transfer offers no improvement
      if (transfer->bf_.none()) {
        // remove it
        transfer = segment_transfers[from_stop_idx - 1U].erase(transfer);
        ++stats.n_transfers_reduced_;
      } else {
        ++transfer;
      }
    }
  }
}

void preprocess_route(timetable const& tt,
                      state& s,
                      route_idx_t const r,
                      profile_idx_t const prf_idx,
                      transfers_t& transfers,
                      stats& stats) {
  transfers.resize(
      std::max(transfers.size(),
               static_cast<std::size_t>(tt.route_transport_ranges_[r].size())));
  for (auto& x : transfers) {
    x.clear();
    x.resize(std::max(x.size(), static_cast<std::size_t>(
                                    tt.route_location_seq_[r].size() - 1U)));
  }

  get_route_neighborhood(tt, r, prf_idx, s.neighborhood_, stats);
  if (s.neighborhood_.empty()) {
    return;
  }

  for (auto const [i, t] : utl::enumerate(tt.route_transport_ranges_[r])) {
    preprocess_transport(tt, s, t, prf_idx, transfers[i], stats);
  }
}

tb_data preprocess(timetable const& tt, profile_idx_t const prf_idx) {
  stats stats;

  auto d = tb_data{};
  d.prf_idx_ = prf_idx;

  // Bitfield deduplication
  auto bitfields = hash_map<bitfield, tb_bitfield_idx_t>{};
  auto const get_or_create_bf = [&](bitfield const& bf) {
    return utl::get_or_create(bitfields, bf, [&]() {
      auto const idx = tb_bitfield_idx_t{d.bitfields_.size()};
      d.bitfields_.emplace_back(bf);
      return idx;
    });
  };

  // Allocate space.
  auto start = segment_idx_t{0U};
  for (auto r = route_idx_t{0U}; r != tt.n_routes(); ++r) {
    auto const stops = tt.route_location_seq_[r];
    auto const transports = tt.route_transport_ranges_[r];
    for (auto const t : transports) {
      d.transport_first_segment_.push_back(start);
      for (auto i = 0U; i != stops.size() - 1U; ++i) {
        d.segment_transports_.push_back(t);
      }
      start += static_cast<segment_idx_t::value_t>(stops.size() - 1U);
    }
  }
  d.transport_first_segment_.push_back(start);

  auto pool = utl::pool<transfers_t>{};
  auto const pt = utl::get_active_progress_tracker();
  pt->status("Compute Transfers").in_high(tt.n_routes());
  utl::parallel_ordered_collect_threadlocal<state>(
      tt.n_routes(),

      // Parallel: computation of route transfers
      [&](state& s, std::size_t const i) {
        auto const r = route_idx_t{i};
        auto route_transfers = pool.get();
        preprocess_route(tt, s, r, prf_idx, route_transfers, stats);
        return route_transfers;
      },

      // Sequential: ordered collect of route transfers
      [&](std::size_t const i, transfers_t&& route_transfers) {
        auto const r = route_idx_t{i};
        auto const transports = tt.route_transport_ranges_[r];
        for (auto const [t, transport_segments] :
             utl::zip(transports,
                      std::span{begin(route_transfers), transports.size()})) {
          auto const segments = d.get_segment_range(t);
          for (auto const [segment_idx, src] : utl::zip(
                   segments,
                   std::span{begin(transport_segments), segments.size()})) {
            auto const dst = d.segment_transfers_.add_back_sized(src.size());
            for (auto const [to, from] : utl::zip(dst, src)) {
              to.to_segment_ =
                  d.transport_first_segment_[from.transport_idx_to_] +
                  from.stop_idx_to_;
              to.traffic_days_ = get_or_create_bf(from.bf_);
              to.route_ = tt.transport_route_[from.transport_idx_to_];
              to.transport_offset_ = static_cast<std::uint16_t>(
                  to_idx(from.transport_idx_to_ -
                         tt.route_transport_ranges_[to.route_].from_));
              to.to_segment_offset_ = from.stop_idx_to_;
              to.set_day_offset(from.day_offset_);
            }
          }
        }
        pool.put(std::move(route_transfers));
      },
      pt->update_fn());

  return d;
}

}  // namespace nigiri::routing::tb