#pragma once

#include "nigiri/common/dial.h"
#include "nigiri/routing/a_star/settings.h"
#include "nigiri/routing/tb/tb_data.h"
#include "nigiri/types.h"

// #define as_debug fmt::println
#define as_debug(...)

namespace nigiri::routing {

using tb_data = tb::tb_data;
using segment_idx_t = tb::segment_idx_t;
using arrival_time_map = hash_map<segment_idx_t, delta>;
using pred_table = hash_map<segment_idx_t, segment_idx_t>;

struct queue_entry {
  queue_entry(segment_idx_t const s, std::uint8_t t)
      : segment_{s}, transfers_{t} {}

  segment_idx_t const segment_;
  std::uint8_t transfers_;
};

struct a_star_state {
  static constexpr auto const startSegmentPredecessor =
      segment_idx_t::invalid();

  a_star_state(tb_data const& tbd)
      : tbd_{tbd}, pq_{0, get_bucket_a_star{*this}} {
    end_reachable_.resize(tbd.segment_transfers_.size());
    settled_segments_.resize(tbd.segment_transfers_.size());
    start_segments_.resize(tbd.segment_transfers_.size());
  }

  bool better_arrival(queue_entry const& qe, delta const& new_arr) {
    auto const cost = cost_function(qe, new_arr);
    return arrival_time_.contains(qe.segment_)
               ? cost < cost_function(qe)
               : cost < worst_cost_ + transfer_factor_ * qe.transfers_;
  }

  uint16_t cost_function(queue_entry const& qe) const {
    return cost_function(qe, arrival_time_.at(qe.segment_));
  }

  uint16_t cost_function(queue_entry const& qe, delta const& arr) const {
    return (use_lower_bounds_
                ? (arr - start_time_).count() + lb_.at(qe.segment_)
                : (arr - start_time_).count()) +
           static_cast<uint16_t>(transfer_factor_ * qe.transfers_);
  }

  void update_segment(segment_idx_t const s,
                      delta const& new_arr,
                      segment_idx_t const pred,
                      uint8_t const transfers) {
    auto const qe = queue_entry{s, transfers};
    if (better_arrival(qe, new_arr)) {
      arrival_time_.insert_or_assign(s, new_arr);
      pred_table_.insert_or_assign(s, pred);
      pq_.push(std::move(qe));
    }
  };

  void setup(delta const start_delta,
             uint16_t const worst_cost,
             uint8_t max_transfers) {
    start_time_ = std::move(start_delta);
    worst_cost_ = worst_cost;
    pq_.n_buckets(worst_cost + std::ceil(max_transfers * transfer_factor_));
    start_segments_.for_each_set_bit([&](segment_idx_t const s) {
      auto const qe = queue_entry{s, 0};
      if (cost_function(qe) >= worst_cost) {
        as_debug("Skipping start segment {} as its cost is too high", s);
      } else {
        pq_.push(std::move(qe));
        pred_table_.emplace(s, startSegmentPredecessor);
      }
    });
  }

  void reset() {
    pred_table_.clear();
    pq_.clear();
    settled_segments_.zero_out();
    start_segments_.zero_out();
    arrival_time_.clear();
    lb_.clear();
    start_time_ = delta{(1U << 5) - 1, (1U << 11) - 1};
  }

  struct get_bucket_a_star {
    using dist_t = std::uint16_t;

    get_bucket_a_star(a_star_state const& state) : state_{state} {}

    dist_t operator()(queue_entry const& q) const {
      return state_.cost_function(q);
    }

  private:
    a_star_state const& state_;
  };

  tb_data const& tbd_;
  arrival_time_map arrival_time_;
  pred_table pred_table_;
  hash_map<segment_idx_t, duration_t> dist_to_dest_;
  dial<queue_entry, get_bucket_a_star> pq_;
  bitvec_map<segment_idx_t> end_reachable_;
  bitvec_map<segment_idx_t> settled_segments_;
  bitvec_map<segment_idx_t> start_segments_;
  float transfer_factor_;
  delta start_time_ = delta{(1U << 5) - 1, (1U << 11) - 1};
  uint16_t worst_cost_;
  hash_map<segment_idx_t, u_int16_t> lb_;
  bool use_lower_bounds_ = false;
};
}  // namespace nigiri::routing