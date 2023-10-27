#pragma once

#include <cstdint>

#include "nigiri/types.h"

namespace nigiri::routing {

struct sorted_transport {
  sorted_transport(day_idx_t const base, transport const& t)
      : day_idx_{static_cast<std::int32_t>(to_idx(t.day_)) -
                 static_cast<std::int32_t>(to_idx(base))},
        transport_{static_cast<std::int32_t>(to_idx(t.t_idx_))} {}

  bool operator<=(sorted_transport const& o) const noexcept {
    return day_idx_ < o.day_idx_ ||
           (day_idx_ == o.day_idx_ && transport_ <= o.transport_);
  }

  bool operator>=(sorted_transport const& o) const noexcept {
    return day_idx_ > o.day_idx_ ||
           (day_idx_ == o.day_idx_ && transport_ >= o.transport_);
  }

  std::int32_t day_idx_ : 2;
  std::int32_t transport_ : 30;
};

struct transport_stop {
  template <direction SeachDir>
  bool dominates(transport_stop const& o) const noexcept {
    return (SeachDir == direction::kForward ? t_ <= o.t_ : t_ >= o.t_) &&
           (SeachDir == direction::kForward ? stop_idx_ <= o.stop_idx_
                                            : stop_idx_ >= o.stop_idx_) &&
           n_transfers_ <= o.n_transfers_;
  }

  sorted_transport t_;
  stop_idx_t stop_idx_;
  std::uint8_t n_transfers_;
};

struct reached {
  template <direction SearchDir>
  stop_idx_t get_stop(day_idx_t const base,
                      route_idx_t const r,
                      transport const x,
                      std::uint8_t const n_transfers) {
    auto const t = sorted_transport{base, x};
    if constexpr (SearchDir == direction::kForward) {
      auto best = std::numeric_limits<stop_idx_t>::max();
      for (auto const& re : route_transport_stops_[r]) {
        if (n_transfers < re.n_transfers_) {
          continue;
        }
        if (re.t_ <= t && re.stop_idx_ < best) {
          best = re.stop_idx_;
        }
      }
      return best;
    } else {
      auto best = std::numeric_limits<stop_idx_t>::min();
      for (auto const& re : route_transport_stops_[r]) {
        if (n_transfers < re.n_transfers_) {
          continue;
        }
        if (re.t_ >= t && re.stop_idx_ > best) {
          best = re.stop_idx_;
        }
      }
      return best;
    }
  }

  template <direction SearchDir>
  void add(day_idx_t const base,
           route_idx_t const r,
           transport const t,
           stop_idx_t const stop_idx,
           std::uint8_t const n_transfers) {
    route_transport_stops_[r].add(
        transport_stop{.t_ = sorted_transport{base, t},
                       .stop_idx_ = stop_idx,
                       .n_transfers_ = n_transfers},
        [](auto&& a, auto&& b) { return a.dominates<SearchDir>(b); });
  }

  vector_map<route_idx_t, pareto_set<transport_stop>> route_transport_stops_;
};

}  // namespace nigiri::routing