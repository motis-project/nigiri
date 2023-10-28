#pragma once

#include <cstdint>

#include "nigiri/routing/pareto_set.h"
#include "nigiri/types.h"

namespace nigiri::routing {

struct sorted_transport {
  sorted_transport() noexcept = default;

  sorted_transport(day_idx_t const base, transport const& t) noexcept
      : day_idx_{static_cast<std::int32_t>(to_idx(t.day_)) -
                 static_cast<std::int32_t>(to_idx(base))},
        t_idx_{static_cast<std::int32_t>(to_idx(t.t_idx_))} {}

  bool operator<=(sorted_transport const& o) const noexcept {
    return day_idx_ < o.day_idx_ ||
           (day_idx_ == o.day_idx_ && t_idx_ <= o.t_idx_);
  }

  bool operator>=(sorted_transport const& o) const noexcept {
    return day_idx_ > o.day_idx_ ||
           (day_idx_ == o.day_idx_ && t_idx_ >= o.t_idx_);
  }

  transport get_transport(day_idx_t const base) const noexcept {
    return {
        .t_idx_ = transport_idx_t{t_idx_},
        .day_ = day_idx_t{static_cast<std::int32_t>(to_idx(base)) + day_idx_}};
  }

  std::int32_t day_idx_ : 2;
  std::int32_t t_idx_ : 30;
};

struct transport_stop {
  template <direction SeachDir>
  bool dominates(transport_stop const& o) const noexcept {
    return active_ >= o.active_ &&
           (SeachDir == direction::kForward ? t_ <= o.t_ : t_ >= o.t_) &&
           (SeachDir == direction::kForward ? stop_idx_ <= o.stop_idx_
                                            : stop_idx_ >= o.stop_idx_) &&
           k_ <= o.k_;
  }

  sorted_transport t_;
  stop_idx_t stop_idx_;
  std::uint8_t k_;
  bool active_{false};
};

struct reached {
  template <direction SearchDir>
  stop_idx_t get_stop(day_idx_t const base,
                      route_idx_t const r,
                      transport const x,
                      std::uint8_t const k) {
    auto const t = sorted_transport{base, x};
    if constexpr (SearchDir == direction::kForward) {
      auto best = std::numeric_limits<stop_idx_t>::max();
      for (auto const& re : route_transport_stops_[r]) {
        if (k < re.k_) {
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
        if (k < re.k_) {
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
  auto add(route_idx_t const r,
           sorted_transport const t,
           stop_idx_t const stop_idx,
           std::uint8_t const k) {
    auto const x = route_transport_stops_[r].add(
        transport_stop{
            .t_ = t, .stop_idx_ = stop_idx, .k_ = k, .active_ = true},
        [](auto&& a, auto&& b) { return a.template dominates<SearchDir>(b); });
    auto const& [added, inserted_it, dominated_by_it] = x;
    if (added) {
      inserted_it->active_ = false;
    }
    return x;
  }

  void activate(route_idx_t const r) {
    for (auto& x : route_transport_stops_[r]) {
      x.active_ = true;
    }
  }

  vector_map<route_idx_t, pareto_set<transport_stop>> route_transport_stops_;
};

}  // namespace nigiri::routing