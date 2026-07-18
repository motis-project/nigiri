#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>

#include "utl/erase_duplicates.h"

#include "nigiri/for_each_meta.h"
#include "nigiri/routing/query.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

struct lb_components {
  static lb_components const& get_cached(timetable const& tt,
                                         profile_idx_t const prf_idx) {
    struct key {
      timetable const* tt_;
      profile_idx_t prf_idx_;
      std::uint32_t n_locations_, n_routes_;
      void const* fp_data_;
      void const* route_data_;
      auto operator<=>(key const&) const = default;
    };
    static auto mutex = std::mutex{};
    static auto cache = std::map<key, std::unique_ptr<lb_components>>{};

    auto const k = key{&tt,
                       prf_idx,
                       tt.n_locations(),
                       tt.n_routes(),
                       tt.locations_.footpaths_out_[prf_idx].data_.data(),
                       tt.route_location_seq_.data_.data()};
    auto const lock = std::scoped_lock{mutex};
    auto& entry = cache[k];
    if (entry == nullptr) {
      entry = std::make_unique<lb_components>(tt, prf_idx);
    }
    return *entry;
  }

  lb_components(timetable const& tt, profile_idx_t const prf_idx) {
    auto const n = tt.n_locations();

    auto parent = std::vector<std::uint32_t>(n);
    std::iota(begin(parent), end(parent), 0U);
    auto const find = [&](std::uint32_t x) {
      while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
      }
      return x;
    };
    auto const unite = [&](std::uint32_t const a, std::uint32_t const b) {
      auto const ra = find(a);
      auto const rb = find(b);
      if (ra != rb) {
        parent[rb] = ra;
      }
    };

    for (auto l = std::uint32_t{0U}; l != n; ++l) {
      auto const li = location_idx_t{l};
      for (auto const fp : tt.locations_.footpaths_out_[prf_idx][li]) {
        unite(l, to_idx(fp.target()));
      }
      for (auto const fp : tt.locations_.footpaths_in_[prf_idx][li]) {
        unite(l, to_idx(fp.target()));
      }
    }

    constexpr auto kNoId = std::numeric_limits<std::uint32_t>::max();
    comp_.resize(n);
    utl::fill(comp_, kNoId);
    n_components_ = 0U;
    for (auto l = std::uint32_t{0U}; l != n; ++l) {
      auto const r = find(l);
      if (comp_[r] == kNoId) {
        comp_[r] = n_components_++;
      }
      comp_[l] = comp_[r];
    }

    comp_loc_offsets_.resize(n_components_ + 1U);
    utl::fill(comp_loc_offsets_, 0U);
    for (auto l = std::uint32_t{0U}; l != n; ++l) {
      ++comp_loc_offsets_[comp_[l] + 1U];
    }
    std::partial_sum(begin(comp_loc_offsets_), end(comp_loc_offsets_),
                     begin(comp_loc_offsets_));
    comp_locations_.resize(n);
    auto pos = comp_loc_offsets_;
    for (auto l = std::uint32_t{0U}; l != n; ++l) {
      comp_locations_[pos[comp_[l]]++] = location_idx_t{l};
    }

    comp_route_offsets_.resize(n_components_ + 1U);
    comp_route_offsets_[0U] = 0U;
    auto routes = std::vector<route_idx_t>{};
    for (auto c = std::uint32_t{0U}; c != n_components_; ++c) {
      routes.clear();
      for (auto i = comp_loc_offsets_[c]; i != comp_loc_offsets_[c + 1U]; ++i) {
        for (auto const r : tt.location_routes_[comp_locations_[i]]) {
          routes.push_back(r);
        }
      }
      utl::erase_duplicates(routes);
      comp_routes_.insert(end(comp_routes_), begin(routes), end(routes));
      comp_route_offsets_[c + 1U] =
          static_cast<std::uint32_t>(comp_routes_.size());
    }
  }

  std::uint32_t n_components_{0U};
  std::vector<std::uint32_t> comp_;  // location -> component id
  std::vector<std::uint32_t> comp_loc_offsets_;
  std::vector<location_idx_t> comp_locations_;
  std::vector<std::uint32_t> comp_route_offsets_;
  std::vector<route_idx_t> comp_routes_;
};

// SearchDir refers to the direction of the main routing query
// fwd: finds the minimum number of transit legs backward from the destination
// bwd: finds the minimum number of transit legs forward from the destination
//
// Locations connected by footpaths are collapsed into components
template <direction SearchDir>
struct lb_transit_legs {
  static constexpr auto kUnreachable = std::numeric_limits<std::uint8_t>::max();
  static constexpr auto kUnknown = kUnreachable - 1U;

  lb_transit_legs(timetable const& tt,
                  query const& q,
                  rt_timetable const* rtt = nullptr,
                  bool const disabled = false)
      : tt_{tt},
        rtt_{rtt != nullptr && rtt->n_rt_transports() > 0U ? rtt : nullptr},
        q_{q},
        comps_{disabled ? nullptr : &lb_components::get_cached(tt, q.prf_idx_)},
        k_{0U},
        end_k_{static_cast<std::uint8_t>(
            std::min(q.max_transfers_, kMaxTransfers) + 2U)},
        any_marked_{false},
        max_finite_lb_{0U},
        exhausted_{false},
        total_time_{0U} {
    if (comps_ == nullptr) {
      return;
    }
    auto const start_time = std::chrono::steady_clock::now();

    lb_.resize(comps_->n_components_);
    utl::fill(lb_, kUnknown);
    comp_mark_.resize(comps_->n_components_);
    utl::fill(comp_mark_.blocks_, 0U);
    route_mark_.resize(tt_.n_routes());
    if (rtt_ != nullptr) {
      rt_transport_mark_.resize(rtt_->n_rt_transports());
    }

    auto const set_terminal = [&](location_idx_t const i) {
      auto const c = comps_->comp_[to_idx(i)];
      lb_[c] = 0U;
      comp_mark_.set(c, true);
    };

    for (auto const& o : q.destination_) {
      for_each_meta(tt_, q.dest_match_mode_, o.target(),
                    [&](location_idx_t const meta) { set_terminal(meta); });
    }

    for (auto const& [l, tds] : q.td_dest_) {
      for (auto const& td : tds) {
        if (td.duration() != footpath::kMaxDuration &&
            td.duration() < q.max_travel_time_) {
          for_each_meta(tt_, q.dest_match_mode_, l,
                        [&](location_idx_t const meta) { set_terminal(meta); });
        }
      }
    }

    total_time_ += std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start_time);

    k_ = 1U;
    for (auto const& s : q.start_) {
      get(s.target());
    }
    for (auto const& td : q.td_start_) {
      get(td.first);
    }
  }

  void run_round() {
    constexpr auto kFwd = SearchDir == direction::kForward;
    auto const start_time = std::chrono::steady_clock::now();

    utl::fill(route_mark_.blocks_, 0U);
    if (rtt_ != nullptr) {
      utl::fill(rt_transport_mark_.blocks_, 0U);
    }

    any_marked_ = false;
    comp_mark_.for_each_set_bit([&](std::uint64_t const c) {
      for (auto i = comps_->comp_route_offsets_[c];
           i != comps_->comp_route_offsets_[c + 1U]; ++i) {
        any_marked_ = true;
        route_mark_.set(to_idx(comps_->comp_routes_[i]), true);
      }
      if (rtt_ != nullptr) {
        for (auto i = comps_->comp_loc_offsets_[c];
             i != comps_->comp_loc_offsets_[c + 1U]; ++i) {
          auto const l = comps_->comp_locations_[i];
          for (auto const rt_t : rtt_->location_rt_transports_[l]) {
            any_marked_ = true;
            rt_transport_mark_.set(to_idx(rt_t), true);
          }
        }
      }
    });
    if (!any_marked_) {
      total_time_ += std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - start_time);
      return;
    }

    utl::fill(comp_mark_.blocks_, 0U);

    any_marked_ = false;
    auto const prev_k = static_cast<std::uint8_t>(k_ - 1U);
    auto const relax_seq = [&](auto const& seq) {
      auto prop = false;
      for (auto x = 0U; x != seq.size(); ++x) {
        auto const pos = kFwd ? seq.size() - x - 1U : x;
        auto const c = comps_->comp_[to_idx(stop{seq[pos]}.location_idx())];
        auto const lb = lb_[c];
        if (lb == prev_k) {
          prop = true;
        } else if (prop) {
          if (k_ < lb) {
            lb_[c] = k_;
            comp_mark_.set(c, true);
            any_marked_ = true;
          }
        }
      }
    };

    route_mark_.for_each_set_bit([&](std::uint64_t const i) {
      relax_seq(tt_.route_location_seq_[route_idx_t{i}]);
    });
    if (rtt_ != nullptr) {
      rt_transport_mark_.for_each_set_bit([&](std::uint64_t const i) {
        relax_seq(rtt_->rt_transport_location_seq_[rt_transport_idx_t{i}]);
      });
    }

    total_time_ += std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start_time);
  }

  void set_end_k(std::uint8_t const end_k) { end_k_ = end_k; }

  bool enabled() const { return comps_ != nullptr; }

  std::uint8_t max_lb() const {
    return comps_ == nullptr ? std::uint8_t{0U}
           : exhausted_      ? max_finite_lb_
                             : kUnreachable;
  }

  void advance() {
    run_round();
    if (any_marked_) {
      max_finite_lb_ = k_;
    }
    ++k_;
    if (!any_marked_) {
      exhausted_ = true;
      for (auto& lb : lb_) {
        if (lb == kUnknown) {
          lb = kUnreachable;
        }
      }
    }
  }

  std::uint8_t get(location_idx_t const l) {
    if (comps_ == nullptr) {
      return 0U;
    }
    auto const c = comps_->comp_[to_idx(l)];
    while (lb_[c] == kUnknown) {
      if (k_ >= end_k_) {
        return kUnreachable;
      }
      advance();
    }
    return lb_[c];
  }

  std::vector<std::uint8_t> const& route_lb() {
    if (comps_ != nullptr && route_lb_.empty()) {
      while (!exhausted_ && k_ < end_k_) {
        advance();
      }
      auto const start_time = std::chrono::steady_clock::now();
      route_lb_.resize(tt_.n_routes());
      utl::fill(route_lb_, kUnreachable);
      for (auto c = std::uint32_t{0U}; c != comps_->n_components_; ++c) {
        auto const lb = lb_[c] == kUnknown ? kUnreachable : lb_[c];
        for (auto i = comps_->comp_route_offsets_[c];
             i != comps_->comp_route_offsets_[c + 1U]; ++i) {
          auto& entry = route_lb_[to_idx(comps_->comp_routes_[i])];
          entry = std::min(entry, lb);
        }
      }
      total_time_ += std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - start_time);
    }
    return route_lb_;
  }

  timetable const& tt_;
  rt_timetable const* rtt_;
  query const& q_;
  lb_components const* comps_;
  std::uint8_t k_;
  std::uint8_t end_k_;
  bool any_marked_;
  std::uint8_t max_finite_lb_;
  bool exhausted_;
  std::chrono::microseconds total_time_;
  bitvec comp_mark_;
  bitvec route_mark_;
  bitvec rt_transport_mark_;
  std::vector<std::uint8_t> lb_;  // component id -> lower bound
  std::vector<std::uint8_t> route_lb_;
};

}  // namespace nigiri::routing
