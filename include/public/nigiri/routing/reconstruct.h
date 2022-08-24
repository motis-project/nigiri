#pragma once

#include "nigiri/routing/limits.h"
#include "nigiri/routing/search_state.h"

namespace nigiri::routing {

template <direction SearchDir>
void reconstruct(timetable const& tt,
                 query const& q,
                 search_state const& s,
                 journey& j) {
  (void)q;
  constexpr auto const kFwd = SearchDir == direction::kForward;

  auto const get_route_transport =
      [&](location_idx_t const l, routing_time const event_time,
          route_idx_t const r,
          unsigned const stop_idx) -> std::optional<transport> {
    for (auto const t : tt.route_transport_ranges_[r]) {
      auto const event_mam =
          tt.event_mam(t, stop_idx, kFwd ? event_type::kArr : event_type::kDep);
      if (event_mam.count() % 1440 != event_time.mam().count()) {
        continue;
      }
    }
    return std::nullopt;
  };

  auto const get_transport =
      [&](location_idx_t const l,
          routing_time const event_time) -> std::optional<journey::leg> {
    for (auto const& r : tt.location_routes_[l]) {
      for (auto const [i, stop] : utl::enumerate(tt.route_location_seq_[r])) {
        if (stop.location_idx() != l) {
          continue;
        }

        auto const transport = get_route_transport(l, event_time, r, i);
        if (transport.has_value()) {
          return journey::leg{};
        }
      }
    }
  };

  auto const get_legs =
      [&](unsigned const k,
          location_idx_t const l) -> std::pair<journey::leg, journey::leg> {
    auto const curr_time = s.round_times_[k][to_idx(l)];
    auto const& fps =
        kFwd ? tt.locations_.footpaths_in_[l] : tt.locations_.footpaths_out_[l];
    for (auto const& fp : fps) {
      auto const fp_start = curr_time - (kFwd ? fp.duration_ : -fp.duration_);
      auto const t = get_transport(fp_start);
      if (t.has_value()) {
        return {journey::leg{
                    .from_ = kFwd ? fp.target_ : l,
                    .to_ = kFwd ? l : fp.target_,
                    .dep_time_ = (kFwd ? fp_start : curr_time).to_unixtime(tt),
                    .arr_time_ = (kFwd ? curr_time : fp_start).to_unixtime(tt),
                    .uses_ = footpath_idx_t::invalid()},
                *t};
      }
    }

    return {};
  };

  auto l = j.dest_;
  for (auto i = 0U; i <= j.transfers_; ++i) {
    auto const k = j.transfers_ - i;
    auto [fp_leg, transport_leg, next_l] = get_legs(k, l);
    j.add(std::move(fp_leg));
    j.add(std::move(transport_leg));
    l = next_l;
  }
}

template <direction SearchDir>
void reconstruct(timetable const& tt, query const& q, search_state& s) {
  auto const starts_in_interval = [&](journey const& j) {
    return q.interval_.contains(j.start_time_);
  };

  for (auto it = begin(s.results_); it != end(s.results_);) {
    if (starts_in_interval(*it)) {
      reconstruct<SearchDir>(tt, q, s, *it);
      ++it;
    } else {
      it = s.results_.erase(it);
    }
  }
}

}  // namespace nigiri::routing
