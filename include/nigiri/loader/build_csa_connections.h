#pragma once

#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::loader {

// build the vector fwd_connections_, after the rest of the timetable is build
void build_csa_connections(timetable& tt) {

  // set size of tt.fwd_connections_
  uint32_t num_con = 0;
  for (auto const r_idx : tt.transport_route_) {
    num_con += tt.route_location_seq_[r_idx].size() - 1;
  }
  tt.fwd_connections_.reserve(num_con);

  // build connections
  tt.n_active_connections_ = 0;
  for (route_idx_t r_idx = route_idx_t{0};
       r_idx < tt.route_stop_time_ranges_.size(); ++r_idx) {
    auto const transport_range = tt.route_transport_ranges_[r_idx];
    auto const n_transports = static_cast<unsigned>(transport_range.size());
    auto const stop_time_range = tt.route_stop_time_ranges_[r_idx];

    auto const stop_seq = tt.route_location_seq_[r_idx];
    for (auto stop_idx = 0U; stop_idx + 1 < stop_seq.size(); ++stop_idx) {
      auto const dep_stop_val = stop_seq[stop_idx];
      auto const arr_stop_val = stop_seq[stop_idx + 1];
      // auto const dep_stop = stop{dep_stop_val};
      // auto const arr_stop = stop{arr_stop_val};
      //  auto const dep_l_idx = dep_stop.location_idx();
      //  auto const arr_l_idx = arr_stop.location_idx();

      for (size_t nth_tran_of_r = 0; nth_tran_of_r < n_transports;
           ++nth_tran_of_r) {
        auto const tran_idx = transport_range[nth_tran_of_r];
        auto const dep_time_idx =
            static_cast<nigiri::vector<nigiri::delta>::access_type>(
                stop_time_range.from_ + n_transports * (stop_idx * 2) +
                nth_tran_of_r);
        auto const arr_time_idx =
            static_cast<nigiri::vector<nigiri::delta>::access_type>(
                stop_time_range.from_ +
                n_transports * ((stop_idx + 1) * 2 - 1) + nth_tran_of_r);
        auto const& dep_time = tt.route_stop_times_[dep_time_idx];
        auto const& arr_time = tt.route_stop_times_[arr_time_idx];

        tt.fwd_connections_.emplace_back(dep_stop_val, arr_stop_val, dep_time,
                                         arr_time, tran_idx, stop_idx);
        tt.n_active_connections_ +=
            tt.bitfields_[tt.transport_traffic_days_[tran_idx]].count();
      }
    }
  }
  assert(num_con == tt.fwd_connections_.size());

  utl::sort(tt.fwd_connections_, [](auto const& a, auto const& b) {
    return a.dep_time_.mam() < b.dep_time_.mam();
  });

  tt.travel_duration_days_.reserve(tt.n_transports());
  for (auto t_idx = transport_idx_t{0}; t_idx < tt.n_transports(); ++t_idx) {
    auto const last_dep_stop_idx = static_cast<stop_idx_t>(
        tt.route_location_seq_[tt.transport_route_[t_idx]].size() - 2);
    delta diff = tt.event_mam(t_idx, last_dep_stop_idx, event_type::kDep) -
                 tt.event_mam(t_idx, stop_idx_t{0}, event_type::kDep);
    assert(diff.days() >= 0);
    auto days = static_cast<uint16_t>(diff.days());
    tt.travel_duration_days_.push_back(days + 1);
  }
}

}  // namespace nigiri::loader