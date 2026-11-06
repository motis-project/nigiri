#pragma once

// Device-resident (rt) timetable storage, shared by gpu raptor and mcraptor.

#include <array>
#include <memory>
#include <vector>

#include "thrust/copy.h"
#include "thrust/device_vector.h"

#include "utl/helpers/algorithm.h"
#include "utl/verify.h"

#include "nigiri/routing/gpu/breadcrumb.h"
#include "nigiri/routing/gpu/device_timetable.cuh"
#include "nigiri/routing/gpu/raptor.h"
#include "nigiri/routing/gpu/types.cuh"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

namespace nigiri::routing::gpu {
// words a device_bitvec<std::uint32_t> needs to hold n bits
constexpr std::uint32_t n_bitvec_words(std::uint32_t const n) {
  return n / 32U + 1U;
}

struct gpu_timetable::impl {
  using t = timetable;
  using fp_t = decltype(t{}.locations_.footpaths_out_[0]);

  static std::vector<std::uint32_t> build_route_stop_offset(
      timetable const& tt) {
    auto v = std::vector<std::uint32_t>(tt.n_routes() + 1U);
    for (auto r = 0U; r < tt.n_routes(); ++r) {
      v[r + 1U] = v[r] + static_cast<std::uint32_t>(
                             tt.route_location_seq_[route_idx_t{r}].size());
    }
    return v;
  }

  static std::vector<std::uint32_t> build_route_of_stop(
      timetable const& tt, std::vector<std::uint32_t> const& off) {
    auto v = std::vector<std::uint32_t>(off.back());
    for (auto r = 0U; r < tt.n_routes(); ++r) {
      for (auto s = off[r]; s < off[r + 1U]; ++s) {
        v[s] = r;
      }
    }
    return v;
  }

  explicit impl(timetable const& tt)
      : n_locations_{tt.n_locations()},
        n_routes_{tt.n_routes()},
        transfer_time_{to_device(tt.locations_.transfer_time_)},
        first_dep_{to_device(tt.transport_first_dep_)},
        transport_pattern_{to_device(tt.transport_pattern_)},
        route_times_{to_device(tt.route_times_)},
        patterns_{to_device(tt.pattern_offsets_)},
        route_clasz_{to_device(tt.route_clasz_)},
        route_section_clasz_{tt.route_section_clasz_},
        route_location_seq_{tt.route_location_seq_},
        location_routes_{tt.location_routes_},
        transport_traffic_days_{to_device(tt.transport_traffic_days_)},
        route_traffic_days_{to_device(tt.route_traffic_days_)},
        transport_route_{to_device(tt.transport_route_)},
        bitfields_{to_device(tt.bitfields_)},
        route_bikes_allowed_{to_device(tt.route_flags_[kBikesAllowed].blocks_)},
        route_cars_allowed_{to_device(tt.route_flags_[kCarsAllowed].blocks_)},
        route_wheelchair_accessible_{
            to_device(tt.route_flags_[kWheelchairAccessible].blocks_)},
        route_reservation_not_required_{
            to_device(tt.route_flags_[kReservationNotRequired].blocks_)},
        route_bike_sections_{tt.route_flags_per_section_[kBikesAllowed]},
        route_car_sections_{tt.route_flags_per_section_[kCarsAllowed]},
        route_wheelchair_sections_{
            tt.route_flags_per_section_[kWheelchairAccessible]},
        route_reservation_not_required_sections_{
            tt.route_flags_per_section_[kReservationNotRequired]},
        internal_interval_days_{tt.internal_interval_days()} {
    auto const off = build_route_stop_offset(tt);
    route_stop_offset_.assign(off.begin(), off.end());
    auto const ros = build_route_of_stop(tt, off);
    route_of_stop_.assign(ros.begin(), ros.end());
    for (auto p = profile_idx_t{0U}; p != kNProfiles; ++p) {
      footpaths_out_[p] = device_vecvec<fp_t>{tt.locations_.footpaths_out_[p]};
      footpaths_in_[p] = device_vecvec<fp_t>{tt.locations_.footpaths_in_[p]};
    }

    // device-resident (launch-arg size, see device_transport_filters)
    auto f = device_transport_filters<route_idx_t>{
        .bike_ = {{to_view(route_bikes_allowed_)},
                  to_view(route_bike_sections_)},
        .car_ = {{to_view(route_cars_allowed_)}, to_view(route_car_sections_)},
        .wheelchair_ = {{to_view(route_wheelchair_accessible_)},
                        to_view(route_wheelchair_sections_)},
        .reservation_not_required_ = {
            {to_view(route_reservation_not_required_)},
            to_view(route_reservation_not_required_sections_)}};
    filters_ctx_.resize(1);
    thrust::copy_n(&f, 1, filters_ctx_.begin());
  }

  device_timetable to_device_timetable() const {
    auto dt = device_timetable{
        .n_locations_ = n_locations_,
        .n_routes_ = n_routes_,
        .transfer_time_ = transfer_time_,
        .first_dep_ = first_dep_,
        .transport_pattern_ = transport_pattern_,
        .route_times_ = route_times_,
        .patterns_ = to_view(patterns_),
        .route_clasz_ = to_view(route_clasz_),
        .route_section_clasz_ = to_view(route_section_clasz_),
        .route_location_seq_ = to_view(route_location_seq_),
        .location_routes_ = to_view(location_routes_),
        .transport_traffic_days_ = to_view(transport_traffic_days_),
        .route_traffic_days_ = to_view(route_traffic_days_),
        .transport_route_ = to_view(transport_route_),
        .bitfields_ = to_view(bitfields_),
        .filters_ = thrust::raw_pointer_cast(filters_ctx_.data()),
        .route_stop_offset_ = to_view(route_stop_offset_),
        .route_of_stop_ = to_view(route_of_stop_),
        .internal_interval_days_ = internal_interval_days_};
    for (auto p = 0U; p != kNProfiles; ++p) {
      dt.footpaths_out_[p] = to_view(footpaths_out_[p]);
      dt.footpaths_in_[p] = to_view(footpaths_in_[p]);
    }
    return dt;
  }

  std::uint32_t n_locations_;
  std::uint32_t n_routes_;

  thrust::device_vector<u8_minutes> transfer_time_;
  std::array<device_vecvec<fp_t>, kNProfiles> footpaths_out_;
  std::array<device_vecvec<fp_t>, kNProfiles> footpaths_in_;

  thrust::device_vector<delta> first_dep_;
  thrust::device_vector<std::uint16_t> transport_pattern_;
  thrust::device_vector<route_times> route_times_;
  thrust::device_vector<delta> patterns_;
  thrust::device_vector<clasz> route_clasz_;
  device_vecvec<decltype(t{}.route_section_clasz_)> route_section_clasz_;

  device_vecvec<decltype(t{}.route_location_seq_)> route_location_seq_;
  device_vecvec<decltype(t{}.location_routes_)> location_routes_;

  thrust::device_vector<bitfield_idx_t> transport_traffic_days_;
  thrust::device_vector<bitfield_idx_t> route_traffic_days_;
  thrust::device_vector<route_idx_t> transport_route_;
  thrust::device_vector<bitfield> bitfields_;
  thrust::device_vector<std::uint64_t> route_bikes_allowed_;
  thrust::device_vector<std::uint64_t> route_cars_allowed_;
  thrust::device_vector<std::uint64_t> route_wheelchair_accessible_;
  thrust::device_vector<std::uint64_t> route_reservation_not_required_;
  device_vecvec<decltype(t{}.route_flags_per_section_[0])> route_bike_sections_;
  device_vecvec<decltype(t{}.route_flags_per_section_[0])> route_car_sections_;
  device_vecvec<decltype(t{}.route_flags_per_section_[0])>
      route_wheelchair_sections_;
  device_vecvec<decltype(t{}.route_flags_per_section_[0])>
      route_reservation_not_required_sections_;
  thrust::device_vector<device_transport_filters<route_idx_t>> filters_ctx_;
  thrust::device_vector<std::uint32_t> route_stop_offset_;
  thrust::device_vector<std::uint32_t> route_of_stop_;

  interval<date::sys_days> internal_interval_days_;
};

struct gpu_rt_timetable::impl {
  using rtt_t = rt_timetable;

  static std::vector<clasz> build_rt_transport_clasz(rt_timetable const& rtt) {
    auto v = std::vector<clasz>{};
    v.reserve(rtt.n_rt_transports());
    for (auto rt_t = 0U; rt_t != rtt.n_rt_transports(); ++rt_t) {
      auto const sections =
          rtt.rt_transport_section_clasz_[rt_transport_idx_t{rt_t}];
      v.push_back(sections.empty() ? clasz::kOther : sections[0]);
    }
    return v;
  }

  impl(timetable const& tt, rt_timetable const& rtt)
      : n_rt_transports_{rtt.n_rt_transports()},
        base_day_idx_{rtt.base_day_idx_},
        location_rt_transports_{
            cista::raw::make_vecvec(rtt.location_rt_transports_)},
        rt_transport_location_seq_{rtt.rt_transport_location_seq_},
        rt_transport_stop_times_{rtt.rt_transport_stop_times_},
        rt_transport_clasz_{to_device(build_rt_transport_clasz(rtt))},
        rt_transport_section_clasz_{rtt.rt_transport_section_clasz_},
        transport_traffic_days_{to_device(rtt.transport_traffic_days_)},
        bitfields_{to_device(rtt.bitfields_)},
        rt_transport_bikes_allowed_{
            to_device(rtt.rt_transport_flags_[kBikesAllowed].blocks_)},
        rt_transport_cars_allowed_{
            to_device(rtt.rt_transport_flags_[kCarsAllowed].blocks_)},
        rt_transport_wheelchair_accessibility_{
            to_device(rtt.rt_transport_flags_[kWheelchairAccessible].blocks_)},
        rt_transport_reservation_not_required_{to_device(
            rtt.rt_transport_flags_[kReservationNotRequired].blocks_)},
        rt_bike_sections_{rtt.rt_flags_per_section_[kBikesAllowed]},
        rt_car_sections_{rtt.rt_flags_per_section_[kCarsAllowed]},
        rt_wheelchair_sections_{
            rtt.rt_flags_per_section_[kWheelchairAccessible]},
        rt_reservation_not_required_sections_{
            rtt.rt_flags_per_section_[kReservationNotRequired]} {
    utl::verify(
        bc_transport_space_fits(tt.transport_route_.size(), n_rt_transports_),
        "transport idx space too small: {} static + {} rt",
        tt.transport_route_.size(), n_rt_transports_);

    // Copy filters.
    auto f = device_transport_filters<rt_transport_idx_t>{
        .bike_ = {{to_view(rt_transport_bikes_allowed_)},
                  to_view(rt_bike_sections_)},
        .car_ = {{to_view(rt_transport_cars_allowed_)},
                 to_view(rt_car_sections_)},
        .wheelchair_ = {{to_view(rt_transport_wheelchair_accessibility_)},
                        to_view(rt_wheelchair_sections_)},
        .reservation_not_required_ = {
            {to_view(rt_transport_reservation_not_required_)},
            to_view(rt_reservation_not_required_sections_)}};
    rt_filters_ctx_.resize(1);
    thrust::copy_n(&f, 1, rt_filters_ctx_.begin());

    // Copy td-footpaths.
    for (auto p = profile_idx_t{0U}; p != kNProfiles; ++p) {
      // for the host-side kernel dispatch
      has_td_fps_[p] = rtt.has_td_footpaths_out_[p].any() ||
                       rtt.has_td_footpaths_in_[p].any();

      if (!has_td_fps_[p]) {
        continue;
      }

      td_footpaths_out_[p] = device_vecvec<td_fp_t>{rtt.td_footpaths_out_[p]};
      has_td_out_[p] = to_device(rtt.has_td_footpaths_out_[p].blocks_);
      td_footpaths_in_[p] = device_vecvec<td_fp_t>{rtt.td_footpaths_in_[p]};
      has_td_in_[p] = to_device(rtt.has_td_footpaths_in_[p].blocks_);
    }

    // device-resident view struct (the launch-parameter struct only carries
    // a pointer; ~480B of views inline cost ~0.5% pong throughput)
    auto td = device_rt_timetable::td_footpaths{};
    for (auto p = 0U; p != kNProfiles; ++p) {
      td.out_[p] = to_view(td_footpaths_out_[p]);
      td.in_[p] = to_view(td_footpaths_in_[p]);
      td.has_out_[p] = {to_view(has_td_out_[p])};
      td.has_in_[p] = {to_view(has_td_in_[p])};
    }
    td_ctx_.resize(1);
    thrust::copy_n(&td, 1, td_ctx_.begin());
  }

  device_rt_timetable to_device_rt_timetable() const {
    auto d = device_rt_timetable{
        .n_rt_transports_ = n_rt_transports_,
        .base_day_idx_ = base_day_idx_,
        .location_rt_transports_ = to_view(location_rt_transports_),
        .rt_transport_location_seq_ = to_view(rt_transport_location_seq_),
        .rt_transport_stop_times_ = to_view(rt_transport_stop_times_),
        .rt_transport_clasz_ = to_view(rt_transport_clasz_),
        .rt_transport_section_clasz_ = to_view(rt_transport_section_clasz_),
        .transport_traffic_days_ = to_view(transport_traffic_days_),
        .bitfields_ = to_view(bitfields_),
        .filters_ = thrust::raw_pointer_cast(rt_filters_ctx_.data()),
        .td_ = thrust::raw_pointer_cast(td_ctx_.data())};
    return d;
  }

  std::uint32_t n_rt_transports_;
  day_idx_t base_day_idx_;

  device_vecvec<vecvec<location_idx_t, rt_transport_idx_t>>
      location_rt_transports_;
  device_vecvec<decltype(rtt_t{}.rt_transport_location_seq_)>
      rt_transport_location_seq_;
  device_vecvec<decltype(rtt_t{}.rt_transport_stop_times_)>
      rt_transport_stop_times_;
  thrust::device_vector<clasz> rt_transport_clasz_;
  device_vecvec<decltype(rtt_t{}.rt_transport_section_clasz_)>
      rt_transport_section_clasz_;

  thrust::device_vector<bitfield_idx_t> transport_traffic_days_;
  thrust::device_vector<bitfield> bitfields_;
  thrust::device_vector<std::uint64_t> rt_transport_bikes_allowed_;
  thrust::device_vector<std::uint64_t> rt_transport_cars_allowed_;
  thrust::device_vector<std::uint64_t> rt_transport_wheelchair_accessibility_;
  thrust::device_vector<std::uint64_t> rt_transport_reservation_not_required_;
  device_vecvec<decltype(rtt_t{}.rt_flags_per_section_[0])> rt_bike_sections_;
  device_vecvec<decltype(rtt_t{}.rt_flags_per_section_[0])> rt_car_sections_;
  device_vecvec<decltype(rtt_t{}.rt_flags_per_section_[0])>
      rt_wheelchair_sections_;
  device_vecvec<decltype(rtt_t{}.rt_flags_per_section_[0])>
      rt_reservation_not_required_sections_;
  thrust::device_vector<device_transport_filters<rt_transport_idx_t>>
      rt_filters_ctx_;

  using td_fp_t = decltype(rtt_t{}.td_footpaths_out_[0]);
  std::array<device_vecvec<td_fp_t>, kNProfiles> td_footpaths_out_;
  std::array<device_vecvec<td_fp_t>, kNProfiles> td_footpaths_in_;
  std::array<thrust::device_vector<std::uint64_t>, kNProfiles> has_td_out_;
  std::array<thrust::device_vector<std::uint64_t>, kNProfiles> has_td_in_;
  std::array<bool, kNProfiles> has_td_fps_{};
  thrust::device_vector<device_rt_timetable::td_footpaths> td_ctx_;
};

}  // namespace nigiri::routing::gpu
