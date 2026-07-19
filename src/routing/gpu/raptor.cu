#include "nigiri/routing/gpu/raptor.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <optional>
#include <string_view>

// date/date.h (pulled in transitively via nigiri/types.h above) leaks a
// `#define NOEXCEPT noexcept`. CCCL's concept machinery builds a requirement
// switch via `_CCCL_PP_CASE(NOEXCEPT)`, where the NOEXCEPT token expands to
// lowercase `noexcept` and produces the undefined
// `_CCCL_CONCEPT_REQUIREMENT_CASE_noexcept`, breaking every <cuda/std/...>
// concept header. Undo the leak before including any CUDA std header.
#undef NOEXCEPT

#include "cuda/std/array"
#include "cuda/std/span"

#include "thrust/copy.h"
#include "thrust/device_vector.h"
#include "thrust/fill.h"
#include "thrust/host_vector.h"

#include "utl/helpers/algorithm.h"
#include "utl/timer.h"

#include "nigiri/for_each_meta.h"
#include "nigiri/logging.h"
#include "nigiri/routing/component_graph.h"
#include "nigiri/routing/gpu/component_lb.cuh"
#include "nigiri/routing/gpu/cuda_check.cuh"
#include "nigiri/routing/gpu/device_buffer.cuh"
#include "nigiri/routing/gpu/device_timetable.cuh"
#include "nigiri/routing/gpu/pinned_host_buffer.cuh"
#include "nigiri/routing/gpu/raptor_impl.cuh"
#include "nigiri/routing/gpu/types.cuh"
#include "nigiri/td_footpath.h"

namespace nigiri::routing::gpu {

namespace {
bool env_enabled(char const* name) {
  auto const* const v = std::getenv(name);
  return v == nullptr || std::string_view{v} != "0";
}
bool env_opt_in(char const* name) {
  auto const* const v = std::getenv(name);
  return v != nullptr && std::string_view{v} == "1";
}
}  // namespace

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
        route_stop_times_{to_device(tt.route_stop_times_)},
        route_stop_time_ranges_{to_device(tt.route_stop_time_ranges_)},
        route_transport_ranges_{to_device(tt.route_transport_ranges_)},
        route_clasz_{to_device(tt.route_clasz_)},
        route_location_seq_{tt.route_location_seq_},
        location_routes_{tt.location_routes_},
        transport_traffic_days_{to_device(tt.transport_traffic_days_)},
        route_traffic_days_{to_device(tt.route_traffic_days_)},
        transport_route_{to_device(tt.transport_route_)},
        bitfields_{to_device(tt.bitfields_)},
        route_bikes_allowed_{to_device(tt.route_bikes_allowed_.blocks_)},
        route_cars_allowed_{to_device(tt.route_cars_allowed_.blocks_)},
        route_wheelchair_accessible_{
            to_device(tt.route_wheelchair_accessible_.blocks_)},
        route_bike_sections_{tt.route_bikes_allowed_per_section_},
        route_car_sections_{tt.route_cars_allowed_per_section_},
        route_wheelchair_sections_{
            tt.route_wheelchair_accessibility_per_section_},
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
                        to_view(route_wheelchair_sections_)}};
    filters_ctx_.resize(1);
    thrust::copy_n(&f, 1, filters_ctx_.begin());

    // component graph for the ping's per-transfer lower bounds
    // (opt-in: measured net-negative on GER/EU, see gpu-ping-lb experiment)
    if (env_opt_in("NIGIRI_GPU_PING_LB")) {
      auto const timer = scoped_timer{"gpu component graph"};
      auto const cg = build_component_graph(tt);
      n_components_ = cg.n_components_;
      location_component_ = to_device(cg.location_component_);
      comp_seqs_ = device_vecvec<decltype(cg.seqs_)>{cg.seqs_};
      comp_durations_ = device_vecvec<decltype(cg.durations_)>{cg.durations_};
      comp_routes_ = device_vecvec<decltype(cg.comp_routes_)>{cg.comp_routes_};
      has_component_graph_ = true;
      log(log_lvl::info, "gpu", "component graph: {} components, {} routes",
          n_components_, cg.seqs_.size());
    }
  }

  device_component_graph to_device_component_graph() const {
    return {.n_components_ = n_components_,
            .location_component_ = {to_view(location_component_)},
            .seqs_ = to_view(comp_seqs_),
            .durations_ = to_view(comp_durations_),
            .comp_routes_ = to_view(comp_routes_)};
  }

  device_timetable to_device_timetable() const {
    auto dt = device_timetable{
        .n_locations_ = n_locations_,
        .n_routes_ = n_routes_,
        .transfer_time_ = transfer_time_,
        .route_stop_times_ = to_view(route_stop_times_),
        .route_stop_time_ranges_ = to_view(route_stop_time_ranges_),
        .route_transport_ranges_ = to_view(route_transport_ranges_),
        .route_clasz_ = to_view(route_clasz_),
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

  thrust::device_vector<delta> route_stop_times_;
  thrust::device_vector<interval<std::uint32_t>> route_stop_time_ranges_;
  thrust::device_vector<interval<transport_idx_t>> route_transport_ranges_;
  thrust::device_vector<clasz> route_clasz_;

  device_vecvec<decltype(t{}.route_location_seq_)> route_location_seq_;
  device_vecvec<decltype(t{}.location_routes_)> location_routes_;

  thrust::device_vector<bitfield_idx_t> transport_traffic_days_;
  thrust::device_vector<bitfield_idx_t> route_traffic_days_;
  thrust::device_vector<route_idx_t> transport_route_;
  thrust::device_vector<bitfield> bitfields_;
  thrust::device_vector<std::uint64_t> route_bikes_allowed_;
  thrust::device_vector<std::uint64_t> route_cars_allowed_;
  thrust::device_vector<std::uint64_t> route_wheelchair_accessible_;
  device_vecvec<decltype(t{}.route_bikes_allowed_per_section_)>
      route_bike_sections_;
  device_vecvec<decltype(t{}.route_cars_allowed_per_section_)>
      route_car_sections_;
  device_vecvec<decltype(t{}.route_wheelchair_accessibility_per_section_)>
      route_wheelchair_sections_;
  thrust::device_vector<device_transport_filters<route_idx_t>> filters_ctx_;
  thrust::device_vector<std::uint32_t> route_stop_offset_;
  thrust::device_vector<std::uint32_t> route_of_stop_;

  // component graph (empty unless NIGIRI_GPU_PING_LB)
  bool has_component_graph_{false};
  std::uint32_t n_components_{0U};
  thrust::device_vector<component_idx_t> location_component_;
  device_vecvec<vecvec<comp_route_idx_t, component_idx_t>> comp_seqs_;
  device_vecvec<vecvec<comp_route_idx_t, std::uint16_t>> comp_durations_;
  device_vecvec<vecvec<component_idx_t, comp_route_idx_t>> comp_routes_;

  interval<date::sys_days> internal_interval_days_;
};

gpu_timetable::gpu_timetable(timetable const& tt)
    : impl_{std::make_unique<impl>(tt)} {}

gpu_timetable::~gpu_timetable() = default;

struct gpu_rt_timetable::impl {
  using rtt_t = rt_timetable;

  static vecvec<location_idx_t, rt_transport_idx_t> build_location_rt(
      timetable const& tt, rt_timetable const& rtt) {
    auto v = vecvec<location_idx_t, rt_transport_idx_t>{};
    auto tmp = std::vector<rt_transport_idx_t>{};
    for (auto l = 0U; l != tt.n_locations(); ++l) {
      tmp.clear();
      for (auto const rt_t : rtt.location_rt_transports_[location_idx_t{l}]) {
        tmp.push_back(rt_t);
      }
      v.emplace_back(tmp);
    }
    return v;
  }

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
        location_rt_transports_{build_location_rt(tt, rtt)},
        rt_transport_location_seq_{rtt.rt_transport_location_seq_},
        rt_transport_stop_times_{rtt.rt_transport_stop_times_},
        rt_transport_clasz_{to_device(build_rt_transport_clasz(rtt))},
        transport_traffic_days_{to_device(rtt.transport_traffic_days_)},
        bitfields_{to_device(rtt.bitfields_)},
        rt_transport_bikes_allowed_{
            to_device(rtt.rt_transport_bikes_allowed_.blocks_)},
        rt_transport_cars_allowed_{
            to_device(rtt.rt_transport_cars_allowed_.blocks_)},
        rt_transport_wheelchair_accessibility_{
            to_device(rtt.rt_transport_wheelchair_accessibility_.blocks_)},
        rt_bike_sections_{rtt.rt_bikes_allowed_per_section_},
        rt_car_sections_{rtt.rt_cars_allowed_per_section_},
        rt_wheelchair_sections_{rtt.rt_wheelchair_accessible_per_section_} {
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
                        to_view(rt_wheelchair_sections_)}};
    rt_filters_ctx_.resize(1);
    thrust::copy_n(&f, 1, rt_filters_ctx_.begin());

    // Copy td-footpaths.
    for (auto p = profile_idx_t{0U}; p != kNProfiles; ++p) {
      if (!rtt.td_footpaths_out_[p].empty()) {
        td_footpaths_out_[p] = device_vecvec<td_fp_t>{rtt.td_footpaths_out_[p]};
        has_td_out_[p] = to_device(rtt.has_td_footpaths_out_[p].blocks_);
      }

      if (!rtt.td_footpaths_in_[p].empty()) {
        td_footpaths_in_[p] = device_vecvec<td_fp_t>{rtt.td_footpaths_in_[p]};
        has_td_in_[p] = to_device(rtt.has_td_footpaths_in_[p].blocks_);
      }

      // for the host-side kernel dispatch
      has_td_fps_[p] = rtt.has_td_footpaths_out_[p].any() ||
                       rtt.has_td_footpaths_in_[p].any();
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

  thrust::device_vector<bitfield_idx_t> transport_traffic_days_;
  thrust::device_vector<bitfield> bitfields_;
  thrust::device_vector<std::uint64_t> rt_transport_bikes_allowed_;
  thrust::device_vector<std::uint64_t> rt_transport_cars_allowed_;
  thrust::device_vector<std::uint64_t> rt_transport_wheelchair_accessibility_;
  device_vecvec<decltype(rtt_t{}.rt_bikes_allowed_per_section_)>
      rt_bike_sections_;
  device_vecvec<decltype(rtt_t{}.rt_cars_allowed_per_section_)>
      rt_car_sections_;
  device_vecvec<decltype(rtt_t{}.rt_wheelchair_accessible_per_section_)>
      rt_wheelchair_sections_;
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

gpu_rt_timetable::gpu_rt_timetable(timetable const& tt, rt_timetable const& rtt)
    : impl_{std::make_unique<impl>(tt, rtt)} {}

gpu_rt_timetable::~gpu_rt_timetable() = default;

std::unique_ptr<void, void (*)(void*)> make_gpu_rtt(timetable const& tt,
                                                    rt_timetable const& rtt) {
  return {new gpu_rt_timetable{tt, rtt},
          [](void* p) { delete static_cast<gpu_rt_timetable*>(p); }};
}

struct gpu_raptor_state::impl {
  explicit impl(gpu_timetable const& gtt)
      : tt_{gtt.impl_->to_device_timetable()} {
    cudaStreamCreate(&stream_);

    auto const n_route_stops = tt_.route_of_stop_.size();
    et_result_.resize(n_route_stops);
    et_task_list_.resize(n_route_stops);
    et_task_count_.resize(1U);
    route_list_.resize(tt_.n_routes_);
    route_list_count_.resize(1U);

    // the state is bound to this timetable for its whole lifetime -> all
    // n_locations/n_routes-sized buffers are allocated once here; the
    // selective-clear scheme requires an all-invalid (0xFF) start
    time_at_dest_.resize(kMaxTransfers + 2);
    tmp_.resize(tt_.n_locations_);
    best_.resize(tt_.n_locations_);
    round_times_.resize(tt_.n_locations_ * (kMaxTransfers + 2));
    cudaMemsetAsync(thrust::raw_pointer_cast(round_times_.data()), 0xFF,
                    round_times_.size() * sizeof(std::uint64_t), stream_);
    cudaMemsetAsync(thrust::raw_pointer_cast(best_.data()), 0xFF,
                    best_.size() * sizeof(std::uint64_t), stream_);
    cudaMemsetAsync(thrust::raw_pointer_cast(tmp_.data()), 0xFF,
                    tmp_.size() * sizeof(std::uint64_t), stream_);
    station_mark_.resize(tt_.n_locations_ / 32U + 1U);
    prev_station_mark_.resize(tt_.n_locations_ / 32U + 1U);
    route_mark_.resize(tt_.n_routes_ / 32U + 1U);
    any_marked_.resize(1U);
    done_.resize(1U);
    n_pruned_.resize(1U);

    // component lower bounds working set (ping pruning)
    has_cg_ = gtt.impl_->has_component_graph_;
    if (has_cg_) {
      cg_ = gtt.impl_->to_device_component_graph();
      auto const n_comp = gtt.impl_->n_components_;
      auto const n_comp_routes = gtt.impl_->comp_seqs_.index_.size() - 1U;
      lb_tt_cur_.resize(n_comp);
      lb_tt_prev_.resize(n_comp);
      lb_ic_.resize(n_comp);
      comp_lb_.resize(n_comp);
      lb_comp_marked_.resize(n_comp / 32U + 1U);
      lb_route_marked_.resize(n_comp_routes / 32U + 1U);
      lb_flags_.resize(2U);  // [0]=any_improved, [1]=done
    }
    adhoc_tp_dev_.resize(tt_.n_locations_ / 64U + 1U);
    prelude_seeds_dev_.resize(1U);

    // is_dest + dist_to_dest differ between ping vs pong -> handled in
    // upload_query (called by the gpu_raptor ctor -> one per dir)
  }

  ~impl() { cudaStreamDestroy(stream_); }

  // the only per-query sizing: the rt timetable may be absent or may have
  // grown (rt updates) since the last query on this state
  void resize_rt(unsigned const n_rt_transports) {
    rt_transport_mark_.resize(n_rt_transports / 32U + 1U);
  }

  void upload_query(
      unsigned const dir /* fwd=0 bwd=1 -> ping/pong can coexist */,
      nigiri::bitvec const& is_dest,
      std::vector<std::uint16_t> const& dist_to_dest,
      hash_map<location_idx_t, std::vector<td_offset>> const& td_dist_to_dest,
      std::vector<std::uint16_t> const& lb) {
    is_intermodal_dest_[dir] = !dist_to_dest.empty();

    // location-level lower bounds (CPU dijkstra, travel time only);
    // empty = disabled (also reset: the state may be reused across queries)
    if (lb.empty()) {
      loc_lb_dev_[dir].clear();
    } else {
      auto* const lb_pin = loc_lb_pin_[dir].ensure(lb.size());
      std::copy(lb.begin(), lb.end(), lb_pin);
      utl::verify(cudaSuccess ==
                      cudaMemcpyAsync(loc_lb_dev_[dir].ensure(lb.size(),
                                                              stream_),
                                      lb_pin, lb.size() * sizeof(std::uint16_t),
                                      cudaMemcpyHostToDevice, stream_),
                  "could not copy location lower bounds");
    }

    // td dest offsets: flatten into sorted (loc, range, data) groups.
    if (td_dist_to_dest.empty()) {
      // reset used sizes: the state may have served a flex query before
      td_dest_locs_dev_[dir].clear();
      td_dest_ranges_dev_[dir].clear();
      td_dest_data_dev_[dir].clear();
    } else {
      auto locs = std::vector<location_idx_t>{};
      locs.reserve(td_dist_to_dest.size());
      for (auto const& [l, _] : td_dist_to_dest) {
        locs.push_back(l);
      }
      utl::sort(locs);  // hash_map order is not deterministic
      auto ranges = std::vector<std::uint32_t>{0U};
      auto td_data = std::vector<td_offset>{};
      for (auto const l : locs) {
        auto const& v = td_dist_to_dest.at(l);
        td_data.insert(end(td_data), begin(v), end(v));
        ranges.push_back(static_cast<std::uint32_t>(td_data.size()));
      }

      auto* const locs_pin = td_dest_locs_pin_[dir].ensure(locs.size());
      std::copy(locs.begin(), locs.end(), locs_pin);
      auto* const ranges_pin = td_dest_ranges_pin_[dir].ensure(ranges.size());
      std::copy(ranges.begin(), ranges.end(), ranges_pin);
      auto* const data_pin = td_dest_data_pin_[dir].ensure(td_data.size());
      std::copy(td_data.begin(), td_data.end(), data_pin);
      utl::verify(
          cudaSuccess ==
                  cudaMemcpyAsync(
                      td_dest_locs_dev_[dir].ensure(locs.size(), stream_),
                      locs_pin, locs.size() * sizeof(location_idx_t),
                      cudaMemcpyHostToDevice, stream_) &&
              cudaSuccess ==
                  cudaMemcpyAsync(
                      td_dest_ranges_dev_[dir].ensure(ranges.size(), stream_),
                      ranges_pin, ranges.size() * sizeof(std::uint32_t),
                      cudaMemcpyHostToDevice, stream_) &&
              cudaSuccess ==
                  cudaMemcpyAsync(
                      td_dest_data_dev_[dir].ensure(td_data.size(), stream_),
                      data_pin, td_data.size() * sizeof(td_offset),
                      cudaMemcpyHostToDevice, stream_),
          "could not copy td dest offsets");
    }

    // Copy is_dest.
    is_dest_[dir].resize(is_dest.blocks_.size());
    auto* const is_dest_pin = is_dest_pin_[dir].ensure(is_dest.blocks_.size());
    std::copy(is_dest.blocks_.begin(), is_dest.blocks_.end(), is_dest_pin);
    utl::verify(
        cudaSuccess == cudaMemcpyAsync(
                           thrust::raw_pointer_cast(is_dest_[dir].data()),
                           is_dest_pin,
                           is_dest.blocks_.size() *
                               sizeof(std::decay_t<decltype(is_dest)>::block_t),
                           cudaMemcpyHostToDevice, stream_),
        "could not copy is_dest");

    // Copy dist_to_dest.
    dist_to_dest_dev_[dir].resize(dist_to_dest.size());
    auto* const dd_pinned = dist_to_dest_[dir].ensure(dist_to_dest.size());
    std::copy(dist_to_dest.begin(), dist_to_dest.end(), dd_pinned);
    utl::verify(cudaSuccess ==
                    cudaMemcpyAsync(
                        thrust::raw_pointer_cast(dist_to_dest_dev_[dir].data()),
                        dd_pinned, dist_to_dest.size() * sizeof(std::uint16_t),
                        cudaMemcpyHostToDevice, stream_),
                "could not copy dist to dest");
  }

  bool is_intermodal_dest_[2];  // per direction: [0]=fwd, [1]=bwd
  thrust::device_vector<std::uint32_t> any_marked_;
  thrust::device_vector<std::uint32_t> done_;

  // 16bit time | 48bit breadcrumb
  thrust::device_vector<std::uint64_t> time_at_dest_;
  thrust::device_vector<std::uint64_t> tmp_;
  thrust::device_vector<std::uint64_t> best_;
  thrust::device_vector<std::uint64_t> round_times_;

  thrust::device_vector<std::uint32_t> station_mark_;
  thrust::device_vector<std::uint32_t> prev_station_mark_;
  thrust::device_vector<std::uint32_t> route_mark_;
  thrust::device_vector<std::uint32_t> rt_transport_mark_;

  // per-direction query data: [0]=fwd, [1]=bwd (ping/pong interleave on the
  // shared state, each direction uploads its slot once in the ctor)
  thrust::device_vector<std::uint64_t> is_dest_[2];
  pinned_host_buffer<std::uint64_t> is_dest_pin_[2];

  thrust::device_vector<std::uint16_t> dist_to_dest_dev_[2];

  // location-level dijkstra lower bounds per direction (ping only in
  // practice); used size 0 = disabled
  device_buffer<std::uint16_t> loc_lb_dev_[2];
  pinned_host_buffer<std::uint16_t> loc_lb_pin_[2];

  // td egress offsets (q.td_dest_), sparse groups per direction;
  // used size 0 = no td offsets (device never touched)
  device_buffer<location_idx_t> td_dest_locs_dev_[2];
  device_buffer<std::uint32_t> td_dest_ranges_dev_[2];
  device_buffer<td_offset> td_dest_data_dev_[2];
  pinned_host_buffer<location_idx_t> td_dest_locs_pin_[2];
  pinned_host_buffer<std::uint32_t> td_dest_ranges_pin_[2];
  pinned_host_buffer<td_offset> td_dest_data_pin_[2];

  pinned_host_buffer<std::uint16_t> dist_to_dest_[2];
  pinned_host_buffer<std::pair<location_idx_t, unixtime_t>> starts_;
  thrust::device_vector<std::pair<location_idx_t, unixtime_t>> starts_dev_;

  // reusable reconstruction buffers
  pinned_host_buffer<location_idx_t> rec_dest_pin_;
  device_buffer<location_idx_t> rec_dest_;
  device_buffer<gpu_journey> rec_out_;
  pinned_host_buffer<gpu_journey> rec_host_out_;

  device_timetable tt_;

  thrust::device_vector<std::uint32_t> et_result_;
  thrust::device_vector<std::uint32_t> et_task_list_;
  thrust::device_vector<std::uint32_t> et_task_count_;
  thrust::device_vector<std::uint32_t> route_list_;
  thrust::device_vector<std::uint32_t> route_list_count_;

  // ping-bounds pruning: 16-bit bounds rows derived from the ping's
  // round_times (see gpu::fill_bounds); allocated lazily on first use so
  // plain (non-pong) searches never pay for it
  device_buffer<delta_t> bounds_dev_;
  thrust::device_vector<unsigned long long> n_pruned_;

  // component lower bounds (see component_lb.cuh)
  bool has_cg_{false};
  device_component_graph cg_;
  thrust::device_vector<std::uint32_t> lb_tt_cur_;
  thrust::device_vector<std::uint32_t> lb_tt_prev_;
  thrust::device_vector<std::uint32_t> lb_ic_;
  thrust::device_vector<std::uint32_t> comp_lb_;
  thrust::device_vector<std::uint32_t> lb_comp_marked_;
  thrust::device_vector<std::uint32_t> lb_route_marked_;
  thrust::device_vector<std::uint32_t> lb_flags_;

  // ad-hoc transfer pattern stop marks (restricted prelude)
  thrust::device_vector<std::uint64_t> adhoc_tp_dev_;
  pinned_host_buffer<std::uint64_t> adhoc_tp_pin_;
  thrust::device_vector<std::uint32_t> prelude_seeds_dev_;
  pinned_host_buffer<std::uint32_t> prelude_seeds_pin_;

  cudaStream_t stream_;
};

gpu_raptor_state::gpu_raptor_state(gpu_timetable const& gtt)
    : impl_{std::make_unique<impl>(gtt)} {}

gpu_raptor_state::~gpu_raptor_state() = default;

template <direction SearchDir>
gpu_raptor<SearchDir>::gpu_raptor(
    timetable const& tt,
    rt_timetable const* rtt,
    gpu_raptor_state& state,
    nigiri::bitvec& is_dest,
    std::array<nigiri::bitvec, kMaxVias> const& /* is_via (GPU: no vias) */,
    std::vector<std::uint16_t> const& dist_to_dest,
    hash_map<location_idx_t, std::vector<td_offset>> const& td_dist_to_dest,
    std::vector<std::uint16_t> const& lb,
    std::vector<via_stop> const& via_stops,
    day_idx_t const base,
    clasz_mask_t const allowed_claszes,
    bool const require_bike_transport,
    bool const require_car_transport,
    bool const is_wheelchair,
    transfer_time_settings const& tts)
    : tt_{tt},
      rtt_{rtt},
      gpu_rtt_{rtt == nullptr ? nullptr
                              : static_cast<gpu_rt_timetable const*>(
                                    rtt->gpu_rtt_.ptr_.get())},
      n_locations_{tt_.n_locations()},
      state_{state},
      is_dest_{is_dest},
      base_{base},
      allowed_claszes_{allowed_claszes},
      require_bike_transport_{require_bike_transport},
      require_car_transport_{require_car_transport},
      is_wheelchair_{is_wheelchair},
      transfer_time_settings_{tts} {
  utl::verify(rtt == nullptr || gpu_rtt_ != nullptr,
              "GPU raptor: rt search requires the uploaded device rt "
              "timetable (rt_timetable::gpu_rtt_)");
  state_.impl_->resize_rt(rtt == nullptr ? 0U : rtt->n_rt_transports());
  reset_arrivals();
  state_.impl_->upload_query(kDirIdx, is_dest, dist_to_dest, td_dist_to_dest,
                             lb);
}

template <direction SearchDir>
__global__ void init_arrivals_kernel(raptor_impl<SearchDir> r,
                                     unixtime_t const worst_time_at_dest) {
  r.init_arrivals(worst_time_at_dest);
}

template <direction SearchDir>
__global__ void reuse_previous_arrivals_kernel(raptor_impl<SearchDir> r,
                                               unsigned const k) {
  if (*r.done_) {
    return;
  }
  r.reuse_previous_arrivals(k);
}

template <direction SearchDir>
__global__ void mark_routes_kernel(raptor_impl<SearchDir> r, unsigned const k) {
  if (*r.done_) {
    return;
  }
  r.mark_routes(k);
}

template <direction SearchDir>
__global__ void mark_rt_transports_kernel(raptor_impl<SearchDir> r,
                                          unsigned const k) {
  if (*r.done_) {
    return;
  }
  r.mark_rt_transports(k);
}

template <direction SearchDir,
          bool WithClaszFilter,
          bool IsWheelchair,
          bool WithFilters>
__global__ void update_rt_transports_kernel(raptor_impl<SearchDir> r,
                                            unsigned const k) {
  if (*r.done_) {
    return;
  }
  r.template update_rt_transports<WithClaszFilter, IsWheelchair, WithFilters>(
      k);
}

template <direction SearchDir>
__global__ void begin_transit_phase_kernel(raptor_impl<SearchDir> r) {
  if (*r.done_) {
    return;
  }
  if (*r.any_marked_ == 0U) {  // no route marked -> search converged
    if (get_global_thread_id() == 0U) {
      *r.done_ = 1U;
    }
    return;
  }
  r.begin_transit_phase();
}

template <direction SearchDir>
__global__ void et_build_route_list_kernel(raptor_impl<SearchDir> r) {
  if (*r.done_) {
    return;
  }
  r.et_build_route_list();
}

template <direction SearchDir, bool IsWheelchair>
__global__ void et_collect_tasks_kernel(raptor_impl<SearchDir> r,
                                        unsigned const k) {
  if (*r.done_) {
    return;
  }
  r.template et_collect_tasks<IsWheelchair>(k);
}

template <direction SearchDir>
__global__ void et_run_lookups_kernel(raptor_impl<SearchDir> r,
                                      unsigned const k) {
  if (*r.done_) {
    return;
  }
  r.et_run_lookups(k);
}

template <direction SearchDir,
          bool WithClaszFilter,
          bool IsWheelchair,
          bool WithFilters>
__global__ void loop_routes_kernel(raptor_impl<SearchDir> r, unsigned const k) {
  if (*r.done_) {
    return;
  }
  r.template loop_routes<WithClaszFilter, IsWheelchair, WithFilters>(k);
}

template <direction SearchDir>
__global__ void begin_footpath_phase_kernel(raptor_impl<SearchDir> r) {
  if (*r.done_) {
    return;
  }
  if (*r.any_marked_ == 0U) {  // no location improved -> search converged
    if (get_global_thread_id() == 0U) {
      *r.done_ = 1U;
    }
    return;
  }
  r.begin_footpath_phase();
}

template <direction SearchDir, bool WithTdDest, bool WithTdFootpaths>
__global__ void transfers_footpaths_kernel(raptor_impl<SearchDir> r,
                                           unsigned const k) {
  if (*r.done_) {
    return;
  }
  r.template update_transfers_and_footpaths<WithTdDest, WithTdFootpaths>(k);
  r.route_mark_.reset();
  r.rt_transport_mark_.reset();
}

// === component lower bound kernels (see component_lb.cuh) ===
template <direction SearchDir>
__global__ void lb_init_kernel(component_lb_impl<SearchDir> r) {
  r.init();
}

template <direction SearchDir>
__global__ void lb_mark_routes_kernel(component_lb_impl<SearchDir> r) {
  if (*r.lb_done_) {
    return;
  }
  r.mark_routes();
}

template <direction SearchDir>
__global__ void lb_scan_kernel(component_lb_impl<SearchDir> r,
                               unsigned const k) {
  if (*r.lb_done_) {
    return;
  }
  r.scan(k);
}

template <direction SearchDir>
__global__ void lb_check_done_kernel(component_lb_impl<SearchDir> r) {
  if (*r.lb_done_) {
    return;
  }
  r.check_done();
}

template <direction SearchDir>
__global__ void lb_finalize_kernel(component_lb_impl<SearchDir> r,
                                   std::uint32_t const max_travel_time) {
  r.finalize(max_travel_time);
}

// Loosen the restricted prelude's time_at_dest_ seeds by one minute: the
// seeds are real journeys but the main search prunes with strict
// comparisons -- an optimal journey that exactly ties a seed (and the seed
// journey itself, which is not in this run's round_times_) must survive.
// Also counts the seeded rounds (entries better than the worst-time init)
// into *n_seeds for the prelude effectiveness stats.
template <direction SearchDir>
__global__ void bump_time_at_dest_kernel(raptor_impl<SearchDir> r,
                                         unixtime_t const worst_time_at_dest,
                                         std::uint32_t* const n_seeds) {
  auto const gid = get_global_thread_id();
  auto const stride = get_global_stride();
  auto const d_worst = unix_to_delta(r.base(), worst_time_at_dest);
  for (auto i = gid; i < kMaxTransfers + 2U; i += stride) {
    auto const t = r.time_at_dest_.get(static_cast<std::uint8_t>(i));
    if (t != kInvalidDelta<SearchDir>) {
      if (SearchDir == direction::kForward ? t < d_worst : t > d_worst) {
        atomicAdd(n_seeds, 1U);
      }
      r.time_at_dest_.data_[i] = device_times<SearchDir, 1U>::pack(
          clamp(t + (SearchDir == direction::kForward ? 1 : -1)), 0U);
    }
  }
}

template <typename Kernel>
std::pair<int, int> launch_dims(Kernel kernel) {
  static auto const dims = [&]() {
    auto blocks = 0;
    auto threads = 0;
    // half + quarter benchmarked with less throughput
    cudaOccupancyMaxPotentialBlockSize(&blocks, &threads, kernel, 0, 0);
    return std::pair{blocks, threads};
  }();
  return dims;
}

template <typename Kernel, typename... Args>
void launch(Kernel kernel, cudaStream_t stream, Args&&... args) {
  auto const [blocks, threads] = launch_dims(kernel);
  kernel<<<blocks, threads, 0, stream>>>(std::forward<Args>(args)...);
}

// === ping-bounds fill (see pong.cc) =======================================
// Strip the 16-bit time keys out of the ping's 64-bit time+breadcrumb round
// entries and make them monotonic over rounds (row k = best over rounds
// <= k). The per-direction key bias makes "smaller key" == "better time" for
// both directions, so the prefix-min runs on raw keys; decode happens once
// per cell on write.
template <direction PingDir>
__global__ void fill_bounds_kernel(std::uint64_t const* const round_times,
                                   delta_t* const bounds,
                                   std::uint32_t const n_locations,
                                   std::uint32_t const n_rows,
                                   std::uint64_t const* const td_stops) {
  auto const gid = get_global_thread_id();
  auto const stride = get_global_stride();
  for (auto l = gid; l < n_locations; l += stride) {
    if (td_stops != nullptr &&
        (td_stops[l >> 6U] & (std::uint64_t{1U} << (l & 63U))) != 0U) {
      // The ping used time-dependent footpaths at this stop, which
      // within_bounds' static footpath rescue cannot invert -> pass
      // everything (see the CPU fill_bounds in pong.cc).
      constexpr auto const kPassAll = PingDir == direction::kForward
                                          ? std::numeric_limits<delta_t>::min()
                                          : std::numeric_limits<delta_t>::max();
      for (auto k = 0U; k != n_rows; ++k) {
        bounds[k * n_locations + l] = kPassAll;
      }
      continue;
    }
    auto best_key = std::uint16_t{0xFFFFU};  // worst key = invalid
    for (auto k = 0U; k != n_rows; ++k) {
      auto const key =
          static_cast<std::uint16_t>(round_times[k * n_locations + l] >> 48U);
      best_key = key < best_key ? key : best_key;
      bounds[k * n_locations + l] =
          device_times<PingDir, 1U>::from_key(best_key);
    }
  }
}

template <direction SearchDir>
delta_t const* fill_bounds(gpu_raptor_state& state,
                           std::size_t const n_rows,
                           rt_timetable const* const rtt,
                           profile_idx_t const prf_idx) {
  auto& s = *state.impl_;
  auto const* td_stops = static_cast<std::uint64_t const*>(nullptr);
  if (rtt != nullptr && prf_idx != 0U && rtt->gpu_rtt_.ptr_ != nullptr) {
    auto const* const gpu_rtt =
        static_cast<gpu_rt_timetable const*>(rtt->gpu_rtt_.ptr_.get());
    auto const& blocks = SearchDir == direction::kForward
                             ? gpu_rtt->impl_->has_td_out_[prf_idx]
                             : gpu_rtt->impl_->has_td_in_[prf_idx];
    if (!blocks.empty()) {
      td_stops = thrust::raw_pointer_cast(blocks.data());
    }
  }
  auto* const bounds = s.bounds_dev_.ensure(
      static_cast<std::size_t>(s.tt_.n_locations_) * (kMaxTransfers + 2U),
      s.stream_);
  launch(fill_bounds_kernel<SearchDir>, s.stream_,
         thrust::raw_pointer_cast(s.round_times_.data()), bounds,
         s.tt_.n_locations_, static_cast<std::uint32_t>(n_rows), td_stops);
  return bounds;
}

template delta_t const* fill_bounds<direction::kForward>(gpu_raptor_state&,
                                                         std::size_t,
                                                         rt_timetable const*,
                                                         profile_idx_t);
template delta_t const* fill_bounds<direction::kBackward>(gpu_raptor_state&,
                                                          std::size_t,
                                                          rt_timetable const*,
                                                          profile_idx_t);

template <typename Fn>
void dispatch_filtered(bool const with_clasz,
                       bool const is_wheelchair,
                       bool const with_filters,
                       Fn&& fn) {
  switch ((static_cast<unsigned>(with_clasz) << 2) |
          (static_cast<unsigned>(is_wheelchair) << 1) |
          static_cast<unsigned>(with_filters)) {
    case 0b000: fn.template operator()<false, false, false>(); break;
    case 0b001: fn.template operator()<false, false, true>(); break;
    case 0b011: fn.template operator()<false, true, true>(); break;
    case 0b100: fn.template operator()<true, false, false>(); break;
    case 0b101: fn.template operator()<true, false, true>(); break;
    case 0b111: fn.template operator()<true, true, true>(); break;
    default: __builtin_unreachable();  // no 010/110 (wheelchair => filters)
  }
}

template <direction SearchDir>
void gpu_raptor<SearchDir>::compute_component_bounds(
    std::uint16_t const max_travel_time) {
  auto& s = *state_.impl_;
  if (!s.has_cg_ || (rtt_ != nullptr && rtt_->n_rt_transports() != 0U)) {
    return;  // rt trips may undercut the static minimum -> not admissible
  }

  auto ev0 = cudaEvent_t{};
  auto ev1 = cudaEvent_t{};
  cudaEventCreate(&ev0);
  cudaEventCreate(&ev1);
  cudaEventRecord(ev0, s.stream_);

  cudaMemsetAsync(thrust::raw_pointer_cast(s.lb_tt_cur_.data()), 0xFF,
                  s.lb_tt_cur_.size() * sizeof(std::uint32_t), s.stream_);
  cudaMemsetAsync(thrust::raw_pointer_cast(s.lb_ic_.data()), 0xFF,
                  s.lb_ic_.size() * sizeof(std::uint32_t), s.stream_);
  cudaMemsetAsync(thrust::raw_pointer_cast(s.lb_comp_marked_.data()), 0x00,
                  s.lb_comp_marked_.size() * sizeof(std::uint32_t), s.stream_);

  auto r = component_lb_impl<SearchDir>{
      .g_ = s.cg_,
      .n_locations_ = n_locations_,
      .is_dest_ = {to_view(s.is_dest_[kDirIdx])},
      .dist_to_end_ = to_view(s.dist_to_dest_dev_[kDirIdx]),
      .td_dest_locs_ = {s.td_dest_locs_dev_[kDirIdx].data(),
                        s.td_dest_locs_dev_[kDirIdx].size()},
      .tt_cur_ = to_mutable_view(s.lb_tt_cur_),
      .tt_prev_ = to_mutable_view(s.lb_tt_prev_),
      .ic_ = to_mutable_view(s.lb_ic_),
      .comp_marked_ = {to_mutable_view(s.lb_comp_marked_)},
      .route_marked_ = {to_mutable_view(s.lb_route_marked_)},
      .any_improved_ = thrust::raw_pointer_cast(s.lb_flags_.data()),
      .lb_done_ = thrust::raw_pointer_cast(s.lb_flags_.data()) + 1U,
      .comp_lb_ = to_mutable_view(s.comp_lb_)};

  launch(lb_init_kernel<SearchDir>, s.stream_, r);
  for (auto k = 1U; k != kMaxTransfers + 2U; ++k) {
    cudaMemsetAsync(thrust::raw_pointer_cast(s.lb_route_marked_.data()), 0x00,
                    s.lb_route_marked_.size() * sizeof(std::uint32_t),
                    s.stream_);
    launch(lb_mark_routes_kernel<SearchDir>, s.stream_, r);
    cudaMemsetAsync(thrust::raw_pointer_cast(s.lb_comp_marked_.data()), 0x00,
                    s.lb_comp_marked_.size() * sizeof(std::uint32_t),
                    s.stream_);
    cudaMemcpyAsync(thrust::raw_pointer_cast(s.lb_tt_prev_.data()),
                    thrust::raw_pointer_cast(s.lb_tt_cur_.data()),
                    s.lb_tt_cur_.size() * sizeof(std::uint32_t),
                    cudaMemcpyDeviceToDevice, s.stream_);
    launch(lb_scan_kernel<SearchDir>, s.stream_, r, k);
    launch(lb_check_done_kernel<SearchDir>, s.stream_, r);
  }
  launch(lb_finalize_kernel<SearchDir>, s.stream_, r,
         static_cast<std::uint32_t>(max_travel_time));
  cudaEventRecord(ev1, s.stream_);
  cudaEventSynchronize(ev1);
  auto lb_ms = 0.F;
  cudaEventElapsedTime(&lb_ms, ev0, ev1);
  stats_.gpu_lb_compute_us_ += static_cast<std::uint64_t>(lb_ms * 1000.F);
  cudaEventDestroy(ev0);
  cudaEventDestroy(ev1);
  CUDA_CHECK(cudaPeekAtLastError());
  use_lb_ = true;
}

template <direction SearchDir>
void gpu_raptor<SearchDir>::add_adhoc_stop(location_idx_t const l) {
  static auto const enabled = env_enabled("NIGIRI_GPU_ADHOC_TP");
  if (!enabled) {
    return;
  }
  if (adhoc_tp_.size() != n_locations_) {
    adhoc_tp_.resize(n_locations_);
  }
  if (!adhoc_tp_.test(to_idx(l))) {
    adhoc_tp_.set(to_idx(l), true);
    adhoc_dirty_ = true;
    adhoc_any_ = true;
  }
}

template <direction SearchDir>
__global__ void reconstruct_kernel(location_idx_t const* const dest_list,
                                   std::uint32_t const n_dest,
                                   std::uint32_t const end_k,
                                   raptor_impl<SearchDir> r,
                                   gpu_journey* const out) {
  auto const tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_dest * end_k) {
    return;
  }
  out[tid].state_ = reconstruction_result::kNotReconstructed;
  auto const k = tid % end_k;
  if (k == 0U) {
    return;
  }
  r.reconstruct_journey(dest_list[tid / end_k], k, &out[tid]);
}

template <direction SearchDir>
void gpu_raptor<SearchDir>::execute(unixtime_t start_time,
                                    std::uint8_t max_transfers,
                                    unixtime_t worst_time_at_dest,
                                    profile_idx_t prf_idx,
                                    pareto_set<journey>& results) {
  auto& s = *state_.impl_;

  // Copy starts.
  auto* const starts_pinned = s.starts_.ensure(starts_.size());
  std::memset(starts_pinned, 0,  // otherwise initcheck flags
              starts_.size() * sizeof(std::pair<location_idx_t, unixtime_t>));
  std::copy(starts_.begin(), starts_.end(), starts_pinned);
  if (s.starts_dev_.size() < starts_.size()) {
    s.starts_dev_.resize(starts_.size());
  }
  cudaMemcpyAsync(
      thrust::raw_pointer_cast(s.starts_dev_.data()), starts_pinned,
      starts_.size() * sizeof(std::pair<location_idx_t, unixtime_t>),
      cudaMemcpyHostToDevice, s.stream_);
  auto const starts =
      cuda::std::span<std::pair<location_idx_t, unixtime_t> const>{
          thrust::raw_pointer_cast(s.starts_dev_.data()), starts_.size()};

  cudaStreamSynchronize(s.stream_);
  CUDA_CHECK(cudaPeekAtLastError());

  auto const rt_active = gpu_rtt_ != nullptr;
  auto const with_td_dest = s.td_dest_locs_dev_[kDirIdx].size() > 0U;
  auto const with_td_fps =
      rt_active && prf_idx != 0U && gpu_rtt_->impl_->has_td_fps_[prf_idx];
  auto r = raptor_impl<SearchDir>{
      .any_marked_ = thrust::raw_pointer_cast(s.any_marked_.data()),
      .done_ = thrust::raw_pointer_cast(s.done_.data()),
      .tt_ = s.tt_,
      .rtt_ = rt_active ? gpu_rtt_->impl_->to_device_rt_timetable()
                        : device_rt_timetable{},
      .transfer_time_settings_ = transfer_time_settings_,
      .max_transfers_ = max_transfers,
      .allowed_claszes_ = allowed_claszes_,
      .prf_idx_ = prf_idx,
      .require_bike_transport_ = require_bike_transport_,
      .require_car_transport_ = require_car_transport_,
      .base_ = base_,
      .starts_ = starts,
      .is_dest_ = {to_view(s.is_dest_[kDirIdx])},
      .dist_to_end_ = to_view(s.dist_to_dest_dev_[kDirIdx]),
      .td_dest_locs_ = {s.td_dest_locs_dev_[kDirIdx].data(),
                        s.td_dest_locs_dev_[kDirIdx].size()},
      .td_dest_ = {.data_ = {s.td_dest_data_dev_[kDirIdx].data(),
                             s.td_dest_data_dev_[kDirIdx].size()},
                   .bucket_starts_ = {s.td_dest_ranges_dev_[kDirIdx].data(),
                                      s.td_dest_ranges_dev_[kDirIdx].size()}},
      .round_times_ = {to_mutable_view(s.round_times_), n_locations_},
      .best_ = {to_mutable_view(s.best_), n_locations_},
      .tmp_ = {to_mutable_view(s.tmp_), n_locations_},
      .time_at_dest_ = {to_mutable_view(s.time_at_dest_), n_locations_},
      .station_mark_ = {to_mutable_view(s.station_mark_)},
      .prev_station_mark_ = {to_mutable_view(s.prev_station_mark_)},
      .route_mark_ = {to_mutable_view(s.route_mark_)},
      .rt_transport_mark_ = {to_mutable_view(s.rt_transport_mark_)},
      .et_result_ = to_mutable_view(s.et_result_),
      .et_task_list_ = to_mutable_view(s.et_task_list_),
      .et_task_count_ = thrust::raw_pointer_cast(s.et_task_count_.data()),
      .route_list_ = to_mutable_view(s.route_list_),
      .route_list_count_ =
          thrust::raw_pointer_cast(s.route_list_count_.data())};

  if (rt_active) {
    r.tt_.transport_traffic_days_ = r.rtt_.transport_traffic_days_;
    r.tt_.bitfields_ = r.rtt_.bitfields_;
  }

  r.loose_pruning_ = loose_pruning_;
  if (bounds_ != nullptr) {
    r.bounds_ = bounds_;
    r.bounds_last_k_ = bounds_last_k_;
    r.bounds_prf_idx_ = bounds_prf_idx_;
    r.n_pruned_ = thrust::raw_pointer_cast(s.n_pruned_.data());
    cudaMemsetAsync(r.n_pruned_, 0, sizeof(unsigned long long), s.stream_);
  }

  if (use_lb_) {
    r.location_component_ = s.cg_.location_component_;
    r.comp_lb_ = to_view(s.comp_lb_);
  }
  r.loc_lb_ = {s.loc_lb_dev_[kDirIdx].data(), s.loc_lb_dev_[kDirIdx].size()};

  auto const end_k =
      static_cast<std::uint32_t>(std::min(max_transfers, kMaxTransfers) + 2U);

  // === ROUTING KERNELS ===
  auto const run_rounds = [&](raptor_impl<SearchDir>& ri) {
    launch(init_arrivals_kernel<SearchDir>, s.stream_, ri, worst_time_at_dest);
    for (auto k = 1U; k != end_k; ++k) {
      launch(reuse_previous_arrivals_kernel<SearchDir>, s.stream_, ri, k);
      launch(mark_routes_kernel<SearchDir>, s.stream_, ri, k);
      if (rt_active) {
        launch(mark_rt_transports_kernel<SearchDir>, s.stream_, ri, k);
      }
      launch(begin_transit_phase_kernel<SearchDir>, s.stream_, ri);
      launch(et_build_route_list_kernel<SearchDir>, s.stream_, ri);
      if (is_wheelchair_) {
        launch(et_collect_tasks_kernel<SearchDir, true>, s.stream_, ri, k);
      } else {
        launch(et_collect_tasks_kernel<SearchDir, false>, s.stream_, ri, k);
      }
      launch(et_run_lookups_kernel<SearchDir>, s.stream_, ri, k);
      {
        auto const with_clasz = allowed_claszes_ != all_clasz_allowed();
        auto const with_filters =
            is_wheelchair_ || require_bike_transport_ || require_car_transport_;
        dispatch_filtered(
            with_clasz, is_wheelchair_, with_filters,
            [&]<bool WithClasz, bool IsWheelchair, bool WithFilters>() {
              launch(loop_routes_kernel<SearchDir, WithClasz, IsWheelchair,
                                        WithFilters>,
                     s.stream_, ri, k);
            });
        if (rt_active) {
          dispatch_filtered(
              with_clasz, is_wheelchair_, with_filters,
              [&]<bool WithClasz, bool IsWheelchair, bool WithFilters>() {
                launch(update_rt_transports_kernel<SearchDir, WithClasz,
                                                   IsWheelchair, WithFilters>,
                       s.stream_, ri, k);
              });
        }
      }
      launch(begin_footpath_phase_kernel<SearchDir>, s.stream_, ri);
      if (!with_td_dest && !with_td_fps) {
        launch(transfers_footpaths_kernel<SearchDir, false, false>, s.stream_,
               ri, k);
      } else if (with_td_dest && !with_td_fps) {
        launch(transfers_footpaths_kernel<SearchDir, true, false>, s.stream_,
               ri, k);
      } else if (!with_td_dest && with_td_fps) {
        launch(transfers_footpaths_kernel<SearchDir, false, true>, s.stream_,
               ri, k);
      } else {
        launch(transfers_footpaths_kernel<SearchDir, true, true>, s.stream_,
               ri, k);
      }
    }
  };

  auto ev_a = cudaEvent_t{};
  auto ev_b = cudaEvent_t{};
  auto ev_c = cudaEvent_t{};
  cudaEventCreate(&ev_a);
  cudaEventCreate(&ev_b);
  cudaEventCreate(&ev_c);
  cudaEventRecord(ev_a, s.stream_);

  // Ad-hoc transfer pattern prelude: run the enter/exit-restricted search
  // first -- its (real, achievable) destination arrivals seed time_at_dest_
  // so the main run prunes from round 1. Everything except time_at_dest_ is
  // wiped in between (the +1 bump keeps ties findable, see bump kernel).
  if (adhoc_any_) {
    if (adhoc_dirty_) {
      auto* const pin = s.adhoc_tp_pin_.ensure(adhoc_tp_.blocks_.size());
      std::copy(adhoc_tp_.blocks_.begin(), adhoc_tp_.blocks_.end(), pin);
      CUDA_CHECK(cudaMemcpyAsync(
          thrust::raw_pointer_cast(s.adhoc_tp_dev_.data()), pin,
          adhoc_tp_.blocks_.size() * sizeof(std::uint64_t),
          cudaMemcpyHostToDevice, s.stream_));
      adhoc_dirty_ = false;
    }
    auto pre = r;
    pre.restricted_ = true;
    pre.adhoc_tp_ = {to_view(s.adhoc_tp_dev_)};
    cudaMemsetAsync(thrust::raw_pointer_cast(s.prelude_seeds_dev_.data()), 0,
                    sizeof(std::uint32_t), s.stream_);
    run_rounds(pre);
    launch(bump_time_at_dest_kernel<SearchDir>, s.stream_, pre,
           worst_time_at_dest,
           thrust::raw_pointer_cast(s.prelude_seeds_dev_.data()));
    CUDA_CHECK(cudaMemcpyAsync(
        s.prelude_seeds_pin_.ensure(1U),
        thrust::raw_pointer_cast(s.prelude_seeds_dev_.data()),
        sizeof(std::uint32_t), cudaMemcpyDeviceToHost, s.stream_));
    cudaMemsetAsync(thrust::raw_pointer_cast(s.round_times_.data()), 0xFF,
                    s.round_times_.size() * sizeof(std::uint64_t), s.stream_);
    cudaMemsetAsync(thrust::raw_pointer_cast(s.best_.data()), 0xFF,
                    s.best_.size() * sizeof(std::uint64_t), s.stream_);
    cudaMemsetAsync(thrust::raw_pointer_cast(s.tmp_.data()), 0xFF,
                    s.tmp_.size() * sizeof(std::uint64_t), s.stream_);
    thrust::fill(thrust::cuda::par.on(s.stream_), begin(s.station_mark_),
                 end(s.station_mark_), 0U);
    thrust::fill(thrust::cuda::par.on(s.stream_), begin(s.prev_station_mark_),
                 end(s.prev_station_mark_), 0U);
    thrust::fill(thrust::cuda::par.on(s.stream_), begin(s.route_mark_),
                 end(s.route_mark_), 0U);
    thrust::fill(thrust::cuda::par.on(s.stream_), begin(s.rt_transport_mark_),
                 end(s.rt_transport_mark_), 0U);
  }

  cudaEventRecord(ev_b, s.stream_);
  run_rounds(r);
  cudaEventRecord(ev_c, s.stream_);
  cudaStreamSynchronize(s.stream_);
  if (adhoc_any_) {
    auto const n_seeds = *s.prelude_seeds_pin_.ensure(1U);
    ++stats_.n_prelude_runs_;
    stats_.n_prelude_with_connection_ += n_seeds != 0U ? 1U : 0U;
    stats_.n_prelude_seed_entries_ += n_seeds;
  }
  auto prelude_ms = 0.F;
  auto main_ms = 0.F;
  cudaEventElapsedTime(&prelude_ms, ev_a, ev_b);
  cudaEventElapsedTime(&main_ms, ev_b, ev_c);
  stats_.gpu_prelude_us_ += static_cast<std::uint64_t>(prelude_ms * 1000.F);
  stats_.gpu_main_us_ += static_cast<std::uint64_t>(main_ms * 1000.F);
  cudaEventDestroy(ev_a);
  cudaEventDestroy(ev_b);
  cudaEventDestroy(ev_c);
  CUDA_CHECK(cudaPeekAtLastError());

  if (bounds_ != nullptr) {
    auto pruned = static_cast<unsigned long long>(0U);
    CUDA_CHECK(cudaMemcpy(&pruned,
                          thrust::raw_pointer_cast(s.n_pruned_.data()),
                          sizeof(pruned), cudaMemcpyDeviceToHost));
    stats_.n_pruned_by_ping_bounds_ += static_cast<std::uint64_t>(pruned);
  }

  // === DEVICE RECONSTRUCT ===
  auto dest_list = std::vector<location_idx_t>{};
  if (s.is_intermodal_dest_[kDirIdx]) {
    dest_list.push_back(get_special_station(special_station::kEnd));
  } else {
    is_dest_.for_each_set_bit(
        [&](auto const i) { dest_list.push_back(location_idx_t{i}); });
  }
  if (dest_list.empty()) {
    return;
  }

  auto const n_dest = static_cast<std::uint32_t>(dest_list.size());
  auto const total = n_dest * end_k;

  auto* const dest_pin = s.rec_dest_pin_.ensure(dest_list.size());
  std::copy(dest_list.begin(), dest_list.end(), dest_pin);
  auto* const dest_dev = s.rec_dest_.ensure(dest_list.size(), s.stream_);
  CUDA_CHECK(cudaMemcpyAsync(dest_dev, dest_pin,
                             dest_list.size() * sizeof(location_idx_t),
                             cudaMemcpyHostToDevice, s.stream_));
  auto* const rec_out_dev = s.rec_out_.ensure(total, s.stream_);
  auto* const rec_host = s.rec_host_out_.ensure(total);

  {
    auto const threads = 128U;
    auto const blocks = (total + threads - 1U) / threads;
    reconstruct_kernel<SearchDir><<<blocks, threads, 0, s.stream_>>>(
        dest_dev, n_dest, end_k, r, rec_out_dev);
    CUDA_CHECK(cudaMemcpyAsync(rec_host, rec_out_dev,
                               total * sizeof(gpu_journey),
                               cudaMemcpyDeviceToHost, s.stream_));
    cudaStreamSynchronize(s.stream_);
  }
  CUDA_CHECK(cudaPeekAtLastError());

  // === CONVERT DEVICE JOURNEYS TO HOST JOURNEYS ===
  for (auto idx = std::uint32_t{0U}; idx != total; ++idx) {
    auto const& gj = rec_host[idx];

    if (gj.state_ == reconstruction_result::kReconstructionFailed) {
      // should not happen
      log(log_lvl::error, "search",
          "reconstruct failed: gpu breadcrumb chain unreconstructable "
          "(dest={}, k={})",
          to_idx(gj.dest_l_), idx % end_k);
      continue;
    }

    if (gj.state_ != reconstruction_result::kOk || gj.n_legs_ == 0U) {
      continue;
    }

    // Construct host journey from device journey.
    // This also reverses legs for forward search.
    // -> Legs are in chronological order from here on.
    auto j = journey{};
    j.start_time_ = start_time;
    j.dest_time_ = delta_to_unix(base(), gj.dest_time_);
    j.dest_ = gj.dest_l_;
    j.transfers_ = gj.transfers_;

    for (auto li2 = 0U; li2 != gj.n_legs_; ++li2) {
      auto const li = (SearchDir == direction::kForward)
                          ? (static_cast<unsigned>(gj.n_legs_) - 1U - li2)
                          : li2;
      auto const& gl = gj.legs_[li];
      auto const from = gl.from_l_;
      auto const to = gl.to_l_;
      auto const dep = delta_to_unix(base(), gl.dep_);
      auto const arr = delta_to_unix(base(), gl.arr_);

      if (gl.is_footpath_) {
        j.legs_.emplace_back(journey::leg{
            SearchDir, from, to, dep, arr,
            footpath{to, duration_t{
                             static_cast<duration_t::rep>(gl.fp_duration_)}}});
      } else if (gl.rt_transport_ != rt_transport_idx_t::invalid()) {
        auto const rt_t = gl.rt_transport_;
        auto const run =
            rt::run{.t_ = rtt_->resolve_static(rt_t),
                    .stop_range_ =
                        interval<stop_idx_t>{
                            stop_idx_t{0},
                            static_cast<stop_idx_t>(
                                rtt_->rt_transport_location_seq_[rt_t].size())},
                    .rt_ = rt_t};
        j.legs_.emplace_back(journey::leg{
            SearchDir, from, to, dep, arr,
            journey::run_enter_exit{run, gl.enter_stop_, gl.exit_stop_}});
      } else {
        auto const t_idx = gl.transport_;
        auto const route = tt_.transport_route_[t_idx];
        auto const route_len =
            static_cast<stop_idx_t>(tt_.route_location_seq_[route].size());
        auto const run = rt::run{
            .t_ = transport{t_idx, gl.day_},
            .stop_range_ = interval<stop_idx_t>{stop_idx_t{0}, route_len}};
        j.legs_.emplace_back(journey::leg{
            SearchDir, from, to, dep, arr,
            journey::run_enter_exit{run, gl.enter_stop_, gl.exit_stop_}});
      }
    }

    // Backward search requires to re-anchor footpath durations
    // to the arrival of the previous trip
    // instead of the departure of the next trip.
    // No-op for forward search.
    for (auto i = std::size_t{1U}; i < j.legs_.size(); ++i) {
      if (std::holds_alternative<footpath>(j.legs_[i].uses_)) {
        auto const dur = std::get<footpath>(j.legs_[i].uses_).duration();
        j.legs_[i].dep_time_ = j.legs_[i - 1U].arr_time_;
        j.legs_[i].arr_time_ = j.legs_[i].dep_time_ + dur;
      }
    }

    // Shorten td footpaths to their actual duration (excluding waiting).
    if (rtt_ != nullptr && prf_idx != 0U &&
        !(SearchDir == direction::kForward ? rtt_->td_footpaths_in_[prf_idx]
                                           : rtt_->td_footpaths_out_[prf_idx])
             .empty()) {
      constexpr auto const kIsFwd = SearchDir == direction::kForward;
      auto const& has_td = kIsFwd ? rtt_->has_td_footpaths_in_[prf_idx]
                                  : rtt_->has_td_footpaths_out_[prf_idx];
      auto const& td_fps = kIsFwd ? rtt_->td_footpaths_in_[prf_idx]
                                  : rtt_->td_footpaths_out_[prf_idx];
      for (auto& lg : j.legs_) {
        if (!std::holds_alternative<footpath>(lg.uses_)) {
          continue;
        }
        auto const key_l = kIsFwd ? lg.to_ : lg.from_;
        auto const target_l = kIsFwd ? lg.from_ : lg.to_;
        if (!has_td.test(key_l)) {
          continue;
        }
        auto const t = lg.arr_time_;
        for_each_footpath<SearchDir>(td_fps[key_l], t, [&](footpath const fp) {
          if (fp.target() != target_l) {
            return utl::cflow::kContinue;
          }
          lg.dep_time_ = t - fp.duration();
          lg.arr_time_ = t;
          lg.uses_ = footpath{lg.to_, fp.duration()};
          return utl::cflow::kBreak;
        });
      }
    }

    // is_reconstructed_ stays false
    // -> only reconstruct() called by pong/search.h might set it to true
    results.add(std::move(j));
  }
}

template <direction SearchDir>
void gpu_raptor<SearchDir>::reset_arrivals() {
  auto& s = *state_.impl_;
  cudaMemsetAsync(thrust::raw_pointer_cast(s.time_at_dest_.data()), 0xFF,
                  s.time_at_dest_.size() * sizeof(std::uint64_t), s.stream_);
  cudaMemsetAsync(thrust::raw_pointer_cast(s.round_times_.data()), 0xFF,
                  s.round_times_.size() * sizeof(std::uint64_t), s.stream_);
}

template <direction SearchDir>
void gpu_raptor<SearchDir>::next_start_time() {
  starts_.clear();
  auto& s = *state_.impl_;
  cudaMemsetAsync(thrust::raw_pointer_cast(s.best_.data()), 0xFF,
                  s.best_.size() * sizeof(std::uint64_t), s.stream_);
  cudaMemsetAsync(thrust::raw_pointer_cast(s.tmp_.data()), 0xFF,
                  s.tmp_.size() * sizeof(std::uint64_t), s.stream_);
  thrust::fill(thrust::cuda::par.on(state_.impl_->stream_),
               begin(state_.impl_->prev_station_mark_),
               end(state_.impl_->prev_station_mark_), 0U);
  thrust::fill(thrust::cuda::par.on(state_.impl_->stream_),
               begin(state_.impl_->station_mark_),
               end(state_.impl_->station_mark_), 0U);
  thrust::fill(thrust::cuda::par.on(state_.impl_->stream_),
               begin(state_.impl_->route_mark_), end(state_.impl_->route_mark_),
               0U);
  thrust::fill(thrust::cuda::par.on(state_.impl_->stream_),
               begin(state_.impl_->rt_transport_mark_),
               end(state_.impl_->rt_transport_mark_), 0U);
}

// First/last mile mumo offset and start footpath legs are added here
// on the host, where the query offsets live.
template <direction SearchDir>
void gpu_raptor<SearchDir>::reconstruct(query const& q, journey& j) {
  // The core legs (boarding station -> alighting station) reconstructed by the
  // GPU kernel (breadcrumb pointer chase).
  utl::verify(!j.legs_.empty(), "gpu reconstruct: journey without core legs");

  // The legs are chronological (materialization reverses forward device order).
  constexpr auto const is_fwd = SearchDir == direction::kForward;

  // Front-side mumo leg: special_station -> first transit stop.
  auto const from = j.legs_.front().from_;
  auto const dep_time = j.legs_.front().dep_time_;
  auto const front_match_mode =
      is_fwd ? q.start_match_mode_ : q.dest_match_mode_;
  if (front_match_mode == location_match_mode::kIntermodal) {
    auto const& offsets = is_fwd ? q.start_ : q.destination_;
    auto const& td_offsets = is_fwd ? q.td_start_ : q.td_dest_;
    auto const special = get_special_station(is_fwd ? special_station::kStart
                                                    : special_station::kEnd);
    auto const o = utl::find_if(offsets, [&](offset const& x) {
      return matches(tt_, front_match_mode, x.target(), from) &&
             (is_fwd
                  // fwd: query start, check feasibility (allows ontrip start)
                  ? dep_time - x.duration() >= j.start_time_
                  // bwd: destination, anchored exactly at j.dest_time_
                  : dep_time - x.duration() == j.dest_time_);
    });
    if (o != end(offsets)) {
      auto const dep = dep_time - o->duration();
      j.legs_.insert(begin(j.legs_), journey::leg{direction::kForward, special,
                                                  from, dep, dep_time, *o});
    } else {
      // td offset (e.g. flex), mirrors the CPU reconstruct:
      // - fwd = query start side (find_start_footpath): evaluated backward
      //   from the first transit departure, feasible if not before the
      //   journey start
      // - bwd = td egress side (get_legs intermodal dest): evaluated forward
      //   from the final label j.dest_time_
      // specify_td_offsets() refines both to the raw entry duration below.
      auto inserted = false;
      for (auto const& [target, tds] : td_offsets) {
        if (!matches(tt_, front_match_mode, target, from)) {
          continue;
        }
        if (is_fwd) {
          auto const fp = get_td_duration<direction::kBackward>(tds, dep_time);
          if (!fp.has_value() || dep_time - fp->first < j.start_time_) {
            continue;
          }
          j.legs_.insert(begin(j.legs_),
                         journey::leg{direction::kForward, special, from,
                                      dep_time - fp->first, dep_time,
                                      offset{target, fp->first,
                                             fp->second.transport_mode_id_}});
        } else {
          auto const t = j.dest_time_;
          auto const fp = get_td_duration<direction::kForward>(tds, t);
          if (!fp.has_value() || t + fp->first > dep_time) {
            continue;  // must reach the stop before the transit departure
          }
          j.legs_.insert(
              begin(j.legs_),
              journey::leg{
                  direction::kForward, special, from, t, t + fp->first,
                  offset{target, fp->first, fp->second.transport_mode_id_}});
        }
        inserted = true;
        break;
      }
      utl::verify(inserted, "gpu reconstruct: no front mumo offset");
    }
  }

  // offset: last transit stop -> special_station.
  auto const to = j.legs_.back().to_;
  auto const arr_time = j.legs_.back().arr_time_;
  auto const back_match_mode =
      is_fwd ? q.dest_match_mode_ : q.start_match_mode_;
  if (back_match_mode == location_match_mode::kIntermodal) {
    auto const& offsets = is_fwd ? q.destination_ : q.start_;
    auto const& td_offsets = is_fwd ? q.td_dest_ : q.td_start_;
    auto const special = get_special_station(is_fwd ? special_station::kEnd
                                                    : special_station::kStart);
    auto const o = utl::find_if(offsets, [&](offset const& x) {
      return matches(tt_, back_match_mode, x.target(), to) &&
             (is_fwd
                  // fwd: destination, anchored exactly at j.dest_time_
                  ? arr_time + x.duration() == j.dest_time_
                  // bwd: query start, anchored by feasibility
                  : arr_time + x.duration() <= j.start_time_);
    });
    if (o != end(offsets)) {
      auto const arr = arr_time + o->duration();
      j.legs_.push_back(
          journey::leg{direction::kForward, to, special, arr_time, arr, *o});
      j.dest_ = special;
    } else {
      // dest td offset
      auto inserted = false;
      for (auto const& [target, tds] : td_offsets) {
        if (!matches(tt_, back_match_mode, target, to)) {
          continue;
        }

        if (is_fwd) {
          auto const fp =
              get_td_duration<direction::kBackward>(tds, j.dest_time_);
          if (!fp.has_value() ||
              j.dest_time_ - fp->first /* duration*/ < arr_time) {
            continue;
          }

          j.legs_.push_back(journey::leg{
              direction::kForward, to, special, j.dest_time_ - fp->first,
              j.dest_time_,
              offset{target, fp->first, fp->second.transport_mode_id_}});
        } else {
          auto const fp = get_td_duration<direction::kForward>(tds, arr_time);
          if (!fp.has_value() || arr_time + fp->first > j.start_time_) {
            continue;
          }

          j.legs_.push_back(journey::leg{
              direction::kForward, to, special, arr_time, arr_time + fp->first,
              offset{target, fp->first, fp->second.transport_mode_id_}});
        }

        j.dest_ = special;
        inserted = true;

        break;
      }
      utl::verify(inserted, "gpu reconstruct: no back mumo offset");
    }
  }

  // use_start_footpaths_ == true:
  // reconstruct the start footpath that seeded round k=0 at the first stop.
  if (q.start_match_mode_ != location_match_mode::kIntermodal) {
    auto const is_journey_start = [&](location_idx_t const l) {
      for (auto const& o : q.start_) {
        if (matches(tt_, q.start_match_mode_, o.target(), l)) {
          return true;
        }
      }
      return false;
    };
    auto const start_l = is_fwd ? j.legs_.front().from_ : j.legs_.back().to_;
    auto const start_t =
        is_fwd ? j.legs_.front().dep_time_ : j.legs_.back().arr_time_;
    auto const direct_start_ok =
        is_fwd ? j.start_time_ <= start_t : j.start_time_ >= start_t;
    if (!is_journey_start(start_l) || !direct_start_ok) {
      auto const fps = is_fwd
                           ? tt_.locations_.footpaths_in_[q.prf_idx_][start_l]
                           : tt_.locations_.footpaths_out_[q.prf_idx_][start_l];
      auto best = std::optional<footpath>{};
      for (auto const fp : fps) {
        if ((!best.has_value() || fp.duration() < best->duration()) &&
            is_journey_start(fp.target())) {
          best = fp;
        }
      }
      if (best.has_value()) {
        auto const dur = duration_t{adjusted_transfer_time(
            q.transfer_time_settings_, best->duration().count())};
        auto const fp_arr = j.start_time_ + (is_fwd ? dur : -dur);
        if (is_fwd ? fp_arr <= start_t : fp_arr >= start_t) {
          auto const lg = journey::leg{
              SearchDir,     best->target(), start_l,
              j.start_time_, fp_arr,         footpath{best->target(), dur}};
          if (is_fwd) {
            j.legs_.insert(begin(j.legs_), lg);
          } else {
            j.legs_.push_back(lg);
          }
        }
      }
    }
  }

  if constexpr (is_fwd) {
    optimize_footpaths(tt_, rtt_, q, j);
    specify_td_offsets(q, j);
  } else {
    // The journey's legs are chronological, but q is in search direction.
    auto journey_q = q;
    journey_q.flip_dir();
    optimize_footpaths(tt_, rtt_, journey_q, j);
    specify_td_offsets(journey_q, j);
  }

  j.is_reconstructed_ = true;
}

template <direction SearchDir>
void gpu_raptor<SearchDir>::add_start(location_idx_t const l,
                                      unixtime_t const t) {
  starts_.emplace_back(l, t);
}

template class gpu_raptor<direction::kForward>;
template class gpu_raptor<direction::kBackward>;

}  // namespace nigiri::routing::gpu