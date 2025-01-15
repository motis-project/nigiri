#include "nigiri/routing/gpu/raptor.h"

#include <cstdio>
#include <iostream>

#include "cuda/std/array"
#include "cuda/std/span"

#include "cooperative_groups.h"

#include "thrust/device_vector.h"
#include "thrust/fill.h"

#include "nigiri/common/delta_t.h"
#include "nigiri/common/flat_matrix_view.h"
#include "nigiri/routing/gpu/bitvec.cuh"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

namespace cg = cooperative_groups;

namespace nigiri::routing::gpu {

static constexpr auto const kUnreachable =
    std::numeric_limits<std::uint16_t>::max();

template <typename T>
using device_vec = thrust::device_vector<T>;

using device_bitvec = cista::basic_bitvec<thrust::device_vector<std::uint64_t>>;
using device_bitvec_view =
    cista::basic_bitvec<cuda::std::span<std::uint64_t const>>;
using device_mutable_bitvec_view =
    cista::basic_bitvec<cuda::std::span<std::uint64_t>>;

template <typename T>
using device_flat_matrix_view = base_flat_matrix_view<cuda::std::span<T>>;

template <typename T>
cuda::std::span<T const> to_view(thrust::device_vector<T> const& v) {
  return {thrust::raw_pointer_cast(v.data()), v.size()};
}

template <typename T>
cuda::std::span<T> to_mutable_view(thrust::device_vector<T>& v) {
  return {thrust::raw_pointer_cast(v.data()), v.size()};
}

template <typename T>
thrust::device_vector<typename T::value_type> to_device(T const& t) {
  return thrust::device_vector<typename T::value_type>(begin(t), end(t));
}

device_bitvec to_device(nigiri::bitvec const& t) {
  return {to_device(t.blocks_), t.size()};
}

device_bitvec_view to_view(device_bitvec const& x) {
  return {to_view(x.blocks_), x.size_};
}

device_mutable_bitvec_view to_mutable_view(device_bitvec& x) {
  return {to_mutable_view(x.blocks_), x.size_};
}

template <typename T, std::size_t N>
auto to_view(std::array<T, N> const& a) {
  auto ret = cuda::std::array<decltype(to_view(a[0U])), N>{};
  for (auto i = 0U; i != N; ++i) {
    ret[i] = to_view(a[i]);
  }
  return ret;
}

template <typename Host>
struct device_vecvec {
  using H = std::decay_t<Host>;
  using data_value_type = typename H::data_value_type;
  using index_value_type = typename H::index_value_type;
  explicit device_vecvec(H const& h)
      : data_{to_device(h.data_)}, index_{to_device(h.bucket_starts_)} {}
  thrust::device_vector<data_value_type> data_;
  thrust::device_vector<index_value_type> index_;
};

template <typename Host>
using d_vecvec_view = cista::basic_vecvec<
    typename std::decay_t<Host>::key,
    cuda::std::span<typename std::decay_t<Host>::data_value_type const>,
    cuda::std::span<typename std::decay_t<Host>::index_value_type const>>;

template <typename Host>
d_vecvec_view<Host> to_view(device_vecvec<Host> const& h) {
  return {.data_ = to_view(h.data_), .bucket_starts_ = to_view(h.index_)};
}

template <typename K, typename V>
struct d_vecmap_view {
  d_vecmap_view(cuda::std::span<V const> data) : data_{data} {}
  d_vecmap_view(thrust::device_vector<V> const& data) : data_{to_view(data)} {}

  __forceinline__ __device__ V const& operator[](K const k) const {
    return data_[to_idx(k)];
  }

  cuda::std::span<V const> data_;
};

struct device_timetable {
  using t = timetable;

  cuda::std::span<delta const> event_times_at_stop(
      route_idx_t const r,
      stop_idx_t const stop_idx,
      event_type const ev_type) const {
    auto const n_transports =
        static_cast<unsigned>(route_transport_ranges_[r].size());
    auto const idx = static_cast<unsigned>(
        route_stop_time_ranges_[r].from_ +
        n_transports * (stop_idx * 2 - (ev_type == event_type::kArr ? 1 : 0)));
    return {&route_stop_times_[idx], n_transports};
  }

  std::uint32_t n_locations_;
  std::uint32_t n_routes_;

  d_vecmap_view<location_idx_t, u8_minutes> transfer_time_;
  d_vecvec_view<decltype(t{}.locations_.footpaths_out_[0])> footpaths_out_;
  d_vecvec_view<decltype(t{}.locations_.footpaths_in_[0])> footpaths_in_;

  cuda::std::span<delta const> route_stop_times_;
  d_vecmap_view<route_idx_t, interval<std::uint32_t>> route_stop_time_ranges_;
  d_vecmap_view<route_idx_t, interval<transport_idx_t>> route_transport_ranges_;
  d_vecmap_view<route_idx_t, clasz> route_clasz_;
  device_bitvec_view route_bikes_allowed_;
  d_vecvec_view<decltype(t{}.route_bikes_allowed_per_section_)>
      route_bikes_allowed_per_section_;

  d_vecvec_view<decltype(t{}.route_location_seq_)> route_location_seq_;
  d_vecvec_view<decltype(t{}.location_routes_)> location_routes_;

  d_vecmap_view<transport_idx_t, bitfield_idx_t> transport_traffic_days_;
  d_vecmap_view<bitfield_idx_t, bitfield> bitfields_;

  interval<date::sys_days> date_range_;
};

struct gpu_timetable::impl {
  using t = timetable;

  explicit impl(timetable const& tt)
      : n_locations_{tt.n_locations()},
        n_routes_{tt.n_routes()},
        transfer_time_{to_device(tt.locations_.transfer_time_)},
        footpaths_out_{tt.locations_.footpaths_out_[0]},
        footpaths_in_{tt.locations_.footpaths_in_[0]},
        route_stop_times_{to_device(tt.route_stop_times_)},
        route_stop_time_ranges_{to_device(tt.route_stop_time_ranges_)},
        route_clasz_{to_device(tt.route_clasz_)},
        route_bikes_allowed_{to_device(tt.route_bikes_allowed_)},
        route_bikes_allowed_per_section_{tt.route_bikes_allowed_per_section_},
        route_location_seq_{tt.route_location_seq_},
        location_routes_{tt.location_routes_},
        transport_traffic_days_{to_device(tt.transport_traffic_days_)},
        bitfields_{to_device(tt.bitfields_)},
        date_range_{tt.date_range_} {}

  device_timetable to_device_timetable() const {
    return {.n_locations_ = n_locations_,
            .n_routes_ = n_routes_,
            .transfer_time_ = transfer_time_,
            .footpaths_out_ = to_view(footpaths_out_),
            .footpaths_in_ = to_view(footpaths_in_),
            .route_stop_times_ = to_view(route_stop_times_),
            .route_stop_time_ranges_ = to_view(route_stop_time_ranges_),
            .route_transport_ranges_ = to_view(route_transport_ranges_),
            .route_clasz_ = to_view(route_clasz_),
            .route_bikes_allowed_ = to_view(route_bikes_allowed_),
            .route_bikes_allowed_per_section_ =
                to_view(route_bikes_allowed_per_section_),
            .route_location_seq_ = to_view(route_location_seq_),
            .location_routes_ = to_view(location_routes_),
            .transport_traffic_days_ = to_view(transport_traffic_days_),
            .bitfields_ = to_view(bitfields_),
            .date_range_ = date_range_};
  }

  std::uint32_t n_locations_;
  std::uint32_t n_routes_;

  thrust::device_vector<u8_minutes> transfer_time_;
  device_vecvec<decltype(t{}.locations_.footpaths_out_[0])> footpaths_out_;
  device_vecvec<decltype(t{}.locations_.footpaths_in_[0])> footpaths_in_;

  thrust::device_vector<delta> route_stop_times_;
  thrust::device_vector<interval<std::uint32_t>> route_stop_time_ranges_;
  thrust::device_vector<interval<transport_idx_t>> route_transport_ranges_;
  thrust::device_vector<clasz> route_clasz_;
  device_bitvec route_bikes_allowed_;
  device_vecvec<decltype(t{}.route_bikes_allowed_per_section_)>
      route_bikes_allowed_per_section_;

  device_vecvec<decltype(t{}.route_location_seq_)> route_location_seq_;
  device_vecvec<decltype(t{}.location_routes_)> location_routes_;

  thrust::device_vector<bitfield_idx_t> transport_traffic_days_;
  thrust::device_vector<bitfield> bitfields_;

  interval<date::sys_days> date_range_;
};

gpu_timetable::gpu_timetable(timetable const& tt)
    : impl_{std::make_unique<impl>(tt)} {}

gpu_timetable::~gpu_timetable() = default;

struct gpu_raptor_state::impl {
  explicit impl(gpu_timetable const& gtt)
      : tt_{gtt.impl_->to_device_timetable()} {}

  void resize(unsigned n_locations,
              unsigned n_routes,
              unsigned n_rt_transports,
              std::array<nigiri::bitvec, kMaxVias> const& is_via,
              std::vector<via_stop> const& via_stops,
              nigiri::bitvec const& is_dest,
              std::vector<std::uint16_t> const& dist_to_dest,
              std::vector<std::uint16_t> const& lb) {
    is_intermodal_dest_ = !dist_to_dest.empty();
    n_locations_ = n_locations;
    tmp_storage_.resize(n_locations * (kMaxVias + 1));
    best_storage_.resize(n_locations * (kMaxVias + 1));
    round_times_storage_.resize(n_locations * (kMaxVias + 1) *
                                (kMaxTransfers + 1));
    station_mark_.resize(n_locations);
    prev_station_mark_.resize(n_locations);
    route_mark_.resize(n_routes);
    rt_transport_mark_.resize(n_rt_transports);
    end_reachable_.resize(n_locations);

    for (auto i = 0U; i != is_via.size(); ++i) {
      is_via_[i].size_ = is_via[i].size_;
      is_via_[i].blocks_.resize(is_via[i].blocks_.size());
      thrust::copy(thrust::cuda::par_nosync, begin(is_via[i].blocks_),
                   end(is_via[i].blocks_), begin(is_via_[i].blocks_));
    }

    via_stops_.resize(via_stops.size());
    thrust::copy(thrust::cuda::par_nosync, begin(via_stops), end(via_stops),
                 begin(via_stops_));

    is_dest_.size_ = is_dest.size_;
    is_dest_.blocks_.resize(is_dest.blocks_.size());
    thrust::copy(thrust::cuda::par_nosync, begin(is_dest.blocks_),
                 end(is_dest.blocks_), begin(is_dest_.blocks_));
  }

  template <via_offset_t Vias>
  cuda::std::span<cuda::std::array<delta_t, Vias + 1>> get_tmp() {
    return {reinterpret_cast<cuda::std::array<delta_t, Vias + 1>*>(
                thrust::raw_pointer_cast(tmp_storage_.data())),
            n_locations_};
  }

  template <via_offset_t Vias>
  cuda::std::span<cuda::std::array<delta_t, Vias + 1>> get_best() {
    return {reinterpret_cast<cuda::std::array<delta_t, Vias + 1>*>(
                thrust::raw_pointer_cast(best_storage_.data())),
            n_locations_};
  }

  template <via_offset_t Vias>
  cuda::std::span<cuda::std::array<delta_t, Vias + 1> const> get_best() const {
    return {reinterpret_cast<cuda::std::array<delta_t, Vias + 1> const*>(
                thrust::raw_pointer_cast(best_storage_.data())),
            n_locations_};
  }

  template <via_offset_t Vias>
  device_flat_matrix_view<cuda::std::array<delta_t, Vias + 1>>
  get_round_times() {
    return {{reinterpret_cast<cuda::std::array<delta_t, Vias + 1>*>(
                 thrust::raw_pointer_cast(round_times_storage_.data())),
             n_locations_ * (kMaxTransfers + 1)},
            kMaxTransfers + 1U,
            n_locations_};
  }

  template <via_offset_t Vias>
  device_flat_matrix_view<cuda::std::array<delta_t, Vias + 1> const>
  get_round_times() const {
    return {{reinterpret_cast<cuda::std::array<delta_t, Vias + 1> const*>(
                 thrust::raw_pointer_cast(round_times_storage_.data())),
             n_locations_ * (kMaxTransfers + 1)},
            kMaxTransfers + 1U,
            n_locations_};
  }

  unsigned n_locations_;
  bool is_intermodal_dest_;
  thrust::device_vector<delta_t> time_at_dest_{kMaxTransfers};
  thrust::device_vector<delta_t> tmp_storage_;
  thrust::device_vector<delta_t> best_storage_;
  thrust::device_vector<delta_t> round_times_storage_;
  thrust::device_vector<bitvec::block_t> station_mark_;
  thrust::device_vector<bitvec::block_t> prev_station_mark_;
  thrust::device_vector<bitvec::block_t> route_mark_;
  thrust::device_vector<bitvec::block_t> rt_transport_mark_;

  device_bitvec end_reachable_;
  device_bitvec is_dest_;
  std::array<device_bitvec, kMaxVias> is_via_;
  thrust::device_vector<via_stop> via_stops_;
  thrust::device_vector<std::uint16_t> dist_to_dest_;
  thrust::device_vector<std::uint16_t> lb_;

  device_timetable tt_;
};

gpu_raptor_state::gpu_raptor_state(gpu_timetable const& gtt)
    : impl_{std::make_unique<impl>(gtt)} {}

gpu_raptor_state::~gpu_raptor_state() = default;

gpu_raptor_state& gpu_raptor_state::resize(
    unsigned const n_locations,
    unsigned const n_routes,
    unsigned const n_rt_transports,
    std::array<nigiri::bitvec, kMaxVias> const& is_via,
    std::vector<via_stop> const& via_stops,
    nigiri::bitvec const& is_dest,
    std::vector<std::uint16_t> const& dist_to_dest,
    std::vector<std::uint16_t> const& lb) {
  impl_->resize(n_locations, n_routes, n_rt_transports, is_via, via_stops,
                is_dest, dist_to_dest, lb);
  return *this;
}

template <direction SearchDir, bool Rt, via_offset_t Vias>
gpu_raptor<SearchDir, Rt, Vias>::gpu_raptor(
    timetable const& tt,
    rt_timetable const* rtt,
    gpu_raptor_state& state,
    nigiri::bitvec& is_dest,
    std::array<nigiri::bitvec, kMaxVias> const& is_via,
    std::vector<std::uint16_t> const& dist_to_dest,
    hash_map<location_idx_t, std::vector<td_offset>> const& td_dist_to_dest,
    std::vector<std::uint16_t> const& lb,
    std::vector<via_stop> const& via_stops,
    day_idx_t const base,
    clasz_mask_t const allowed_claszes,
    bool const require_bike_transport,
    bool const is_wheelchair,
    transfer_time_settings const& tts)
    : tt_{tt},
      rtt_{rtt},
      n_days_{tt_.internal_interval_days().size().count()},
      n_locations_{tt_.n_locations()},
      n_routes_{tt.n_routes()},
      n_rt_transports_{Rt ? rtt->n_rt_transports() : 0U},
      state_{state.resize(n_locations_,
                          n_routes_,
                          n_rt_transports_,
                          is_via,
                          via_stops,
                          is_dest,
                          dist_to_dest,
                          lb)},
      is_dest_{is_dest},
      is_via_{is_via},
      dist_to_end_{dist_to_dest},
      td_dist_to_end_{td_dist_to_dest},
      lb_{lb},
      via_stops_{via_stops},
      base_{base},
      allowed_claszes_{allowed_claszes},
      require_bike_transport_{require_bike_transport},
      is_wheelchair_{is_wheelchair},
      transfer_time_settings_{tts} {}

__device__ __forceinline__ unsigned get_block_thread_id() {
  return threadIdx.x + (blockDim.x * threadIdx.y);
}

__device__ __forceinline__ unsigned get_global_thread_id() {
  return get_block_thread_id() + (blockDim.x * blockDim.y * blockIdx.x);
}

__device__ __forceinline__ unsigned get_block_stride() {
  return blockDim.x * blockDim.y;
}

__device__ __forceinline__ unsigned get_global_stride() {
  return get_block_stride() * gridDim.x * gridDim.y;
}

template <direction SearchDir>
constexpr static bool is_better(auto a, auto b) {
  return SearchDir == direction::kForward ? a < b : a > b;
}

template <direction SearchDir>
constexpr static bool is_better_or_eq(auto a, auto b) {
  return SearchDir == direction::kForward ? a <= b : a >= b;
}

template <direction SearchDir>
constexpr auto get_best(auto a, auto b) {
  return is_better<SearchDir>(a, b) ? a : b;
}

template <direction SearchDir>
constexpr auto get_best(auto x, auto... y) {
  ((x = get_best<SearchDir>(x, y)), ...);
  return x;
}

constexpr auto min(auto x, auto y) { return x <= y ? x : y; }

constexpr int as_int(location_idx_t const d) { return static_cast<int>(d.v_); }
constexpr int as_int(day_idx_t const d) { return static_cast<int>(d.v_); }

template <direction SearchDir>
constexpr auto dir(auto const a) {
  return (SearchDir == direction::kForward ? 1 : -1) * a;
}

template <direction SearchDir>
__device__ transport
get_earliest_transport(device_timetable const& tt,
                       cuda::std::span<delta_t> time_at_dest,
                       cuda::std::span<std::uint16_t> lb,
                       unsigned const k,
                       route_idx_t const r,
                       stop_idx_t const stop_idx,
                       day_idx_t const day_at_stop,
                       minutes_after_midnight_t const mam_at_stop,
                       location_idx_t const l) {
  constexpr auto const kFwd = SearchDir == direction::kForward;

  auto const event_times = tt.event_times_at_stop(
      r, stop_idx, kFwd ? event_type::kDep : event_type::kArr);

  auto const get_begin_it = [&](auto&& t) {
    if constexpr (kFwd) {
      return t.begin();
    } else {
      return t.rbegin();
    }
  };

  auto const get_end_it = [&](auto&& t) {
    if constexpr (kFwd) {
      return t.end();
    } else {
      return t.rend();
    }
  };

  auto const seek_first_day = [&]() {
    return linear_lb(get_begin_it(event_times), get_end_it(event_times),
                     mam_at_stop,
                     [&](delta const a, minutes_after_midnight_t const b) {
                       return is_better<SearchDir>(a.mam(), b.count());
                     });
  };

  constexpr auto const kNDaysToIterate = day_idx_t::value_t{2U};
  for (auto i = day_idx_t::value_t{0U}; i != kNDaysToIterate; ++i) {
    auto const ev_time_range =
        it_range{i == 0U ? seek_first_day() : get_begin_it(event_times),
                 get_end_it(event_times)};
    if (ev_time_range.empty()) {
      continue;
    }

    auto const day = kFwd ? day_at_stop + i : day_at_stop - i;
    for (auto it = begin(ev_time_range); it != end(ev_time_range); ++it) {
      auto const t_offset = static_cast<std::size_t>(&*it - event_times.data());
      auto const ev = *it;
      auto const ev_mam = ev.mam();

      if (is_better_or_eq(time_at_dest[k],
                          to_delta(day, ev_mam) + dir(lb[to_idx(l)]))) {
        return {transport_idx_t::invalid(), day_idx_t::invalid()};
      }

      auto const t = tt.route_transport_ranges_[r][t_offset];
      if (i == 0U && !is_better_or_eq(mam_at_stop.count(), ev_mam)) {
        continue;
      }

      auto const ev_day_offset = ev.days();
      auto const start_day =
          static_cast<std::size_t>(as_int(day) - ev_day_offset);
      if (!is_transport_active(t, start_day)) {
        continue;
      }

      return {t, static_cast<day_idx_t>(as_int(day) - ev_day_offset)};
    }
  }
  return {};
}

template <direction SearchDir, via_offset_t Vias, bool WithSectionBikeFilter>
__device__ bool update_route(
    device_timetable const& tt,
    unsigned const k,
    route_idx_t const r,
    bool const is_wheelchair,
    bool const is_intermodal_dest,
    cuda::std::span<std::pair<location_idx_t, unixtime_t> const> starts,
    cuda::std::array<device_bitvec_view, kMaxVias> is_via,
    cuda::std::span<via_stop const> via_stops,
    device_bitvec_view is_dest,
    device_bitvec_view end_reachable,
    cuda::std::span<std::uint16_t> dist_to_dest,
    cuda::std::span<std::uint16_t> lb,
    device_flat_matrix_view<cuda::std::array<delta_t, Vias + 1>> round_times,
    cuda::std::span<cuda::std::array<delta_t, Vias + 1>> best,
    cuda::std::span<cuda::std::array<delta_t, Vias + 1>> tmp,
    cuda::std::span<delta_t> time_at_dest,
    bitvec station_mark,
    bitvec prev_station_mark,
    bitvec route_mark) {
  constexpr auto const kInvalid = kInvalidDelta<SearchDir>;
  constexpr auto const kFwd = SearchDir == direction::kForward;

  auto const stop_seq = tt.route_location_seq_[r];
  bool any_marked = false;

  auto et = cuda::std::array<transport, Vias + 1>{};
  auto v_offset = cuda::std::array<std::size_t, Vias + 1>{};

  for (auto i = 0U; i != stop_seq.size(); ++i) {
    auto const stop_idx =
        static_cast<stop_idx_t>(kFwd ? i : stop_seq.size() - i - 1U);
    auto const stp = stop{stop_seq[stop_idx]};
    auto const l_idx = cista::to_idx(stp.location_idx());
    auto const is_first = i == 0U;
    auto const is_last = i == stop_seq.size() - 1U;

    auto current_best = cuda::std::array<delta_t, Vias + 1>{};
    current_best.fill(kInvalid);

    // v = via state when entering the transport
    // v + v_offset = via state at the current stop after entering the
    // transport (v_offset > 0 if the transport passes via stops)
    for (auto j = 0U; j != Vias + 1; ++j) {
      auto const v = Vias - j;
      if (!et[v].is_valid() && !prev_station_mark[l_idx]) {
        continue;
      }

      if constexpr (WithSectionBikeFilter) {
        if (!is_first &&
            !tt.route_bikes_allowed_per_section_[r][kFwd ? stop_idx - 1
                                                         : stop_idx]) {
          et[v] = {};
          v_offset[v] = 0;
        }
      }

      auto target_v = v + v_offset[v];

      if (et[v].is_valid() && stp.can_finish<SearchDir>(is_wheelchair)) {
        auto const by_transport = time_at_stop(
            r, et[v], stop_idx, kFwd ? event_type::kArr : event_type::kDep);

        auto const stop_is_via = target_v != Vias && is_via[target_v][l_idx];
        auto const is_no_stay_via =
            stop_is_via && via_stops[target_v].stay_ == duration_t{0};

        // special case: stop is via with stay > 0m + destination
        auto const is_via_and_dest =
            stop_is_via && !is_no_stay_via &&
            (is_dest[l_idx] || (is_intermodal_dest && end_reachable[l_idx]));

        if (is_no_stay_via) {
          ++v_offset[v];
          ++target_v;
        }

        current_best[v] = get_best(round_times[k - 1][l_idx][target_v],
                                   tmp[l_idx][target_v], best[l_idx][target_v]);

        auto higher_v_best = kInvalid;
        for (auto higher_v = Vias; higher_v != target_v; --higher_v) {
          higher_v_best =
              get_best(higher_v_best, round_times[k - 1][l_idx][higher_v],
                       tmp[l_idx][higher_v], best[l_idx][higher_v]);
        }

        assert(by_transport != std::numeric_limits<delta_t>::min() &&
               by_transport != std::numeric_limits<delta_t>::max());
        if (is_better<SearchDir>(by_transport, current_best[v]) &&
            is_better<SearchDir>(by_transport, time_at_dest[k]) &&
            is_better<SearchDir>(by_transport, higher_v_best) &&
            lb[l_idx] != kUnreachable &&
            is_better<SearchDir>(by_transport + dir<SearchDir>(lb[l_idx]),
                                 time_at_dest[k])) {
          tmp[l_idx][target_v] = get_best(by_transport, tmp[l_idx][target_v]);
          station_mark.mark(l_idx);
          current_best[v] = by_transport;
          any_marked = true;
        }

        if (is_via_and_dest) {
          auto const dest_v = target_v + 1;
          auto const best_dest =
              get_best(round_times[k - 1][l_idx][dest_v], tmp[l_idx][dest_v],
                       best[l_idx][dest_v]);

          if (is_better<SearchDir>(by_transport, best_dest) &&
              is_better<SearchDir>(by_transport, time_at_dest[k]) &&
              lb[l_idx] != kUnreachable &&
              is_better<SearchDir>(by_transport + dir<SearchDir>(lb[l_idx]),
                                   time_at_dest[k])) {
            tmp[l_idx][dest_v] = get_best(by_transport, tmp[l_idx][dest_v]);
            station_mark.mark(l_idx);
            any_marked = true;
          }
        }
      }
    }

    if (is_last || !stp.can_start<SearchDir>(is_wheelchair) ||
        !prev_station_mark[l_idx]) {
      continue;
    }

    if (lb[l_idx] == kUnreachable) {
      break;
    }

    for (auto v = 0U; v != Vias + 1; ++v) {
      if (!et[v].is_valid() && !prev_station_mark[l_idx]) {
        continue;
      }

      auto const target_v = v + v_offset[v];
      auto const et_time_at_stop =
          et[v].is_valid()
              ? time_at_stop(r, et[v], stop_idx,
                             kFwd ? event_type::kDep : event_type::kArr)
              : kInvalid;
      auto const prev_round_time = round_times[k - 1][l_idx][target_v];
      if (prev_round_time != kInvalid &&
          is_better_or_eq<SearchDir>(prev_round_time, et_time_at_stop)) {
        auto const [day, mam] = split(prev_round_time);
        auto const new_et = get_earliest_transport(k, r, stop_idx, day, mam,
                                                   stp.location_idx());
        current_best[v] = get_best<SearchDir>(
            current_best[v], best[l_idx][target_v], tmp[l_idx][target_v]);
        if (new_et.is_valid() &&
            (current_best[v] == kInvalid ||
             is_better_or_eq<SearchDir>(
                 time_at_stop(r, new_et, stop_idx,
                              kFwd ? event_type::kDep : event_type::kArr),
                 et_time_at_stop))) {
          et[v] = new_et;
          v_offset[v] = 0;
        }
      }
    }
  }
  return any_marked;
}

template <direction SearchDir, bool Rt, via_offset_t Vias>
__global__ void exec_raptor(
    device_timetable const tt,
    std::uint8_t const max_transfers,
    clasz_mask_t const allowed_claszes,
    bool const require_bike_transport,
    date::sys_days const base,
    unixtime_t const worst_time_at_dest,
    bool const is_intermodal_dest,
    cuda::std::span<std::pair<location_idx_t, unixtime_t> const> starts,
    cuda::std::array<device_bitvec_view, kMaxVias> is_via,
    cuda::std::span<via_stop const> via_stops,
    device_bitvec_view is_dest,
    device_bitvec_view end_reachable,
    cuda::std::span<std::uint16_t const> dist_to_dest,
    cuda::std::span<std::uint16_t const> lb,
    device_flat_matrix_view<cuda::std::array<delta_t, Vias + 1>> round_times,
    cuda::std::span<cuda::std::array<delta_t, Vias + 1>> best,
    cuda::std::span<cuda::std::array<delta_t, Vias + 1>> tmp,
    cuda::std::span<delta_t> time_at_dest,
    bitvec station_mark,
    bitvec prev_station_mark,
    bitvec route_mark) {
  auto const global_t_id = get_global_thread_id();
  auto const global_stride = get_global_stride();

  auto const update_time_at_dest = [&](size_t const k, delta_t const t) {
    for (auto i = k; i != time_at_dest.size(); ++i) {
      time_at_dest[i] = get_best<SearchDir>(time_at_dest[i], t);
    }
  };

  auto const zero_out = [&](bitvec& v) {
    for (auto i = global_t_id; i < station_mark.blocks_.size();
         i += global_stride) {
      v.blocks_[i] = 0U;
    }
  };

  for (auto i = global_t_id; i < starts.size(); i += global_stride) {
    auto const [l, t] = starts[i];
    auto const v = (Vias != 0 && is_via[0][to_idx(l)]) ? 1U : 0U;
    best[to_idx(l)][v] = unix_to_delta(base, t);
    round_times[0U][to_idx(l)][v] = unix_to_delta(base, t);
    station_mark.mark(to_idx(l));
  }

  auto const d_worst_at_dest = unix_to_delta(base, worst_time_at_dest);
  for (auto i = global_t_id; i < kMaxTransfers + 1U; i += global_stride) {
    time_at_dest[i] = get_best<SearchDir>(d_worst_at_dest, time_at_dest[i]);
  }

  cg::this_grid().sync();

  auto const end_k = min(max_transfers, kMaxTransfers) + 1U;
  for (auto k = 1U; k != end_k; ++k) {
    // ==================
    // RAPTOR ROUND START
    // ------------------

    // Reuse best time from previous time at start (for range queries).
    for (auto i = global_t_id; i < tt.n_locations_; i += global_stride) {
      for (auto v = 0U; v != Vias + 1; ++v) {
        best[i][v] = get_best<SearchDir>(round_times[k][i][v], best[i][v]);
      }
      if (is_dest[i]) {
        update_time_at_dest(k, best[i][Vias]);
      }
    }

    // Mark every route at all stations marked in the previous round.
    auto any_marked = cuda::std::atomic_bool{false};
    for (auto i = global_t_id; i < tt.n_locations_; i += global_stride) {
      if (station_mark[i]) {
        if (!any_marked) {
          any_marked = true;
        }
        for (auto r : tt.location_routes_[location_idx_t{i}]) {
          route_mark.mark(to_idx(r));
        }
      }
    }

    cg::this_grid().sync();

    if (!any_marked) {
      return;
    }

    cuda::std::swap(prev_station_mark, station_mark);
    zero_out(station_mark);

    cg::this_grid().sync();

    // Loop routes.
    for (auto i = global_t_id; i < tt.n_routes_; i += global_stride) {
      if (!route_mark[i]) {
        continue;
      }

      auto const r = route_idx_t{i};
      if (!is_allowed(allowed_claszes, tt.route_clasz_[r])) {
        return;
      }

      auto section_bike_filter = false;
      if (require_bike_transport) {
        auto const bikes_allowed_on_all_sections =
            tt.route_bikes_allowed_.test(i * 2);
        if (!bikes_allowed_on_all_sections) {
          auto const bikes_allowed_on_some_sections =
              tt.route_bikes_allowed_.test(i * 2 + 1);
          if (!bikes_allowed_on_some_sections) {
            return;
          }
          section_bike_filter = true;
        }
      }

      any_marked |= section_bike_filter
                        ? update_route<SearchDir, Vias, true>(k, r)
                        : update_route<SearchDir, Vias, false>(k, r);
    }

    cg::this_grid().sync();

    // ----------------
    // RAPTOR ROUND END
    // ================
  }
}

template <direction SearchDir, bool Rt, via_offset_t Vias>
void gpu_raptor<SearchDir, Rt, Vias>::execute(
    unixtime_t const start_time,
    std::uint8_t const max_transfers,
    unixtime_t const worst_time_at_dest,
    profile_idx_t const prf_idx,
    pareto_set<journey>& results) {
  auto const starts =
      thrust::device_vector<std::pair<location_idx_t, unixtime_t>>{starts_};
  cudaDeviceSynchronize();

  auto& s = *state_.impl_;
  exec_raptor<SearchDir, Rt, Vias><<<1, 1>>>(
      s.tt_, max_transfers, allowed_claszes_, require_bike_transport_, base(),
      worst_time_at_dest, s.is_intermodal_dest_, to_view(starts),
      to_view(s.is_via_), to_view(s.via_stops_), to_view(s.is_dest_),
      to_view(s.end_reachable_), to_view(s.dist_to_dest_), to_view(s.lb_),
      s.get_round_times<Vias>(), s.get_best<Vias>(), s.get_tmp<Vias>(),
      to_mutable_view(s.time_at_dest_), s.station_mark_, s.prev_station_mark_,
      s.route_mark_);
}

template <direction SearchDir, bool Rt, via_offset_t Vias>
void gpu_raptor<SearchDir, Rt, Vias>::reset_arrivals() {
  thrust::fill(thrust::cuda::par_nosync, begin(state_.impl_->time_at_dest_),
               end(state_.impl_->time_at_dest_), kInvalid);
  thrust::fill(thrust::cuda::par_nosync,
               begin(state_.impl_->round_times_storage_),
               end(state_.impl_->round_times_storage_), kInvalid);
}

template <direction SearchDir, bool Rt, via_offset_t Vias>
void gpu_raptor<SearchDir, Rt, Vias>::next_start_time() {
  starts_.clear();
  thrust::fill(thrust::cuda::par_nosync, begin(state_.impl_->best_storage_),
               end(state_.impl_->best_storage_), kInvalid);
  thrust::fill(thrust::cuda::par_nosync, begin(state_.impl_->tmp_storage_),
               end(state_.impl_->tmp_storage_), kInvalid);
  thrust::fill(thrust::cuda::par_nosync,
               begin(state_.impl_->prev_station_mark_),
               end(state_.impl_->prev_station_mark_), 0U);
  thrust::fill(thrust::cuda::par_nosync, begin(state_.impl_->station_mark_),
               end(state_.impl_->station_mark_), 0U);
  thrust::fill(thrust::cuda::par_nosync, begin(state_.impl_->route_mark_),
               end(state_.impl_->route_mark_), 0U);
  if constexpr (Rt) {
    thrust::fill(thrust::cuda::par_nosync,
                 begin(state_.impl_->rt_transport_mark_),
                 end(state_.impl_->rt_transport_mark_), 0U);
  }
}

template <direction SearchDir, bool Rt, via_offset_t Vias>
void gpu_raptor<SearchDir, Rt, Vias>::reconstruct(query const&, journey&) {}

template <direction SearchDir, bool Rt, via_offset_t Vias>
void gpu_raptor<SearchDir, Rt, Vias>::add_start(location_idx_t const l,
                                                unixtime_t const t) {
  starts_.emplace_back(l, t);
}

template class gpu_raptor<direction::kForward, true, 0U>;
template class gpu_raptor<direction::kForward, true, 1U>;
template class gpu_raptor<direction::kForward, true, 2U>;
template class gpu_raptor<direction::kForward, false, 0U>;
template class gpu_raptor<direction::kForward, false, 1U>;
template class gpu_raptor<direction::kForward, false, 2U>;
template class gpu_raptor<direction::kBackward, true, 0U>;
template class gpu_raptor<direction::kBackward, true, 1U>;
template class gpu_raptor<direction::kBackward, true, 2U>;
template class gpu_raptor<direction::kBackward, false, 0U>;
template class gpu_raptor<direction::kBackward, false, 1U>;
template class gpu_raptor<direction::kBackward, false, 2U>;

}  // namespace nigiri::routing::gpu