#include "nigiri/loader/virt/rewrite_routes.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "utl/to_vec.h"

#include "nigiri/loader/import_context.h"
#include "nigiri/common/interval.h"
#include "nigiri/shapes_storage.h"
#include "nigiri/stop.h"
#include "nigiri/timetable.h"

namespace nigiri::loader {

void rewrite_routes(
    timetable& tt,
    import_context& ctx,
    shapes_storage* shapes,
    vector_map<route_idx_t, route_idx_t> const& base_route,
    vecvec<route_idx_t, transport_idx_t> const& route_transports) {
  auto const n_routes = tt.route_transport_ranges_.size();
  auto const& starts = route_transports.bucket_starts_;

  auto const old_transports = std::exchange(tt.route_transport_ranges_, {});
  auto pieces = hash_map<route_idx_t, std::vector<route_idx_t>>{};
  for (auto r = route_idx_t{n_routes}; r != base_route.size(); ++r) {
    pieces[base_route[r]].push_back(r);
  }
  auto& ranges = ctx.route_stop_time_ranges_;
  auto& times = ctx.route_stop_times_;
  ranges.resize(base_route.size());
  auto block = std::vector<delta>{};
  for (auto const& [o, split] : pieces) {
    auto const n = static_cast<unsigned>(old_transports[o].size());
    auto const n_events =
        static_cast<unsigned>(tt.route_location_seq_[o].size() * 2U - 2U);
    auto i = ranges[o].from_;
    block.assign(times.begin() + i, times.begin() + i + n_events * n);
    auto const write = [&](route_idx_t const r) {
      auto const from = i;
      for (auto e = 0U; e != n_events; ++e) {
        for (auto const t : route_transports[r]) {
          times[i++] = block[e * n + to_idx(t - old_transports[o].from_)];
        }
      }
      ranges[r] = {from, i};
    };
    write(o);
    for (auto const r : split) {
      write(r);
    }
  }

  for (auto r = route_idx_t{0U}; r != base_route.size(); ++r) {
    auto const o = base_route[r];
    tt.route_transport_ranges_.emplace_back(
        transport_idx_t{starts[to_idx(r)]},
        transport_idx_t{starts[to_idx(r) + 1U]});

    if (r < n_routes) {
      continue;
    }
    auto const clasz = tt.route_clasz_[o];
    tt.route_clasz_.push_back(clasz);
    auto const section_clasz = utl::to_vec(tt.route_section_clasz_[o]);
    tt.route_section_clasz_.emplace_back(section_clasz);
    for (auto f = 0U; f != kNumRouteFlags; ++f) {
      auto& flags = tt.route_flags_[f];
      flags.resize(flags.size() + 2U);
      flags.set(to_idx(r) * 2U, flags.test(to_idx(o) * 2U));
      flags.set(to_idx(r) * 2U + 1U, flags.test(to_idx(o) * 2U + 1U));
      auto const per_section = utl::to_vec(tt.route_flags_per_section_[f][o]);
      tt.route_flags_per_section_[f].emplace_back(per_section);
    }
    if (shapes != nullptr && shapes->route_bboxes_.size() == to_idx(r)) {
      auto const bbox = shapes->route_bboxes_[o];
      auto const segments = utl::to_vec(shapes->route_segment_bboxes_[o]);
      shapes->route_bboxes_.push_back(bbox);
      shapes->route_segment_bboxes_.emplace_back(segments);
    }
  }

  auto const& order = route_transports.data_;
  auto inverse = vector_map<transport_idx_t, transport_idx_t>{};
  inverse.resize(order.size());
  for (auto t = transport_idx_t{0U}; t != order.size(); ++t) {
    inverse[order[to_idx(t)]] = t;
  }
  auto const permute = [&](auto& v) {
    auto copy = v;
    for (auto t = transport_idx_t{0U}; t != order.size(); ++t) {
      copy[t] = v[order[to_idx(t)]];
    }
    v = std::move(copy);
  };
  auto const permute_buckets = [&](auto& v, auto empty) {
    for (auto const t : order) {
      empty.emplace_back(v[t]);
    }
    v = std::move(empty);
  };
  permute(tt.transport_first_dep_offset_);
  permute(tt.transport_traffic_days_);
  permute_buckets(tt.transport_to_trip_section_,
                  decltype(tt.transport_to_trip_section_){});
  permute_buckets(tt.transport_section_attributes_,
                  decltype(tt.transport_section_attributes_)::with_default({}));
  permute(tt.transport_section_providers_.sequence_);
  permute(tt.transport_section_directions_.sequence_);
  for (auto r = route_idx_t{0U}; r != tt.n_routes(); ++r) {
    for (auto const t : tt.route_transport_ranges_[r]) {
      tt.transport_route_[t] = r;
    }
  }
  for (auto trip = trip_idx_t{0U}; trip != ctx.trip_transport_ranges_.size();
       ++trip) {
    for (auto& range : ctx.trip_transport_ranges_[trip]) {
      range.first = inverse[range.first];
    }
  }

  auto location_routes = mutable_fws_multimap<location_idx_t, route_idx_t>{};
  for (auto r = route_idx_t{0U}; r != tt.n_routes(); ++r) {
    for (auto const s : tt.route_location_seq_[r]) {
      auto routes = location_routes[stop{s}.location_idx()];
      if (routes.empty() || routes.back() != r) {
        routes.push_back(r);
      }
    }
  }
  tt.location_routes_.clear();
  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    tt.location_routes_.emplace_back(location_routes[l]);
  }
}

}  // namespace nigiri::loader
