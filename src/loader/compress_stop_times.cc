#include "nigiri/loader/compress_stop_times.h"

#include <vector>

#include "utl/get_or_create.h"

#include "nigiri/timetable.h"

namespace nigiri::loader {

void compress_stop_times(timetable& tt) {
  auto const n_transports = tt.transport_route_.size();
  tt.transport_first_dep_.resize(n_transports);
  tt.transport_pattern_.resize(n_transports);
  tt.route_times_.reserve(tt.n_routes());

  auto offsets = std::vector<delta>{};
  auto route_patterns = vecvec<unsigned, delta>{};
  auto patterns = hash_map<std::vector<delta>, std::uint16_t>{};
  for (auto r = route_idx_t{0U}; r != tt.n_routes(); ++r) {
    auto const transports = tt.route_transport_ranges_[r];
    auto const n = static_cast<unsigned>(transports.size());
    auto const n_ev =
        static_cast<unsigned>(tt.route_location_seq_[r].size() * 2U - 2U);
    auto const base = tt.route_stop_time_ranges_[r].from_;
    auto const stop_time = [&](unsigned const e, transport_idx_t const t) {
      return tt.route_stop_times_[base + e * n + to_idx(t - transports.from_)];
    };

    offsets.resize(n_ev);
    route_patterns.clear();
    patterns.clear();
    for (auto const t : transports) {
      tt.transport_first_dep_[t] = stop_time(0U, t);
      for (auto e = 0U; e != n_ev; ++e) {
        offsets[e] = stop_time(e, t) - tt.transport_first_dep_[t];
      }

      tt.transport_pattern_[t] = utl::get_or_create(patterns, offsets, [&] {
        route_patterns.emplace_back(offsets);
        return static_cast<std::uint16_t>(route_patterns.size() - 1U);
      });
    }

    auto const n_patterns = static_cast<std::uint16_t>(route_patterns.size());
    tt.route_times_.push_back({.transports_ = transports,
                               .pattern_start_ = static_cast<std::uint32_t>(
                                   tt.pattern_offsets_.size()),
                               .n_patterns_ = n_patterns});
    for (auto e = 0U; e != n_ev; ++e) {
      for (auto p = 0U; p != n_patterns; ++p) {
        tt.pattern_offsets_.push_back(route_patterns[p][e]);
      }
    }
  }

  tt.route_stop_time_ranges_ = {};
  tt.route_stop_times_ = {};
}

}  // namespace nigiri::loader
