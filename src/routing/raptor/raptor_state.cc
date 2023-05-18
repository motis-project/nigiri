#include "nigiri/routing/raptor/raptor_state.h"

#include "fmt/core.h"

#include "utl/helpers/algorithm.h"

#include "nigiri/routing/limits.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

void raptor_state::reset(unsigned const n_locations,
                         unsigned const n_routes,
                         delta_t invalid) {
  tmp_.resize(n_locations);
  utl::fill(tmp_, invalid);

  station_mark_.resize(n_locations);
  utl::fill(station_mark_, false);

  prev_station_mark_.resize(n_locations);
  utl::fill(prev_station_mark_, false);

  route_mark_.resize(n_routes);
  utl::fill(route_mark_, false);

  best_.resize(n_locations);
  utl::fill(best_, invalid);

  round_times_.resize(kMaxTransfers + 1U, n_locations);
  round_times_.reset(invalid);
}

void raptor_state::print(timetable const& tt,
                         date::sys_days const base,
                         delta_t const invalid) {
  auto const has_empty_rounds = [&](std::uint32_t const l) {
    return false;
    for (auto k = 0U; k != kMaxTransfers + 1U; ++k) {
      if (round_times_[k][l] != invalid) {
        return false;
      }
    }
    return true;
  };

  auto const print_delta = [&](delta_t const d) {
    if (d == invalid) {
      fmt::print("________________");
    } else {
      fmt::print("{:16}", delta_to_unix(base, d));
    }
  };

  for (auto l = 0U; l != tt.n_locations(); ++l) {
    if (best_[l] == invalid && has_empty_rounds(l)) {
      continue;
    }

    fmt::print("{:40}  ", location{tt, location_idx_t{l}});

    auto const b = best_[l];
    fmt::print("best=");
    print_delta(b);
    fmt::print(", round_times: ");
    for (auto i = 0U; i != kMaxTransfers + 1U; ++i) {
      auto const t = round_times_[i][l];
      print_delta(t);
      fmt::print(" ");
    }
    fmt::print("\n");
  }
}

}  // namespace nigiri::routing
