#pragma once

#include <filesystem>
#include <mutex>

#include "cista/memory_holder.h"

#include "nigiri/routing/journey.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::routing {

struct reach_info {
  reach_info();

  bool valid() const noexcept {
    return start_end_ != location_idx_t::invalid();
  }

  void update(float const new_reach,
              routing::journey const&,
              location_idx_t const start_end,
              location_idx_t const stop_in_route);

  std::mutex mutex_;
  float reach_{-1.0};
  routing::journey j_;
  location_idx_t start_end_{location_idx_t::invalid()};
  location_idx_t stop_in_route_{location_idx_t::invalid()};
};

std::vector<reach_info> compute_reach_values(
    timetable const& tt,
    std::vector<location_idx_t> const& source_locations,
    interval<date::sys_days> const search_interval);

std::pair<std::vector<unsigned>, float> get_separation_fn(
    timetable const& tt,
    std::vector<reach_info> const& route_reachs,
    double const reach_factor,
    double const outlier_percent);

void write_reach_values(timetable const&,
                        float y,
                        float x_slope,
                        std::vector<reach_info> const& route_reachs,
                        std::filesystem::path const&);

cista::wrapped<vector_map<route_idx_t, unsigned>> read_reach_values(
    cista::memory_holder&&);

}  // namespace nigiri::routing