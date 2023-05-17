#pragma once

#include "nigiri/routing/journey.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/routing/start_times.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::routing {

struct search_state {
  void reset(timetable const& tt) {
    is_destination_.resize(tt.n_locations());
    utl::fill(is_destination_, false);

    travel_time_lower_bound_.resize(tt.n_locations());
    utl::fill(travel_time_lower_bound_, duration_t{0});

    starts_.clear();
    destinations_.clear();
    results_.clear();
  }

  std::vector<duration_t> travel_time_lower_bound_;
  std::vector<bool> is_destination_;
  std::vector<start> starts_;
  std::vector<std::set<location_idx_t>> destinations_;
  std::vector<pareto_set<journey>> results_;
  interval<unixtime_t> search_interval_;
};

template <typename Algo>
struct search {
  search(search_state& s) : search_state_{s} {}
  search_state& search_state_;
};

}  // namespace nigiri::routing