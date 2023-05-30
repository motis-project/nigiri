#include "nigiri/loader/init_finish.h"

#include <execution>

#include "nigiri/loader/build_footpaths.h"
#include "nigiri/loader/build_lb_graph.h"
#include "nigiri/special_stations.h"
#include "nigiri/timetable.h"

namespace nigiri::loader {

void register_special_stations(timetable& tt) {
  auto empty_idx_vec = vector<location_idx_t>{};
  auto empty_footpath_vec = vector<footpath>{};
  for (auto const& name : special_stations_names) {
    tt.locations_.register_location(location{name,
                                             name,
                                             {0.0, 0.0},
                                             source_idx_t::invalid(),
                                             location_type::kStation,
                                             osm_node_id_t::invalid(),
                                             location_idx_t::invalid(),
                                             timezone_idx_t::invalid(),
                                             0_minutes,
                                             it_range{empty_idx_vec},
                                             it_range{empty_footpath_vec},
                                             it_range{empty_footpath_vec}});
  }
  tt.location_routes_.resize(tt.n_locations());
}

void finalize(timetable& tt) {
  tt.location_routes_.resize(tt.n_locations());

  {
    auto const timer = scoped_timer{"loader.sort_trip_ids"};
    std::sort(
#if __cpp_lib_execution
        std::execution::par_unseq,
#endif
        begin(tt.trip_id_to_idx_), end(tt.trip_id_to_idx_),
        [&](pair<trip_id_idx_t, trip_idx_t> const& a,
            pair<trip_id_idx_t, trip_idx_t> const& b) {
          return tt.trip_id_strings_[a.first].view() <
                 tt.trip_id_strings_[b.first].view();
        });
  }
  build_footpaths(tt);
  build_lb_graph<direction::kForward>(tt);
  build_lb_graph<direction::kBackward>(tt);
}

}  // namespace nigiri::loader