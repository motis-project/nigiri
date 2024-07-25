#include "nigiri/loader/init_finish.h"

#include <execution>

#include "nigiri/loader/build_footpaths.h"
#include "nigiri/loader/build_lb_graph.h"
#include "nigiri/special_stations.h"
#include "nigiri/timetable.h"

namespace nigiri::loader {

void register_special_stations(timetable& tt) {
  auto empty_idx_vec = vector<location_idx_t>{};
  for (auto const& name : special_stations_names) {
    tt.locations_.register_location(location{name,
                                             name,
                                             {0.0, 0.0},
                                             source_idx_t::invalid(),
                                             location_type::kStation,
                                             location_idx_t::invalid(),
                                             timezone_idx_t::invalid(),
                                             0_minutes,
                                             it_range{empty_idx_vec}});
  }
  tt.location_routes_.resize(tt.n_locations());
  tt.bitfields_.emplace_back(bitfield{});  // bitfield_idx 0 = 000...00 bitfield
}

void finalize(timetable& tt,
              bool const adjust_footpaths,
              bool const merge_duplicates,
              std::uint16_t const max_footpath_length) {
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
          return std::tuple(tt.trip_id_src_[a.first],
                            tt.trip_id_strings_[a.first].view()) <
                 std::tuple(tt.trip_id_src_[b.first],
                            tt.trip_id_strings_[b.first].view());
        });
  }
  build_footpaths(tt, adjust_footpaths, merge_duplicates, max_footpath_length);
  build_lb_graph<direction::kForward>(tt);
  build_lb_graph<direction::kBackward>(tt);
  std::sort(begin(tt.locations_.sorted_by_location_id_),
            end(tt.locations_.sorted_by_location_id_),
            [&](auto const& a, auto const& b) {
              return std::string_view{begin(tt.locations_.ids_[a]),
                                      end(tt.locations_.ids_[a])} <
                     std::string_view{begin(tt.locations_.ids_[b]),
                                      end(tt.locations_.ids_[b])};
            });
}

}  // namespace nigiri::loader