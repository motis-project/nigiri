#include "nigiri/loader/init_finish.h"

#include <execution>

#include "nigiri/loader/build_footpaths.h"
#include "nigiri/loader/build_lb_graph.h"
#include "nigiri/special_stations.h"
#include "nigiri/timetable.h"

namespace nigiri::loader {

template <typename IdIdxT, typename IdxT>
void sorted_ids(vector<pair<IdIdxT, IdxT>>& id_to_idx,
                vecvec<IdIdxT, char> const& id_strings,
                vector_map<IdIdxT, source_idx_t>& id_src) {
  std::sort(
#if __cpp_lib_execution
      std::execution::par_unseq,
#endif
      begin(id_to_idx), end(id_to_idx),
      [&](pair<IdIdxT, IdxT> const& a, pair<IdIdxT, IdxT> const& b) {
        return std::tuple(id_src[a.first], id_strings[a.first].view()) <
               std::tuple(id_src[b.first], id_strings[b.first].view());
      });
}

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

void finalize(timetable& tt, finalize_options const opt) {
  assert(route_id_idx_t{tt.route_id_strings_.size()} == tt.next_route_id_idx_);
  assert(route_id_idx_t{tt.route_id_src_.size()} == tt.next_route_id_idx_);
  tt.location_routes_.resize(tt.n_locations());

  {
    auto const timer = scoped_timer{"loader.sort_ids"};
    sorted_ids(tt.route_id_to_idx_, tt.route_id_strings_, tt.route_id_src_);
    sorted_ids(tt.trip_id_to_idx_, tt.trip_id_strings_, tt.trip_id_src_);
  }
  build_footpaths(tt, opt);
  build_lb_graph<direction::kForward>(tt);
  build_lb_graph<direction::kBackward>(tt);
}

void finalize(timetable& tt,
              bool const adjust_footpaths,
              bool const merge_dupes_intra_src,
              bool const merge_dupes_inter_src,
              std::uint16_t const max_footpath_length) {
  finalize(tt, {adjust_footpaths, merge_dupes_intra_src, merge_dupes_inter_src,
                max_footpath_length});
}

}  // namespace nigiri::loader