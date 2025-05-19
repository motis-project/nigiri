#include "nigiri/loader/init_finish.h"

#include <execution>

#include "utl/enumerate.h"

#include "geo/box.h"

#include "nigiri/loader/build_footpaths.h"
#include "nigiri/loader/build_lb_graph.h"
#include "nigiri/flex.h"
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

void build_location_tree(timetable& tt) {
  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    auto box = geo::box{};
    box.extend(tt.locations_.coordinates_[l]);
    tt.locations_.rtree_.insert(box.min_.lnglat_float(),
                                box.max_.lnglat_float(), l);
  }
}

void assign_stops_to_flex_areas(timetable& tt) {
  for (auto const [i, bbox] : utl::enumerate(tt.flex_area_bbox_)) {
    auto const flex_area = flex_area_idx_t{i};
    tt.flex_area_locations_.emplace_back(
        std::initializer_list<location_idx_t>{});
    tt.locations_.rtree_.search(
        bbox.min_.lnglat_float(), bbox.max_.lnglat_float(),
        [&](auto, auto, location_idx_t const l) {
          if (is_in_flex_area(tt, flex_area, tt.locations_.coordinates_[l])) {
            tt.flex_area_locations_.back().push_back(l);
          }
          return true;
        });
  }
}

void finalize(timetable& tt, finalize_options const opt) {
  tt.strings_.cache_.clear();
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
  {
    auto const timer = scoped_timer{"loader.sort_providers"};
    std::sort(
#if __cpp_lib_execution
        std::execution::par_unseq,
#endif
        begin(tt.provider_id_to_idx_), end(tt.provider_id_to_idx_),
        [&](provider_idx_t const a, provider_idx_t const b) {
          return tt.strings_.get(tt.providers_[a].short_name_) <
                 tt.strings_.get(tt.providers_[b].short_name_);
        });
  }
  build_footpaths(tt, opt);
  build_lb_graph<direction::kForward>(tt);
  build_lb_graph<direction::kBackward>(tt);
  build_location_tree(tt);
  assign_stops_to_flex_areas(tt);
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