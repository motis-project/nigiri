#include "nigiri/loader/gtfs/shape.h"

#include "geo/latlng.h"

#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/for_each.h"
#include "utl/progress_tracker.h"
#include "utl/sort_by.h"

#include "nigiri/common/cached_lookup.h"
#include "nigiri/logging.h"
#include "nigiri/shapes_storage.h"

namespace nigiri::loader::gtfs {

shape_loader_state parse_shapes(std::string_view const data,
                                shapes_storage& shapes_data) {
  auto& shapes = shapes_data.data_;
  struct shape_entry {
    utl::csv_col<utl::cstr, UTL_NAME("shape_id")> id_;
    utl::csv_col<double, UTL_NAME("shape_pt_lat")> lat_;
    utl::csv_col<double, UTL_NAME("shape_pt_lon")> lon_;
    utl::csv_col<std::uint32_t, UTL_NAME("shape_pt_sequence")> seq_;
    utl::csv_col<double, UTL_NAME("shape_dist_traveled")> distance_;
  };

  auto states = shape_loader_state{
      .index_offset_ = static_cast<shape_idx_t::value_t>(shapes.size()),
  };
  auto lookup = cached_lookup(states.id_map_);
  auto seq = vector_map<relative_shape_idx_t, std::vector<std::uint32_t>>{};

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Parse Shapes")
      .out_bounds(37.F, 38.F)
      .in_high(data.size());
  utl::line_range{utl::make_buf_reader(data, progress_tracker->update_fn())}  //
      | utl::csv<shape_entry>()  //
      | utl::for_each([&](shape_entry const entry) {
          auto const shape_idx = lookup(entry.id_->view(), [&] {
            auto const idx = static_cast<shape_idx_t>(shapes.size());
            shapes.emplace_back_empty();
            states.distances_.emplace_back();
            seq.emplace_back();
            return idx;
          });
          auto polyline = shapes[shape_idx];
          polyline.push_back(geo::latlng{*entry.lat_, *entry.lon_});
          auto const state_idx = states.get_relative_idx(shape_idx);
          seq[state_idx].push_back(*entry.seq_);
          auto& distances = states.distances_[state_idx];
          if (distances.empty()) {
            if (*entry.distance_ != 0.0) {
              for (auto i = 0U; i != polyline.size(); ++i) {
                distances.push_back(0.0);
              }
              distances.back() = *entry.distance_;
            }
          } else {
            distances.push_back(*entry.distance_);
          }
        });

  auto polyline = std::vector<geo::latlng>();
  for (auto const i :
       interval{relative_shape_idx_t{0U},
                relative_shape_idx_t{states.distances_.size()}}) {
    if (utl::is_sorted(seq[i], std::less<>{})) {
      continue;
    }
    auto const shape_idx = states.get_shape_idx(i);

    polyline.resize(shapes[shape_idx].size());
    for (auto j = 0U; j != shapes[shape_idx].size(); ++j) {
      polyline[j] = shapes[shape_idx][j];
    }

    std::tie(seq[i], states.distances_[i], polyline) =
        utl::sort_by(seq[i], states.distances_[i], polyline);
    std::copy(begin(polyline), end(polyline), begin(shapes[shape_idx]));
  }

  return states;
}

}  // namespace nigiri::loader::gtfs
