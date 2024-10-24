#include "nigiri/loader/gtfs/shape.h"

#include "geo/latlng.h"

#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/for_each.h"
#include "utl/progress_tracker.h"

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
    utl::csv_col<std::size_t, UTL_NAME("shape_pt_sequence")> seq_;
    utl::csv_col<double, UTL_NAME("shape_dist_traveled")> distance_;
  };

  auto const index_offset = static_cast<shape_idx_t>(shapes.size());
  auto states = shape_loader_state{
      .index_offset_ = index_offset,
  };
  auto lookup = cached_lookup(states.id_map_);

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Parse Shapes")
      .out_bounds(37.F, 38.F)
      .in_high(data.size());
  utl::line_range{utl::make_buf_reader(data, progress_tracker->update_fn())}  //
      | utl::csv<shape_entry>()  //
      | utl::for_each([&](shape_entry const entry) {
          auto& state = lookup(entry.id_->view(), [&] {
            auto const index = static_cast<shape_idx_t>(shapes.size());
            shapes.emplace_back_empty();
            states.distances_.emplace_back_empty();
            return shape_state{index, 0U};
          });
          auto const seq = *entry.seq_;
          auto bucket = shapes[state.index_];
          if (!bucket.empty() && state.last_seq_ >= seq) {
            log(log_lvl::error, "loader.gtfs.shape",
                "Non monotonic sequence for shape_id '{}': Sequence number {} "
                "followed by {}",
                entry.id_->to_str(), state.last_seq_, seq);
          }
          bucket.push_back(geo::latlng{*entry.lat_, *entry.lon_});
          state.last_seq_ = seq;
          auto distances = states.distances_[state.index_ - index_offset];
          if (distances.empty()) {
            if (*entry.distance_ != 0.0) {
              for (auto i = 0U; i != bucket.size(); ++i) {
                distances.push_back(0.0);
              }
              distances.back() = *entry.distance_;
            }
          } else {
            distances.push_back(*entry.distance_);
          }
        });
  return states;
}

}  // namespace nigiri::loader::gtfs
