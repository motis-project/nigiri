#include "nigiri/loader/gtfs/shape.h"

#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/for_each.h"
#include "utl/progress_tracker.h"

#include "nigiri/logging.h"

namespace nigiri::loader::gtfs {

shape_id_map_t const parse_shapes(std::string_view const data,
                                  shape_vecvec_t& vecvec) {
  struct shape_entry {
    utl::csv_col<utl::cstr, UTL_NAME("shape_id")> id_;
    utl::csv_col<double, UTL_NAME("shape_pt_lat")> lat_;
    utl::csv_col<double, UTL_NAME("shape_pt_lon")> lon_;
    utl::csv_col<size_t, UTL_NAME("shape_pt_sequence")> seq_;
  };

  struct shape_state {
    shape_idx_t index_{};
    size_t last_seq_{};
  };
  auto states = hash_map<utl::cstr, shape_state>{};

  auto const store_to_map = [&states, &vecvec](shape_entry const entry) {
    if (auto found = states.find(entry.id_->view()); found != states.end()) {
      auto& state = found->second;
      auto const seq = entry.seq_.val();
      if (state.last_seq_ >= seq) {
        log(log_lvl::error, "loader.gtfs.shape",
            "Non monotonic sequence for shape_id '{}': Sequence number {} "
            "followed by {}",
            entry.id_.val().to_str(), state.last_seq_, seq);
      }
      vecvec[state.index_].push_back(
          geo::latlng{entry.lat_.val(), entry.lon_.val()});
      state.last_seq_ = seq;
    } else {
      auto const index = static_cast<shape_idx_t>(vecvec.size());
      auto bucket = vecvec.add_back_sized(0u);
      states.insert({entry.id_->view(), {index, entry.seq_.val()}});
      bucket.push_back(geo::latlng{entry.lat_.val(), entry.lon_.val()});
    }
  };

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Parse Shapes")
      .out_bounds(37.F, 38.F)
      .in_high(data.size());
  utl::line_range{utl::make_buf_reader(data, progress_tracker->update_fn())} |
      utl::csv<shape_entry>() | utl::for_each(store_to_map);

  auto shapes{shape_id_map_t{}};
  for (auto const& [id, state] : states) {
    shapes.insert({id.to_str(), state.index_});
  }
  return shapes;
}

}  // namespace nigiri::loader::gtfs
