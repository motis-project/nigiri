#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/for_each.h"
#include "utl/pipes/transform.h"
#include "utl/progress_tracker.h"

#include "nigiri/loader/gtfs/shape.h"
#include "nigiri/logging.h"

namespace nigiri::loader::gtfs {

struct shape_point {
  shape::id_type const id;
  shape::stored_type const coordinate;
  size_t const seq;
  struct entry {
    utl::csv_col<shape::id_type, UTL_NAME("shape_id")> id;
    utl::csv_col<double, UTL_NAME("shape_pt_lat")> lat;
    utl::csv_col<double, UTL_NAME("shape_pt_lon")> lon;
    utl::csv_col<size_t, UTL_NAME("shape_pt_sequence")> seq;
  };
  static const shape_point from_entry(entry const&);
};

const shape_point shape_point::from_entry(entry const& entry) {
  return shape_point{
      entry.id->view(),
      {
          entry.lat.val(),
          entry.lon.val(),
      },
      entry.seq.val(),
  };
}

auto load_shapes(const std::string_view data, shape::mmap_vecvec& vecvec) {
  struct state {
    shape::key_type offset{};
    size_t last_seq{};
  };
  hash_map<shape::id_type, state> states;

  auto store_to_map = [&vecvec, &states](shape_point const point) {
    if (auto found = states.find(point.id); found != states.end()) {
      auto& state = found->second;
      if (state.last_seq >= point.seq) {
        log(log_lvl::error, "loader.gtfs.shape",
            "Non monotonic sequence for shape_id '{}': Sequence number {} "
            "followed by {}",
            point.id.to_str(), state.last_seq, point.seq);
      }
      vecvec[state.offset].push_back(point.coordinate);
      state.last_seq = point.seq;
    } else {
      shape::key_type offset{static_cast<shape::key_type>(states.size())};
      auto bucket = vecvec.add_back_sized(0u);
      states.insert({point.id, {offset, point.seq}});
      bucket.push_back(point.coordinate);
    }
  };

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Parse Shapes")
      .out_bounds(0.F, 1.F)
      .in_high(data.size());
  utl::line_range{utl::make_buf_reader(data, progress_tracker->update_fn())} |
      utl::csv<shape_point::entry>() |
      utl::transform([&](shape_point::entry const& entry) {
        return shape_point::from_entry(entry);
      }) |
      utl::for_each(store_to_map);

  return states;
}

shape::value_type shape::operator()() const {
  return value_type{bucket_.begin(), bucket_.end()};
}

shape::builder_t shape::get_builder() {
  return [](const id_type&) { return std::nullopt; };
}

shape::builder_t shape::get_builder(const std::string_view data,
                                    mmap_vecvec* vecvec) {
  if (vecvec == nullptr) {
    return get_builder();
  }
  auto const map = load_shapes(data, *vecvec);
  return [vecvec, map](const id_type& id) {
    auto found = map.find(id);
    return (found != map.end())
               ? std::make_optional(shape{(*vecvec)[found->second.offset]})
               : std::nullopt;
  };
}

}  // namespace nigiri::loader::gtfs