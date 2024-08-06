#include "nigiri/loader/gtfs/shape.h"
#include "nigiri/logging.h"

#include <optional>
#include <ranges>

#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/for_each.h"
#include "utl/pipes/transform.h"

#include "geo/latlng.h"

namespace nigiri::loader::gtfs {

namespace helper {

using precision_t = decltype(shape::coordinate_precision);

/* Code duplicated from 'osmium/osm/location.hpp' */
constexpr precision_t double_to_fix(double const c) noexcept {
  return static_cast<int32_t>(std::round(c * shape::coordinate_precision));
}

constexpr double fix_to_double(precision_t const c) noexcept {
  return static_cast<double>(c) / shape::coordinate_precision;
}
}  // namespace helper

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
          helper::double_to_fix(entry.lat.val()),
          helper::double_to_fix(entry.lon.val()),
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

  utl::line_range{utl::make_buf_reader(data, utl::noop_progress_consumer{})} |
      utl::csv<shape_point::entry>() |
      utl::transform([&](shape_point::entry const& entry) {
        return shape_point::from_entry(entry);
      }) |
      utl::for_each(store_to_map);

  return states;
}

shape::shape(mmap_vecvec::bucket bucket) : bucket_{bucket} {}

shape::value_type shape::get() const {
  auto coordinates =
      bucket_ | std::views::transform([](shape::coordinate const& c) {
        return geo::latlng{helper::fix_to_double(c.lat),
                           helper::fix_to_double(c.lon)};
      });
  return value_type{coordinates.begin(), coordinates.end()};
}

std::function<std::optional<shape>(const shape::id_type&)>
shape::get_builder() {
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
    return (found != map.end()) ? std::make_optional<shape>(
                                      shape{(*vecvec)[found->second.offset]})
                                : std::nullopt;
  };
}

}  // namespace nigiri::loader::gtfs
