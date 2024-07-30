#include <cista/containers/mmap_vec.h>
#include <cista/mmap.h>
#include <sys/types.h>
#include <cstddef>
#include <format>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

#include "cista/mmap.h"
#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/for_each.h"
#include "utl/progress_tracker.h"

#include "shared.h"
#include "utl/pipes/transform.h"

const std::string in_file{"../dev/shapes.txt"};
constexpr std::string_view cache_file_template{"shape-cache.{}.dat"};
constexpr std::string_view id_map_file{"shape-id.dat"};  // Might be anything?

using datatype = int32_t;
using id_type = uint16_t;

class InvalidShapesFormat final : public std::runtime_error {
public:
  InvalidShapesFormat(const std::string& msg) : std::runtime_error{msg} {}
  ~InvalidShapesFormat() override;
};
InvalidShapesFormat::~InvalidShapesFormat() = default;

struct ShapePoint {
  const id_type id;
  const datatype lat, lon;
  const size_t seq;

  struct Shape {
    utl::csv_col<id_type, UTL_NAME("shape_id")> id;
    utl::csv_col<double, UTL_NAME("shape_pt_lat")> lat;
    utl::csv_col<double, UTL_NAME("shape_pt_lon")> lon;
    utl::csv_col<size_t, UTL_NAME("shape_pt_sequence")> seq;
  };

  static constexpr ShapePoint from_shape(const Shape& shape) {
    return ShapePoint{
        shape.id.val(),
        double_to_fix(shape.lat.val()),
        double_to_fix(shape.lon.val()),
        shape.seq.val(),
    };
  }

  operator std::string() const {
    return std::format("ShapePoint(id={}, {}, {}, seq={})", id, lat, lon, seq);
  }
};

void progress_lines(const auto& file_content, auto func) {
  auto const progress_tracker = utl::activate_progress_tracker("writer");
  progress_tracker->status("Parse Agencies")
      .out_bounds(0.F, 1.F)
      .in_high(file_content.size());
  utl::line_range{
      utl::make_buf_reader(file_content, progress_tracker->update_fn())}  //
      | utl::csv<ShapePoint::Shape>()  //
      | utl::transform([&](ShapePoint::Shape const& shape) {
          return ShapePoint::from_shape(shape);
        }) |
      utl::for_each(func);
}

auto get_cache(cista::mmap::protection mode) {
  return mm_vecvec<std::size_t, datatype>{
      cista::basic_mmap_vec<datatype, std::size_t>{
          cista::mmap{std::format(cache_file_template, "values").data(), mode}},
      cista::basic_mmap_vec<std::size_t, std::size_t>{cista::mmap{
          std::format(cache_file_template, "metadata").data(), mode}}};
}

auto get_cache_writer() { return get_cache(cista::mmap::protection::WRITE); }

auto get_mapper(cista::mmap::protection mode) {
  return cista::mmap_vec<id_type>{cista::mmap{id_map_file.data(), mode}};
}

auto get_map_writer() { return get_mapper(cista::mmap::protection::WRITE); }

void show_stats(auto& cache) {
  std::cout << std::format("Added {} buckets", cache.size()) << std::endl;
  auto entries =
      std::accumulate(cache.begin(), cache.end(), 0u,
                      [](auto count, auto b) { return count + b.size(); });
  std::cout << std::format("Number of entries: {}", entries) << std::endl;
}

void store_ids(auto ids) {
  auto mapper = get_map_writer();
  mapper.reserve(static_cast<unsigned int>(2u * ids.size()));
  for (auto [id, pos] : ids) {
    mapper.push_back(id);
    mapper.push_back(pos);
  }
}

int main() {
  cista::mmap_vec<char> shaped_data{
      cista::mmap{in_file.data(), cista::mmap::protection::READ}};
  auto cache = get_cache_writer();

  size_t last_id{0u};
  auto bucket = cache.add_back_sized(0u);
  std::unordered_map<id_type, id_type> ids;

  auto store_entry = [&bucket, &cache, &ids, &last_id](const auto& point) {
    if (last_id == 0u) {
      ids.insert({point.id, 0u});
      last_id = point.id;
    } else if (last_id != point.id) {
      ids.insert({point.id, cache.size()});
      bucket = cache.add_back_sized(0u);
      last_id = point.id;
    }
    bucket.push_back(point.lat);
    bucket.push_back(point.lon);
  };
  progress_lines(std::string_view(shaped_data), store_entry);

  store_ids(ids);

  show_stats(cache);
  std::cout << "Testing some ids …" << std::endl;
  for (auto key : {1, 134, 573}) {
    std::cout << std::format("Key {:>5} at position {:>5}", key,
                             ids.at(static_cast<id_type>(key)))
              << std::endl;
  }
  return 0;
}
