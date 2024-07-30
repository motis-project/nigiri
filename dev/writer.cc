#include <cista/containers/mmap_vec.h>
#include <cista/mmap.h>
#include <cstddef>
#include <algorithm>
#include <numeric>
#include <ranges>
#include <iostream>
#include <format>
#include <stdexcept>
#include <string_view>
#include <vector>

// #include <sys/resource.h>

#include "cista/mmap.h"
#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/for_each.h"

#include "shared.h"
#include "utl/pipes/transform.h"

const std::string in_file{"../dev/shapes.txt"};
constexpr std::string_view cache_file_template{"shape-cache.{}.dat"};

using datatype = int32_t;

class InvalidShapesFormat final : public std::runtime_error {
public:
    InvalidShapesFormat(const std::string& msg) : std::runtime_error{msg} {}
    ~InvalidShapesFormat() override;
};
InvalidShapesFormat::~InvalidShapesFormat() = default;

struct ShapePoint {
  const size_t id;
  const datatype lat, lon;
  const size_t seq;

  struct Shape {
    utl::csv_col<utl::cstr, UTL_NAME("shape_id")> id;
    utl::csv_col<utl::cstr, UTL_NAME("shape_pt_lat")> lat;
    utl::csv_col<utl::cstr, UTL_NAME("shape_pt_lon")> lon;
    utl::csv_col<utl::cstr, UTL_NAME("shape_pt_sequence")> seq;
  };

  // template <typename T>
  // struct CsvIterator {
  //   CsvIterator(T&& t) : t_{std::move(t)} {}
  //   auto operator*() {
  //     return t_.value();
  //   }
  //   T t_;
  // }

  static constexpr std::string_view sep{","};

  static constexpr ShapePoint from_string(const std::string_view row) {
    auto to_string_view = [](auto item) { return std::string_view(item); };
    auto item_view = std::views::split(row, sep)
        | std::views::transform(to_string_view)
    ;
    std::vector<std::string_view> items{item_view.begin(), item_view.end()};
    if (items.size() < 4 || items.size() > 5) {
      throw InvalidShapesFormat{std::format("Invalid format in row '{}'", row)};
    }
    assert(items.size() == 4);

    return ShapePoint{
        std::stoul(items[0].data()),
        double_to_fix(std::stod(items[1].data())),
        double_to_fix(std::stod(items[2].data())),
        std::stoul(items[3].data()),
    };
  }

  static constexpr ShapePoint from_shape(const Shape& shape) {
    return ShapePoint{
        std::stoul(shape.id->data()),
        double_to_fix(std::stod(shape.lat->data())),
        double_to_fix(std::stod(shape.lon->data())),
        std::stoul(shape.seq->data()),
    };
  }

  operator std::string() const {
    return std::format("ShapePoint(id={}, {}, {}, seq={})", id, lat, lon, seq);
  }
};

auto read_lines(auto& data_source) {

  auto filter_cr = [](const char c) { return c != '\r'; };
  auto pred = [](const char curr, const char next) { return curr != '\n' and next != '\n'; };
  auto not_empty = [](const auto& view) { return view.begin() != view.end(); };
  auto starts_with_int = [](const auto& view) {
      char first = view.front();
      return first >= '0' && first <= '9';
  };
  auto join = [](const auto& view) { return std::string{view.begin(), view.end()}; };

  return data_source
      | std::views::filter(filter_cr)
      | std::views::chunk_by(pred)
      | std::views::transform(join)
      | std::views::filter(not_empty)
      | std::views::filter(starts_with_int)
      | std::views::take(200'000)
      | std::views::transform(ShapePoint::from_string);
}

void progress_lines(const auto& data_source, auto func) {
  utl::noop_progress_consumer x;
  utl::line_range(utl::make_buf_reader(data_source.data(), std::move(x)))
    | utl::csv<ShapePoint::Shape>()
    | utl::transform([](const ShapePoint::Shape& s){ return ShapePoint::from_shape(s); })
    | utl::for_each(func)
  ;
}

auto get_cache(cista::mmap::protection mode) {
  return mm_vecvec<std::size_t, datatype>{
      cista::basic_mmap_vec<datatype, std::size_t>{cista::mmap{std::format(cache_file_template, "values").data(), mode}},
      cista::basic_mmap_vec<std::size_t, std::size_t>{cista::mmap{std::format(cache_file_template, "metadata").data(), mode}}
  };
}

auto get_cache_writer() {
  return get_cache(cista::mmap::protection::WRITE);
}

int main() {
  // rlimit limit{50, RLIM_INFINITY};
  // std::cout << std::format("Limits applied: {}", ::setrlimit(RLIMIT_RSS, &limit)) << std::endl;

    cista::mmap_vec<char> shaped_data{cista::mmap{in_file.data(), cista::mmap::protection::READ}};
    auto cache = get_cache_writer();

    size_t last_id{0u};
    auto bucket = cache.add_back_sized(0u);

    auto store_entry =
        [&bucket, &cache, &last_id](const auto& x) {
          if(last_id == 0u) {
            last_id = x.id;
          } else if (last_id != x.id) {
            bucket = cache.add_back_sized(0u);
            last_id = x.id;
          }
          bucket.push_back(x.lat);
          bucket.push_back(x.lon);
        }
    ;
    constexpr bool custom{true};
    if (custom) {
      std::ranges::for_each(read_lines(shaped_data), store_entry);
    } else {
      progress_lines(shaped_data, store_entry);
    }

    std::cout << std::format("Added {} buckets", cache.size()) << std::endl;
    auto entries = std::accumulate(cache.begin(), cache.end(), 0u, [](auto count, auto b) { return count + b.size(); });
    std::cout << std::format("Number of entries: {}", entries);
    return 0;
}
