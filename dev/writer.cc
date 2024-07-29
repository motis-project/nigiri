#include <cista/containers/mmap_vec.h>
#include <cista/mmap.h>
#include <cstddef>
#include <algorithm>
#include <exception>
#include <fstream>
#include <ios>
#include <numeric>
#include <ranges>
#include <iostream>
#include <format>
#include <stdexcept>
#include <vector>

// #include <sys/resource.h>

#include "cista/mmap.h"

#include "shared.h"

const std::string in_file{"../dev/shapes.txt"};
constexpr std::string_view cache_file_template{"shape-cache.{}.dat"};

using datatype = int32_t;

class InvalidShapesFormat final : public std::runtime_error {
public:
    // InvalidShapesFormat(const std::string& msg);
    InvalidShapesFormat(const std::string& msg) : std::runtime_error{msg} {}
    ~InvalidShapesFormat() override;
    // ~InvalidShapesFormat() override = default;
    // ~InvalidShapesFormat() override = {};
    // virtual const char * what() const override {
    //     return std::runtime_error::what();
    // }
};
InvalidShapesFormat::~InvalidShapesFormat() = default;

struct ShapePoint {
  const size_t id;
  const datatype lat, lon;
  const size_t seq;

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
        | std::views::filter(not_empty)
        | std::views::filter(starts_with_int)
        | std::views::transform(join)
        // | std::views::take(1'000'000)
        | std::views::transform(ShapePoint::from_string)
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

    std::ranges::for_each(
        read_lines(shaped_data),
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
    );
    std::cout << std::format("Added {} buckets", cache.size()) << std::endl;
    auto entries = std::accumulate(cache.begin(), cache.end(), 0u, [](auto count, auto b) { return count + b.size(); });
    std::cout << std::format("Number of entries: {}", entries);
    return 0;
}