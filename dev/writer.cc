#include <cista/mmap.h>
#include <cstddef>
#include <algorithm>
#include <exception>
#include <fstream>
#include <ios>
#include <ranges>
#include <iostream>
#include <format>
#include <stdexcept>
#include <vector>

#include "cista/mmap.h"

#include "shared.h"

const std::string in_file{"../dev/shapes.txt"};

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
  const int32_t lat, lon;
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
        | std::views::take(3)
        | std::views::transform(ShapePoint::from_string)
    ;
}

int main() {
    cista::mmap_vec<char> shaped_data{cista::mmap{in_file.data(), cista::mmap::protection::READ}};

    std::ranges::for_each(
        read_lines(shaped_data),
        [](const auto& x) { std::cout << static_cast<std::string>(x) << std::endl; }
    );
    return 0;
}
