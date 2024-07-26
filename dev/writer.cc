#include <cista/mmap.h>
#include <cstddef>
#include <algorithm>
#include <fstream>
#include <ios>
#include <ranges>
#include <iostream>
#include <format>
#include <vector>

#include "cista/containers/mmap_vec.h"

#include "shared.h"

const std::string in_file{"../dev/shapes.txt"};

struct ShapePoint {
    const size_t id;
    const int32_t lat, lon;
    const size_t seq;
    
    static constexpr std::string_view sep{","};
    
    static ShapePoint from_string(const std::string_view row) {
        std::cout << std::format("Raw: '{}'", row) << std::endl;
        auto item_view = std::views::split(row, sep) | std::views::transform([](auto item) { return std::string_view(item); });
        std::vector<std::string_view> items{item_view.begin(), item_view.end()};
        assert(items.size() == 4);
        for (auto& i : items) {
            std::cout << std::format("Item: '{}'", i) << std::endl;
        }
        return ShapePoint{
            std::stoul(items[0].data()),
            double_to_fix(std::stod(items[1].data())),
            double_to_fix(std::stod(items[2].data())),
            std::stoul(items[3].data()),
        };
        // for (const auto item : std::views::split(row, sep)) {
        // // auto view = std::views::split(row, sep); // | std::ranges::to<std::vector>();
        // // for (const auto& item : view) {
        //     // std::cout << std::format("Got item: '{}'", item) << std::endl;
        //     std::cout << std::format("Got item: '{}'", std::string_view(item)) << std::endl;
        // }
        // return ShapePoint{0u, 0, 0, 0u};
        // std::vector<std::string> items;
        // items.insert()
        // items.insert(items.end(), view.cbegin(), view.cend());
        // return ShapePoint{
        //     view
        // }
    }
        
    operator std::string() const {
        return std::format("ShapePoint(id={}, {}, {}, seq={})", id, lat, lon, seq);
    }
};

int main() {
    cista::mmap_vec<char> in{cista::mmap{in_file.data(), cista::mmap::protection::READ}};
    auto filter_cr = [](const char c) { return c != '\r'; };
    auto pred = [](const char curr, const char next) { return curr != '\n' and next != '\n'; };
    auto not_empty = [](const auto& view) { return view.begin() != view.end(); };
    auto starts_with_int = [](const auto& view) {
        char first = view.front();
        return first >= '0' && first <= '9';
    };
    auto join = [](const auto& view) { return std::string{view.begin(), view.end()}; };
    std::ranges::for_each(
        in
            | std::views::filter(filter_cr)
            | std::views::chunk_by(pred)
            | std::views::filter(not_empty)
            | std::views::filter(starts_with_int)
            | std::views::transform(join)
            | std::views::take(3)
            | std::views::transform(ShapePoint::from_string)
        ,
        [](const auto& x) { std::cout << static_cast<std::string>(x) << std::endl; }
    );
    return 0;
}