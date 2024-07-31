#include <format>
#include <iostream>
#include <vector>

#include "cista/containers/mmap_vec.h"

#include "shared.h"

auto load_ids() {
    auto mapper = get_map_reader();
    id_map_type ids{};
    std::string_view v{mapper};
    size_t start{0u}, end;
    while ((end = v.find('\0', start)) != v.npos) {
        // std::cout << "Debug: " << start << " " << end << std::endl;
        ids.push_back(id_type{v.substr(start, end - start)});
        start = end + 1;
    }
    // Testing read data
    for (auto pos : std::vector<size_t>{0u, 1u, 756, 2763, 3659}) {
        std::cout << std::format("Position {:>5} contains {}", pos, ids.at(pos)) << std::endl;
    }
    std::cout << std::format("Total items: {}", ids.size()) << std::endl;

    return ids;
}

auto load_shape_data() {
    return get_cache_reader();
}

void test_id(std::string id, auto const& id_map, auto const& shapes) {
    for (auto x : shapes.at(id_map.at(id))) {
        std::cout << x << ", " << std::endl;
    }
}

void test_id_entries(std::vector<std::string> ids, auto const& id_map, auto const& shapes) {
    for (auto id : ids) {
        std::cout << std::format("Found {:>4} entries for id {}", shapes.at(id_map.at(id)).size(), id) << std::endl;
    }
}

int main() {
    auto ids = load_ids();
    auto id_map = vec_to_map(ids);
    auto shapes = load_shape_data();
    test_id("243", id_map, shapes);
    test_id_entries({"243", "1844", "450", "1771", "202"}, id_map, shapes);
    return 0;
}