#include <ranges>

#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/shape.h"
#include <cstdio>
#include <filesystem>

// #include "./test_data.h"

using namespace nigiri::loader::gtfs;

TEST(gtfs, shapeImport_validData_storeToMap) {
    std::string shapes_data{R"("shape_id","shape_pt_lat","shape_pt_lon","shape_pt_sequence"
243,51.543652,7.217830,0
243,51.478609,7.223275,1
3105,50.553822,6.356876,0
3105,50.560999,6.355028,1
3105,50.560999,6.355028,2
3105,50.565724,6.364605,3
3105,50.578249,6.383394,7
3105,50.578249,6.383394,8
3105,50.581956,6.379866,11
)"};
    std::string memory_map_base{std::tmpnam(nullptr)};
    std::vector<std::string> paths{memory_map_base + ".coordinates", memory_map_base + ".metadata"};
    ShapePoint::MemoryMapConfig config{paths[0], paths[1]};
    auto shape_map = create_shape_memory_map(config, cista::mmap::protection::WRITE);

    shape_load_map(shapes_data, shape_map);

    EXPECT_EQ(2, shape_map.size());
    EXPECT_EQ(2, shape_map.at(0).size());
    EXPECT_EQ(7, shape_map.at(1).size());
    std::vector<ShapePoint::Coordinate> points{
        {515436520, 72178300},
        {514786090, 72232750},
        {505538220, 63568760},
        {505609990, 63550280},
        {505609990, 63550280},
        {505657240, 63646050},
        {505782490, 63833940},
        {505782490, 63833940},
        {505819560, 63798660},
    };
    for (auto pos : std::views::iota(0u, 2u)) {
        EXPECT_EQ(points[pos], shape_map.at(0).at(pos));
    }
    for (auto pos : std::views::iota(2u, points.size())) {
        EXPECT_EQ(points[pos], shape_map.at(1).at(pos - 2));
    }

    for (auto path : {config.coordinates_file, config.metadata_file}) {
        if (std::filesystem::exists(path)) {
            std::filesystem::remove(path);
        }
    }
}