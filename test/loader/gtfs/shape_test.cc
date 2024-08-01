#include <cstdio>
#include <filesystem>
#include <ranges>
#include <vector>

#include "gtest/gtest.h"

#include "geo/latlng.h"

#include "nigiri/loader/gtfs/shape.h"

// #include "./test_data.h"

using namespace nigiri::loader::gtfs;

struct DataGuard {
    DataGuard(const std::function<void()> f) : f_(f) {}
    ~DataGuard() { f_(); }
    const std::function<void()> f_;
};

void cleanup_paths(const ShapeMap::Paths& paths) {
    for (auto path : std::vector<std::filesystem::path>{paths.id_file, paths.shape_data_file, paths.shape_metadata_file}) {
        if (std::filesystem::exists(path)) {
            std::filesystem::remove(path);
        }
    }
}

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
    // std::string base_path{std::tmpnam(nullptr)};
    std::string base_path{"shape-test-create"};
    ShapeMap::Paths paths{
        base_path + "-id.dat",
        base_path + "-shape-data.dat",
        base_path + "-shape-metadata.dat",
    };
    const DataGuard guard{[&paths]() { cleanup_paths(paths); }};

    ShapeMap shapes(shapes_data, paths);

    std::vector<std::vector<geo::latlng>> shape_points{
        {
            {51.543652,7.217830},
            {51.478609,7.223275},
        },
        {
            {50.553822,6.356876},
            {50.560999,6.355028},
            {50.560999,6.355028},
            {50.565724,6.364605},
            {50.578249,6.383394},
            {50.578249,6.383394},
            {50.581956,6.379866},
        },
    };
    EXPECT_EQ(2, shapes.size());
    EXPECT_TRUE(shapes.contains("243"));
    EXPECT_TRUE(shapes.contains("3105"));
    EXPECT_FALSE(shapes.contains("1234"));
    EXPECT_EQ(shape_points.at(0), shapes.at("243"));
    EXPECT_EQ(shape_points.at(1), shapes.at("3105"));
}