#include <filesystem>
#include <numeric>
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
  for (auto path : std::vector<std::filesystem::path>{
           paths.id_file, paths.shape_data_file, paths.shape_metadata_file}) {
    if (std::filesystem::exists(path)) {
      std::filesystem::remove(path);
    }
  }
}

ShapeMap::Paths get_paths(std::string base_path) {
  return {
      base_path + "-id.dat",
      base_path + "-shape-data.dat",
      base_path + "-shape-metadata.dat",
  };
}

TEST(gtfs, shapeConstruct_createData_canAccessData) {
  std::string shapes_data{
      R"("shape_id","shape_pt_lat","shape_pt_lon","shape_pt_sequence"
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
  auto paths{get_paths("shape-test-create")};
  const DataGuard guard{[&paths]() { cleanup_paths(paths); }};

  ShapeMap shapes(shapes_data, paths);

  std::vector<std::vector<geo::latlng>> shape_points{
      {
          {51.543652, 7.217830},
          {51.478609, 7.223275},
      },
      {
          {50.553822, 6.356876},
          {50.560999, 6.355028},
          {50.560999, 6.355028},
          {50.565724, 6.364605},
          {50.578249, 6.383394},
          {50.578249, 6.383394},
          {50.581956, 6.379866},
      },
  };
  EXPECT_EQ(2, shapes.size());
  EXPECT_TRUE(shapes.contains("243"));
  EXPECT_TRUE(shapes.contains("3105"));
  EXPECT_FALSE(shapes.contains("1234"));
  EXPECT_EQ(shape_points.at(0), shapes.at("243"));
  EXPECT_EQ(shape_points.at(1), shapes.at("3105"));
  size_t loop_count{}, loop_sum{};
  for (const auto shape : shapes) {
    // Reminder: Internal order can be random
    EXPECT_TRUE(shape == shape_points.at(0) || shape == shape_points.at(1));
    ++loop_count;
    loop_sum += shape.size();
  }
  EXPECT_EQ(2, loop_count);
  EXPECT_EQ(9, loop_sum);
  auto points_total =
      std::accumulate(shapes.begin(), shapes.end(), 0u,
                      [](auto sum, auto shape) { return sum + shape.size(); });
  EXPECT_EQ(9, points_total);
}

TEST(gtfs, shapeConstruct_storeAndLoadData_canAccessData) {
  std::string shapes_data{
      R"("shape_id","shape_pt_lat","shape_pt_lon","shape_pt_sequence"
120,50.769767,6.073793,53
3104,50.553822,6.356876,0
3104,50.560999,6.355028,1
3104,50.560999,6.355028,2
3104,50.568805,6.374001,6
3104,50.578249,6.383394,7
3104,50.578249,6.383394,8
3104,50.581956,6.379866,11
3104,50.581956,6.379866,12
3104,50.589090,6.378158,14
3104,50.584129,6.372146,15
3104,50.585341,6.364319,17
3104,50.585341,6.364319,18
3104,50.584388,6.361445,20
3104,50.581905,6.353209,25
138,51.256676,7.166106,3261
137,50.767436,6.089977,4
137,51.194829,6.521109,988
)"};
  auto paths{get_paths("shape-test-store-and-reload")};
  const DataGuard guard{[&paths]() { cleanup_paths(paths); }};

  // Store only
  ShapeMap::write_shapes(shapes_data, paths);
  // Restore only
  ShapeMap shapes(paths);

  std::vector<std::string> ids{"120", "3104", "138", "137"};
  std::vector<std::vector<geo::latlng>> shape_points{{
                                                         {50.769767, 6.073793},
                                                     },
                                                     {
                                                         {50.553822, 6.356876},
                                                         {50.560999, 6.355028},
                                                         {50.560999, 6.355028},
                                                         {50.568805, 6.374001},
                                                         {50.578249, 6.383394},
                                                         {50.578249, 6.383394},
                                                         {50.581956, 6.379866},
                                                         {50.581956, 6.379866},
                                                         {50.589090, 6.378158},
                                                         {50.584129, 6.372146},
                                                         {50.585341, 6.364319},
                                                         {50.585341, 6.364319},
                                                         {50.584388, 6.361445},
                                                         {50.581905, 6.353209},
                                                     },
                                                     {
                                                         {51.256676, 7.166106},
                                                     },
                                                     {
                                                         {50.767436, 6.089977},
                                                         {51.194829, 6.521109},
                                                     }};
  EXPECT_EQ(ids.size(), shapes.size());
  for (auto [pos, id] : std::ranges::enumerate_view(ids)) {
    EXPECT_TRUE(shapes.contains(id));
    EXPECT_EQ(shape_points.at(static_cast<size_t>(pos)), shapes.at(id));
  }
}

TEST(gtfs, shapeParse_validIDs_parseData) {
  std::string shapes_data{
      R"("shape_id","shape_pt_lat","shape_pt_lon","shape_pt_sequence"
test id,50.553822,6.356876,0
----,50.560999,6.355028,1
)"
      "\x07\x13\x41\x08"
      R"(,50.560999,6.355028,2
ルーティング,50.565724,6.364605,3
,50.565724,6.364605,4
🚀,51.543652,7.217830,0
🚏,51.478609,7.223275,1
)"};
  auto paths{get_paths("shape-test-valid-ids")};
  const DataGuard guard{[&paths]() { cleanup_paths(paths); }};

  ShapeMap shapes(shapes_data, paths);

  EXPECT_EQ(7, shapes.size());
  EXPECT_TRUE(shapes.contains("test id"));
  EXPECT_TRUE(shapes.contains("----"));
  EXPECT_TRUE(shapes.contains("\x07\x13\x41\x08"));
  EXPECT_TRUE(shapes.contains("ルーティング"));
  EXPECT_TRUE(shapes.contains(""));
  EXPECT_TRUE(shapes.contains("🚀"));
  EXPECT_TRUE(shapes.contains("🚏"));
}

TEST(gtfs, shapeParse_randomColumOrder_parseCorrectly) {
  std::string shapes_data{
      R"("shape_pt_sequence","shape_pt_lon","shape_id","shape_pt_lat"
6,6.089410,123,50.767212
74,6.074227,123,50.775187
230,6.094470,123,50.871905
277,6.070844,123,50.890206
339,6.023209,123,50.896437
367,5.995949,123,50.890583
410,5.978670,123,50.890088
481,5.909033,123,50.879289
663,5.705982,123,50.849446
721,5.716989,123,50.838980
)"};
  auto paths{get_paths("shape-test-random-column-order")};
  const DataGuard guard{[&paths]() { cleanup_paths(paths); }};

  ShapeMap shapes(shapes_data, paths);

  EXPECT_EQ(1, shapes.size());
  EXPECT_TRUE(shapes.contains("123"));
  EXPECT_EQ(10, shapes.at("123").size());
}

TEST(gtfs, shapeParse_notAscendingSequence_throwException) {
  std::string shapes_data{
      R"("shape_id","shape_pt_lat","shape_pt_lon","shape_pt_sequence"
1,50.636512,6.473487,1
1,50.636259,6.473668,0
)"};
  auto paths{get_paths("shape-test-not-ascending-sequence")};
  const DataGuard guard{[&paths]() { cleanup_paths(paths); }};

  EXPECT_THROW(ShapeMap shapes(shapes_data, paths), InvalidShapesFormat);
}

// // Currently not testable
// TEST(gtfs, shapeParse_missingColumn_throwException) {
//     std::string shapes_data{R"("shape_id","shape_pt_lat","shape_pt_sequence"
// 1,50.636259,0
// )"};
//     auto paths{get_paths("shape-test-missing-column")};
//     const DataGuard guard{[&paths]() { cleanup_paths(paths); }};

//     EXPECT_THROW(ShapeMap shapes(shapes_data, paths), InvalidShapesFormat);
// }

TEST(gtfs, shapeParse_shuffledRows_parseAllData) {
  std::string shapes_data{
      R"("shape_id","shape_pt_lat","shape_pt_lon","shape_pt_sequence"
234,51.473214,7.139521,0
241,51.504903,7.102455,0
241,51.473214,7.139521,1
243,51.543652,7.217830,0
244,51.473214,7.139521,0
244,51.504903,7.102455,1
243,51.478609,7.223275,1
235,51.478609,7.223275,0
234,51.459894,7.153535,1
240,51.459894,7.153535,0
240,51.473214,7.139521,1
235,51.543652,7.217830,1
)"};
  auto paths{get_paths("shape-test-shuffled-rows")};
  const DataGuard guard{[&paths]() { cleanup_paths(paths); }};

  ShapeMap shapes(shapes_data, paths);

  std::unordered_map<std::string, std::vector<geo::latlng>> shape_points{
      {"240",
       {
           {51.459894, 7.153535},
           {51.473214, 7.139521},
       }},
      {"234",
       {
           {51.473214, 7.139521},
           {51.459894, 7.153535},
       }},
      {"244",
       {
           {51.473214, 7.139521},
           {51.504903, 7.102455},
       }},
      {"235",
       {
           {51.478609, 7.223275},
           {51.543652, 7.217830},
       }},
      {"241",
       {
           {51.504903, 7.102455},
           {51.473214, 7.139521},
       }},
      {"243",
       {
           {51.543652, 7.217830},
           {51.478609, 7.223275},
       }},
  };
  EXPECT_EQ(shape_points.size(), shapes.size());
  for (auto [id, coordinates] : shape_points) {
    EXPECT_TRUE(shapes.contains(id));
    EXPECT_EQ(coordinates, shapes.at(id));
  }
}

TEST(gtfs, shapeParse_delayedInsertWithNotAscendingSequence_throwException) {
  std::string shapes_data{
      R"("shape_id","shape_pt_lat","shape_pt_lon","shape_pt_sequence"
1,50.636512,6.473487,1
2,51.473214,7.139521,0
1,50.636259,6.473668,0
)"};
  auto paths{get_paths("shape-test-not-ascending-sequence")};
  const DataGuard guard{[&paths]() { cleanup_paths(paths); }};

  EXPECT_THROW(ShapeMap shapes(shapes_data, paths), InvalidShapesFormat);
}