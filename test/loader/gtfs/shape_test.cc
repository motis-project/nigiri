#include <iostream>
#include <optional>
#include <sstream>
#include <vector>

#include "gtest/gtest.h"

#include "geo/polyline.h"

#include "utl/raii.h"

#include "./shape_test.h"

#include "./test_data.h"

using namespace nigiri::loader::gtfs;

const std::string_view shapes_data_aachen{
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

const std::unordered_map<std::string, shape::value_type> shape_points_aachen{
    {"243",
     {
         {51.543652, 7.217830},
         {51.478609, 7.223275},
     }},
    {"3105",
     {
         {50.553822, 6.356876},
         {50.560999, 6.355028},
         {50.560999, 6.355028},
         {50.565724, 6.364605},
         {50.578249, 6.383394},
         {50.578249, 6.383394},
         {50.581956, 6.379866},
     }},
};

TEST(gtfs, shapeBuilder_withoutData_getNull) {
  auto builder = shape::get_builder();

  auto shape = builder("1");
  EXPECT_EQ(std::nullopt, shape);
}

TEST(gtfs, shapeBuilder_withData_getExistingShapePoints) {
  auto [mmap, paths] = create_temporary_paths("shape-test-builder");
  auto guard = utl::make_raii(paths, cleanup_paths);

  auto builder = shape::get_builder(shapes_data_aachen, &mmap);

  auto shape_not_existing = builder("1");
  auto shape_243 = builder("243");
  auto shape_3105 = builder("3105");

  EXPECT_EQ(std::nullopt, shape_not_existing);
  EXPECT_TRUE(shape_243.has_value());
  EXPECT_TRUE(shape_3105.has_value());
  EXPECT_EQ(shape_points_aachen.at("243"), shape_243.value()());
  EXPECT_EQ(shape_points_aachen.at("3105"), shape_3105.value()());
}

TEST(gtfs, shapeGet_unusualShapeIds_getAllIds) {
  using std::literals::operator""s;
  std::string shapes_data{
      R"("shape_id","shape_pt_lat","shape_pt_lon","shape_pt_sequence"
test id,50.553822,6.356876,0
----,50.560999,6.355028,1
)"
      "\x07\x13\x41\x08"
      R"(,50.560999,6.355028,2
ルーティング,50.565724,6.364605,3
,50.565724,6.364605,4
)"
      "\0"s
      R"(,50.578249,6.383394,7
🚀,51.543652,7.217830,0
🚏,51.478609,7.223275,1
)"};
  auto [mmap, paths] = create_temporary_paths("shape-test-unicode-ids");
  auto guard = utl::make_raii(paths, cleanup_paths);

  auto builder = shape::get_builder(shapes_data, &mmap);

  std::vector<std::string> ids{"test id"s,      "----"s, "\x07\x13\x41\x08"s,
                               "ルーティング"s, ""s,     "\0"s,
                               "🚀"s,           "🚏"s};
  for (auto const& id : ids) {
    auto shape = builder(id);
    EXPECT_TRUE(shape.has_value());
    EXPECT_EQ(1, (*shape)().size());
  }
}

TEST(gtfs, shapeParse_notAscendingSequence_progressAndLogError) {
  std::string shapes_data{
      R"("shape_id","shape_pt_lat","shape_pt_lon","shape_pt_sequence"
1,50.636512,6.473487,1
1,50.636259,6.473668,0
)"};
  auto [mmap, paths] =
      create_temporary_paths("shape-test-not-ascending-sequence");
  auto guard = utl::make_raii(paths, cleanup_paths);
  std::stringstream buffer{};
  auto backup = std::clog.rdbuf(buffer.rdbuf());
  auto buffer_guard = utl::make_raii(
      backup, [](const decltype(backup)& buf) { std::clog.rdbuf(buf); });

  auto builder = shape::get_builder(shapes_data, &mmap);

  shape::value_type shape_points{{50.636512, 6.473487}, {50.636259, 6.473668}};
  std::clog.flush();
  std::string_view log{buffer.str()};
  auto shape = builder("1");
  EXPECT_TRUE(shape.has_value());
  EXPECT_EQ(shape_points, shape.value()());
  EXPECT_TRUE(
      log.contains("Non monotonic sequence for shape_id '1': Sequence number 1 "
                   "followed by 0"));
}

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
  auto [mmap, paths] = create_temporary_paths("shape-test-shuffled-rows");
  auto guard = utl::make_raii(paths, cleanup_paths);

  auto builder = shape::get_builder(shapes_data, &mmap);

  std::unordered_map<std::string, geo::polyline> shape_points{
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
  for (auto [id, coordinates] : shape_points) {
    auto shape = builder(id);
    EXPECT_TRUE(shape.has_value());
    EXPECT_EQ(coordinates, (*shape)());
  }
}

TEST(gtfs,
     shapeParse_delayedInsertWithNotAscendingSequence_progressAndLogError) {
  std::string shapes_data{
      R"("shape_id","shape_pt_lat","shape_pt_lon","shape_pt_sequence"
1,50.636512,6.473487,1
2,51.473214,7.139521,0
1,50.636259,6.473668,0
)"};
  auto [mmap, paths] =
      create_temporary_paths("shape-test-not-ascending-sequence");
  auto guard = utl::make_raii(paths, cleanup_paths);
  std::stringstream buffer{};
  auto backup = std::clog.rdbuf(buffer.rdbuf());
  auto buffer_guard = utl::make_raii(
      backup, [](const decltype(backup)& buf) { std::clog.rdbuf(buf); });

  auto builder = shape::get_builder(shapes_data, &mmap);

  std::clog.flush();
  std::string_view log{buffer.str()};
  EXPECT_TRUE(builder("1").has_value());
  EXPECT_TRUE(builder("2").has_value());
  EXPECT_TRUE(
      log.contains("Non monotonic sequence for shape_id '1': Sequence number 1 "
                   "followed by 0"));
}