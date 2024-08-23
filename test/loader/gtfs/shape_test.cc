#include <ranges>
#include <sstream>
#include <vector>

#include "gtest/gtest.h"

#include "geo/polyline.h"

#include "utl/raii.h"

#include "nigiri/loader/gtfs/shape.h"

#include "./shape_test.h"

using namespace nigiri::loader::gtfs;

void assert_polyline_eq(geo::polyline const& line1,
                        geo::polyline const& line2) {
  EXPECT_EQ(line1.size(), line2.size());
  for (auto [p1, p2] : std::views::zip(line1, line2)) {
    EXPECT_DOUBLE_EQ(p1.lat(), p2.lat());
    EXPECT_DOUBLE_EQ(p1.lng(), p2.lng());
  }
}

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

const std::unordered_map<std::string, geo::polyline> shape_points_aachen{
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
  auto const shapes = parse_shapes("", nullptr);

  auto const index_it = shapes.find("1");
  EXPECT_EQ(shapes.end(), index_it);
}

TEST(gtfs, shapeBuilder_withData_getExistingShapePoints) {
  auto mmap = shape_test_mmap{"shape-test-builder"};
  auto& vecvec = mmap.get_vecvec();

  auto const shapes = parse_shapes(shapes_data_aachen, &vecvec);

  auto const shape_not_existing_it = shapes.find("1");
  auto const shape_243_it = shapes.find("243");
  auto const shape_3105_it = shapes.find("3105");

  EXPECT_EQ(shapes.end(), shape_not_existing_it);
  EXPECT_NE(shapes.end(), shape_243_it);
  EXPECT_NE(shapes.end(), shape_3105_it);
  auto const shape_243 = vecvec[shape_243_it->second];
  auto const shape_3105 = vecvec[shape_3105_it->second];
  assert_polyline_eq(shape_points_aachen.at("243"),
                     geo::polyline{shape_243.begin(), shape_243.end()});
  assert_polyline_eq(shape_points_aachen.at("3105"),
                     geo::polyline{shape_3105.begin(), shape_3105.end()});
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
  auto mmap = shape_test_mmap{"shape-test-unicode-ids"};
  auto& vecvec = mmap.get_vecvec();

  auto const shapes = parse_shapes(shapes_data, &vecvec);

  std::vector<std::string> ids{"test id"s,      "----"s, "\x07\x13\x41\x08"s,
                               "ルーティング"s, ""s,     "\0"s,
                               "🚀"s,           "🚏"s};
  for (auto const& id : ids) {
    auto shape_it = shapes.find(id);
    EXPECT_NE(shapes.end(), shape_it);
    EXPECT_EQ(1, vecvec[shape_it->second].size());
  }
}

TEST(gtfs, shapeParse_notAscendingSequence_progressAndLogError) {
  std::string shapes_data{
      R"("shape_id","shape_pt_lat","shape_pt_lon","shape_pt_sequence"
1,50.636512,6.473487,1
1,50.636259,6.473668,0
)"};
  auto mmap = shape_test_mmap{"shape-test-not-ascending-sequence"};
  auto& vecvec = mmap.get_vecvec();
  std::stringstream buffer{};
  auto backup = std::clog.rdbuf(buffer.rdbuf());
  auto buffer_guard = utl::make_raii(
      backup, [](const decltype(backup)& buf) { std::clog.rdbuf(buf); });

  auto const shapes = parse_shapes(shapes_data, &vecvec);

  auto const shape_points =
      geo::polyline{{50.636512, 6.473487}, {50.636259, 6.473668}};
  std::clog.flush();
  std::string log{buffer.str()};
  auto const shape_it = shapes.find("1");
  EXPECT_NE(shapes.end(), shape_it);
  auto const shape = vecvec[shape_it->second];
  assert_polyline_eq(shape_points, geo::polyline{shape.begin(), shape.end()});
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
  auto mmap = shape_test_mmap{"shape-test-shuffled-rows"};
  auto& vecvec = mmap.get_vecvec();

  auto const shapes = parse_shapes(shapes_data, &vecvec);

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
  for (auto [id, polyline] : shape_points) {
    auto const shape_it = shapes.find(id);
    EXPECT_NE(shapes.end(), shape_it);
    auto const shape = vecvec[shape_it->second];
    assert_polyline_eq(polyline, geo::polyline{shape.begin(), shape.end()});
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
  auto mmap = shape_test_mmap{"shape-test-not-ascending-sequence"};
  std::stringstream buffer{};
  auto backup = std::clog.rdbuf(buffer.rdbuf());
  auto buffer_guard = utl::make_raii(
      backup, [](const decltype(backup)& buf) { std::clog.rdbuf(buf); });

  auto const shapes = parse_shapes(shapes_data, &mmap.get_vecvec());

  std::clog.flush();
  std::string log{buffer.str()};
  EXPECT_NE(shapes.find("1"), shapes.end());
  EXPECT_NE(shapes.find("2"), shapes.end());
  EXPECT_TRUE(
      log.contains("Non monotonic sequence for shape_id '1': Sequence number 1 "
                   "followed by 0"));
}