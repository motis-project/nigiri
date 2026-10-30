#include "gtest/gtest.h"

#include <filesystem>
#include <random>
#include <ranges>
#include <sstream>
#include <vector>

#include "fmt/std.h"

#include "geo/polyline.h"

#include "nigiri/loader/gtfs/shape.h"
#include "nigiri/common/span_cmp.h"
#include "nigiri/shapes_storage.h"

namespace fs = std::filesystem;
using namespace nigiri;
using namespace nigiri::loader::gtfs;

namespace {

// Unique per run: fixed names in the shared temp dir collide with
// directories left behind by other users (e.g. on self-hosted CI runners).
struct temp_dir {
  explicit temp_dir(std::string_view name)
      : path_{fs::temp_directory_path() /
              fmt::format("{}-{:x}", name, std::random_device{}())} {}
  temp_dir(temp_dir const&) = delete;
  temp_dir& operator=(temp_dir const&) = delete;
  ~temp_dir() {
    auto ec = std::error_code{};
    fs::remove_all(path_, ec);
  }
  fs::path path_;
};

}  // namespace

TEST(gtfs, shape_get_existing_shape_points) {
  constexpr auto const kShapesData =
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
)";

  auto const dir = temp_dir{"shape-test-builder"};
  auto shapes_data = shapes_storage{dir.path_, cista::mmap::protection::WRITE};
  auto const shape_states = parse_shapes(kShapesData, shapes_data);
  auto const& shapes = shape_states.id_map_;

  EXPECT_EQ(end(shapes), shapes.find("1"));

  EXPECT_EQ((geo::polyline{
                {51.543652, 7.217830},
                {51.478609, 7.223275},
            }),
            shapes_data.get_shape(to_scoped_shape_idx(
                shapes.at("243"), shape_source::kTimetable)));

  EXPECT_EQ((geo::polyline{
                {50.553822, 6.356876},
                {50.560999, 6.355028},
                {50.560999, 6.355028},
                {50.565724, 6.364605},
                {50.578249, 6.383394},
                {50.578249, 6.383394},
                {50.581956, 6.379866},
            }),
            shapes_data.get_shape((to_scoped_shape_idx(
                shapes.at("3105"), shape_source::kTimetable))));
}

TEST(gtfs, shape_not_ascending_sequence) {
  constexpr auto const kShapesData =
      R"("shape_id","shape_pt_lat","shape_pt_lon","shape_pt_sequence"
1,50.636512,6.473487,1
1,50.636259,6.473668,0
)";

  auto const dir = temp_dir{"shape-test-not-ascending-sequence"};
  auto shapes_data = shapes_storage{dir.path_, cista::mmap::protection::WRITE};
  auto const shape_states = parse_shapes(kShapesData, shapes_data);
  auto const& shapes = shape_states.id_map_;

  EXPECT_EQ((geo::polyline{{50.636259, 6.473668}, {50.636512, 6.473487}}),
            shapes_data.get_shape((to_scoped_shape_idx(
                shapes.at("1"), shape_source::kTimetable))));
}

TEST(gtfs, shape_shuffled_rows) {
  constexpr auto const kShapesData =
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
)";

  auto const dir = temp_dir{"shape-test-shuffled-rows"};
  auto shapes_data = shapes_storage{dir.path_, cista::mmap::protection::WRITE};
  auto const shape_states = parse_shapes(kShapesData, shapes_data);
  auto const& shapes = shape_states.id_map_;

  auto const shape_points =
      std::initializer_list<std::pair<std::string_view, geo::polyline>>{
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
    EXPECT_EQ(polyline, shapes_data.get_shape((to_scoped_shape_idx(
                            shapes.at(id), shape_source::kTimetable))));
  }
}

TEST(gtfs, shape_delay_insert_no_ascending_sequence) {
  constexpr auto const kShapesData =
      R"("shape_id","shape_pt_lat","shape_pt_lon","shape_pt_sequence"
1,50.636512,6.473487,1
2,51.473214,7.139521,0
1,50.636259,6.473668,0
)";
  auto const dir = temp_dir{"shape-test-not-ascending-sequence"};
  auto shapes_data = shapes_storage{dir.path_, cista::mmap::protection::WRITE};
  auto const shape_states = parse_shapes(kShapesData, shapes_data);
  auto const& shapes = shape_states.id_map_;

  auto const shape_idx = shapes.find("1");
  EXPECT_NE(shape_idx, end(shapes));
  EXPECT_NE(shapes.find("2"), end(shapes));
  EXPECT_EQ((geo::polyline{{50.636259, 6.473668}, {50.636512, 6.473487}}),
            shapes_data.get_shape((to_scoped_shape_idx(
                shape_idx->second, shape_source::kTimetable))));
}
