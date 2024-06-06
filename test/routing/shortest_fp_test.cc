#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/timetable.h"

#include "../raptor_search.h"
#include "./test_data.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::routing;
using nigiri::test::raptor_intermodal_search;

TEST(routing, raptor_shortest_fp_forward) {
  constexpr auto const src = source_idx_t{0U};
  auto const config = loader_config{};

  timetable tt;
  tt.date_range_ = shortest_fp_period();
  register_special_stations(tt);
  gtfs::load_timetable(config, src, shortest_fp_files(), tt);
  finalize(tt);

  auto const results = raptor_intermodal_search(
      tt, nullptr,
      {{tt.locations_.location_id_to_idx_.at({.id_ = "A0", .src_ = src}),
        3_minutes, 23U},
       {tt.locations_.location_id_to_idx_.at({.id_ = "A1", .src_ = src}),
        5_minutes, 23U},
       {tt.locations_.location_id_to_idx_.at({.id_ = "A2", .src_ = src}),
        10_minutes, 23U}},
      {{tt.locations_.location_id_to_idx_.at({.id_ = "C3", .src_ = src}),
        30_minutes, 42U},
       {tt.locations_.location_id_to_idx_.at({.id_ = "C4", .src_ = src}),
        20_minutes, 42U},
       {tt.locations_.location_id_to_idx_.at({.id_ = "C5", .src_ = src}),
        10_minutes, 42U}},
      interval{unixtime_t{sys_days{2024_y / June / 8}} + 1_hours,
               unixtime_t{sys_days{2024_y / June / 8} + 3_hours}},
      direction::kForward);

  EXPECT_EQ(1U, results.size());

  std::stringstream ss;
  ss << "\n";
  for (auto const& x : results) {
    x.print(ss, tt);
    ss << "\n\n";
  }
  std::cout << ss.str() << "\n";
}