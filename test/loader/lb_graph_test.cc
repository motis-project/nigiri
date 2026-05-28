#include "gtest/gtest.h"

#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/loader/init_finish.h"

#include "hrd/hrd_timetable.h"

using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::hrd;
using namespace nigiri::test_data::hrd_timetable;

TEST(routing, lb_graph_distances_check) {
  auto tt = timetable{};
  tt.date_range_ = full_period();
  load_timetable(source_idx_t{0U}, hrd_5_20_26, files_simple(), tt);
  finalize(tt);

  using distance_table_t =
      std::map<std::pair<std::string, std::string>, duration_t>;
  auto distances = distance_table_t{};
  for (auto i = location_idx_t{0U}; i != tt.locations_.ids_.size(); ++i) {
    if (tt.fwd_search_lb_graph_.at(kDefaultProfile).at(i).empty()) {
      continue;
    }
    for (auto const& fp : tt.fwd_search_lb_graph_.at(kDefaultProfile)[i]) {
      distances[{std::string{tt.get_default_name(i)},
                 std::string{tt.get_default_name(fp.target())}}] =
          fp.duration();
    }
  }

  EXPECT_EQ(
      (distance_table_t{{{"A", "B"}, 60_minutes}, {{"B", "A"}, 60_minutes}}),
      distances);
}
