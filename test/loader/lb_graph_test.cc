#include "gtest/gtest.h"

#include "nigiri/loader/hrd/load_timetable.h"

#include "hrd/hrd_timetable.h"

using namespace nigiri;
using namespace nigiri::loader::hrd;
using namespace nigiri::test_data::hrd_timetable;

TEST(lb_graph, distances_check) {
  auto tt = timetable{};
  tt.date_range_ = full_period();
  load_timetable(source_idx_t{0U}, hrd_5_20_26, files_simple(), tt);

  std::stringstream ss;
  for (auto i = location_idx_t{0U}; i != tt.locations_.ids_.size(); ++i) {
    if (tt.fwd_search_lb_graph_.at(i).empty()) {
      continue;
    }
    ss << location{tt, i} << "\n";
    for (auto const& fp : tt.fwd_search_lb_graph_[i]) {
      ss << "  " << location{tt, fp.target_} << " " << fp.duration_.count()
         << "\n";
    }
  }

  constexpr auto const raw = R"((A, 0000001)
  (B, 0000002) 60
(B, 0000002)
  (A, 0000001) 60
)";

  EXPECT_EQ(std::string_view{raw}, ss.str());
}
