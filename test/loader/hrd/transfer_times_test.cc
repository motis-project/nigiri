#include "gtest/gtest.h"

#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/timetable.h"

#include "./hrd_timetable.h"

using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::hrd;
using namespace nigiri::test_data::hrd_timetable;

namespace {

// station E (0000005) is served by 01337 (ICE), 00815 (RE) and 03374 (IC).
// the default transfer time there is 2 minutes; the rule below asks for 15
// minutes between 01337 and 00815 only.
constexpr auto const umsteigb_content = R"(
9999999 02 02 STANDARD
0000005 02 02 E
)";

constexpr auto const umsteigz_content = R"(
0000005 001337 80____ 000815 80____ 015
)";

mem_dir with_transfer_rules(std::string_view umsteigb,
                            std::string_view umsteigz) {
  auto d = files();
  auto const& b = hrd_5_20_26.core_data_;
  d.dir_[(b / hrd_5_20_26.transfers_.station_[0]).string()] = umsteigb;
  d.dir_[(b / hrd_5_20_26.transfers_.trip_[1]).string()] = umsteigz;
  return d;
}

timetable load(mem_dir&& d) {
  auto tt = timetable{};
  tt.date_range_ = full_period();
  register_special_stations(tt);
  load_timetable(source_idx_t{0U}, hrd_5_20_26, d, tt);
  finalize(tt);
  return tt;
}

}  // namespace

TEST(hrd, transfer_rules_create_virtual_locations) {
  auto const tt = load(with_transfer_rules(umsteigb_content, umsteigz_content));

  auto const e = tt.find(location_id{"0000005", source_idx_t{0U}});
  ASSERT_TRUE(e.has_value());

  auto const& children = tt.locations_.children_[*e];
  // 01337 and 00815 behave differently towards each other, 03374 shares
  // 00815's behaviour, so E splits into exactly two classes.
  ASSERT_EQ(2U, children.size())
      << "umsteigz must split station E into two behaviour classes";
  for (auto const& c : children) {
    EXPECT_EQ(location_type::kVirt, tt.locations_.types_[c]);
    EXPECT_EQ(*e, tt.locations_.parents_[c]);
  }

  // exactly one cell deviates from the 2 minute default, the other direction
  // and the diagonal are left to the hubs
  auto edges = std::vector<footpath>{};
  for (auto const& c : children) {
    for (auto const& fp : tt.locations_.transfer_rule_fps_[c]) {
      edges.emplace_back(fp);
    }
  }
  ASSERT_EQ(1U, edges.size()) << "only the deviating cell must be materialized";
  EXPECT_EQ(15_minutes, edges.front().duration());
}

TEST(hrd, transfer_rules_without_rules_no_virtual_locations) {
  auto const tt = load(files());
  auto const e = tt.find(location_id{"0000005", source_idx_t{0U}});
  ASSERT_TRUE(e.has_value());
  EXPECT_TRUE(tt.locations_.children_[*e].empty())
      << "no umsteigz -> no behaviour classes";
}
