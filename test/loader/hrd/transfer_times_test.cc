#include "gtest/gtest.h"

#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/timetable.h"

#include "../../raptor_search.h"
#include "./hrd_timetable.h"

using namespace date;
using namespace std::chrono_literals;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::hrd;
using namespace nigiri::test_data::hrd_timetable;
using nigiri::test::raptor_search;

namespace {

// 01337 ends at E (0000005) at 11:00, 00815 leaves E at 11:05 for F, 03374
// at 11:06. The default transfer time at E is 2 minutes; the rule below asks
// for 15 minutes between 01337 and 00815 only.
constexpr auto const services = R"(
*Z 01337 80____                                           %
*A VE 0000001 0000005 000003                              %
*G ICE 0000001 0000005                                    %
0000001 A                            01000                %
0000005 E                     01100                       %
*Z 00815 80____                                           %
*A VE 0000007 0000006 000003                              %
*G RE  0000007 0000006                                    %
0000007 G                            01000                %
0000005 E                     01104  01105                %
0000006 F                     01130                       %
*Z 03374 80____                                           %
*A VE 0000008 0000006 000003                              %
*G IC  0000008 0000006                                    %
0000008 H                            01000                %
0000005 E                     01103  01106                %
0000006 F                     01200                       %
)";

constexpr auto const umsteigb_content = R"(
9999999 02 02 STANDARD
0000005 02 02 E
)";

constexpr auto const umsteigz_content = R"(
0000005 001337 80____ 000815 80____ 015
)";

mem_dir with_transfer_rules(std::string_view umsteigb,
                            std::string_view umsteigz) {
  auto const& c = hrd_5_20_26;
  auto d = base().add({c.fplan_ / "services.101", services});
  d.dir_[(c.core_data_ / c.transfers_.station_[0]).string()] = umsteigb;
  if (!umsteigz.empty()) {
    d.dir_[(c.core_data_ / c.transfers_.trip_[1]).string()] = umsteigz;
  }
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

// the earliest arrival at F leaving A at 09:55 local time on 2020-03-30
unixtime_t best_arrival(timetable const& tt) {
  auto const results =
      raptor_search(tt, nullptr, "0000001", "0000006",
                    unixtime_t{sys_days{2020_y / March / 30} + 7h + 55min});
  auto best = unixtime_t::max();
  for (auto const& j : results) {
    best = std::min(best, j.dest_time_);
  }
  return best;
}

}  // namespace

TEST(hrd, transfer_rules_decide_the_change) {
  // without the rule, 00815 at 11:05 is reachable: F at 11:30
  EXPECT_EQ(unixtime_t{sys_days{2020_y / March / 30} + 9h + 30min},
            best_arrival(load(with_transfer_rules(umsteigb_content, ""))));

  // with the 15 minute rule only 03374 at 11:06 remains: F at 12:00
  auto const tt = load(with_transfer_rules(umsteigb_content, umsteigz_content));
  EXPECT_EQ(unixtime_t{sys_days{2020_y / March / 30} + 10h}, best_arrival(tt));

  // the rule needs a location of its own at E
  auto const e = tt.find(location_id{"0000005", source_idx_t{0U}});
  ASSERT_TRUE(e.has_value());
  auto const& children = tt.locations_.children_[*e];
  ASSERT_FALSE(children.empty());
  for (auto const& c : children) {
    EXPECT_EQ(location_type::kVirt, tt.locations_.types_[c]);
    EXPECT_EQ(*e, tt.locations_.parents_[c]);
  }
}

TEST(hrd, transfer_rules_without_rules_no_virtual_locations) {
  auto const tt = load(with_transfer_rules(umsteigb_content, ""));
  auto const e = tt.find(location_id{"0000005", source_idx_t{0U}});
  ASSERT_TRUE(e.has_value());
  EXPECT_TRUE(tt.locations_.children_[*e].empty())
      << "no umsteigz -> no behaviour classes";
}
