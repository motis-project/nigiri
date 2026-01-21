#include "nigiri/routing/ch/saw.h"
#include <cstdint>
#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"

#include "nigiri/routing/ch/ch_data.h"
#include "nigiri/routing/limits.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

using namespace nigiri;
using namespace date;
using namespace std::chrono_literals;
using namespace std::string_view_literals;
using namespace nigiri::routing;

TEST(ch, saw_traffic_days_test) {
  auto tt = timetable{};

  auto const s1 = owning_saw<routing::saw_type::kTrafficDays>{
      {{1001U, u16_minutes{10}, bitfield_idx_t{0}},
       {1000U, u16_minutes{5}, bitfield_idx_t{1}},
       {1000U, u16_minutes{5}, bitfield_idx_t{0}},
       {1000U, u16_minutes{5}, bitfield_idx_t{1}},
       {999U, u16_minutes{10}, bitfield_idx_t{1}}},
      u16_minutes{0}};
  auto const s2 = owning_saw<routing::saw_type::kTrafficDays>{
      {{1001U, u16_minutes{10}, bitfield_idx_t{0}},
       {997U, u16_minutes{10}, bitfield_idx_t{1}}},
      u16_minutes{0}};
  auto traffic_days =
      vector_map<bitfield_idx_t, std::pair<bitfield, std::uint16_t>>{};
  traffic_days.emplace_back(bitfield{"000001100100000"}, 9);
  traffic_days.emplace_back(bitfield{"000000110000000"}, 8);

  EXPECT_FALSE(traffic_days.at(bitfield_idx_t{0}).first.test(4));
  EXPECT_TRUE(traffic_days.at(bitfield_idx_t{0}).first.test(5));
  EXPECT_FALSE(traffic_days.at(bitfield_idx_t{0}).first.test(6));
  EXPECT_FALSE(traffic_days.at(bitfield_idx_t{0}).first.test(7));
  EXPECT_TRUE(traffic_days.at(bitfield_idx_t{0}).first.test(8));
  EXPECT_TRUE(traffic_days.at(bitfield_idx_t{0}).first.test(9));

  EXPECT_FALSE(s1.to_saw(traffic_days) < s1.to_saw(traffic_days));
  EXPECT_TRUE(s1.to_saw(traffic_days).less(s1.to_saw(traffic_days), true));
  EXPECT_TRUE(s1.to_saw(traffic_days) <= s1.to_saw(traffic_days));
  EXPECT_FALSE(s1.to_saw(traffic_days).leq(s1.to_saw(traffic_days), true));

  EXPECT_FALSE(s1.to_saw(traffic_days) < s2.to_saw(traffic_days));
  EXPECT_FALSE(s1.to_saw(traffic_days).less(s2.to_saw(traffic_days), true));
  EXPECT_TRUE(s1.to_saw(traffic_days) <= s2.to_saw(traffic_days));
  EXPECT_TRUE(s1.to_saw(traffic_days).leq(s2.to_saw(traffic_days), true));

  EXPECT_EQ(s1.to_saw(traffic_days).max().count(), 2888);
  EXPECT_EQ(s1.to_saw(traffic_days).min().count(), 5);

  {
    auto const s3 = owning_saw<routing::saw_type::kTrafficDays>{
        {{1001U, u16_minutes{10}, bitfield_idx_t{1}}}, u16_minutes{0}};
    EXPECT_EQ(s3.to_saw(traffic_days).max().count(), 1450);

    auto traffic_days_1 =
        vector_map<bitfield_idx_t, std::pair<bitfield, std::uint16_t>>{};
    traffic_days_1.emplace_back(bitfield{"000001000000000"}, 9);
    traffic_days_1.emplace_back(bitfield{"001000000100000"}, 12);

    auto const s4 = owning_saw<routing::saw_type::kTrafficDays>{
        {{1001U, u16_minutes{10}, bitfield_idx_t{0}}}, u16_minutes{0}};
    EXPECT_EQ(s4.to_saw(traffic_days_1).max().count(), kMaxTravelTime.count());

    auto const s5 = owning_saw<routing::saw_type::kTrafficDays>{
        {{1001U, u16_minutes{10}, bitfield_idx_t{1}}}, u16_minutes{0}};
    EXPECT_EQ(s5.to_saw(traffic_days_1).max().count(), kMaxTravelTime.count());

    auto const s6 =
        owning_saw<routing::saw_type::kTrafficDays>{{}, u16_minutes{0}};
    EXPECT_EQ(s6.to_saw(traffic_days_1).max().count(), kMaxTravelTime.count());
  }

  {
    auto tmp = std::vector<tooth>{};
    s1.to_saw(traffic_days).simplify(s2.to_saw(traffic_days), tmp);

    auto expected =
        std::vector<tooth>{{1001U, u16_minutes{10}, bitfield_idx_t{0}},
                           {1000U, u16_minutes{5}, bitfield_idx_t{1}},
                           {1000U, u16_minutes{5}, bitfield_idx_t{0}}};
    EXPECT_EQ(tmp.size(), expected.size());
    for (auto i = 0U; i < expected.size(); ++i) {
      EXPECT_EQ(expected[i], tmp[i]);
    }
  }

  {
    auto const s3 = owning_saw<routing::saw_type::kTrafficDays>{
        {{1000U, u16_minutes{3}, bitfield_idx_t{1}},
         {999U, u16_minutes{13}, bitfield_idx_t{1}}},
        u16_minutes{0}};

    auto tmp = std::vector<tooth>{};
    s1.to_saw(traffic_days).simplify(s3.to_saw(traffic_days), tmp);

    auto expected =
        std::vector<tooth>{{1001U, u16_minutes{10}, bitfield_idx_t{0}},
                           {1000U, u16_minutes{3}, bitfield_idx_t{1}},
                           {1000U, u16_minutes{5}, bitfield_idx_t{0}}};
    EXPECT_EQ(tmp.size(), expected.size());
    for (auto i = 0U; i < expected.size(); ++i) {
      EXPECT_EQ(expected[i], tmp[i]);
    }
  }

  {
    auto tmp = std::vector<tooth>{};
    s1.to_saw(traffic_days).concat(s2.to_saw(traffic_days), true, tmp);

    auto expected =
        std::vector<tooth>{{1001U, u16_minutes{2886}, bitfield_idx_t{0}},
                           {1000U, u16_minutes{1447}, bitfield_idx_t{1}}};
    EXPECT_EQ(tmp.size(), expected.size());
    for (auto i = 0U; i < expected.size(); ++i) {
      EXPECT_EQ(expected[i], tmp[i]);
    }
  }

  {
    auto tmp = std::vector<tooth>{};
    s1.to_saw(traffic_days).concat(s2.to_saw(traffic_days), false, tmp);

    auto expected =
        std::vector<tooth>{{1001U, u16_minutes{1450}, bitfield_idx_t{0}},
                           {1000U, u16_minutes{1447}, bitfield_idx_t{1}}};
    EXPECT_EQ(tmp.size(), expected.size());
    for (auto i = 0U; i < expected.size(); ++i) {
      EXPECT_EQ(expected[i], tmp[i]);
    }
  }

  {
    auto const s3 = owning_saw<routing::saw_type::kTrafficDays>{
        {{1001U, u16_minutes{10}, bitfield_idx_t{0}}}, u16_minutes{0}};
    auto const s4 = owning_saw<routing::saw_type::kTrafficDays>{
        {{1000U, u16_minutes{10}, bitfield_idx_t{0}}}, u16_minutes{0}};

    auto traffic_days_1 =
        vector_map<bitfield_idx_t, std::pair<bitfield, std::uint16_t>>{};
    traffic_days_1.emplace_back(bitfield{"001000000100000"}, 12);

    auto tmp = std::vector<tooth>{};
    s3.to_saw(traffic_days_1).concat(s4.to_saw(traffic_days_1), true, tmp);

    EXPECT_EQ(tmp.size(), 0);
  }

  {
    auto tmp = std::vector<tooth>{};
    auto const s3 = owning_saw<routing::saw_type::kConstant>{
        saw<nigiri::routing::saw_type::kConstant>::of(duration_t{1000}),
        u16_minutes{0}};
    s2.to_saw(traffic_days)
        .concat_const(kReverse, s3.to_saw(traffic_days), tmp);

    auto expected =
        std::vector<tooth>{{1U, u16_minutes{1010}, bitfield_idx_t{0}},
                           {1437U, u16_minutes{1010}, bitfield_idx_t{2}}};
    EXPECT_EQ(tmp.size(), expected.size());
    for (auto i = 0U; i < expected.size(); ++i) {
      EXPECT_EQ(expected[i], tmp[i]);
    }
    ASSERT_EQ(traffic_days.size(), 3);
    EXPECT_FALSE(traffic_days.at(bitfield_idx_t{2}).first.test(4));
    EXPECT_FALSE(traffic_days.at(bitfield_idx_t{2}).first.test(5));
    EXPECT_TRUE(traffic_days.at(bitfield_idx_t{2}).first.test(6));
    EXPECT_TRUE(traffic_days.at(bitfield_idx_t{2}).first.test(7));
    EXPECT_FALSE(traffic_days.at(bitfield_idx_t{2}).first.test(8));
    EXPECT_EQ(traffic_days.at(bitfield_idx_t{2}).second, 7);
  }
}
