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
  auto td = nigiri::routing::traffic_days{};
  td.get_or_create(bitfield{"000001100100000"}, 9);
  td.get_or_create(bitfield{"000000110000000"}, 8);

  EXPECT_EQ(saw<nigiri::routing::saw_type::kTrafficDays>::last_set_bit(
                td.bitfields_.at(bitfield_idx_t{0}).first),
            9);
  EXPECT_EQ(saw<nigiri::routing::saw_type::kTrafficDays>::last_set_bit(
                td.bitfields_.at(bitfield_idx_t{1}).first),
            8);

  EXPECT_FALSE(td.bitfields_.at(bitfield_idx_t{0}).first.test(4));
  EXPECT_TRUE(td.bitfields_.at(bitfield_idx_t{0}).first.test(5));
  EXPECT_FALSE(td.bitfields_.at(bitfield_idx_t{0}).first.test(6));
  EXPECT_FALSE(td.bitfields_.at(bitfield_idx_t{0}).first.test(7));
  EXPECT_TRUE(td.bitfields_.at(bitfield_idx_t{0}).first.test(8));
  EXPECT_TRUE(td.bitfields_.at(bitfield_idx_t{0}).first.test(9));

  EXPECT_FALSE(s1.to_saw(td) < s1.to_saw(td));
  EXPECT_TRUE(s1.to_saw(td).less(s1.to_saw(td), true));
  EXPECT_TRUE(s1.to_saw(td) <= s1.to_saw(td));
  EXPECT_FALSE(s1.to_saw(td).leq(s1.to_saw(td), true));

  EXPECT_FALSE(s1.to_saw(td) < s2.to_saw(td));
  EXPECT_FALSE(s1.to_saw(td).less(s2.to_saw(td), true));
  EXPECT_TRUE(s1.to_saw(td) <= s2.to_saw(td));
  EXPECT_TRUE(s1.to_saw(td).leq(s2.to_saw(td), true));

  EXPECT_EQ(s1.to_saw(td).max().count(), 2888);
  EXPECT_EQ(s1.to_saw(td).min().count(), 5);

  {
    auto const s3 = owning_saw<routing::saw_type::kTrafficDays>{
        {{1001U, u16_minutes{10}, bitfield_idx_t{1}}}, u16_minutes{0}};
    EXPECT_EQ(s3.to_saw(td).max().count(), 1450);

    auto traffic_days_1 = traffic_days{};
    traffic_days_1.bitfields_.emplace_back(bitfield{"000001000000000"}, 9);
    traffic_days_1.bitfields_.emplace_back(bitfield{"001000000100000"}, 12);

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
    s1.to_saw(td).simplify(s2.to_saw(td), tmp);

    auto expected =
        std::vector<tooth>{{1001U, u16_minutes{10}, bitfield_idx_t{0}},
                           {1000U, u16_minutes{5}, bitfield_idx_t{1}},
                           {1000U, u16_minutes{5}, bitfield_idx_t{0}}};
    ASSERT_EQ(tmp.size(), expected.size());
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
    s1.to_saw(td).simplify(s3.to_saw(td), tmp);

    auto expected =
        std::vector<tooth>{{1001U, u16_minutes{10}, bitfield_idx_t{0}},
                           {1000U, u16_minutes{3}, bitfield_idx_t{1}},
                           {1000U, u16_minutes{5}, bitfield_idx_t{0}}};
    ASSERT_EQ(tmp.size(), expected.size());
    for (auto i = 0U; i < expected.size(); ++i) {
      EXPECT_EQ(expected[i], tmp[i]);
    }
  }

  {
    auto tmp = std::vector<tooth>{};
    s1.to_saw(td).concat(s2.to_saw(td), true, tmp);

    auto expected =
        std::vector<tooth>{{1001U, u16_minutes{2886}, bitfield_idx_t{0}},
                           {1000U, u16_minutes{1447}, bitfield_idx_t{1}}};
    ASSERT_EQ(tmp.size(), expected.size());
    for (auto i = 0U; i < expected.size(); ++i) {
      EXPECT_EQ(expected[i], tmp[i]);
    }
  }

  {
    auto tmp = std::vector<tooth>{};
    s1.to_saw(td).concat(s2.to_saw(td), false, tmp);

    auto expected =
        std::vector<tooth>{{1001U, u16_minutes{1450}, bitfield_idx_t{0}},
                           {1000U, u16_minutes{1447}, bitfield_idx_t{1}}};
    ASSERT_EQ(tmp.size(), expected.size());
    for (auto i = 0U; i < expected.size(); ++i) {
      EXPECT_EQ(expected[i], tmp[i]);
    }
  }

  {
    auto const s3 = owning_saw<routing::saw_type::kTrafficDays>{
        {{1001U, u16_minutes{10}, bitfield_idx_t{0}}}, u16_minutes{0}};
    auto const s4 = owning_saw<routing::saw_type::kTrafficDays>{
        {{1000U, u16_minutes{10}, bitfield_idx_t{0}}}, u16_minutes{0}};

    auto traffic_days_1 = traffic_days{};
    traffic_days_1.bitfields_.emplace_back(bitfield{"001000000100000"}, 12);

    auto tmp = std::vector<tooth>{};
    s3.to_saw(traffic_days_1).concat(s4.to_saw(traffic_days_1), true, tmp);

    EXPECT_EQ(tmp.size(), 0);
  }

  {
    auto tmp = std::vector<tooth>{};
    auto const s3 = owning_saw<routing::saw_type::kConstant>{
        saw<nigiri::routing::saw_type::kConstant>::of(duration_t{1000}),
        u16_minutes{0}};
    s2.to_saw(td).concat_const(kReverse, s3.to_saw(td), tmp);

    auto expected =
        std::vector<tooth>{{1U, u16_minutes{1010}, bitfield_idx_t{0}},
                           {1437U, u16_minutes{1010}, bitfield_idx_t{2}}};
    ASSERT_EQ(tmp.size(), expected.size());
    for (auto i = 0U; i < expected.size(); ++i) {
      EXPECT_EQ(expected[i], tmp[i]);
    }
    ASSERT_EQ(td.bitfields_.size(), 3);
    EXPECT_FALSE(td.bitfields_.at(bitfield_idx_t{2}).first.test(4));
    EXPECT_FALSE(td.bitfields_.at(bitfield_idx_t{2}).first.test(5));
    EXPECT_TRUE(td.bitfields_.at(bitfield_idx_t{2}).first.test(6));
    EXPECT_TRUE(td.bitfields_.at(bitfield_idx_t{2}).first.test(7));
    EXPECT_FALSE(td.bitfields_.at(bitfield_idx_t{2}).first.test(8));
    EXPECT_EQ(td.bitfields_.at(bitfield_idx_t{2}).second, 7);
  }
}

TEST(ch, saw_traffic_days_power_test) {
  auto tt = timetable{};

  auto const s1 = owning_saw<routing::saw_type::kTrafficDaysPower>{
      {{1001U, u16_minutes{10}, bitfield_idx_t{0}},
       {1000U, u16_minutes{5}, bitfield_idx_t{1}},
       {1000U, u16_minutes{5}, bitfield_idx_t{0}},
       {1000U, u16_minutes{5}, bitfield_idx_t{1}},
       {999U, u16_minutes{10}, bitfield_idx_t{1}}},
      u16_minutes{0}};
  auto const s2 = owning_saw<routing::saw_type::kTrafficDaysPower>{
      {{1001U, u16_minutes{10}, bitfield_idx_t{0}},
       {997U, u16_minutes{10}, bitfield_idx_t{1}}},
      u16_minutes{0}};

  {
    auto td = nigiri::routing::traffic_days{};
    td.get_or_create(bitfield{"000001100100000"}, 9);
    td.get_or_create(bitfield{"000000110000000"}, 8);

    EXPECT_FALSE(s1.to_saw(td) < s1.to_saw(td));
    EXPECT_TRUE(s1.to_saw(td).less(s1.to_saw(td), true));
    EXPECT_TRUE(s1.to_saw(td) <= s1.to_saw(td));
    EXPECT_FALSE(s1.to_saw(td).leq(s1.to_saw(td), true));

    EXPECT_FALSE(s1.to_saw(td) < s2.to_saw(td));
    EXPECT_FALSE(s1.to_saw(td).less(s2.to_saw(td), true));
    EXPECT_TRUE(s1.to_saw(td) <= s2.to_saw(td));
    EXPECT_TRUE(s1.to_saw(td).leq(s2.to_saw(td), true));

    EXPECT_EQ(s1.to_saw(td).max().count(), 2888);
    EXPECT_EQ(s1.to_saw(td).min().count(), 5);

    auto tmp = std::vector<tooth>{};
    s1.to_saw(td).simplify(s2.to_saw(td), tmp);

    auto expected =
        std::vector<tooth>{{1001U, u16_minutes{10}, bitfield_idx_t{0}},
                           {1000U, u16_minutes{5}, bitfield_idx_t{2}}};

    EXPECT_EQ(td.bitfields_.at(bitfield_idx_t{2}).first.count(), 4);
    EXPECT_TRUE(td.bitfields_.at(bitfield_idx_t{2}).first.test(5));
    EXPECT_TRUE(td.bitfields_.at(bitfield_idx_t{2}).first.test(7));
    EXPECT_TRUE(td.bitfields_.at(bitfield_idx_t{2}).first.test(8));
    EXPECT_TRUE(td.bitfields_.at(bitfield_idx_t{2}).first.test(9));
    ASSERT_EQ(tmp.size(), expected.size());
    for (auto i = 0U; i < expected.size(); ++i) {
      EXPECT_EQ(expected[i], tmp[i]);
    }
  }

  {
    auto const s3 = owning_saw<routing::saw_type::kTrafficDaysPower>{
        {{1000U, u16_minutes{3}, bitfield_idx_t{1}},
         {999U, u16_minutes{13}, bitfield_idx_t{1}}},
        u16_minutes{0}};

    auto tmp = std::vector<tooth>{};
    auto td = nigiri::routing::traffic_days{};
    td.get_or_create(bitfield{"000001100100000"}, 9);
    td.get_or_create(bitfield{"000000110000000"}, 8);

    s1.to_saw(td).simplify(s3.to_saw(td), tmp);

    auto expected =
        std::vector<tooth>{{1001U, u16_minutes{10}, bitfield_idx_t{0}},
                           {1000U, u16_minutes{3}, bitfield_idx_t{1}},
                           {1000U, u16_minutes{5}, bitfield_idx_t{2}}};

    EXPECT_EQ(td.bitfields_.at(bitfield_idx_t{2}).first.count(), 2);
    EXPECT_TRUE(td.bitfields_.at(bitfield_idx_t{2}).first.test(5));
    EXPECT_TRUE(td.bitfields_.at(bitfield_idx_t{2}).first.test(9));
    ASSERT_EQ(tmp.size(), expected.size());
    for (auto i = 0U; i < expected.size(); ++i) {
      EXPECT_EQ(expected[i], tmp[i]);
    }
  }

  {
    auto tmp = std::vector<tooth>{};
    auto td = nigiri::routing::traffic_days{};
    td.get_or_create(bitfield{"000001100100000"}, 9);
    td.get_or_create(bitfield{"000000110000000"}, 8);

    s1.to_saw(td).concat(s2.to_saw(td), true, tmp);

    auto expected = std::vector<tooth>{
        {1001U, u16_minutes{1450}, bitfield_idx_t{2}},
        {1001U, u16_minutes{2886}, bitfield_idx_t{3}},
        {1000U, u16_minutes{1447}, bitfield_idx_t{4}},
    };

    EXPECT_EQ(td.bitfields_.at(bitfield_idx_t{2}).first.count(), 1);
    EXPECT_TRUE(td.bitfields_.at(bitfield_idx_t{2}).first.test(8));

    EXPECT_EQ(td.bitfields_.at(bitfield_idx_t{3}).first.count(), 1);
    EXPECT_TRUE(td.bitfields_.at(bitfield_idx_t{3}).first.test(5));

    EXPECT_EQ(td.bitfields_.at(bitfield_idx_t{4}).first.count(), 1);
    EXPECT_TRUE(td.bitfields_.at(bitfield_idx_t{4}).first.test(7));

    ASSERT_EQ(tmp.size(), expected.size());
    for (auto i = 0U; i < expected.size(); ++i) {
      EXPECT_EQ(expected[i], tmp[i]);
    }
  }

  {
    auto tmp = std::vector<tooth>{};
    auto td = nigiri::routing::traffic_days{};
    td.get_or_create(bitfield{"000001100100000"}, 9);
    td.get_or_create(bitfield{"000000110000000"}, 8);

    s1.to_saw(td).concat(s2.to_saw(td), false, tmp);

    auto expected = std::vector<tooth>{
        {1001U, u16_minutes{1450}, bitfield_idx_t{2}},
        {1001U, u16_minutes{2886}, bitfield_idx_t{3}},
        {1000U, u16_minutes{1447},
         bitfield_idx_t{4}},  // TODO could it happen that same mam entries are
                              // inserted in wrong order for disjunct bitfields?
    };

    EXPECT_EQ(td.bitfields_.at(bitfield_idx_t{2}).first.count(), 1);
    std::cout << "MAIN " << td.bitfields_.at(bitfield_idx_t{2}).first
              << std::endl;
    EXPECT_TRUE(td.bitfields_.at(bitfield_idx_t{2}).first.test(8));

    EXPECT_EQ(td.bitfields_.at(bitfield_idx_t{3}).first.count(), 1);
    EXPECT_TRUE(td.bitfields_.at(bitfield_idx_t{3}).first.test(5));

    EXPECT_EQ(td.bitfields_.at(bitfield_idx_t{4}).first.count(), 1);
    EXPECT_TRUE(td.bitfields_.at(bitfield_idx_t{4}).first.test(7));

    ASSERT_EQ(tmp.size(), expected.size());
    for (auto i = 0U; i < expected.size(); ++i) {
      EXPECT_EQ(expected[i], tmp[i]);
    }
  }

  {
    auto const s3 = owning_saw<routing::saw_type::kTrafficDaysPower>{
        {{1001U, u16_minutes{10}, bitfield_idx_t{0}}}, u16_minutes{0}};
    auto const s4 = owning_saw<routing::saw_type::kTrafficDaysPower>{
        {{1000U, u16_minutes{10}, bitfield_idx_t{0}}}, u16_minutes{0}};

    auto td = traffic_days{};
    td.bitfields_.emplace_back(bitfield{"001000000100000"}, 12);

    auto tmp = std::vector<tooth>{};
    s3.to_saw(td).concat(s4.to_saw(td), true, tmp);

    ASSERT_EQ(tmp.size(), 0);
  }

  {
    auto tmp = std::vector<tooth>{};
    auto td = nigiri::routing::traffic_days{};
    td.get_or_create(bitfield{"000001100100000"}, 9);
    td.get_or_create(bitfield{"000000110000000"}, 8);

    auto const s3 = owning_saw<routing::saw_type::kConstant>{
        saw<nigiri::routing::saw_type::kConstant>::of(duration_t{1000}),
        u16_minutes{0}};
    s2.to_saw(td).concat_const(kReverse, s3.to_saw(td), tmp);

    auto expected =
        std::vector<tooth>{{1U, u16_minutes{1010}, bitfield_idx_t{0}},
                           {1437U, u16_minutes{1010}, bitfield_idx_t{2}}};
    ASSERT_EQ(tmp.size(), expected.size());
    for (auto i = 0U; i < expected.size(); ++i) {
      EXPECT_EQ(expected[i], tmp[i]);
    }
    ASSERT_EQ(td.bitfields_.size(), 3);
    EXPECT_FALSE(td.bitfields_.at(bitfield_idx_t{2}).first.test(4));
    EXPECT_FALSE(td.bitfields_.at(bitfield_idx_t{2}).first.test(5));
    EXPECT_TRUE(td.bitfields_.at(bitfield_idx_t{2}).first.test(6));
    EXPECT_TRUE(td.bitfields_.at(bitfield_idx_t{2}).first.test(7));
    EXPECT_FALSE(td.bitfields_.at(bitfield_idx_t{2}).first.test(8));
    EXPECT_EQ(td.bitfields_.at(bitfield_idx_t{2}).second, 7);
  }
}
