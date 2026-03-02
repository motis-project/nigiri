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

// TODO test automatic const, same mam concat

tooth metadata_tooth(std::uint16_t const val = u16_minutes::max().count()) {
  return {std::numeric_limits<std::int16_t>::max(),
          u16_minutes{val},
          bitfield_idx_t::invalid(),
          ch_edge_idx_t::invalid(),
          transport_idx_t::invalid(),
          ch_edge_idx_t::invalid(),
          transport_idx_t::invalid()};
}

TEST(ch, saw_day_test) {
  auto tt = timetable{};

  auto const s1 = owning_saw<routing::saw_type::kDay>{
      {metadata_tooth(0U),
       metadata_tooth(),
       metadata_tooth(),
       {1001U, u16_minutes{10}, bitfield_idx_t::invalid()},
       {1000U, u16_minutes{5}, bitfield_idx_t::invalid()},
       {999U, u16_minutes{10}, bitfield_idx_t::invalid()}},
      u16_minutes{0}};
  auto const s2 = owning_saw<routing::saw_type::kDay>{
      {metadata_tooth(0U),
       metadata_tooth(),
       metadata_tooth(),
       {1001U, u16_minutes{10}, bitfield_idx_t::invalid()},
       {997U, u16_minutes{10}, bitfield_idx_t::invalid()}},
      u16_minutes{0}};
  auto td = nigiri::routing::traffic_days{};

  EXPECT_FALSE(s1.to_saw(td) < s1.to_saw(td));
  EXPECT_FALSE(s1.to_saw(td) > s1.to_saw(td));
  EXPECT_TRUE(s1.to_saw(td).less(s1.to_saw(td), true));
  EXPECT_TRUE(s1.to_saw(td) <= s1.to_saw(td));
  EXPECT_FALSE(s1.to_saw(td).leq(s1.to_saw(td), true));

  EXPECT_FALSE(s1.to_saw(td) < s2.to_saw(td));
  EXPECT_FALSE(s2.to_saw(td) > s1.to_saw(td));
  EXPECT_FALSE(s1.to_saw(td).less(s2.to_saw(td), true));
  EXPECT_TRUE(s1.to_saw(td) <= s2.to_saw(td));
  EXPECT_TRUE(s1.to_saw(td).leq(s2.to_saw(td), true));

  EXPECT_EQ(s1.to_saw(td).max().count(), 1448);
  EXPECT_EQ(s1.to_saw(td).min().count(), 5);
  {
    auto const s3 = owning_saw<routing::saw_type::kDay>{
        {metadata_tooth(0U),
         metadata_tooth(),
         metadata_tooth(),
         {1001U, u16_minutes{10}, bitfield_idx_t::invalid()}},
        u16_minutes{0}};
    EXPECT_EQ(s3.to_saw(td).max().count(), 1450);

    auto const s6 = owning_saw<routing::saw_type::kDay>{{}, u16_minutes{0}};
    EXPECT_EQ(s6.to_saw(td).max().count(), kMaxTravelTime.count());
  }
  {
    auto tmp = std::vector<tooth>{};
    s1.to_saw(td).simplify(s2.to_saw(td), false, tmp);

    auto expected = std::vector<tooth>{
        metadata_tooth(0U),
        metadata_tooth(
            saw<nigiri::routing::saw_type::kDay>{tmp, td}.min().count()),
        metadata_tooth(
            saw<nigiri::routing::saw_type::kDay>{tmp, td}.max().count()),
        {1001U, u16_minutes{10}, bitfield_idx_t::invalid()},
        {1000U, u16_minutes{5}, bitfield_idx_t::invalid()}};
    ASSERT_EQ(tmp.size(), expected.size());
    for (auto i = 0U; i < expected.size(); ++i) {
      EXPECT_EQ(expected[i], tmp[i]);
    }
  }

  {
    auto const s3 = owning_saw<routing::saw_type::kDay>{
        {metadata_tooth(0U),
         metadata_tooth(),
         metadata_tooth(),
         {1000U, u16_minutes{3}, bitfield_idx_t::invalid()},
         {999U, u16_minutes{13}, bitfield_idx_t::invalid()}},
        u16_minutes{0}};

    auto tmp = std::vector<tooth>{};
    s1.to_saw(td).simplify(s3.to_saw(td), false, tmp);

    auto expected = std::vector<tooth>{
        metadata_tooth(0U),
        metadata_tooth(
            saw<nigiri::routing::saw_type::kDay>{tmp, td}.min().count()),
        metadata_tooth(
            saw<nigiri::routing::saw_type::kDay>{tmp, td}.max().count()),
        {1001U, u16_minutes{10}, bitfield_idx_t::invalid()},
        {1000U, u16_minutes{3}, bitfield_idx_t::invalid()}};
    ASSERT_EQ(tmp.size(), expected.size());
    for (auto i = 0U; i < expected.size(); ++i) {
      EXPECT_EQ(expected[i], tmp[i]);
    }
  }

  {
    auto tmp = std::vector<tooth>{};
    s1.to_saw(td).concat(s2.to_saw(td), ch_edge_idx_t::invalid(),
                         ch_edge_idx_t::invalid(), true, tmp);

    auto expected = std::vector<tooth>{
        metadata_tooth(0U),
        metadata_tooth(1446U),
        metadata_tooth(),
        {1001U, u16_minutes{1446}, bitfield_idx_t::invalid()}};
    ASSERT_EQ(tmp.size(), expected.size());
    for (auto i = 0U; i < expected.size(); ++i) {
      EXPECT_EQ(expected[i], tmp[i]);
    }

    EXPECT_TRUE(
        s1.to_saw(td).less(saw<nigiri::routing::saw_type::kDay>{tmp, td}));
    EXPECT_TRUE(
        s2.to_saw(td).less(saw<nigiri::routing::saw_type::kDay>{tmp, td}));
  }

  {
    auto tmp = std::vector<tooth>{};
    s1.to_saw(td).concat(s2.to_saw(td), ch_edge_idx_t::invalid(),
                         ch_edge_idx_t::invalid(), false, tmp);

    auto expected = std::vector<tooth>{
        metadata_tooth(0U),
        metadata_tooth(1446U),
        metadata_tooth(),
        {1001U, u16_minutes{1446}, bitfield_idx_t::invalid()}};
    ASSERT_EQ(tmp.size(), expected.size());
    for (auto i = 0U; i < expected.size(); ++i) {
      EXPECT_EQ(expected[i], tmp[i]);
    }

    EXPECT_TRUE(
        s1.to_saw(td).less(saw<nigiri::routing::saw_type::kDay>{tmp, td}));
    EXPECT_TRUE(
        s2.to_saw(td).less(saw<nigiri::routing::saw_type::kDay>{tmp, td}));
  }

  {
    auto const s3 = owning_saw<routing::saw_type::kDay>{
        {metadata_tooth(0U),
         metadata_tooth(),
         metadata_tooth(),
         {1001U, u16_minutes{250}, bitfield_idx_t::invalid()},
         {1000U, u16_minutes{200}, bitfield_idx_t::invalid()},
         {999U, u16_minutes{10}, bitfield_idx_t::invalid()}},
        u16_minutes{0}};
    auto const s4 = owning_saw<routing::saw_type::kDay>{
        {metadata_tooth(0U),
         metadata_tooth(),
         metadata_tooth(),
         {1300U, u16_minutes{10}, bitfield_idx_t::invalid()},
         {1160U, u16_minutes{60}, bitfield_idx_t::invalid()},
         {1150U, u16_minutes{50}, bitfield_idx_t::invalid()},
         {1090U, u16_minutes{110}, bitfield_idx_t::invalid()},
         {1009U, u16_minutes{10}, bitfield_idx_t::invalid()}},
        u16_minutes{0}};

    auto tmp = std::vector<tooth>{};

    s3.to_saw(td).concat(s4.to_saw(td), ch_edge_idx_t::invalid(),
                         ch_edge_idx_t::invalid(), false, tmp);

    auto expected = std::vector<tooth>{
        metadata_tooth(0U),
        metadata_tooth(20U),
        metadata_tooth(),
        {1001U, u16_minutes{309}, bitfield_idx_t::invalid()},
        {999U, u16_minutes{20}, bitfield_idx_t::invalid()},
    };

    ASSERT_EQ(tmp.size(), expected.size());
    for (auto i = 0U; i < expected.size(); ++i) {
      EXPECT_EQ(expected[i], tmp[i]);
    }

    EXPECT_TRUE(
        s3.to_saw(td).less(saw<nigiri::routing::saw_type::kDay>{tmp, td}));
    EXPECT_TRUE(
        s4.to_saw(td).less(saw<nigiri::routing::saw_type::kDay>{tmp, td}));
  }

  {
    auto tmp = std::vector<tooth>{};
    auto const s3 = owning_saw<routing::saw_type::kConstant>{
        saw<nigiri::routing::saw_type::kConstant>::of(duration_t{1000}),
        u16_minutes{0}};
    s2.to_saw(td).concat_const(kReverse, s3.to_saw(td),
                               ch_edge_idx_t::invalid(),
                               ch_edge_idx_t::invalid(), tmp);

    auto expected = std::vector<tooth>{
        metadata_tooth(0U),
        metadata_tooth(),
        metadata_tooth(),
        {1U, u16_minutes{1010}, bitfield_idx_t::invalid()},
        {1437U, u16_minutes{1010}, bitfield_idx_t::invalid()}};
    ASSERT_EQ(tmp.size(), expected.size());
    for (auto i = 0U; i < expected.size(); ++i) {
      EXPECT_EQ(expected[i], tmp[i]);
    }
  }
}

TEST(ch, saw_traffic_days_test) {
  auto tt = timetable{};

  auto const s1 = owning_saw<routing::saw_type::kTrafficDays>{
      {metadata_tooth(9U),
       metadata_tooth(),
       metadata_tooth(),
       {1001U, u16_minutes{10}, bitfield_idx_t{0}},
       {1000U, u16_minutes{5}, bitfield_idx_t{1}},
       {1000U, u16_minutes{5}, bitfield_idx_t{0}},
       {1000U, u16_minutes{5}, bitfield_idx_t{1}},
       {999U, u16_minutes{10}, bitfield_idx_t{1}}},
      u16_minutes{0}};
  auto const s2 = owning_saw<routing::saw_type::kTrafficDays>{
      {metadata_tooth(9U),
       metadata_tooth(),
       metadata_tooth(),
       {1001U, u16_minutes{10}, bitfield_idx_t{0}},
       {997U, u16_minutes{10}, bitfield_idx_t{1}}},
      u16_minutes{0}};
  auto td = nigiri::routing::traffic_days{};
  td.get_or_create(bitfield{"000001100100000"}, 9);
  td.get_or_create(bitfield{"000000110000000"}, 8);

  EXPECT_EQ(saw<nigiri::routing::saw_type::kTrafficDays>::last_set_bit(
                td.at(bitfield_idx_t{0}).first),
            9);
  EXPECT_EQ(saw<nigiri::routing::saw_type::kTrafficDays>::last_set_bit(
                td.at(bitfield_idx_t{1}).first),
            8);

  EXPECT_FALSE(td.at(bitfield_idx_t{0}).first.test(4));
  EXPECT_TRUE(td.at(bitfield_idx_t{0}).first.test(5));
  EXPECT_FALSE(td.at(bitfield_idx_t{0}).first.test(6));
  EXPECT_FALSE(td.at(bitfield_idx_t{0}).first.test(7));
  EXPECT_TRUE(td.at(bitfield_idx_t{0}).first.test(8));
  EXPECT_TRUE(td.at(bitfield_idx_t{0}).first.test(9));

  EXPECT_FALSE(s1.to_saw(td) < s1.to_saw(td));
  EXPECT_FALSE(s1.to_saw(td) > s1.to_saw(td));
  EXPECT_TRUE(s1.to_saw(td).less(s1.to_saw(td), true));
  EXPECT_TRUE(s1.to_saw(td) <= s1.to_saw(td));
  EXPECT_FALSE(s1.to_saw(td).leq(s1.to_saw(td), true));

  EXPECT_FALSE(s1.to_saw(td) < s2.to_saw(td));
  EXPECT_FALSE(s2.to_saw(td) > s1.to_saw(td));
  EXPECT_TRUE(s1.to_saw(td).less(s2.to_saw(td), true));
  EXPECT_TRUE(s1.to_saw(td) <= s2.to_saw(td));
  EXPECT_TRUE(s1.to_saw(td).leq(s2.to_saw(td), true));

  EXPECT_EQ(s1.to_saw(td).max().count(), 2888);
  EXPECT_EQ(s1.to_saw(td).min().count(), 5);
  {
    auto const s3 = owning_saw<routing::saw_type::kTrafficDays>{
        {metadata_tooth(12U),
         metadata_tooth(),
         metadata_tooth(),
         {1001U, u16_minutes{10}, bitfield_idx_t{1}}},
        u16_minutes{0}};
    EXPECT_EQ(s3.to_saw(td).max().count(), 1450);

    auto traffic_days_1 = traffic_days{};
    traffic_days_1.bitfields_.emplace_back(bitfield{"000001000000000"}, 9);
    traffic_days_1.bitfields_.emplace_back(bitfield{"001000000100000"}, 12);

    auto const s4 = owning_saw<routing::saw_type::kTrafficDays>{
        {metadata_tooth(9U),
         metadata_tooth(),
         metadata_tooth(),
         {1001U, u16_minutes{10}, bitfield_idx_t{0}}},
        u16_minutes{0}};
    EXPECT_EQ(s4.to_saw(traffic_days_1).max().count(), kMaxTravelTime.count());

    auto const s5 = owning_saw<routing::saw_type::kTrafficDays>{
        {metadata_tooth(9U),
         metadata_tooth(),
         metadata_tooth(),
         {1001U, u16_minutes{10}, bitfield_idx_t{1}}},
        u16_minutes{0}};
    EXPECT_EQ(s5.to_saw(traffic_days_1).max().count(), kMaxTravelTime.count());

    auto const s6 =
        owning_saw<routing::saw_type::kTrafficDays>{{}, u16_minutes{0}};
    EXPECT_EQ(s6.to_saw(traffic_days_1).max().count(), kMaxTravelTime.count());
  }
  {
    auto const s3 = owning_saw<routing::saw_type::kTrafficDays>{
        {metadata_tooth(3U)}, u16_minutes{0}};

    auto const s4 = owning_saw<routing::saw_type::kTrafficDays>{
        {metadata_tooth(20U)}, u16_minutes{0}};

    auto const s5 = owning_saw<routing::saw_type::kTrafficDays>{
        {metadata_tooth(3000U)}, u16_minutes{0}};

    EXPECT_FALSE(s1.to_saw(td) < s3.to_saw(td));
    EXPECT_TRUE(s3.to_saw(td) < s1.to_saw(td));

    EXPECT_FALSE(s1.to_saw(td) < s4.to_saw(td));
    EXPECT_FALSE(s4.to_saw(td) < s1.to_saw(td));

    EXPECT_TRUE(s1.to_saw(td) < s5.to_saw(td));
    EXPECT_FALSE(s5.to_saw(td) < s1.to_saw(td));

    {
      auto tmp = std::vector<tooth>{};
      s1.to_saw(td).simplify(s3.to_saw(td), false, tmp);

      auto expected = std::vector<tooth>{metadata_tooth(3U)};
      ASSERT_EQ(tmp.size(), expected.size());
      for (auto i = 0U; i < expected.size(); ++i) {
        EXPECT_EQ(expected[i], tmp[i]);
      }
    }
    {
      auto tmp = std::vector<tooth>{};
      s1.to_saw(td).simplify(s3.to_saw(td), true, tmp);

      auto expected = std::vector<tooth>{metadata_tooth(3U)};
      ASSERT_EQ(tmp.size(), expected.size());
      for (auto i = 0U; i < expected.size(); ++i) {
        EXPECT_EQ(expected[i], tmp[i]);
      }
    }
    {
      auto tmp = std::vector<tooth>{};
      s1.to_saw(td).simplify(s4.to_saw(td), false, tmp);

      auto expected = std::vector<tooth>{metadata_tooth(5U)};
      ASSERT_EQ(tmp.size(), expected.size());
      for (auto i = 0U; i < expected.size(); ++i) {
        EXPECT_EQ(expected[i], tmp[i]);
      }
    }
    {
      auto tmp = std::vector<tooth>{};
      s1.to_saw(td).simplify(s4.to_saw(td), true, tmp);

      auto expected = std::vector<tooth>{metadata_tooth(20U)};
      ASSERT_EQ(tmp.size(), expected.size());
      for (auto i = 0U; i < expected.size(); ++i) {
        EXPECT_EQ(expected[i], tmp[i]);
      }
    }
    {
      auto tmp = std::vector<tooth>{};
      s1.to_saw(td).simplify(s5.to_saw(td), false, tmp);

      auto expected = std::vector<tooth>{metadata_tooth(5U)};
      ASSERT_EQ(tmp.size(), expected.size());
      for (auto i = 0U; i < expected.size(); ++i) {
        EXPECT_EQ(expected[i], tmp[i]);
      }
    }
    {
      auto tmp = std::vector<tooth>{};
      s1.to_saw(td).simplify(s5.to_saw(td), true, tmp);

      auto expected = std::vector<tooth>{metadata_tooth(3000U)};
      ASSERT_EQ(tmp.size(), expected.size());
      for (auto i = 0U; i < expected.size(); ++i) {
        EXPECT_EQ(expected[i], tmp[i]);
      }
    }
  }
  {
    auto tmp = std::vector<tooth>{};
    s1.to_saw(td).simplify(s2.to_saw(td), false, tmp);

    auto expected = std::vector<tooth>{
        metadata_tooth(9U),
        metadata_tooth(saw<nigiri::routing::saw_type::kTrafficDays>{tmp, td}
                           .min()
                           .count()),
        metadata_tooth(saw<nigiri::routing::saw_type::kTrafficDays>{tmp, td}
                           .max()
                           .count()),
        {1001U, u16_minutes{10}, bitfield_idx_t{0}},
        {1000U, u16_minutes{5}, bitfield_idx_t{1}},
        {1000U, u16_minutes{5}, bitfield_idx_t{0}}};
    ASSERT_EQ(tmp.size(), expected.size());
    for (auto i = 0U; i < expected.size(); ++i) {
      EXPECT_EQ(expected[i], tmp[i]);
    }
  }

  {
    auto const s3 = owning_saw<routing::saw_type::kTrafficDays>{
        {metadata_tooth(12U),
         metadata_tooth(),
         metadata_tooth(),
         {1000U, u16_minutes{3}, bitfield_idx_t{1}},
         {999U, u16_minutes{13}, bitfield_idx_t{1}}},
        u16_minutes{0}};

    auto tmp = std::vector<tooth>{};
    s1.to_saw(td).simplify(s3.to_saw(td), false, tmp);

    auto expected = std::vector<tooth>{
        metadata_tooth(12U),
        metadata_tooth(saw<nigiri::routing::saw_type::kTrafficDays>{tmp, td}
                           .min()
                           .count()),
        metadata_tooth(saw<nigiri::routing::saw_type::kTrafficDays>{tmp, td}
                           .max()
                           .count()),
        {1001U, u16_minutes{10}, bitfield_idx_t{0}},
        {1000U, u16_minutes{3}, bitfield_idx_t{1}},
        {1000U, u16_minutes{5}, bitfield_idx_t{0}}};
    ASSERT_EQ(tmp.size(), expected.size());
    for (auto i = 0U; i < expected.size(); ++i) {
      EXPECT_EQ(expected[i], tmp[i]);
    }
  }

  {
    auto tmp = std::vector<tooth>{};
    s1.to_saw(td).concat(s2.to_saw(td), ch_edge_idx_t::invalid(),
                         ch_edge_idx_t::invalid(), true, tmp);

    auto expected = std::vector<tooth>{
        metadata_tooth(9U),
        metadata_tooth(1450U),
        metadata_tooth(),
        {1001U, u16_minutes{1450}, bitfield_idx_t{0}},  // 2886
        {1000U, u16_minutes{1451}, bitfield_idx_t{1}}};
    ASSERT_EQ(tmp.size(), expected.size());
    for (auto i = 0U; i < expected.size(); ++i) {
      EXPECT_EQ(expected[i], tmp[i]);
    }

    EXPECT_TRUE(s1.to_saw(td).less(
        saw<nigiri::routing::saw_type::kTrafficDays>{tmp, td}));
    EXPECT_TRUE(s2.to_saw(td).less(
        saw<nigiri::routing::saw_type::kTrafficDays>{tmp, td}));
  }

  {
    auto tmp = std::vector<tooth>{};
    s1.to_saw(td).concat(s2.to_saw(td), ch_edge_idx_t::invalid(),
                         ch_edge_idx_t::invalid(), false, tmp);

    auto expected =
        std::vector<tooth>{metadata_tooth(9U),
                           metadata_tooth(1447U),
                           metadata_tooth(),
                           {1001U, u16_minutes{1450}, bitfield_idx_t{0}},
                           {1000U, u16_minutes{1447}, bitfield_idx_t{1}}};
    ASSERT_EQ(tmp.size(), expected.size());
    for (auto i = 0U; i < expected.size(); ++i) {
      EXPECT_EQ(expected[i], tmp[i]);
    }

    EXPECT_TRUE(s1.to_saw(td).less(
        saw<nigiri::routing::saw_type::kTrafficDays>{tmp, td}));
    EXPECT_TRUE(s2.to_saw(td).less(
        saw<nigiri::routing::saw_type::kTrafficDays>{tmp, td}));
  }

  {
    auto const s3 = owning_saw<routing::saw_type::kTrafficDays>{
        {metadata_tooth(13U),
         metadata_tooth(),
         metadata_tooth(),
         {1001U, u16_minutes{250}, bitfield_idx_t{0}},
         {1000U, u16_minutes{50}, bitfield_idx_t{2}},
         {1000U, u16_minutes{100}, bitfield_idx_t{3}},
         {1000U, u16_minutes{100}, bitfield_idx_t{4}},
         {1000U, u16_minutes{200}, bitfield_idx_t{5}},
         {999U, u16_minutes{10}, bitfield_idx_t{1}}},
        u16_minutes{0}};
    auto const s4 = owning_saw<routing::saw_type::kTrafficDays>{
        {metadata_tooth(13U),
         metadata_tooth(),
         metadata_tooth(),
         {1300U, u16_minutes{10}, bitfield_idx_t{0}},
         {1300U, u16_minutes{10}, bitfield_idx_t{5}},
         {1160U, u16_minutes{60}, bitfield_idx_t{4}},
         {1150U, u16_minutes{50}, bitfield_idx_t{3}},
         {1090U, u16_minutes{110}, bitfield_idx_t{2}},
         {1009U, u16_minutes{10}, bitfield_idx_t{1}}},
        u16_minutes{0}};

    auto tmp = std::vector<tooth>{};
    auto td1 = nigiri::routing::traffic_days{};
    td1.get_or_create(bitfield{"000001100100000"}, 9);
    td1.get_or_create(bitfield{"000000110000000"}, 8);
    td1.get_or_create(bitfield{"000010001000000"}, 10);
    td1.get_or_create(bitfield{"000100010000000"}, 11);
    td1.get_or_create(bitfield{"001000100000000"}, 12);
    td1.get_or_create(bitfield{"010001000000000"}, 13);

    s3.to_saw(td1).concat(s4.to_saw(td1), ch_edge_idx_t::invalid(),
                          ch_edge_idx_t::invalid(), false, tmp);

    auto expected = std::vector<tooth>{
        metadata_tooth(13U),
        metadata_tooth(20U),
        metadata_tooth(),
        {1001U, u16_minutes{309}, bitfield_idx_t{0}},
        {1000U, u16_minutes{200}, bitfield_idx_t{2}},
        {1000U, u16_minutes{200}, bitfield_idx_t{3}},
        {1000U, u16_minutes{220}, bitfield_idx_t{4}},
        {1000U, u16_minutes{310}, bitfield_idx_t{5}},
        {999U, u16_minutes{20}, bitfield_idx_t{1}},
    };

    ASSERT_EQ(tmp.size(), expected.size());
    for (auto i = 0U; i < expected.size(); ++i) {
      EXPECT_EQ(expected[i], tmp[i]);
    }

    EXPECT_TRUE(s3.to_saw(td1).less(
        saw<nigiri::routing::saw_type::kTrafficDays>{tmp, td1}));
    EXPECT_TRUE(s4.to_saw(td1).less(
        saw<nigiri::routing::saw_type::kTrafficDays>{tmp, td1}));
  }

  {
    auto const s3 = owning_saw<routing::saw_type::kTrafficDays>{
        {metadata_tooth(12U),
         metadata_tooth(),
         metadata_tooth(),
         {1001U, u16_minutes{10}, bitfield_idx_t{0}}},
        u16_minutes{0}};
    auto const s4 = owning_saw<routing::saw_type::kTrafficDays>{
        {metadata_tooth(12U),
         metadata_tooth(),
         metadata_tooth(),
         {1000U, u16_minutes{10}, bitfield_idx_t{0}}},
        u16_minutes{0}};

    auto traffic_days_1 = traffic_days{};
    traffic_days_1.bitfields_.emplace_back(bitfield{"001000000100000"}, 12);

    auto tmp = std::vector<tooth>{};
    s3.to_saw(traffic_days_1)
        .concat(s4.to_saw(traffic_days_1), ch_edge_idx_t::invalid(),
                ch_edge_idx_t::invalid(), true, tmp);

    EXPECT_EQ(tmp.size(), 0);
  }

  {
    auto tmp = std::vector<tooth>{};
    auto const s3 = owning_saw<routing::saw_type::kConstant>{
        saw<nigiri::routing::saw_type::kConstant>::of(duration_t{1000}),
        u16_minutes{0}};
    s2.to_saw(td).concat_const(kReverse, s3.to_saw(td),
                               ch_edge_idx_t::invalid(),
                               ch_edge_idx_t::invalid(), tmp);

    auto const tmp_bfi = bitfield_idx_t::invalid() - 1U;
    auto expected =
        std::vector<tooth>{metadata_tooth(9U),
                           metadata_tooth(),
                           metadata_tooth(),
                           {1U, u16_minutes{1010}, bitfield_idx_t{0}},
                           {1437U, u16_minutes{1010}, tmp_bfi}};
    ASSERT_EQ(tmp.size(), expected.size());
    for (auto i = 0U; i < expected.size(); ++i) {
      EXPECT_EQ(expected[i], tmp[i]);
    }
    ASSERT_EQ(td.bitfields_.size(), 2);
    ASSERT_EQ(td.tmp_bitfields_.size(), 1);
    EXPECT_FALSE(td.at(tmp_bfi).first.test(4));
    EXPECT_FALSE(td.at(tmp_bfi).first.test(5));
    EXPECT_TRUE(td.at(tmp_bfi).first.test(6));
    EXPECT_TRUE(td.at(tmp_bfi).first.test(7));
    EXPECT_FALSE(td.at(tmp_bfi).first.test(8));
    EXPECT_EQ(td.at(tmp_bfi).second, 7);
  }
}

TEST(ch, saw_traffic_days_power_test) {
  auto tt = timetable{};

  auto const s1 = owning_saw<routing::saw_type::kTrafficDaysPower>{
      {metadata_tooth(9U),
       metadata_tooth(),
       metadata_tooth(),
       {1001U, u16_minutes{10}, bitfield_idx_t{0}},
       {1000U, u16_minutes{5}, bitfield_idx_t{1}},
       {1000U, u16_minutes{5}, bitfield_idx_t{0}},
       {1000U, u16_minutes{5}, bitfield_idx_t{1}},
       {999U, u16_minutes{10}, bitfield_idx_t{1}}},
      u16_minutes{0}};
  auto const s2 = owning_saw<routing::saw_type::kTrafficDaysPower>{
      {metadata_tooth(9U),
       metadata_tooth(),
       metadata_tooth(),
       {1001U, u16_minutes{10}, bitfield_idx_t{0}},
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
    EXPECT_TRUE(s1.to_saw(td).less(s2.to_saw(td), true));
    EXPECT_TRUE(s1.to_saw(td) <= s2.to_saw(td));
    EXPECT_TRUE(s1.to_saw(td).leq(s2.to_saw(td), true));

    EXPECT_EQ(s1.to_saw(td).max().count(), 2888);
    EXPECT_EQ(s1.to_saw(td).min().count(), 5);

    auto tmp = std::vector<tooth>{};
    s1.to_saw(td).simplify(s2.to_saw(td), false, tmp);

    auto expected = std::vector<tooth>{
        metadata_tooth(9U),
        metadata_tooth(
            saw<nigiri::routing::saw_type::kTrafficDaysPower>{tmp, td}
                .min()
                .count()),
        metadata_tooth(
            saw<nigiri::routing::saw_type::kTrafficDaysPower>{tmp, td}
                .max()
                .count()),
        {1001U, u16_minutes{10}, bitfield_idx_t{0}},
        {1000U, u16_minutes{5}, bitfield_idx_t{2}}};

    EXPECT_EQ(td.at(bitfield_idx_t{2}).first.count(), 4);
    EXPECT_TRUE(td.at(bitfield_idx_t{2}).first.test(5));
    EXPECT_TRUE(td.at(bitfield_idx_t{2}).first.test(7));
    EXPECT_TRUE(td.at(bitfield_idx_t{2}).first.test(8));
    EXPECT_TRUE(td.at(bitfield_idx_t{2}).first.test(9));
    ASSERT_EQ(tmp.size(), expected.size());
    for (auto i = 0U; i < expected.size(); ++i) {
      EXPECT_EQ(expected[i], tmp[i]);
    }

    auto tmp2 = std::vector<tooth>{};
    saw<nigiri::routing::saw_type::kTrafficDaysPower>{tmp, td}.simplify(
        s2.to_saw(td), false, tmp2);
    EXPECT_TRUE(tmp2 == tmp);

    tmp2.clear();
    auto tmp3 = std::vector<tooth>{};
    saw<nigiri::routing::saw_type::kTrafficDaysPower>{tmp, td}.simplify(
        saw<nigiri::routing::saw_type::kTrafficDaysPower>{tmp3, td}, false,
        tmp2);
    EXPECT_TRUE(tmp2 == tmp);
  }

  {
    auto const s3 = owning_saw<routing::saw_type::kTrafficDaysPower>{
        {metadata_tooth(9U),
         metadata_tooth(),
         metadata_tooth(),
         {1000U, u16_minutes{3}, bitfield_idx_t{1}},
         {999U, u16_minutes{13}, bitfield_idx_t{1}}},
        u16_minutes{0}};

    auto tmp = std::vector<tooth>{};
    auto td = nigiri::routing::traffic_days{};
    td.get_or_create(bitfield{"000001100100000"}, 9);
    td.get_or_create(bitfield{"000000110000000"}, 8);

    s1.to_saw(td).simplify(s3.to_saw(td), false, tmp);

    auto expected = std::vector<tooth>{
        metadata_tooth(9U),
        metadata_tooth(saw<nigiri::routing::saw_type::kTrafficDays>{tmp, td}
                           .min()
                           .count()),
        metadata_tooth(saw<nigiri::routing::saw_type::kTrafficDays>{tmp, td}
                           .max()
                           .count()),
        {1001U, u16_minutes{10}, bitfield_idx_t{0}},
        {1000U, u16_minutes{3}, bitfield_idx_t{1}},
        {1000U, u16_minutes{5}, bitfield_idx_t{2}}};

    EXPECT_EQ(td.at(bitfield_idx_t{2}).first.count(), 2);
    EXPECT_TRUE(td.at(bitfield_idx_t{2}).first.test(5));
    EXPECT_TRUE(td.at(bitfield_idx_t{2}).first.test(9));
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

    s1.to_saw(td).concat(s2.to_saw(td), ch_edge_idx_t::invalid(),
                         ch_edge_idx_t::invalid(), true, tmp);

    auto expected = std::vector<tooth>{
        metadata_tooth(9U),
        metadata_tooth(1447U),
        metadata_tooth(),
        {1001U, u16_minutes{1450},
         traffic_days::invert_tmp_idx(bitfield_idx_t{0U})},
        //{1001U, u16_minutes{2886}, bitfield_idx_t{3}},  // ommitted due to
        // maxwaittime
        {1000U, u16_minutes{1447},
         traffic_days::invert_tmp_idx(bitfield_idx_t{1U})},
    };

    EXPECT_EQ(
        td.at(traffic_days::invert_tmp_idx(bitfield_idx_t{0U})).first.count(),
        1);
    EXPECT_TRUE(
        td.at(traffic_days::invert_tmp_idx(bitfield_idx_t{0U})).first.test(8));

    // EXPECT_EQ(td.at(bitfield_idx_t{3}).first.count(), 1);
    // EXPECT_TRUE(td.at(bitfield_idx_t{3}).first.test(5));

    EXPECT_EQ(
        td.at(traffic_days::invert_tmp_idx(bitfield_idx_t{1U})).first.count(),
        1);
    EXPECT_TRUE(
        td.at(traffic_days::invert_tmp_idx(bitfield_idx_t{1U})).first.test(7));

    ASSERT_EQ(tmp.size(), expected.size());
    for (auto i = 0U; i < expected.size(); ++i) {
      EXPECT_EQ(expected[i], tmp[i]);
    }

    EXPECT_TRUE(s1.to_saw(td).less(
        saw<nigiri::routing::saw_type::kTrafficDaysPower>{tmp, td}));
    EXPECT_TRUE(s2.to_saw(td).less(
        saw<nigiri::routing::saw_type::kTrafficDaysPower>{tmp, td}));
  }

  {
    auto tmp = std::vector<tooth>{};
    auto td = nigiri::routing::traffic_days{};
    td.get_or_create(bitfield{"000001100100000"}, 9);
    td.get_or_create(bitfield{"000000110000000"}, 8);

    s1.to_saw(td).concat(s2.to_saw(td), ch_edge_idx_t::invalid(),
                         ch_edge_idx_t::invalid(), false, tmp);

    auto expected = std::vector<tooth>{
        metadata_tooth(9U),
        metadata_tooth(1447U),
        metadata_tooth(),
        {1001U, u16_minutes{1450},
         traffic_days::invert_tmp_idx(bitfield_idx_t{0U})},
        //{1001U, u16_minutes{2886}, bitfield_idx_t{3}},  // TODO idem
        {1000U, u16_minutes{1447},
         traffic_days::invert_tmp_idx(bitfield_idx_t{1U})},
    };

    EXPECT_EQ(
        td.at(traffic_days::invert_tmp_idx(bitfield_idx_t{0U})).first.count(),
        1);
    EXPECT_TRUE(
        td.at(traffic_days::invert_tmp_idx(bitfield_idx_t{0U})).first.test(8));

    // EXPECT_EQ(td.at(bitfield_idx_t{3}).first.count(), 1);
    // EXPECT_TRUE(td.at(bitfield_idx_t{3}).first.test(5));

    EXPECT_EQ(
        td.at(traffic_days::invert_tmp_idx(bitfield_idx_t{1U})).first.count(),
        1);
    EXPECT_TRUE(
        td.at(traffic_days::invert_tmp_idx(bitfield_idx_t{1U})).first.test(7));

    ASSERT_EQ(tmp.size(), expected.size());
    for (auto i = 0U; i < expected.size(); ++i) {
      EXPECT_EQ(expected[i], tmp[i]);
    }

    EXPECT_TRUE(s1.to_saw(td).less(
        saw<nigiri::routing::saw_type::kTrafficDaysPower>{tmp, td}));
    EXPECT_TRUE(s2.to_saw(td).less(
        saw<nigiri::routing::saw_type::kTrafficDaysPower>{tmp, td}));
  }

  {
    auto const s3 = owning_saw<routing::saw_type::kTrafficDaysPower>{
        {metadata_tooth(13U),
         metadata_tooth(),
         metadata_tooth(),
         {1001U, u16_minutes{250}, bitfield_idx_t{0}},
         {1000U, u16_minutes{50}, bitfield_idx_t{2}},
         {1000U, u16_minutes{100}, bitfield_idx_t{3}},
         {1000U, u16_minutes{100}, bitfield_idx_t{4}},
         {1000U, u16_minutes{200}, bitfield_idx_t{5}},
         {999U, u16_minutes{10}, bitfield_idx_t{1}}},
        u16_minutes{0}};
    auto const s4 = owning_saw<routing::saw_type::kTrafficDaysPower>{
        {metadata_tooth(13U),
         metadata_tooth(),
         metadata_tooth(),
         {1300U, u16_minutes{10}, bitfield_idx_t{1}},
         {1300U, u16_minutes{10}, bitfield_idx_t{5}},
         {1160U, u16_minutes{60}, bitfield_idx_t{4}},
         {1150U, u16_minutes{50}, bitfield_idx_t{3}},
         {1090U, u16_minutes{110}, bitfield_idx_t{2}},
         {1009U, u16_minutes{10}, bitfield_idx_t{1}}},
        u16_minutes{0}};

    auto tmp = std::vector<tooth>{};
    auto td = nigiri::routing::traffic_days{};
    td.get_or_create(bitfield{"000001100100000"}, 9);
    td.get_or_create(bitfield{"000000110000000"}, 8);
    td.get_or_create(bitfield{"000010001000000"}, 10);
    td.get_or_create(bitfield{"000100010000000"}, 11);
    td.get_or_create(bitfield{"001000100000000"}, 12);
    td.get_or_create(bitfield{"010001000000000"}, 13);
    td.get_or_create(bitfield{"000110011000000"}, 11);
    td.get_or_create(bitfield{"010000000000000"}, 13);
    td.get_or_create(bitfield{"000001100000000"}, 9);

    s3.to_saw(td).concat(s4.to_saw(td), ch_edge_idx_t::invalid(),
                         ch_edge_idx_t::invalid(), false, tmp);

    auto expected = std::vector<tooth>{
        metadata_tooth(13U),
        metadata_tooth(20U),
        metadata_tooth(),
        {1001U, u16_minutes{309}, bitfield_idx_t{8}},
        {1000U, u16_minutes{200}, bitfield_idx_t{6}},
        {1000U, u16_minutes{220}, bitfield_idx_t{4}},
        {1000U, u16_minutes{310}, bitfield_idx_t{7}},
        {999U, u16_minutes{20}, bitfield_idx_t{1}},
    };

    ASSERT_EQ(tmp.size(), expected.size());
    for (auto i = 0U; i < expected.size(); ++i) {
      EXPECT_EQ(expected[i], tmp[i]);
    }

    EXPECT_TRUE(s3.to_saw(td).less(
        saw<nigiri::routing::saw_type::kTrafficDaysPower>{tmp, td}));
    EXPECT_TRUE(s4.to_saw(td).less(
        saw<nigiri::routing::saw_type::kTrafficDaysPower>{tmp, td}));
  }

  {
    auto const s3 = owning_saw<routing::saw_type::kTrafficDaysPower>{
        {metadata_tooth(12U),
         metadata_tooth(),
         metadata_tooth(),
         {930U, u16_minutes{1}, bitfield_idx_t{0}},
         {920U, u16_minutes{1}, bitfield_idx_t{1}},
         {885U, u16_minutes{1}, bitfield_idx_t{0}},
         {40U, u16_minutes{1}, bitfield_idx_t{0}}},
        u16_minutes{0}};
    auto const s4 = owning_saw<routing::saw_type::kTrafficDaysPower>{
        {metadata_tooth(12U),
         metadata_tooth(),
         metadata_tooth(),
         {931U, u16_minutes{7}, bitfield_idx_t{0}},
         {921U, u16_minutes{7}, bitfield_idx_t{1}},
         {886U, u16_minutes{7}, bitfield_idx_t{0}},
         {41U, u16_minutes{7}, bitfield_idx_t{0}}},
        u16_minutes{0}};

    auto tmp = std::vector<tooth>{};
    auto td = nigiri::routing::traffic_days{};
    td.get_or_create(bitfield{"000001111100000"}, 9);
    td.get_or_create(bitfield{"001110000000000"}, 12);

    s3.to_saw(td).concat(s4.to_saw(td), ch_edge_idx_t::invalid(),
                         ch_edge_idx_t::invalid(), false, tmp);

    auto expected = std::vector<tooth>{
        metadata_tooth(12U),
        metadata_tooth(8U),
        metadata_tooth(),
        {930U, u16_minutes{8}, bitfield_idx_t{0}},
        {920U, u16_minutes{8}, bitfield_idx_t{1}},
        {885U, u16_minutes{8}, bitfield_idx_t{0}},
        {40U, u16_minutes{8}, bitfield_idx_t{0}},
    };

    ASSERT_EQ(tmp.size(), expected.size());
    for (auto i = 0U; i < expected.size(); ++i) {
      EXPECT_EQ(expected[i], tmp[i]);
    }

    EXPECT_TRUE(s3.to_saw(td).less(
        saw<nigiri::routing::saw_type::kTrafficDaysPower>{tmp, td}));
    EXPECT_TRUE(s4.to_saw(td).less(
        saw<nigiri::routing::saw_type::kTrafficDaysPower>{tmp, td}));
  }

  {
    auto const s3 = owning_saw<routing::saw_type::kTrafficDaysPower>{
        {metadata_tooth(9U),
         metadata_tooth(),
         metadata_tooth(),
         {1001U, u16_minutes{10}, bitfield_idx_t{0}}},
        u16_minutes{0}};
    auto const s4 = owning_saw<routing::saw_type::kTrafficDaysPower>{
        {metadata_tooth(9U),
         metadata_tooth(),
         metadata_tooth(),
         {1000U, u16_minutes{10}, bitfield_idx_t{0}}},
        u16_minutes{0}};

    auto td = traffic_days{};
    td.bitfields_.emplace_back(bitfield{"001000000100000"}, 12);

    auto tmp = std::vector<tooth>{};
    s3.to_saw(td).concat(s4.to_saw(td), ch_edge_idx_t::invalid(),
                         ch_edge_idx_t::invalid(), true, tmp);

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
    s2.to_saw(td).concat_const(kReverse, s3.to_saw(td),
                               ch_edge_idx_t::invalid(),
                               ch_edge_idx_t::invalid(), tmp);

    auto const tmp_bfi = bitfield_idx_t::invalid() - 1U;

    auto expected =
        std::vector<tooth>{metadata_tooth(9U),
                           metadata_tooth(),
                           metadata_tooth(),
                           {1U, u16_minutes{1010}, bitfield_idx_t{0}},
                           {1437U, u16_minutes{1010}, tmp_bfi}};
    ASSERT_EQ(tmp.size(), expected.size());
    for (auto i = 0U; i < expected.size(); ++i) {
      EXPECT_EQ(expected[i], tmp[i]);
    }
    ASSERT_EQ(td.bitfields_.size(), 2);
    ASSERT_EQ(td.tmp_bitfields_.size(), 1);
    EXPECT_FALSE(td.at(tmp_bfi).first.test(4));
    EXPECT_FALSE(td.at(tmp_bfi).first.test(5));
    EXPECT_TRUE(td.at(tmp_bfi).first.test(6));
    EXPECT_TRUE(td.at(tmp_bfi).first.test(7));
    EXPECT_FALSE(td.at(tmp_bfi).first.test(8));
    EXPECT_EQ(td.at(tmp_bfi).second, 7);
  }

  {
    auto td = nigiri::routing::traffic_days{};

    auto const s3 = owning_saw<routing::saw_type::kTrafficDaysPower>{
        {metadata_tooth(9U),
         metadata_tooth(),
         metadata_tooth(),
         {1001U, u16_minutes{10}, bitfield_idx_t{0}},
         {1000U, u16_minutes{5}, bitfield_idx_t{1}},
         {1000U, u16_minutes{5}, bitfield_idx_t{0}},
         {1000U, u16_minutes{5}, bitfield_idx_t{1}},
         {999U, u16_minutes{10}, bitfield_idx_t{1}}},
        u16_minutes{0}};

    ASSERT_FALSE(s1.to_saw(td) == s2.to_saw(td));
    ASSERT_TRUE(s1.to_saw(td) != s2.to_saw(td));
    ASSERT_TRUE(s1.to_saw(td) == s3.to_saw(td));
    ASSERT_FALSE(s1.to_saw(td) != s3.to_saw(td));
  }
}
