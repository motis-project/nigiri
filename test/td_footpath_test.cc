#include "gtest/gtest.h"

#include "nigiri/routing/query.h"
#include "nigiri/td_footpath.h"
#include "nigiri/types.h"

using namespace nigiri;
using namespace date;
using namespace std::chrono_literals;

TEST(td_footpath, simple) {
  auto td_footpath_out = vecvec<location_idx_t, td_footpath>{};

  auto const a = location_idx_t{0};
  auto const b = location_idx_t{1};
  auto const c = location_idx_t{2};
  auto const d = location_idx_t{3};
  auto const e = location_idx_t{4};
  auto const f = location_idx_t{5};

  td_footpath_out.emplace_back({
      /*
       *                      |  2024/06/15 10:00 am   |   5 min
       * 2024/06/15 10:00 am  |  2024/06/15  2:00 pm   |  10 min
       * 2024/06/15  2:00 pm  |  2024/06/16 11:00 am   |   5 min
       * 2024/06/16 11:00 am  |  2024/06/17 12:00 am   |  not possible
       * 2024/06/17 12:00 am  |                        |   5 min
       */
      td_footpath{b, kNull, 5min},
      td_footpath{b, sys_days{June / 15 / 2024_y} + 10h, 10min},
      td_footpath{b, sys_days{June / 15 / 2024_y} + 14h, 5min},
      td_footpath{b, sys_days{June / 16 / 2024_y} + 11h,
                  footpath::kMaxDuration},
      td_footpath{b, sys_days{June / 17 / 2024_y} + 12h, 5min},

      /*
       *                      |  2024/06/21 12:00 am   |   not possible
       * 2024/06/17 12:00 am  |                        |   5 min
       */
      td_footpath{c, kNull, footpath::kMaxDuration},
      td_footpath{c, sys_days{June / 15 / 2024_y} + 13h, 5min},

      /*
       * not possible
       */
      td_footpath{d, kNull, footpath::kMaxDuration},

      /*
       * start > t
       */
      td_footpath{e, sys_days{June / 15 / 2024_y} + 12h, 5min},
      td_footpath{e, sys_days{June / 15 / 2024_y} + 13h, 10min},

      /*
       * start > t
       */
      td_footpath{f, sys_days{June / 15 / 2024_y} + 9h, 5min},
      td_footpath{f, sys_days{June / 15 / 2024_y} + 10h, 7min},
  });

  auto count = 0U;
  auto const map = vector_map<location_idx_t, unixtime_t>{
      kNull,
      sys_days{June / 15 / 2024_y} + 11h + 10min,
      sys_days{June / 15 / 2024_y} + 13h + 5min,
      kNull,
      sys_days{June / 15 / 2024_y} + 12h + 5min,
      sys_days{June / 15 / 2024_y} + 11h + 7min,
  };
  auto const now = sys_days{June / 15 / 2024_y} + 11h;
  for_each_footpath<direction::kForward>(
      td_footpath_out[a], now, [&](footpath const fp) {
        EXPECT_EQ(map[fp.target()], now + fp.duration());
        ++count;
        return utl::cflow::kContinue;
      });
  EXPECT_EQ(4, count);
}

TEST(td_footpath, backward_single) {
  auto const fps = std::vector<td_footpath>{
      {.target_ = location_idx_t{0U},
       .valid_from_ = sys_days{1970_y / January / 1},
       .duration_ = footpath::kMaxDuration},
      {.target_ = location_idx_t{0U},
       .valid_from_ = sys_days{2020_y / March / 30} + 10h,
       .duration_ = 10min}};

  auto called = false;
  for_each_footpath<direction::kBackward>(
      fps, sys_days{2020_y / March / 30} + 7h, [&](auto&&) {
        called = true;
        return utl::cflow::kBreak;
      });
  EXPECT_TRUE(!called);

  called = false;
  auto x = footpath{};
  for_each_footpath<direction::kBackward>(
      fps, sys_days{2020_y / March / 30} + 11h, [&](footpath const fp) {
        called = true;
        x = fp;
        return utl::cflow::kBreak;
      });
  EXPECT_TRUE(called);
  EXPECT_EQ(10min, x.duration());

  EXPECT_FALSE(get_td_duration<direction::kBackward>(
      fps, sys_days{2020_y / March / 30} + 10h));
}

TEST(td_footpath, backward_1) {
  auto const fps = std::vector<routing::td_offset>{
      {.valid_from_ = sys_days{1970_y / January / 1},
       .duration_ = 10min,
       .transport_mode_id_ = 0},
      {.valid_from_ = sys_days{2024_y / June / 19} + 7h + 30min,
       .duration_ = footpath::kMaxDuration,
       .transport_mode_id_ = 0},
      {.valid_from_ = sys_days{2024_y / June / 19} + 12h + 00min,
       .duration_ = 10min,
       .transport_mode_id_ = 0}};

  auto const d = get_td_duration<direction::kBackward>(
      fps, sys_days{2024_y / June / 19} + 8h);
  ASSERT_TRUE(d.has_value());
  EXPECT_EQ(31min, d->first);
}

TEST(td_footpath, backward) {
  auto const fps = std::vector<td_footpath>{
      {.target_ = location_idx_t{0U},
       .valid_from_ = sys_days{1970_y / January / 1},
       .duration_ = footpath::kMaxDuration},
      {.target_ = location_idx_t{0U},
       .valid_from_ = sys_days{2020_y / March / 30} + 10h,
       .duration_ = 10min},
      {.target_ = location_idx_t{0U},
       .valid_from_ = sys_days{2020_y / March / 30} + 12h,
       .duration_ = footpath::kMaxDuration}};

  auto called = false;
  auto x = footpath{};
  for_each_footpath<direction::kBackward>(
      fps, sys_days{2020_y / March / 30} + 12h, [&](footpath const fp) {
        called = true;
        x = fp;
        return utl::cflow::kBreak;
      });
  EXPECT_TRUE(called);
  EXPECT_EQ(10min, x.duration());

  called = false;
  for_each_footpath<direction::kBackward>(
      fps, sys_days{2020_y / March / 30} + 7h, [&](auto&&) {
        called = true;
        return utl::cflow::kBreak;
      });
  EXPECT_TRUE(!called);

  called = false;
  for_each_footpath<direction::kBackward>(
      fps, sys_days{2020_y / March / 30} + 11h, [&](footpath const fp) {
        called = true;
        x = fp;
        return utl::cflow::kBreak;
      });
  EXPECT_TRUE(called);
  EXPECT_EQ(10min, x.duration());

  called = false;
  for_each_footpath<direction::kBackward>(
      fps, sys_days{2020_y / March / 30} + 13h, [&](footpath const fp) {
        called = true;
        x = fp;
        return utl::cflow::kBreak;
      });
  EXPECT_TRUE(called);
  EXPECT_EQ(1h + 1min, x.duration());
}

TEST(td_footpath, backward_last) {
  auto const l = location_idx_t{0U};
  auto const fps = std::vector<td_footpath>{
      {l, date::sys_days{2025_y / August / 11} + 7h + 22min, 25min},
      {l, date::sys_days{2025_y / August / 11} + 7h + 47min,
       footpath::kMaxDuration},
      {l, date::sys_days{2025_y / August / 11} + 8h + 14min, 25min},
      {l, date::sys_days{2025_y / August / 11} + 8h + 47min,
       footpath::kMaxDuration},
      {l, date::sys_days{2025_y / August / 11} + 9h + 22min, 25min},
      {l, date::sys_days{2025_y / August / 11} + 10h + 8min,
       footpath::kMaxDuration},
      {l, date::sys_days{2025_y / August / 11} + 10h + 22min, 25min},
      {l, date::sys_days{2025_y / August / 11} + 11h + 17min,
       footpath::kMaxDuration},
      {l, date::sys_days{2025_y / August / 11} + 11h + 22min, 25min},
      {l, date::sys_days{2025_y / August / 11} + 12h + 17min,
       footpath::kMaxDuration},
      {l, date::sys_days{2025_y / August / 11} + 12h + 22min, 25min},
      {l, date::sys_days{2025_y / August / 11} + 13h + 8min,
       footpath::kMaxDuration},
      {l, date::sys_days{2025_y / August / 11} + 13h + 22min, 25min},
      {l, date::sys_days{2025_y / August / 11} + 14h + 8min,
       footpath::kMaxDuration},
      {l, date::sys_days{2025_y / August / 11} + 14h + 22min, 25min},
      {l, date::sys_days{2025_y / August / 11} + 15h + 8min,
       footpath::kMaxDuration},
      {l, date::sys_days{2025_y / August / 11} + 15h + 22min, 25min},
      {l, date::sys_days{2025_y / August / 11} + 16h + 8min,
       footpath::kMaxDuration},
      {l, date::sys_days{2025_y / August / 11} + 16h + 22min, 25min},
      {l, date::sys_days{2025_y / August / 11} + 16h + 47min,
       footpath::kMaxDuration},
      {l, date::sys_days{2025_y / August / 11} + 17h + 22min, 25min},
      {l, date::sys_days{2025_y / August / 11} + 17h + 47min,
       footpath::kMaxDuration},
      {l, date::sys_days{2025_y / August / 11} + 18h + 22min, 25min},
      {l, date::sys_days{2025_y / August / 11} + 18h + 47min,
       footpath::kMaxDuration}};

  auto const x = date::sys_days{2025_y / August / 12} + 4h + 8min;
  auto const fp = get_td_duration<direction::kBackward>(fps, x);

  ASSERT_TRUE(fp.has_value());
}