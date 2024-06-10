#include "gtest/gtest.h"

#include "nigiri/qa/qa.h"
#include "nigiri/types.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::routing;

TEST(qa, test0) {
  auto a = pareto_set<journey>{};
  auto b = pareto_set<journey>{};

  EXPECT_DOUBLE_EQ(0.0, qa::rate(a, b));
  EXPECT_DOUBLE_EQ(0.0, qa::rate(b, a));

  auto a0 = journey{};
  a0.start_time_ = unixtime_t{sys_days{2024_y / June / 10} + 10_hours};
  a0.dest_time_ = unixtime_t{sys_days{2024_y / June / 10} + 12_hours};
  a0.transfers_ = 0U;
  a.add(std::move(a0));

  EXPECT_DOUBLE_EQ(qa::kMaxRating, qa::rate(a, b));
  EXPECT_DOUBLE_EQ(qa::kMinRating, qa::rate(b, a));
}

TEST(qa, test1) {
  auto a0 = journey{};
  a0.start_time_ = unixtime_t{sys_days{2024_y / June / 10} + 10_hours};
  a0.dest_time_ = unixtime_t{sys_days{2024_y / June / 10} + 12_hours};
  a0.transfers_ = 0U;
  auto a1 = journey{};
  a1.start_time_ = unixtime_t{sys_days{2024_y / June / 10} + 10_hours};
  a1.dest_time_ = unixtime_t{sys_days{2024_y / June / 10} + 12_hours};
  a1.transfers_ = 0U;

  auto a = pareto_set<journey>{};
  a.add(std::move(a0));
  a.add(std::move(a1));

  auto b0 = journey{};
  b0.start_time_ =
      unixtime_t{sys_days{2024_y / June / 10} + 9_hours + 33_minutes};
  b0.dest_time_ =
      unixtime_t{sys_days{2024_y / June / 10} + 11_hours + 34_minutes};
  b0.transfers_ = 0U;
  auto b1 = journey{};
  b1.start_time_ =
      unixtime_t{sys_days{2024_y / June / 10} + 9_hours + 45_minutes};
  b1.dest_time_ =
      unixtime_t{sys_days{2024_y / June / 10} + 12_hours + 16_minutes};
  b1.transfers_ = 1U;

  auto b = pareto_set<journey>{};
  b.add(std::move(b0));
  b.add(std::move(b1));

  EXPECT_DOUBLE_EQ(0.342008418450396, qa::rate(a, b));
  EXPECT_DOUBLE_EQ(-0.342008418450396, qa::rate(b, a));
}

TEST(qa, test2) {
  auto a0 = journey{};
  a0.start_time_ = unixtime_t{sys_days{2024_y / June / 10} + 10_hours};
  a0.dest_time_ = unixtime_t{sys_days{2024_y / June / 10} + 12_hours};
  a0.transfers_ = 0U;
  auto a1 = journey{};
  a1.start_time_ = unixtime_t{sys_days{2024_y / June / 10} + 10_hours};
  a1.dest_time_ = unixtime_t{sys_days{2024_y / June / 10} + 11_hours};
  a1.transfers_ = 1U;

  auto a = pareto_set<journey>{};
  a.add(std::move(a0));
  a.add(std::move(a1));

  auto b0 = journey{};
  b0.start_time_ =
      unixtime_t{sys_days{2024_y / June / 10} + 9_hours + 33_minutes};
  b0.dest_time_ =
      unixtime_t{sys_days{2024_y / June / 10} + 11_hours + 34_minutes};
  b0.transfers_ = 0U;
  auto b1 = journey{};
  b1.start_time_ =
      unixtime_t{sys_days{2024_y / June / 10} + 9_hours + 45_minutes};
  b1.dest_time_ =
      unixtime_t{sys_days{2024_y / June / 10} + 12_hours + 16_minutes};
  b1.transfers_ = 1U;

  auto b = pareto_set<journey>{};
  b.add(std::move(b0));
  b.add(std::move(b1));

  EXPECT_DOUBLE_EQ(15.116357209650212, qa::rate(a, b));
  EXPECT_DOUBLE_EQ(-15.116357209650212, qa::rate(b, a));
}

TEST(qa, test3) {
  auto a0 = journey{};
  a0.start_time_ = unixtime_t{sys_days{2024_y / June / 10} + 10_hours};
  a0.dest_time_ = unixtime_t{sys_days{2024_y / June / 10} + 12_hours};
  a0.transfers_ = 0U;
  auto a1 = journey{};
  a1.start_time_ = unixtime_t{sys_days{2024_y / June / 10} + 10_hours};
  a1.dest_time_ = unixtime_t{sys_days{2024_y / June / 10} + 11_hours};
  a1.transfers_ = 1U;

  auto a = pareto_set<journey>{};
  a.add(std::move(a0));
  a.add(std::move(a1));

  auto b1 = journey{};
  b1.start_time_ =
      unixtime_t{sys_days{2024_y / June / 10} + 9_hours + 45_minutes};
  b1.dest_time_ =
      unixtime_t{sys_days{2024_y / June / 10} + 12_hours + 16_minutes};
  b1.transfers_ = 1U;

  auto b = pareto_set<journey>{};
  b.add(std::move(b1));

  EXPECT_DOUBLE_EQ(31.478651986610316, qa::rate(a, b));
  EXPECT_DOUBLE_EQ(-31.478651986610316, qa::rate(b, a));
}

TEST(qa, test4) {
  auto a0 = journey{};
  a0.start_time_ = unixtime_t{sys_days{2024_y / June / 10} + 10_hours};
  a0.dest_time_ = unixtime_t{sys_days{2024_y / June / 10} + 12_hours};
  a0.transfers_ = 0U;
  auto a1 = journey{};
  a1.start_time_ = unixtime_t{sys_days{2024_y / June / 10} + 10_hours};
  a1.dest_time_ = unixtime_t{sys_days{2024_y / June / 10} + 11_hours};
  a1.transfers_ = 1U;
  auto a2 = journey{};
  a2.start_time_ =
      unixtime_t{sys_days{2024_y / June / 10} + 9_hours + 45_minutes};
  a2.dest_time_ =
      unixtime_t{sys_days{2024_y / June / 10} + 10_hours + 45_minutes};
  a2.transfers_ = 3U;

  auto a = pareto_set<journey>{};
  a.add(std::move(a0));
  a.add(std::move(a1));
  a.add(std::move(a2));

  auto b0 = journey{};
  b0.start_time_ =
      unixtime_t{sys_days{2024_y / June / 10} + 9_hours + 33_minutes};
  b0.dest_time_ =
      unixtime_t{sys_days{2024_y / June / 10} + 11_hours + 34_minutes};
  b0.transfers_ = 0U;
  auto b1 = journey{};
  b1.start_time_ =
      unixtime_t{sys_days{2024_y / June / 10} + 9_hours + 45_minutes};
  b1.dest_time_ =
      unixtime_t{sys_days{2024_y / June / 10} + 12_hours + 16_minutes};
  b1.transfers_ = 1U;

  auto b = pareto_set<journey>{};
  b.add(std::move(b0));
  b.add(std::move(b1));

  EXPECT_DOUBLE_EQ(20.839157331515052, qa::rate(a, b));
  EXPECT_DOUBLE_EQ(-20.839157331515052, qa::rate(b, a));
}

TEST(qa, test5) {
  auto a0 = journey{};
  a0.start_time_ = unixtime_t{sys_days{2024_y / June / 10} + 1_hours};
  a0.dest_time_ = unixtime_t{sys_days{2024_y / June / 10} + 23_hours};
  a0.transfers_ = 0U;
  auto a1 = journey{};
  a1.start_time_ = unixtime_t{sys_days{2024_y / June / 10} + 2_hours};
  a1.dest_time_ = unixtime_t{sys_days{2024_y / June / 10} + 22_hours};
  a1.transfers_ = 1U;
  auto a2 = journey{};
  a2.start_time_ = unixtime_t{sys_days{2024_y / June / 10} + 3_hours};
  a2.dest_time_ = unixtime_t{sys_days{2024_y / June / 10} + 21_hours};
  a2.transfers_ = 2U;
  auto a3 = journey{};
  a3.start_time_ = unixtime_t{sys_days{2024_y / June / 10} + 4_hours};
  a3.dest_time_ = unixtime_t{sys_days{2024_y / June / 10} + 20_hours};
  a3.transfers_ = 3U;
  auto a4 = journey{};
  a4.start_time_ = unixtime_t{sys_days{2024_y / June / 10} + 5_hours};
  a4.dest_time_ = unixtime_t{sys_days{2024_y / June / 10} + 19_hours};
  a4.transfers_ = 4U;
  auto a5 = journey{};
  a5.start_time_ = unixtime_t{sys_days{2024_y / June / 10} + 6_hours};
  a5.dest_time_ = unixtime_t{sys_days{2024_y / June / 10} + 18_hours};
  a5.transfers_ = 5U;

  auto a = pareto_set<journey>{};
  a.add(std::move(a0));
  a.add(std::move(a1));
  a.add(std::move(a2));
  a.add(std::move(a3));
  a.add(std::move(a4));
  a.add(std::move(a5));

  auto b0 = journey{};
  b0.start_time_ =
      unixtime_t{sys_days{2024_y / June / 10} + 9_hours + 33_minutes};
  b0.dest_time_ =
      unixtime_t{sys_days{2024_y / June / 10} + 11_hours + 34_minutes};
  b0.transfers_ = 0U;
  auto b1 = journey{};
  b1.start_time_ =
      unixtime_t{sys_days{2024_y / June / 10} + 9_hours + 45_minutes};
  b1.dest_time_ =
      unixtime_t{sys_days{2024_y / June / 10} + 12_hours + 16_minutes};
  b1.transfers_ = 1U;

  auto b = pareto_set<journey>{};
  b.add(std::move(b0));
  b.add(std::move(b1));

  EXPECT_DOUBLE_EQ(-32.37407751772509, qa::rate(a, b));
  EXPECT_DOUBLE_EQ(32.37407751772509, qa::rate(b, a));
}