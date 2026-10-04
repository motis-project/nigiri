#include "gtest/gtest.h"

#include <string_view>

#include "nigiri/routing/query.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

#include "../raptor_search.h"
#include "../transfer_rules_util.h"

// GPU search with transfer rules. In CUDA builds, test::raptor_search compares
// the device search with the host search on the same query.

using namespace nigiri;
using nigiri::test::add_empty_profile;
using nigiri::test::at;
using nigiri::test::feed;
using nigiri::test::kDay;
using nigiri::test::lidx;
using nigiri::test::load_feeds;

constexpr auto const kProfile = profile_idx_t{1U};

// GT1 arrives at GX 10:30, GT2 leaves it 10:40, with the given transfers.txt
// rows. kProfile is a second profile without footpaths that ignores
// transfers.txt.
timetable load_gx(std::string_view const transfers) {
  auto tt = load_feeds(
      {feed({{"GX", 65.0, 24.0}, {"GA", 65.1, 24.0}, {"GB", 65.2, 24.0}},
            {{"GT1", "R1", {{"GA", "10:00"}, {"GX", "10:30"}}},
             {"GT2", "R2", {{"GX", "10:40"}, {"GB", "11:00"}}}},
            transfers)});
  add_empty_profile(tt, kProfile);
  return tt;
}

// Changing vehicles at GX is banned by an unqualified same-stop row, and no
// rule is qualified, so the timetable has no virtual location. The ban is GX's
// transfer time, which every profile has: the other profile finds no journey
// either - on the host as on the device.
TEST(gpu_transfer_rules, other_profile_without_virtual_locations) {
  auto const tt = load_gx("GX,GX,3,,,,,\n");
  auto const res = test::raptor_search(
      tt, nullptr,
      routing::query{.start_time_ = at("10:00"),
                     .start_ = {{lidx(tt, "GA"), 0_minutes, 0U}},
                     .destination_ = {{lidx(tt, "GB"), 0_minutes, 0U}},
                     .prf_idx_ = kProfile});
  EXPECT_EQ(0U, res.size());
}

// W2's own transfer time is 0 (type 1 row), and the R9 -> R5 rule gives R5's
// departures (GE) a virtual location. A start at that virtual location reaches
// W2, where GD leaves, through W2's 0 min hub. With min_transfer_time_ = 5
// min, that start walk takes 5 min like any transfer: from 09:55, GD (10:02)
// is reached. The device's start leg has to be the host's.
TEST(gpu_transfer_rules, start_leg_through_zero_min_hub) {
  auto const tt =
      load_feeds({feed({{"W2", 65.5, 24.5}, {"WD", 65.6, 24.5}},
                       {{"GD", "R6", {{"W2", "10:02"}, {"WD", "10:30"}}},
                        {"GE", "R5", {{"W2", "11:02"}, {"WD", "11:30"}}}},
                       "W2,W2,1,,,,,\n"
                       "W2,W2,2,300,R9,R5,,\n",
                       {"R9"})});
  auto virt = location_idx_t::invalid();
  tt.locations_.for_each_virt(lidx(tt, "W2"),
                              [&](location_idx_t const l) { virt = l; });
  ASSERT_NE(location_idx_t::invalid(), virt);

  auto const res = test::raptor_search(
      tt, nullptr,
      routing::query{.start_time_ = at("09:55"),
                     .use_start_footpaths_ = true,
                     .start_ = {{virt, 0_minutes, 0U}},
                     .destination_ = {{lidx(tt, "WD"), 0_minutes, 0U}},
                     .transfer_time_settings_ = {
                         .default_ = false, .min_transfer_time_ = 5_minutes}});
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(at("10:30"), begin(res)->dest_time_);
  EXPECT_EQ(at("10:00"), begin(res)->legs_.front().arr_time_);
}

#if defined(NIGIRI_CUDA)

// A profile other than the default one with real-time time-dependent
// footpaths (an elevator outage at GX) makes the device pong fill its bounds
// from the bit vector of locations with such footpaths. The kernel also walks
// the label slots of real-time virtual locations, which that bit vector does
// not cover: it must not read past its end (visible under compute-sanitizer).
TEST(gpu_transfer_rules, fill_bounds_with_td_footpaths) {
  auto const tt = load_gx("");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  auto const gx = lidx(tt, "GX");
  rtt.has_td_footpaths_out_[kProfile].set(gx, true);
  rtt.has_td_footpaths_in_[kProfile].set(gx, true);
  rtt.td_footpaths_out_[kProfile].resize(tt.n_locations());
  rtt.td_footpaths_in_[kProfile].resize(tt.n_locations());

  auto const res = test::raptor_search(
      tt, &rtt,
      routing::query{.start_time_ = interval{at("09:30"), at("10:30")},
                     .start_ = {{lidx(tt, "GA"), 0_minutes, 0U}},
                     .destination_ = {{lidx(tt, "GB"), 0_minutes, 0U}},
                     .prf_idx_ = kProfile},
      direction::kForward);
  EXPECT_EQ(1U, res.size());
}

#endif
