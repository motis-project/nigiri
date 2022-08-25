#include "doctest/doctest.h"

#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/routing/raptor.h"

#include "../loader/hrd/hrd_timetable.h"

using namespace nigiri;

constexpr auto const services = R"(
*Z 01337 80____       048 030                             %
*A VE 0000001 0000002 000005                              %
*G RE  0000001 0000002                                    %
0000001 A                            00230                %
0000002 B                     00330                       %
*Z 07331 80____       092 015                             %
*A VE 0000002 0000003 000005                              %
*G RE  0000002 0000003                                    %
0000002 B                            00230                %
0000003 C                     00330                       %
)";

constexpr auto const fwd_journeys = R"(
[2020-03-30 05:30, 2020-03-30 07:47]
TRANSFERS: 1
     FROM: (A, 0000001) [2020-03-30 05:30]
       TO: (C, 0000003) [2020-03-30 07:47]
leg 0: (A, 0000001) [2020-03-30 05:30] -> (B, 0000002) [2020-03-30 06:30]
  ROUTE=2
   0: 0000001 A...............................................                               d: 30.03 05:30 [30.03 07:30]  [{name=RE 1337, day=2020-03-30, id=80____/1337/0000001/05:30, src=0}]
   1: 0000002 B............................................... a: 30.03 06:30 [30.03 08:30]
leg 1: (B, 0000002) [2020-03-30 06:30] -> (B, 0000002) [2020-03-30 06:32]
  FOOTPATH
leg 2: (B, 0000002) [2020-03-30 06:45] -> (C, 0000003) [2020-03-30 07:45]
  ROUTE=0
   0: 0000002 B...............................................                               d: 30.03 06:45 [30.03 08:45]  [{name=RE 7331, day=2020-03-30, id=80____/7331/0000002/06:45, src=0}]
   1: 0000003 C............................................... a: 30.03 07:45 [30.03 09:45]
leg 3: (C, 0000003) [2020-03-30 07:45] -> (C, 0000003) [2020-03-30 07:47]
  FOOTPATH


[2020-03-30 05:00, 2020-03-30 07:17]
TRANSFERS: 1
     FROM: (A, 0000001) [2020-03-30 05:00]
       TO: (C, 0000003) [2020-03-30 07:17]
leg 0: (A, 0000001) [2020-03-30 05:00] -> (B, 0000002) [2020-03-30 06:00]
  ROUTE=2
   0: 0000001 A...............................................                               d: 30.03 05:00 [30.03 07:00]  [{name=RE 1337, day=2020-03-30, id=80____/1337/0000001/05:00, src=0}]
   1: 0000002 B............................................... a: 30.03 06:00 [30.03 08:00]
leg 1: (B, 0000002) [2020-03-30 06:00] -> (B, 0000002) [2020-03-30 06:02]
  FOOTPATH
leg 2: (B, 0000002) [2020-03-30 06:15] -> (C, 0000003) [2020-03-30 07:15]
  ROUTE=0
   0: 0000002 B...............................................                               d: 30.03 06:15 [30.03 08:15]  [{name=RE 7331, day=2020-03-30, id=80____/7331/0000002/06:15, src=0}]
   1: 0000003 C............................................... a: 30.03 07:15 [30.03 09:15]
leg 3: (C, 0000003) [2020-03-30 07:15] -> (C, 0000003) [2020-03-30 07:17]
  FOOTPATH


)";

constexpr auto const bwd_journeys = R"(
[2020-03-30 05:00, 2020-03-30 02:28]
TRANSFERS: 1
     FROM: (A, 0000001) [2020-03-30 02:28]
       TO: (C, 0000003) [2020-03-30 05:00]
leg 0: (A, 0000001) [2020-03-30 02:28] -> (A, 0000001) [2020-03-30 02:30]
  FOOTPATH
leg 1: (A, 0000001) [2020-03-30 02:30] -> (B, 0000002) [2020-03-30 03:30]
  ROUTE=2
   0: 0000001 A...............................................                               d: 30.03 02:30 [30.03 04:30]  [{name=RE 1337, day=2020-03-30, id=80____/1337/0000001/02:30, src=0}]
   1: 0000002 B............................................... a: 30.03 03:30 [30.03 05:30]
leg 2: (B, 0000002) [2020-03-30 03:58] -> (B, 0000002) [2020-03-30 04:00]
  FOOTPATH
leg 3: (B, 0000002) [2020-03-30 04:00] -> (C, 0000003) [2020-03-30 05:00]
  ROUTE=0
   0: 0000002 B...............................................                               d: 30.03 04:00 [30.03 06:00]  [{name=RE 7331, day=2020-03-30, id=80____/7331/0000002/04:00, src=0}]
   1: 0000003 C............................................... a: 30.03 05:00 [30.03 07:00]


[2020-03-30 05:30, 2020-03-30 02:58]
TRANSFERS: 1
     FROM: (A, 0000001) [2020-03-30 02:58]
       TO: (C, 0000003) [2020-03-30 05:30]
leg 0: (A, 0000001) [2020-03-30 02:58] -> (A, 0000001) [2020-03-30 03:00]
  FOOTPATH
leg 1: (A, 0000001) [2020-03-30 03:00] -> (B, 0000002) [2020-03-30 04:00]
  ROUTE=2
   0: 0000001 A...............................................                               d: 30.03 03:00 [30.03 05:00]  [{name=RE 1337, day=2020-03-30, id=80____/1337/0000001/03:00, src=0}]
   1: 0000002 B............................................... a: 30.03 04:00 [30.03 06:00]
leg 2: (B, 0000002) [2020-03-30 04:28] -> (B, 0000002) [2020-03-30 04:30]
  FOOTPATH
leg 3: (B, 0000002) [2020-03-30 04:30] -> (C, 0000003) [2020-03-30 05:30]
  ROUTE=0
   0: 0000002 B...............................................                               d: 30.03 04:30 [30.03 06:30]  [{name=RE 7331, day=2020-03-30, id=80____/7331/0000002/04:30, src=0}]
   1: 0000003 C............................................... a: 30.03 05:30 [30.03 07:30]


[2020-03-30 05:45, 2020-03-30 03:28]
TRANSFERS: 1
     FROM: (A, 0000001) [2020-03-30 03:28]
       TO: (C, 0000003) [2020-03-30 05:45]
leg 0: (A, 0000001) [2020-03-30 03:28] -> (A, 0000001) [2020-03-30 03:30]
  FOOTPATH
leg 1: (A, 0000001) [2020-03-30 03:30] -> (B, 0000002) [2020-03-30 04:30]
  ROUTE=2
   0: 0000001 A...............................................                               d: 30.03 03:30 [30.03 05:30]  [{name=RE 1337, day=2020-03-30, id=80____/1337/0000001/03:30, src=0}]
   1: 0000002 B............................................... a: 30.03 04:30 [30.03 06:30]
leg 2: (B, 0000002) [2020-03-30 04:43] -> (B, 0000002) [2020-03-30 04:45]
  FOOTPATH
leg 3: (B, 0000002) [2020-03-30 04:45] -> (C, 0000003) [2020-03-30 05:45]
  ROUTE=0
   0: 0000002 B...............................................                               d: 30.03 04:45 [30.03 06:45]  [{name=RE 7331, day=2020-03-30, id=80____/7331/0000002/04:45, src=0}]
   1: 0000003 C............................................... a: 30.03 05:45 [30.03 07:45]


)";

TEST_CASE("raptor, simple_search") {
  using namespace date;
  auto tt = std::make_shared<timetable>();
  auto const src = source_idx_t{0U};
  load_timetable(
      src, loader::hrd::hrd_5_20_26,
      test_data::hrd_timetable::base().add(
          {loader::hrd::hrd_5_20_26.fplan_ / "services.101", services}),
      *tt);
  auto state = routing::search_state{};

  auto fwd_r = routing::raptor<direction::kForward>{
      tt, state,
      routing::query{
          .interval_ = {unixtime_t{sys_days{2020_y / March / 30}} + 5_hours,
                        unixtime_t{sys_days{2020_y / March / 30}} + 6_hours},
          .start_ = {nigiri::routing::offset{
              .location_ = tt->locations_.location_id_to_idx_.at(
                  {.id_ = "0000001", .src_ = src}),
              .offset_ = 0_minutes,
              .type_ = 0U}},
          .destinations_ = {{nigiri::routing::offset{
              .location_ = tt->locations_.location_id_to_idx_.at(
                  {.id_ = "0000003", .src_ = src}),
              .offset_ = 0_minutes,
              .type_ = 0U}}},
          .via_destinations_ = {},
          .allowed_classes_ = bitset<kNumClasses>::max(),
          .max_transfers_ = nigiri::routing::kMaxTransfers,
          .min_connection_count_ = 0U,
          .extend_interval_earlier_ = false,
          .extend_interval_later_ = false}};
  fwd_r.route();

  std::stringstream fws_ss("\n");
  fws_ss << "\n";
  for (auto const& x : fwd_r.state_.results_) {
    x.print(fws_ss, *tt);
    fws_ss << "\n\n";
  }
  CHECK_EQ(std::string_view{fwd_journeys}, fws_ss.str());

  auto bwd_r = routing::raptor<direction::kBackward>{
      tt, state,
      routing::query{
          .interval_ = {unixtime_t{sys_days{2020_y / March / 30}} + 5_hours,
                        unixtime_t{sys_days{2020_y / March / 30}} + 6_hours},
          .start_ = {nigiri::routing::offset{
              .location_ = tt->locations_.location_id_to_idx_.at(
                  {.id_ = "0000003", .src_ = src}),
              .offset_ = 0_minutes,
              .type_ = 0U}},
          .destinations_ = {{nigiri::routing::offset{
              .location_ = tt->locations_.location_id_to_idx_.at(
                  {.id_ = "0000001", .src_ = src}),
              .offset_ = 0_minutes,
              .type_ = 0U}}},
          .via_destinations_ = {},
          .allowed_classes_ = bitset<kNumClasses>::max(),
          .max_transfers_ = nigiri::routing::kMaxTransfers,
          .min_connection_count_ = 0U,
          .extend_interval_earlier_ = false,
          .extend_interval_later_ = false}};
  bwd_r.route();

  std::stringstream bwd_ss;
  bwd_ss << "\n";
  for (auto const& x : bwd_r.state_.results_) {
    x.print(bwd_ss, *tt);
    bwd_ss << "\n\n";
  }
  CHECK_EQ(std::string_view{bwd_journeys}, bwd_ss.str());
};