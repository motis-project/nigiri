#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/loader/init_finish.h"

#include "nigiri/routing/direct.h"
#include "nigiri/routing/raptor/pong.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/rt_timetable.h"
#include "../raptor_search.h"
#include "results_to_string.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace std::chrono_literals;
using nigiri::test::raptor_search;

namespace {

// time-dependent footpath at start: works [9:30-9:45], duration=10min
// time-dependent foopath at end: does not work [14:30-15:30], duration=10min
//
// Interchange at B
// A  --> B --> C
// A  --> D --> C
//
// A->B->C
// T1: A->B 10:00-11:00
// T2: B->C 11:30-12:00
// T3: B->C 12:00-12:30
//
// A->D->C
// T4: A->D 10:00-12:00
// T5: D->C 13:00-15:00
//
// Scenario 1:
// Everything works
// A@10:00 --T1--> 11:00 @ B @ 11:30 --T2--> 12:00 @ C --> 12:10
//
// Scenario 2:
// Elevator at B blocked completely, journey via D
// A@10:00 --T4--> 12:00 @ D @ 13:00 --T5--> 15:00 @ C --wait--> 15:30 --> 15:40
//
// Scenario 3:
// Elevator at B blocked until 11:25, 10min footpath = 11:35 arrival at B2
// A@10:00 --T1--> 11:00 @ B1 @ 11:00
// wait for evelator to work 11:00 - 11:25
// use elevator +10min       11:25 - 11:35
// B2 @ 12:00 --T3--> 12:30 @ C
mem_dir test_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B1,B1,,2.0,3.0,,
B2,B2,,2.0,3.0,,
C,C,,4.0,5.0,,
D,D,,6.0,7.0,,

# calendar_dates.txt
service_id,date,exception_type
S,20240619,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,RE 1,,,2
R2,DB,RE 2,,,2
R3,DB,RE 1,,,2
R4,DB,RE 2,,,2
R5,DB,RE 1,,,2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id,wheelchair_accessible
R1,S,T1,RE 1,,1
R2,S,T2,RE 2,,1
R3,S,T3,RE 3,,1
R4,S,T4,RE 4,,1
R5,S,T5,RE 5,,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T1,10:00:00,10:00:00,A,1,0,0
T1,11:00:00,11:00:00,B1,2,0,0
T2,11:30:00,11:30:00,B2,1,0,0
T2,12:00:00,12:00:00,C,2,0,0
T3,12:00:00,12:00:00,B2,1,0,0
T3,12:30:00,12:30:00,C,2,0,0
T4,10:00:00,10:00:00,A,1,0,0
T4,12:00:00,12:00:00,D,2,0,0
T5,13:00:00,13:00:00,D,1,0,0
T5,15:00:00,15:00:00,C,2,0,0
)");
}

}  // namespace

// clang-format-off
constexpr auto const kEverythingWorks = R"(
[2024-06-19 07:00, 2024-06-19 10:10]
TRANSFERS: 1
     FROM: (START, START) [2024-06-19 07:44]
       TO: (END, END) [2024-06-19 10:10]
leg 0: (START, START) [2024-06-19 07:44] -> (A, A) [2024-06-19 07:54]
  MUMO (payload=0, duration=10)
leg 1: (A, A) [2024-06-19 08:00] -> (B1, B1) [2024-06-19 09:00]
   0: A       A...............................................                               d: 19.06 08:00 [19.06 10:00]  [{name=RE 1, day=2024-06-19, id=T1, src=0}]
   1: B1      B1.............................................. a: 19.06 09:00 [19.06 11:00]
leg 2: (B1, B1) [2024-06-19 09:00] -> (B2, B2) [2024-06-19 09:20]
  FOOTPATH (duration=20)
leg 3: (B2, B2) [2024-06-19 09:30] -> (C, C) [2024-06-19 10:00]
   0: B2      B2..............................................                               d: 19.06 09:30 [19.06 11:30]  [{name=RE 2, day=2024-06-19, id=T2, src=0}]
   1: C       C............................................... a: 19.06 10:00 [19.06 12:00]
leg 4: (C, C) [2024-06-19 10:00] -> (END, END) [2024-06-19 10:10]
  MUMO (payload=0, duration=10)

)";

constexpr auto const kElevatorOutOfOrder = R"(
[2024-06-19 07:00, 2024-06-19 13:40]
TRANSFERS: 1
     FROM: (START, START) [2024-06-19 07:44]
       TO: (END, END) [2024-06-19 13:40]
leg 0: (START, START) [2024-06-19 07:44] -> (A, A) [2024-06-19 07:54]
  MUMO (payload=0, duration=10)
leg 1: (A, A) [2024-06-19 08:00] -> (D, D) [2024-06-19 10:00]
   0: A       A...............................................                               d: 19.06 08:00 [19.06 10:00]  [{name=RE 2, day=2024-06-19, id=T4, src=0}]
   1: D       D............................................... a: 19.06 10:00 [19.06 12:00]
leg 2: (D, D) [2024-06-19 10:00] -> (D, D) [2024-06-19 10:02]
  FOOTPATH (duration=2)
leg 3: (D, D) [2024-06-19 11:00] -> (C, C) [2024-06-19 13:00]
   0: D       D...............................................                               d: 19.06 11:00 [19.06 13:00]  [{name=RE 1, day=2024-06-19, id=T5, src=0}]
   1: C       C............................................... a: 19.06 13:00 [19.06 15:00]
leg 4: (C, C) [2024-06-19 13:30] -> (END, END) [2024-06-19 13:40]
  MUMO (payload=0, duration=10)

)";

constexpr auto const kElevatorStartsWorkingAt1125 = R"(
[2024-06-19 07:00, 2024-06-19 10:40]
TRANSFERS: 1
     FROM: (START, START) [2024-06-19 07:44]
       TO: (END, END) [2024-06-19 10:40]
leg 0: (START, START) [2024-06-19 07:44] -> (A, A) [2024-06-19 07:54]
  MUMO (payload=0, duration=10)
leg 1: (A, A) [2024-06-19 08:00] -> (B1, B1) [2024-06-19 09:00]
   0: A       A...............................................                               d: 19.06 08:00 [19.06 10:00]  [{name=RE 1, day=2024-06-19, id=T1, src=0}]
   1: B1      B1.............................................. a: 19.06 09:00 [19.06 11:00]
leg 2: (B1, B1) [2024-06-19 09:25] -> (B2, B2) [2024-06-19 09:35]
  FOOTPATH (duration=10)
leg 3: (B2, B2) [2024-06-19 10:00] -> (C, C) [2024-06-19 10:30]
   0: B2      B2..............................................                               d: 19.06 10:00 [19.06 12:00]  [{name=RE 1, day=2024-06-19, id=T3, src=0}]
   1: C       C............................................... a: 19.06 10:30 [19.06 12:30]
leg 4: (C, C) [2024-06-19 10:30] -> (END, END) [2024-06-19 10:40]
  MUMO (payload=0, duration=10)

)";

TEST(routing, td_footpath) {
  constexpr auto const kProfile = profile_idx_t{2U};

  timetable tt;
  tt.date_range_ = {date::sys_days{2024_y / June / 18},
                    date::sys_days{2024_y / June / 20}};
  register_special_stations(tt);
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  tt.fwd_search_lb_graph_[kWheelchairProfile] =
      tt.fwd_search_lb_graph_[kDefaultProfile];
  tt.bwd_search_lb_graph_[kWheelchairProfile] =
      tt.bwd_search_lb_graph_[kDefaultProfile];

  auto const find_loc = [&](std::string_view id) {
    auto const idx = tt.find(location_id{id, source_idx_t{0U}});
    EXPECT_TRUE(idx.has_value()) << id;
    return idx.value_or(location_idx_t::invalid());
  };
  auto const A = find_loc("A");
  auto const C = find_loc("C");
  auto const B1 = find_loc("B1");
  auto const B2 = find_loc("B2");

  tt.locations_.footpaths_out_[kProfile].resize(tt.n_locations());
  tt.locations_.footpaths_in_[kProfile].resize(tt.n_locations());
  tt.locations_.footpaths_out_[kProfile][B1].push_back(footpath{B2, 20min});
  tt.locations_.footpaths_in_[kProfile][B2].push_back(footpath{B1, 20min});

  auto rtt = rt::create_rt_timetable(tt, sys_days{2024_y / June / 19});

  auto const run_search = [&]() {
    return raptor_search(
        tt, &rtt,
        routing::query{
            .start_time_ = unixtime_t{sys_days{2024_y / June / 19}} + 7h,
            .start_match_mode_ = routing::location_match_mode::kIntermodal,
            .dest_match_mode_ = routing::location_match_mode::kIntermodal,
            .use_start_footpaths_ = false,
            .td_start_ =
                {{{A,
                   {{.valid_from_ = sys_days{1970_y / January / 1},
                     .duration_ = footpath::kMaxDuration,
                     .transport_mode_payload_ = 0},
                    {.valid_from_ = sys_days{2024_y / June / 19} + 7h + 30min,
                     .duration_ = 10min,
                     .transport_mode_payload_ = 0},
                    {.valid_from_ = sys_days{2024_y / June / 19} + 7h + 45min,
                     .duration_ = footpath::kMaxDuration,
                     .transport_mode_payload_ = 0}}}}},
            .td_dest_ =
                {{{C,
                   {{.valid_from_ = sys_days{1970_y / January / 1},
                     .duration_ = 10min,
                     .transport_mode_payload_ = 0},
                    {.valid_from_ = sys_days{2024_y / June / 19} + 12h + 30min,
                     .duration_ = footpath::kMaxDuration,
                     .transport_mode_payload_ = 0},
                    {.valid_from_ = sys_days{2024_y / June / 19} + 13h + 30min,
                     .duration_ = 10min,
                     .transport_mode_payload_ = 0}}}}},
            .prf_idx_ = 2U},
        direction::kForward);
  };

  // Base: elevator available, no real-time information.
  EXPECT_EQ(kEverythingWorks, to_string(tt, run_search()));

  // Switch to real-time footpaths but don't add any footpaths.
  // Represents "elevator broken forever".
  rtt.has_td_footpaths_in_[kProfile].set(B1, true);
  rtt.has_td_footpaths_in_[kProfile].set(B2, true);
  rtt.has_td_footpaths_out_[kProfile].set(B1, true);
  rtt.has_td_footpaths_out_[kProfile].set(B2, true);
  rtt.td_footpaths_out_[kProfile].resize(tt.n_locations());
  rtt.td_footpaths_in_[kProfile].resize(tt.n_locations());

  EXPECT_EQ(kElevatorOutOfOrder, to_string(tt, run_search()));

  // Add elevator available beginning with 11:25 with 10min footpath length.
  rtt.td_footpaths_out_[kProfile][B1].push_back(td_footpath{
      B2, unixtime_t{sys_days{2024_y / June / 19} + 9h + 25min}, 10min});
  rtt.td_footpaths_in_[kProfile][B2].push_back(td_footpath{
      B1, unixtime_t{sys_days{2024_y / June / 19} + 9h + 25min}, 10min});

  EXPECT_EQ(kElevatorStartsWorkingAt1125, to_string(tt, run_search()));
}

// clang-format off
constexpr auto const kPongElevatorStartsWorkingAt1125 = R"(
[2024-06-19 08:00, 2024-06-19 10:30]
TRANSFERS: 1
     FROM: (A, A) [2024-06-19 08:00]
       TO: (C, C) [2024-06-19 10:30]
leg 0: (A, A) [2024-06-19 08:00] -> (B1, B1) [2024-06-19 09:00]
   0: A       A...............................................                               d: 19.06 08:00 [19.06 10:00]  [{name=RE 1, day=2024-06-19, id=T1, src=0}]
   1: B1      B1.............................................. a: 19.06 09:00 [19.06 11:00]
leg 1: (B1, B1) [2024-06-19 09:50] -> (B2, B2) [2024-06-19 10:00]
  FOOTPATH (duration=10)
leg 2: (B2, B2) [2024-06-19 10:00] -> (C, C) [2024-06-19 10:30]
   0: B2      B2..............................................                               d: 19.06 10:00 [19.06 12:00]  [{name=RE 1, day=2024-06-19, id=T3, src=0}]
   1: C       C............................................... a: 19.06 10:30 [19.06 12:30]

)";
// clang-format on

TEST(routing, td_footpath_pong_keeps_the_wait) {
  // Scenario 3 with PONG (local times):
  // T1 arrives at B1 at 11:00, but the footpath B1 -> B2 (elevator) is only
  // usable from 11:25. The journey must wait at B1 and walk 11:50-12:00 to
  // catch T3. PONG builds its journeys from a backward search, which used
  // to move the walk to 11:00. Also, the static footpath (5min) ignores the
  // outage, so it must not replace the time-dependent one (10min).
  constexpr auto const kProfile = profile_idx_t{2U};

  timetable tt;
  tt.date_range_ = {date::sys_days{2024_y / June / 18},
                    date::sys_days{2024_y / June / 20}};
  register_special_stations(tt);
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  tt.fwd_search_lb_graph_[kWheelchairProfile] =
      tt.fwd_search_lb_graph_[kDefaultProfile];
  tt.bwd_search_lb_graph_[kWheelchairProfile] =
      tt.bwd_search_lb_graph_[kDefaultProfile];

  auto const A = tt.find(location_id{"A", source_idx_t{0U}}).value();
  auto const C = tt.find(location_id{"C", source_idx_t{0U}}).value();
  auto const B1 = tt.find(location_id{"B1", source_idx_t{0U}}).value();
  auto const B2 = tt.find(location_id{"B2", source_idx_t{0U}}).value();

  tt.locations_.footpaths_out_[kProfile].resize(tt.n_locations());
  tt.locations_.footpaths_in_[kProfile].resize(tt.n_locations());
  tt.locations_.footpaths_out_[kProfile][B1].push_back(footpath{B2, 5min});
  tt.locations_.footpaths_in_[kProfile][B2].push_back(footpath{B1, 5min});

  auto rtt = rt::create_rt_timetable(tt, sys_days{2024_y / June / 19});
  for (auto const l : {B1, B2}) {
    rtt.has_td_footpaths_in_[kProfile].set(l, true);
    rtt.has_td_footpaths_out_[kProfile].set(l, true);
  }
  rtt.td_footpaths_out_[kProfile].resize(tt.n_locations());
  rtt.td_footpaths_in_[kProfile].resize(tt.n_locations());
  rtt.td_footpaths_out_[kProfile][B1].push_back(td_footpath{
      B2, unixtime_t{sys_days{2024_y / June / 19} + 9h + 25min}, 10min});
  rtt.td_footpaths_in_[kProfile][B2].push_back(td_footpath{
      B1, unixtime_t{sys_days{2024_y / June / 19} + 9h + 25min}, 10min});

  auto search_state = routing::search_state{};
  auto raptor_state = routing::raptor_state{};
  auto const result = routing::pong_search(
      tt, &rtt, search_state, raptor_state,
      routing::query{
          .start_time_ =
              interval<unixtime_t>{sys_days{2024_y / June / 19} + 7h,
                                   sys_days{2024_y / June / 19} + 9h},
          .start_match_mode_ = routing::location_match_mode::kEquivalent,
          .dest_match_mode_ = routing::location_match_mode::kEquivalent,
          .start_ = {{A, 0min, 0U}},
          .destination_ = {{C, 0min, 0U}},
          .prf_idx_ = kProfile},
      direction::kForward);
  EXPECT_EQ(kPongElevatorStartsWorkingAt1125,
            to_string(tt, &rtt, *result.journeys_));
}

TEST(routing, td_footpath_lookup_keeps_the_wait) {
  // The wait for a time-dependent footpath is spent at the transport's stop,
  // so the footpath leg only covers the walk.
  constexpr auto const kProfile = profile_idx_t{2U};

  timetable tt;
  tt.date_range_ = {date::sys_days{2024_y / June / 18},
                    date::sys_days{2024_y / June / 20}};
  register_special_stations(tt);
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  auto const B1 = tt.find(location_id{"B1", source_idx_t{0U}}).value();
  auto const B2 = tt.find(location_id{"B2", source_idx_t{0U}}).value();
  auto const day = sys_days{2024_y / June / 19};

  tt.locations_.footpaths_out_[kProfile].resize(tt.n_locations());
  tt.locations_.footpaths_in_[kProfile].resize(tt.n_locations());

  auto rtt = rt::create_rt_timetable(tt, day);
  for (auto const l : {B1, B2}) {
    rtt.has_td_footpaths_in_[kProfile].set(l, true);
    rtt.has_td_footpaths_out_[kProfile].set(l, true);
  }
  rtt.td_footpaths_out_[kProfile].resize(tt.n_locations());
  rtt.td_footpaths_in_[kProfile].resize(tt.n_locations());

  // Alighting at B1 at 09:00, the footpath to B2 is usable from 09:25:
  // wait until 09:25, then walk.
  rtt.td_footpaths_in_[kProfile][B2].push_back(
      td_footpath{B1, unixtime_t{day + 9h + 25min}, 10min});
  auto const q = routing::query{.prf_idx_ = kProfile};
  auto const alighting = routing::lookup_footpath(
      B1, unixtime_t{day + 9h}, routing::side::kAlighting, tt, &rtt, q,
      {{B2, 0min, 0U}}, routing::location_match_mode::kExact, true);
  ASSERT_TRUE(alighting.has_value());
  EXPECT_EQ(B1, alighting->from_);
  EXPECT_EQ(B2, alighting->to_);
  EXPECT_EQ(unixtime_t{day + 9h + 25min}, alighting->dep_time_);
  EXPECT_EQ(unixtime_t{day + 9h + 35min}, alighting->arr_time_);

  // Boarding at B2 at 10:00, the footpath from B1 is only usable until 09:40:
  // walk until 09:49 at the latest, then wait.
  rtt.td_footpaths_out_[kProfile][B1].push_back(
      td_footpath{B2, unixtime_t{day + 9h + 25min}, 10min});
  rtt.td_footpaths_out_[kProfile][B1].push_back(
      td_footpath{B2, unixtime_t{day + 9h + 40min}, footpath::kMaxDuration});
  auto const boarding = routing::lookup_footpath(
      B2, unixtime_t{day + 10h}, routing::side::kBoarding, tt, &rtt, q,
      {{B1, 0min, 0U}}, routing::location_match_mode::kExact, true);
  ASSERT_TRUE(boarding.has_value());
  EXPECT_EQ(B1, boarding->from_);
  EXPECT_EQ(B2, boarding->to_);
  EXPECT_EQ(unixtime_t{day + 9h + 39min}, boarding->dep_time_);
  EXPECT_EQ(unixtime_t{day + 9h + 49min}, boarding->arr_time_);
}

TEST(routing, td_offset_lookup_keeps_the_wait) {
  // The wait for a time-dependent offset is spent at the transport's stop, so
  // the offset leg only covers the walk.
  auto const day = sys_days{2024_y / June / 19};
  auto const l = location_idx_t{42U};
  auto const td_offsets =
      routing::td_offsets_t{{l,
                             {{.valid_from_ = sys_days{1970_y / January / 1},
                               .duration_ = footpath::kMaxDuration,
                               .transport_mode_payload_ = 0},
                              {.valid_from_ = day + 9h + 25min,
                               .duration_ = 10min,
                               .transport_mode_payload_ = 0},
                              {.valid_from_ = day + 9h + 40min,
                               .duration_ = footpath::kMaxDuration,
                               .transport_mode_payload_ = 0}}}};

  // Alighting at 09:00, the offset is usable from 09:25: wait until 09:25,
  // then walk.
  auto const alighting = routing::lookup_offset(
      l, unixtime_t{day + 9h}, routing::side::kAlighting, {}, td_offsets);
  ASSERT_TRUE(alighting.has_value());
  EXPECT_EQ(l, alighting->from_);
  EXPECT_EQ(get_special_station(special_station::kEnd), alighting->to_);
  EXPECT_EQ(unixtime_t{day + 9h + 25min}, alighting->dep_time_);
  EXPECT_EQ(unixtime_t{day + 9h + 35min}, alighting->arr_time_);
  EXPECT_EQ(10min, std::get<routing::offset>(alighting->uses_).duration());

  // Boarding at 10:00, the offset is only usable until 09:40: walk until 09:49
  // at the latest, then wait.
  auto const boarding = routing::lookup_offset(
      l, unixtime_t{day + 10h}, routing::side::kBoarding, {}, td_offsets);
  ASSERT_TRUE(boarding.has_value());
  EXPECT_EQ(get_special_station(special_station::kStart), boarding->from_);
  EXPECT_EQ(l, boarding->to_);
  EXPECT_EQ(unixtime_t{day + 9h + 39min}, boarding->dep_time_);
  EXPECT_EQ(unixtime_t{day + 9h + 49min}, boarding->arr_time_);
  EXPECT_EQ(10min, std::get<routing::offset>(boarding->uses_).duration());
}

TEST(routing, td_footpath_pong_earliest_alternative_keeps_the_wait) {
  // With three transports, PONG replaces the middle one with its earliest
  // alternative (`get_earliest_alternative`). The time-dependent footpath
  // B1→B2 is only usable from 11:25 (Berlin), after the arrival at B1 (11:00):
  // the footpath before the middle transport must not start at the arrival.
  constexpr auto const kProfile = profile_idx_t{2U};

  timetable tt;
  tt.date_range_ = {date::sys_days{2024_y / June / 18},
                    date::sys_days{2024_y / June / 20}};
  register_special_stations(tt);
  load_timetable({}, source_idx_t{0}, mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B1,B1,,2.0,3.0,,
B2,B2,,2.0,3.0,,
C,C,,4.0,5.0,,
D,D,,6.0,7.0,,

# calendar_dates.txt
service_id,date,exception_type
S,20240619,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,RE 1,,,2
R2,DB,RE 2,,,2
R3,DB,RE 3,,,2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id,wheelchair_accessible
R1,S,T1,RE 1,,1
R2,S,T2,RE 2,,1
R3,S,T3,RE 3,,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T1,10:00:00,10:00:00,A,1,0,0
T1,11:00:00,11:00:00,B1,2,0,0
T2,12:00:00,12:00:00,B2,1,0,0
T2,12:30:00,12:30:00,C,2,0,0
T3,12:40:00,12:40:00,C,1,0,0
T3,13:00:00,13:00:00,D,2,0,0
)"),
                 tt);
  finalize(tt);

  tt.fwd_search_lb_graph_[kWheelchairProfile] =
      tt.fwd_search_lb_graph_[kDefaultProfile];
  tt.bwd_search_lb_graph_[kWheelchairProfile] =
      tt.bwd_search_lb_graph_[kDefaultProfile];

  auto const A = tt.find(location_id{"A", source_idx_t{0U}}).value();
  auto const B1 = tt.find(location_id{"B1", source_idx_t{0U}}).value();
  auto const B2 = tt.find(location_id{"B2", source_idx_t{0U}}).value();
  auto const D = tt.find(location_id{"D", source_idx_t{0U}}).value();
  auto const day = sys_days{2024_y / June / 19};

  // The static footpath doesn't know about the elevator outage.
  tt.locations_.footpaths_out_[kProfile].resize(tt.n_locations());
  tt.locations_.footpaths_in_[kProfile].resize(tt.n_locations());
  tt.locations_.footpaths_out_[kProfile][B1].push_back(footpath{B2, 5min});
  tt.locations_.footpaths_in_[kProfile][B2].push_back(footpath{B1, 5min});

  auto rtt = rt::create_rt_timetable(tt, day);
  for (auto const l : {B1, B2}) {
    rtt.has_td_footpaths_in_[kProfile].set(l, true);
    rtt.has_td_footpaths_out_[kProfile].set(l, true);
  }
  rtt.td_footpaths_out_[kProfile].resize(tt.n_locations());
  rtt.td_footpaths_in_[kProfile].resize(tt.n_locations());
  rtt.td_footpaths_out_[kProfile][B1].push_back(
      td_footpath{B2, unixtime_t{day + 9h + 25min}, 10min});
  rtt.td_footpaths_in_[kProfile][B2].push_back(
      td_footpath{B1, unixtime_t{day + 9h + 25min}, 10min});

  auto search_state = routing::search_state{};
  auto raptor_state = routing::raptor_state{};
  auto const result = routing::pong_search(
      tt, &rtt, search_state, raptor_state,
      routing::query{
          .start_time_ = interval<unixtime_t>{day + 7h, day + 9h},
          .start_match_mode_ = routing::location_match_mode::kEquivalent,
          .dest_match_mode_ = routing::location_match_mode::kEquivalent,
          .start_ = {{A, 0min, 0U}},
          .destination_ = {{D, 0min, 0U}},
          .prf_idx_ = kProfile},
      direction::kForward);
  ASSERT_EQ(1U, result.journeys_->size());

  auto const& legs = result.journeys_->begin()->legs_;
  auto const fp = utl::find_if(
      legs, [&](routing::journey::leg const& l) { return l.from_ == B1; });
  ASSERT_NE(fp, end(legs)) << to_string(tt, &rtt, *result.journeys_);
  EXPECT_EQ(B2, fp->to_);
  EXPECT_GE(fp->dep_time_, unixtime_t{day + 9h + 25min})
      << to_string(tt, &rtt, *result.journeys_);
  EXPECT_EQ(10min, fp->arr_time_ - fp->dep_time_);
}
