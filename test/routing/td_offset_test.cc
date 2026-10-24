#include "gtest/gtest.h"

#include "utl/helpers/algorithm.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/loader/init_finish.h"

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
// T0 -> T1 at A and T1 -> T2 at B are forbidden, which puts T1 on virtual
// locations of A and B. The journeys name A and B nonetheless.
mem_dir test_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,2.0,3.0,,
Z,Z,,4.0,1.0,,
Y,Y,,6.0,3.0,,

# calendar_dates.txt
service_id,date,exception_type
S,20240619,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DB,RE 0,,,2
R1,DB,RE 1,,,2
R2,DB,RE 2,,,2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,S,T0,RE 0,
R1,S,T1,RE 1,
R2,S,T2,RE 2,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T0,09:00:00,09:00:00,Z,1,0,0
T0,09:55:00,09:55:00,A,2,0,0
T1,10:00:00,10:00:00,A,1,0,0
T1,11:00:00,11:00:00,B,2,0,0
T2,11:05:00,11:05:00,B,1,0,0
T2,11:30:00,11:30:00,Y,2,0,0

# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time,from_route_id,to_route_id,from_trip_id,to_trip_id
A,A,3,,R0,R1,,
B,B,3,,R1,R2,,
)");
}

bool t1_at_virtual_locations(timetable const& tt) {
  return utl::any_of(tt.route_location_seq_, [&](auto const& seq) {
    return utl::all_of(seq, [&](stop::value_type const s) {
      return tt.locations_.types_[stop{s}.location_idx()] ==
             location_type::kVirt;
    });
  });
}

}  // namespace

constexpr auto kExpStartFwd = R"(
[2024-06-19 07:00, 2024-06-19 09:00]
TRANSFERS: 0
     FROM: (START, START) [2024-06-19 07:50]
       TO: (END, END) [2024-06-19 09:00]
leg 0: (START, START) [2024-06-19 07:50] -> (A, A) [2024-06-19 08:00]
  MUMO (payload=5, duration=10)
leg 1: (A, A) [2024-06-19 08:00] -> (B, B) [2024-06-19 09:00]
   0: A       A...............................................                               d: 19.06 08:00 [19.06 10:00]  [{name=RE 1, day=2024-06-19, id=T1, src=0}]
   1: B       B............................................... a: 19.06 09:00 [19.06 11:00]
leg 2: (B, B) [2024-06-19 09:00] -> (END, END) [2024-06-19 09:00]
  MUMO (payload=0, duration=0)

)";

TEST(routing, td_start_fwd) {
  timetable tt;
  tt.date_range_ = {date::sys_days{2024_y / June / 18},
                    date::sys_days{2024_y / June / 20}};
  register_special_stations(tt);
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);
  ASSERT_TRUE(t1_at_virtual_locations(tt));

  auto const A = tt.find(location_id{"A", source_idx_t{}}).value();
  auto const B = tt.find(location_id{"B", source_idx_t{}}).value();

  auto const run_search = [&]() {
    return raptor_search(
        tt, nullptr,
        routing::query{
            .start_time_ = unixtime_t{sys_days{2024_y / June / 19}} + 7h,
            .start_match_mode_ = routing::location_match_mode::kIntermodal,
            .dest_match_mode_ = routing::location_match_mode::kIntermodal,
            .use_start_footpaths_ = false,
            .destination_ = {{B, duration_t{0},
                              routing::transport_mode_t::payload_t{0}}},
            .td_start_ =
                {{{A,
                   {{.valid_from_ = sys_days{2024_y / June / 19} + 7h + 45min,
                     .duration_ = 10min,
                     .transport_mode_payload_ = 5},
                    {.valid_from_ = sys_days{2024_y / June / 19} + 7h + 55min,
                     .duration_ = kTdFootpathBan,
                     .transport_mode_payload_ = 5}}}}},
            .prf_idx_ = 0U},
        direction::kForward);
  };

  std::cout << "\n" << to_string(tt, run_search()) << "\n";

  EXPECT_EQ(kExpStartFwd, to_string(tt, run_search()));
}

constexpr auto kExpDestFwd = R"(
[2024-06-19 07:00, 2024-06-19 09:15]
TRANSFERS: 0
     FROM: (START, START) [2024-06-19 08:00]
       TO: (END, END) [2024-06-19 09:15]
leg 0: (START, START) [2024-06-19 08:00] -> (A, A) [2024-06-19 08:00]
  MUMO (payload=0, duration=0)
leg 1: (A, A) [2024-06-19 08:00] -> (B, B) [2024-06-19 09:00]
   0: A       A...............................................                               d: 19.06 08:00 [19.06 10:00]  [{name=RE 1, day=2024-06-19, id=T1, src=0}]
   1: B       B............................................... a: 19.06 09:00 [19.06 11:00]
leg 2: (B, B) [2024-06-19 09:05] -> (END, END) [2024-06-19 09:15]
  MUMO (payload=5, duration=10)

)";

TEST(routing, td_dest_fwd) {
  timetable tt;
  tt.date_range_ = {date::sys_days{2024_y / June / 18},
                    date::sys_days{2024_y / June / 20}};
  register_special_stations(tt);
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);
  ASSERT_TRUE(t1_at_virtual_locations(tt));

  auto const A = tt.find({"A", {}}).value();
  auto const B = tt.find({"B", {}}).value();

  auto const run_search = [&]() {
    return raptor_search(
        tt, nullptr,
        routing::query{
            .start_time_ = unixtime_t{sys_days{2024_y / June / 19}} + 7h,
            .start_match_mode_ = routing::location_match_mode::kIntermodal,
            .dest_match_mode_ = routing::location_match_mode::kIntermodal,
            .use_start_footpaths_ = false,
            .start_ = {{A, duration_t{0},
                        routing::transport_mode_t::payload_t{0}}},
            .td_dest_ =
                {{{B,
                   {{.valid_from_ = sys_days{2024_y / June / 19} + 9h + 5min,
                     .duration_ = 10min,
                     .transport_mode_payload_ = 5},
                    {.valid_from_ = sys_days{2024_y / June / 19} + 9h + 15min,
                     .duration_ = kTdFootpathBan,
                     .transport_mode_payload_ = 5}}}}},
            .prf_idx_ = 0U},
        direction::kForward);
  };

  std::cout << "\n" << to_string(tt, run_search()) << "\n";

  EXPECT_EQ(kExpDestFwd, to_string(tt, run_search()));
}

constexpr auto kExpStartBwd = R"(
[2024-06-19 08:00, 2024-06-19 10:00]
TRANSFERS: 0
     FROM: (END, END) [2024-06-19 08:00]
       TO: (START, START) [2024-06-19 09:15]
leg 0: (END, END) [2024-06-19 08:00] -> (A, A) [2024-06-19 08:00]
  MUMO (payload=0, duration=0)
leg 1: (A, A) [2024-06-19 08:00] -> (B, B) [2024-06-19 09:00]
   0: A       A...............................................                               d: 19.06 08:00 [19.06 10:00]  [{name=RE 1, day=2024-06-19, id=T1, src=0}]
   1: B       B............................................... a: 19.06 09:00 [19.06 11:00]
leg 2: (B, B) [2024-06-19 09:05] -> (START, START) [2024-06-19 09:15]
  MUMO (payload=5, duration=10)

)";

TEST(routing, td_start_bwd) {
  timetable tt;
  tt.date_range_ = {date::sys_days{2024_y / June / 18},
                    date::sys_days{2024_y / June / 20}};
  register_special_stations(tt);
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);
  ASSERT_TRUE(t1_at_virtual_locations(tt));

  auto const A = tt.find({"A", {}}).value();
  auto const B = tt.find({"B", {}}).value();

  auto const run_search = [&]() {
    return raptor_search(
        tt, nullptr,
        routing::query{
            .start_time_ = unixtime_t{sys_days{2024_y / June / 19}} + 10h,
            .start_match_mode_ = routing::location_match_mode::kIntermodal,
            .dest_match_mode_ = routing::location_match_mode::kIntermodal,
            .use_start_footpaths_ = false,
            .destination_ = {{A, duration_t{0},
                              routing::transport_mode_t::payload_t{0}}},
            .td_start_ =
                {{{B,
                   {{.valid_from_ = sys_days{2024_y / June / 19} + 9h + 5min,
                     .duration_ = 10min,
                     .transport_mode_payload_ = 5},
                    {.valid_from_ = sys_days{2024_y / June / 19} + 9h + 15min,
                     .duration_ = kTdFootpathBan,
                     .transport_mode_payload_ = 5}}}}},
            .prf_idx_ = 0U},
        direction::kBackward);
  };

  std::cout << "\n" << to_string(tt, run_search()) << "\n";

  EXPECT_EQ(kExpStartBwd, to_string(tt, run_search()));
}

constexpr auto kExpDestBwd = R"(
[2024-06-19 07:50, 2024-06-19 10:00]
TRANSFERS: 0
     FROM: (END, END) [2024-06-19 07:50]
       TO: (START, START) [2024-06-19 09:00]
leg 0: (END, END) [2024-06-19 07:50] -> (A, A) [2024-06-19 08:00]
  MUMO (payload=5, duration=10)
leg 1: (A, A) [2024-06-19 08:00] -> (B, B) [2024-06-19 09:00]
   0: A       A...............................................                               d: 19.06 08:00 [19.06 10:00]  [{name=RE 1, day=2024-06-19, id=T1, src=0}]
   1: B       B............................................... a: 19.06 09:00 [19.06 11:00]
leg 2: (B, B) [2024-06-19 09:00] -> (START, START) [2024-06-19 09:00]
  MUMO (payload=0, duration=0)

)";

TEST(routing, td_dest_bwd) {
  timetable tt;
  tt.date_range_ = {date::sys_days{2024_y / June / 18},
                    date::sys_days{2024_y / June / 20}};
  register_special_stations(tt);
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);
  ASSERT_TRUE(t1_at_virtual_locations(tt));

  auto const A = tt.find({"A", {}}).value();
  auto const B = tt.find({"B", {}}).value();

  auto const run_search = [&]() {
    return raptor_search(
        tt, nullptr,
        routing::query{
            .start_time_ = unixtime_t{sys_days{2024_y / June / 19}} + 10h,
            .start_match_mode_ = routing::location_match_mode::kIntermodal,
            .dest_match_mode_ = routing::location_match_mode::kIntermodal,
            .use_start_footpaths_ = false,
            .start_ = {{B, duration_t{0},
                        routing::transport_mode_t::payload_t{0}}},
            .td_dest_ =
                {{{A,
                   {{.valid_from_ = sys_days{2024_y / June / 19} + 7h + 45min,
                     .duration_ = 10min,
                     .transport_mode_payload_ = 5},
                    {.valid_from_ = sys_days{2024_y / June / 19} + 7h + 55min,
                     .duration_ = kTdFootpathBan,
                     .transport_mode_payload_ = 5}}}}},
            .prf_idx_ = 0U},
        direction::kBackward);
  };

  std::cout << "\n" << to_string(tt, run_search()) << "\n";

  EXPECT_EQ(kExpDestBwd, to_string(tt, run_search()));
}

constexpr auto kExpFirstMileFwd1minValidity = R"(
[2024-06-19 07:30, 2024-06-19 09:00]
TRANSFERS: 0
     FROM: (START, START) [2024-06-19 07:30]
       TO: (END, END) [2024-06-19 09:00]
leg 0: (START, START) [2024-06-19 07:30] -> (A, A) [2024-06-19 07:40]
  MUMO (payload=5, duration=10)
leg 1: (A, A) [2024-06-19 08:00] -> (B, B) [2024-06-19 09:00]
   0: A       A...............................................                               d: 19.06 08:00 [19.06 10:00]  [{name=RE 1, day=2024-06-19, id=T1, src=0}]
   1: B       B............................................... a: 19.06 09:00 [19.06 11:00]
leg 2: (B, B) [2024-06-19 09:00] -> (END, END) [2024-06-19 09:00]
  MUMO (payload=0, duration=0)

)";

TEST(routing, first_mile_fwd_1min_validity) {
  timetable tt;
  tt.date_range_ = {date::sys_days{2024_y / June / 18},
                    date::sys_days{2024_y / June / 20}};
  register_special_stations(tt);
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);
  ASSERT_TRUE(t1_at_virtual_locations(tt));

  auto const A = tt.find({"A", {}}).value();
  auto const B = tt.find({"B", {}}).value();

  auto const run_search = [&]() {
    return raptor_search(
        tt, nullptr,
        routing::query{
            .start_time_ = interval<unixtime_t>{{sys_days{2024_y / June / 19}},
                                                {sys_days{2024_y / June / 20}}},
            .start_match_mode_ = routing::location_match_mode::kIntermodal,
            .dest_match_mode_ = routing::location_match_mode::kIntermodal,
            .use_start_footpaths_ = false,
            .destination_ = {{B, duration_t{0},
                              routing::transport_mode_t::payload_t{0}}},
            .td_start_ =
                {{{A,
                   {{.valid_from_ = unixtime_t{0h},
                     .duration_ = kTdFootpathBan,
                     .transport_mode_payload_ = 5},
                    {.valid_from_ = sys_days{2024_y / June / 19} + 7h + 30min,
                     .duration_ = 10min,
                     .transport_mode_payload_ = 5},
                    {.valid_from_ = sys_days{2024_y / June / 19} + 7h + 31min,
                     .duration_ = kTdFootpathBan,
                     .transport_mode_payload_ = 5}}}}},
            .prf_idx_ = 0U},
        direction::kForward);
  };

  std::cout << "\n" << to_string(tt, run_search()) << "\n";

  EXPECT_EQ(kExpFirstMileFwd1minValidity, to_string(tt, run_search()));
}

constexpr auto kExpFirstMileBwd1minValidity = R"(
[2024-06-19 07:30, 2024-06-19 09:00]
TRANSFERS: 0
     FROM: (END, END) [2024-06-19 07:30]
       TO: (START, START) [2024-06-19 09:00]
leg 0: (END, END) [2024-06-19 07:30] -> (A, A) [2024-06-19 07:40]
  MUMO (payload=5, duration=10)
leg 1: (A, A) [2024-06-19 08:00] -> (B, B) [2024-06-19 09:00]
   0: A       A...............................................                               d: 19.06 08:00 [19.06 10:00]  [{name=RE 1, day=2024-06-19, id=T1, src=0}]
   1: B       B............................................... a: 19.06 09:00 [19.06 11:00]
leg 2: (B, B) [2024-06-19 09:00] -> (START, START) [2024-06-19 09:00]
  MUMO (payload=0, duration=0)

)";

TEST(routing, first_mile_bwd_1min_validity) {
  timetable tt;
  tt.date_range_ = {date::sys_days{2024_y / June / 18},
                    date::sys_days{2024_y / June / 20}};
  register_special_stations(tt);
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);
  ASSERT_TRUE(t1_at_virtual_locations(tt));

  auto const A = tt.find({"A", {}}).value();
  auto const B = tt.find({"B", {}}).value();

  auto const run_search = [&]() {
    return raptor_search(
        tt, nullptr,
        routing::query{
            .start_time_ = interval<unixtime_t>{{sys_days{2024_y / June / 19}},
                                                {sys_days{2024_y / June / 20}}},
            .start_match_mode_ = routing::location_match_mode::kIntermodal,
            .dest_match_mode_ = routing::location_match_mode::kIntermodal,
            .use_start_footpaths_ = false,
            .start_ = {{B, duration_t{0},
                        routing::transport_mode_t::payload_t{0}}},
            .td_dest_ =
                {{{A,
                   {{.valid_from_ = unixtime_t{0h},
                     .duration_ = kTdFootpathBan,
                     .transport_mode_payload_ = 5},
                    {.valid_from_ = sys_days{2024_y / June / 19} + 7h + 30min,
                     .duration_ = 10min,
                     .transport_mode_payload_ = 5},
                    {.valid_from_ = sys_days{2024_y / June / 19} + 7h + 31min,
                     .duration_ = kTdFootpathBan,
                     .transport_mode_payload_ = 5}}}}},
            .prf_idx_ = 0U},
        direction::kBackward);
  };

  std::cout << "\n" << to_string(tt, run_search()) << "\n";

  EXPECT_EQ(kExpFirstMileBwd1minValidity, to_string(tt, run_search()));
}

constexpr auto kExpLastMileFwd1minValidity = R"(
[2024-06-19 08:00, 2024-06-19 09:40]
TRANSFERS: 0
     FROM: (START, START) [2024-06-19 08:00]
       TO: (END, END) [2024-06-19 09:40]
leg 0: (START, START) [2024-06-19 08:00] -> (A, A) [2024-06-19 08:00]
  MUMO (payload=0, duration=0)
leg 1: (A, A) [2024-06-19 08:00] -> (B, B) [2024-06-19 09:00]
   0: A       A...............................................                               d: 19.06 08:00 [19.06 10:00]  [{name=RE 1, day=2024-06-19, id=T1, src=0}]
   1: B       B............................................... a: 19.06 09:00 [19.06 11:00]
leg 2: (B, B) [2024-06-19 09:30] -> (END, END) [2024-06-19 09:40]
  MUMO (payload=5, duration=10)

)";

TEST(routing, last_mile_fwd_1min_validity) {
  timetable tt;
  tt.date_range_ = {date::sys_days{2024_y / June / 18},
                    date::sys_days{2024_y / June / 20}};
  register_special_stations(tt);
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);
  ASSERT_TRUE(t1_at_virtual_locations(tt));

  auto const A = tt.find({"A", {}}).value();
  auto const B = tt.find({"B", {}}).value();

  auto const run_search = [&]() {
    return raptor_search(
        tt, nullptr,
        routing::query{
            .start_time_ = interval<unixtime_t>{{sys_days{2024_y / June / 19}},
                                                {sys_days{2024_y / June / 20}}},
            .start_match_mode_ = routing::location_match_mode::kIntermodal,
            .dest_match_mode_ = routing::location_match_mode::kIntermodal,
            .use_start_footpaths_ = false,
            .start_ = {{A, duration_t{0},
                        routing::transport_mode_t::payload_t{0}}},
            .td_dest_ =
                {{{B,
                   {{.valid_from_ = unixtime_t{0h},
                     .duration_ = kTdFootpathBan,
                     .transport_mode_payload_ = 5},
                    {.valid_from_ = sys_days{2024_y / June / 19} + 9h + 30min,
                     .duration_ = 10min,
                     .transport_mode_payload_ = 5},
                    {.valid_from_ = sys_days{2024_y / June / 19} + 9h + 31min,
                     .duration_ = kTdFootpathBan,
                     .transport_mode_payload_ = 5}}}}},
            .prf_idx_ = 0U},
        direction::kForward);
  };

  std::cout << "\n" << to_string(tt, run_search()) << "\n";

  EXPECT_EQ(kExpLastMileFwd1minValidity, to_string(tt, run_search()));
}

constexpr auto kExpLastMileBwd1minValidity = R"(
[2024-06-19 08:00, 2024-06-19 09:40]
TRANSFERS: 0
     FROM: (END, END) [2024-06-19 08:00]
       TO: (START, START) [2024-06-19 09:40]
leg 0: (END, END) [2024-06-19 08:00] -> (A, A) [2024-06-19 08:00]
  MUMO (payload=0, duration=0)
leg 1: (A, A) [2024-06-19 08:00] -> (B, B) [2024-06-19 09:00]
   0: A       A...............................................                               d: 19.06 08:00 [19.06 10:00]  [{name=RE 1, day=2024-06-19, id=T1, src=0}]
   1: B       B............................................... a: 19.06 09:00 [19.06 11:00]
leg 2: (B, B) [2024-06-19 09:30] -> (START, START) [2024-06-19 09:40]
  MUMO (payload=5, duration=10)

)";

TEST(routing, last_mile_bwd_1min_validity) {
  timetable tt;
  tt.date_range_ = {date::sys_days{2024_y / June / 18},
                    date::sys_days{2024_y / June / 20}};
  register_special_stations(tt);
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);
  ASSERT_TRUE(t1_at_virtual_locations(tt));

  auto const A = tt.find({"A", {}}).value();
  auto const B = tt.find({"B", {}}).value();

  auto const run_search = [&]() {
    return raptor_search(
        tt, nullptr,
        routing::query{
            .start_time_ = interval<unixtime_t>{{sys_days{2024_y / June / 19}},
                                                {sys_days{2024_y / June / 20}}},
            .start_match_mode_ = routing::location_match_mode::kIntermodal,
            .dest_match_mode_ = routing::location_match_mode::kIntermodal,
            .use_start_footpaths_ = false,
            .destination_ = {{A, duration_t{0},
                              routing::transport_mode_t::payload_t{0}}},
            .td_start_ =
                {{{B,
                   {{.valid_from_ = unixtime_t{0h},
                     .duration_ = kTdFootpathBan,
                     .transport_mode_payload_ = 5},
                    {.valid_from_ = sys_days{2024_y / June / 19} + 9h + 30min,
                     .duration_ = 10min,
                     .transport_mode_payload_ = 5},
                    {.valid_from_ = sys_days{2024_y / June / 19} + 9h + 31min,
                     .duration_ = kTdFootpathBan,
                     .transport_mode_payload_ = 5}}}}},
            .prf_idx_ = 0U},
        direction::kBackward);
  };

  std::cout << "\n" << to_string(tt, run_search()) << "\n";

  EXPECT_EQ(kExpLastMileBwd1minValidity, to_string(tt, run_search()));
}

constexpr auto kExpDestFwdBeyondFootpathMaxDuration = R"(
[2024-06-19 07:00, 2024-06-19 14:05]
TRANSFERS: 0
     FROM: (START, START) [2024-06-19 08:00]
       TO: (END, END) [2024-06-19 14:05]
leg 0: (START, START) [2024-06-19 08:00] -> (A, A) [2024-06-19 08:00]
  MUMO (payload=0, duration=0)
leg 1: (A, A) [2024-06-19 08:00] -> (B, B) [2024-06-19 09:00]
   0: A       A...............................................                               d: 19.06 08:00 [19.06 10:00]  [{name=RE 1, day=2024-06-19, id=T1, src=0}]
   1: B       B............................................... a: 19.06 09:00 [19.06 11:00]
leg 2: (B, B) [2024-06-19 09:05] -> (END, END) [2024-06-19 14:05]
  MUMO (payload=5, duration=300)

)";

TEST(routing, td_dest_fwd_beyond_footpath_max_duration) {
  // A 300 min last mile, more than footpath::kMax, is reconstructed.
  timetable tt;
  tt.date_range_ = {date::sys_days{2024_y / June / 18},
                    date::sys_days{2024_y / June / 20}};
  register_special_stations(tt);
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  auto const A = tt.find({"A", {}}).value();
  auto const B = tt.find({"B", {}}).value();

  auto const result = raptor_search(
      tt, nullptr,
      routing::query{
          .start_time_ = unixtime_t{sys_days{2024_y / June / 19}} + 7h,
          .start_match_mode_ = routing::location_match_mode::kIntermodal,
          .dest_match_mode_ = routing::location_match_mode::kIntermodal,
          .use_start_footpaths_ = false,
          .start_ = {{A, duration_t{0},
                      routing::transport_mode_t::payload_t{0}}},
          .td_dest_ =
              {{{B,
                 {{.valid_from_ = sys_days{2024_y / June / 19} + 9h + 5min,
                   .duration_ = 300min,
                   .transport_mode_payload_ = 5},
                  {.valid_from_ = sys_days{2024_y / June / 19} + 9h + 15min,
                   .duration_ = kTdFootpathBan,
                   .transport_mode_payload_ = 5}}}}},
          .prf_idx_ = 0U},
      direction::kForward);

  EXPECT_EQ(kExpDestFwdBeyondFootpathMaxDuration, to_string(tt, result));
}

constexpr auto kExpStartFwdExactlyFootpathMaxDuration = R"(
[2024-06-19 03:45, 2024-06-19 09:00]
TRANSFERS: 0
     FROM: (START, START) [2024-06-19 03:45]
       TO: (END, END) [2024-06-19 09:00]
leg 0: (START, START) [2024-06-19 03:45] -> (A, A) [2024-06-19 07:55]
  MUMO (payload=5, duration=250)
leg 1: (A, A) [2024-06-19 08:00] -> (B, B) [2024-06-19 09:00]
   0: A       A...............................................                               d: 19.06 08:00 [19.06 10:00]  [{name=RE 1, day=2024-06-19, id=T1, src=0}]
   1: B       B............................................... a: 19.06 09:00 [19.06 11:00]
leg 2: (B, B) [2024-06-19 09:00] -> (END, END) [2024-06-19 09:00]
  MUMO (payload=0, duration=0)

)";

TEST(routing, td_start_fwd_exactly_footpath_max_duration) {
  // The first mile is usable until 03:45 and takes 250 min: with the 5 min
  // wait for T1 at A that is exactly 255 min (footpath::kBan), a real duration.
  timetable tt;
  tt.date_range_ = {date::sys_days{2024_y / June / 18},
                    date::sys_days{2024_y / June / 20}};
  register_special_stations(tt);
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  auto const A = tt.find(location_id{"A", source_idx_t{}}).value();
  auto const B = tt.find(location_id{"B", source_idx_t{}}).value();

  auto const result = raptor_search(
      tt, nullptr,
      routing::query{
          .start_time_ =
              interval<unixtime_t>{sys_days{2024_y / June / 19} + 3h,
                                   sys_days{2024_y / June / 19} + 4h},
          .start_match_mode_ = routing::location_match_mode::kIntermodal,
          .dest_match_mode_ = routing::location_match_mode::kIntermodal,
          .use_start_footpaths_ = false,
          .destination_ = {{B, duration_t{0},
                            routing::transport_mode_t::payload_t{0}}},
          .td_start_ = {{{A,
                          {{.valid_from_ = sys_days{2024_y / June / 19} + 3h,
                            .duration_ = 250min,
                            .transport_mode_payload_ = 5},
                           {.valid_from_ =
                                sys_days{2024_y / June / 19} + 3h + 46min,
                            .duration_ = kTdFootpathBan,
                            .transport_mode_payload_ = 5}}}}},
          .prf_idx_ = 0U},
      direction::kForward);

  EXPECT_EQ(kExpStartFwdExactlyFootpathMaxDuration, to_string(tt, result));
}
