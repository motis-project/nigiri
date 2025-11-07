#include "gtest/gtest.h"

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
mem_dir test_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,2.0,3.0,,

# calendar_dates.txt
service_id,date,exception_type
S,20240619,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,RE 1,,,2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R1,S,T1,RE 1,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T1,10:00:00,10:00:00,A,1,0,0
T1,11:00:00,11:00:00,B,2,0,0
)");
}

}  // namespace

constexpr auto const kExpStartFwd = R"(
[2024-06-19 07:00, 2024-06-19 09:00]
TRANSFERS: 0
     FROM: (START, START) [2024-06-19 07:45]
       TO: (END, END) [2024-06-19 09:00]
leg 0: (START, START) [2024-06-19 07:45] -> (A, A) [2024-06-19 08:00]
  MUMO (id=5, duration=15)
leg 1: (A, A) [2024-06-19 08:00] -> (B, B) [2024-06-19 09:00]
   0: A       A...............................................                               d: 19.06 08:00 [19.06 10:00]  [{name=RE 1, day=2024-06-19, id=T1, src=0}]
   1: B       B............................................... a: 19.06 09:00 [19.06 11:00]
leg 2: (B, B) [2024-06-19 09:00] -> (END, END) [2024-06-19 09:00]
  MUMO (id=0, duration=0)

)";

TEST(routing, td_start_fwd) {
  timetable tt;
  tt.date_range_ = {date::sys_days{2024_y / June / 18},
                    date::sys_days{2024_y / June / 20}};
  register_special_stations(tt);
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  auto const A = tt.locations_.get({"A", {}}).l_;
  auto const B = tt.locations_.get({"B", {}}).l_;

  auto const run_search = [&]() {
    return raptor_search(
        tt, nullptr,
        routing::query{
            .start_time_ = unixtime_t{sys_days{2024_y / June / 19}} + 7h,
            .start_match_mode_ = routing::location_match_mode::kIntermodal,
            .dest_match_mode_ = routing::location_match_mode::kIntermodal,
            .use_start_footpaths_ = false,
            .destination_ = {{B, duration_t{0}, transport_mode_id_t{0}}},
            .td_start_ =
                {{{A,
                   {{.valid_from_ = sys_days{2024_y / June / 19} + 7h + 45min,
                     .duration_ = 10min,
                     .transport_mode_id_ = 5},
                    {.valid_from_ = sys_days{2024_y / June / 19} + 7h + 55min,
                     .duration_ = footpath::kMaxDuration,
                     .transport_mode_id_ = 5}}}}},
            .prf_idx_ = 0U},
        direction::kForward);
  };

  EXPECT_EQ(kExpStartFwd, to_string(tt, run_search()));
}

constexpr auto const kExpDestFwd = R"(
[2024-06-19 07:00, 2024-06-19 09:15]
TRANSFERS: 0
     FROM: (START, START) [2024-06-19 07:00]
       TO: (END, END) [2024-06-19 09:15]
leg 0: (START, START) [2024-06-19 07:00] -> (A, A) [2024-06-19 07:00]
  MUMO (id=0, duration=0)
leg 1: (A, A) [2024-06-19 08:00] -> (B, B) [2024-06-19 09:00]
   0: A       A...............................................                               d: 19.06 08:00 [19.06 10:00]  [{name=RE 1, day=2024-06-19, id=T1, src=0}]
   1: B       B............................................... a: 19.06 09:00 [19.06 11:00]
leg 2: (B, B) [2024-06-19 09:05] -> (END, END) [2024-06-19 09:15]
  MUMO (id=5, duration=10)

)";

TEST(routing, td_dest_fwd) {
  timetable tt;
  tt.date_range_ = {date::sys_days{2024_y / June / 18},
                    date::sys_days{2024_y / June / 20}};
  register_special_stations(tt);
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  auto const A = tt.locations_.get({"A", {}}).l_;
  auto const B = tt.locations_.get({"B", {}}).l_;

  auto const run_search = [&]() {
    return raptor_search(
        tt, nullptr,
        routing::query{
            .start_time_ = unixtime_t{sys_days{2024_y / June / 19}} + 7h,
            .start_match_mode_ = routing::location_match_mode::kIntermodal,
            .dest_match_mode_ = routing::location_match_mode::kIntermodal,
            .use_start_footpaths_ = false,
            .start_ = {{A, duration_t{0}, transport_mode_id_t{0}}},
            .td_dest_ =
                {{{B,
                   {{.valid_from_ = sys_days{2024_y / June / 19} + 9h + 5min,
                     .duration_ = 10min,
                     .transport_mode_id_ = 5},
                    {.valid_from_ = sys_days{2024_y / June / 19} + 9h + 15min,
                     .duration_ = footpath::kMaxDuration,
                     .transport_mode_id_ = 5}}}}},
            .prf_idx_ = 0U},
        direction::kForward);
  };

  EXPECT_EQ(kExpDestFwd, to_string(tt, run_search()));
}

constexpr auto const kExpStartBwd [[maybe_unused]] = R"(
[2024-06-19 08:00, 2024-06-19 10:00]
TRANSFERS: 0
     FROM: (END, END) [2024-06-19 08:00]
       TO: (START, START) [2024-06-19 09:15]
leg 0: (END, END) [2024-06-19 08:00] -> (A, A) [2024-06-19 08:00]
  MUMO (id=0, duration=0)
leg 1: (A, A) [2024-06-19 08:00] -> (B, B) [2024-06-19 09:00]
   0: A       A...............................................                               d: 19.06 08:00 [19.06 10:00]  [{name=RE 1, day=2024-06-19, id=T1, src=0}]
   1: B       B............................................... a: 19.06 09:00 [19.06 11:00]
leg 2: (B, B) [2024-06-19 09:00] -> (START, START) [2024-06-19 09:15]
  MUMO (id=5, duration=15)

)";

TEST(routing, td_start_bwd) {
  timetable tt;
  tt.date_range_ = {date::sys_days{2024_y / June / 18},
                    date::sys_days{2024_y / June / 20}};
  register_special_stations(tt);
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  auto const A = tt.locations_.get({"A", {}}).l_;
  auto const B = tt.locations_.get({"B", {}}).l_;

  auto const run_search = [&]() {
    return raptor_search(
        tt, nullptr,
        routing::query{
            .start_time_ = unixtime_t{sys_days{2024_y / June / 19}} + 10h,
            .start_match_mode_ = routing::location_match_mode::kIntermodal,
            .dest_match_mode_ = routing::location_match_mode::kIntermodal,
            .use_start_footpaths_ = false,
            .destination_ = {{A, duration_t{0}, transport_mode_id_t{0}}},
            .td_start_ =
                {{{B,
                   {{.valid_from_ = sys_days{2024_y / June / 19} + 9h + 5min,
                     .duration_ = 10min,
                     .transport_mode_id_ = 5},
                    {.valid_from_ = sys_days{2024_y / June / 19} + 9h + 15min,
                     .duration_ = footpath::kMaxDuration,
                     .transport_mode_id_ = 5}}}}},
            .prf_idx_ = 0U},
        direction::kBackward);
  };

  EXPECT_EQ(kExpStartBwd, to_string(tt, run_search()));
}

constexpr auto const kExpDestBwd = R"(
[2024-06-19 07:45, 2024-06-19 10:00]
TRANSFERS: 0
     FROM: (END, END) [2024-06-19 07:45]
       TO: (START, START) [2024-06-19 10:00]
leg 0: (END, END) [2024-06-19 07:45] -> (A, A) [2024-06-19 07:55]
  MUMO (id=5, duration=10)
leg 1: (A, A) [2024-06-19 08:00] -> (B, B) [2024-06-19 09:00]
   0: A       A...............................................                               d: 19.06 08:00 [19.06 10:00]  [{name=RE 1, day=2024-06-19, id=T1, src=0}]
   1: B       B............................................... a: 19.06 09:00 [19.06 11:00]
leg 2: (B, B) [2024-06-19 10:00] -> (START, START) [2024-06-19 10:00]
  MUMO (id=0, duration=0)

)";

TEST(routing, td_dest_bwd) {
  timetable tt;
  tt.date_range_ = {date::sys_days{2024_y / June / 18},
                    date::sys_days{2024_y / June / 20}};
  register_special_stations(tt);
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  auto const A = tt.locations_.get({"A", {}}).l_;
  auto const B = tt.locations_.get({"B", {}}).l_;

  auto const run_search = [&]() {
    return raptor_search(
        tt, nullptr,
        routing::query{
            .start_time_ = unixtime_t{sys_days{2024_y / June / 19}} + 10h,
            .start_match_mode_ = routing::location_match_mode::kIntermodal,
            .dest_match_mode_ = routing::location_match_mode::kIntermodal,
            .use_start_footpaths_ = false,
            .start_ = {{B, duration_t{0}, transport_mode_id_t{0}}},
            .td_dest_ =
                {{{A,
                   {{.valid_from_ = sys_days{2024_y / June / 19} + 7h + 45min,
                     .duration_ = 10min,
                     .transport_mode_id_ = 5},
                    {.valid_from_ = sys_days{2024_y / June / 19} + 7h + 55min,
                     .duration_ = footpath::kMaxDuration,
                     .transport_mode_id_ = 5}}}}},
            .prf_idx_ = 0U},
        direction::kBackward);
  };

  EXPECT_EQ(kExpDestBwd, to_string(tt, run_search()));
}

TEST(routing, td_start_fwd_1min_validity) {
  timetable tt;
  tt.date_range_ = {date::sys_days{2024_y / June / 18},
                    date::sys_days{2024_y / June / 20}};
  register_special_stations(tt);
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  auto const A = tt.locations_.get({"A", {}}).l_;
  auto const B = tt.locations_.get({"B", {}}).l_;

  auto const td_start = [&]() {
    auto td_offsets = routing::td_offsets_t{};
    auto tdo_ride =
        routing::td_offset{.valid_from_ = sys_days{2024_y / June / 19} + 6h,
                           .duration_ = 7min,
                           .transport_mode_id_ = 5};
    auto tdo_off = routing::td_offset{
        .valid_from_ = sys_days{2024_y / June / 19} + 6h + 1min,
        .duration_ = footpath::kMaxDuration,
        .transport_mode_id_ = 5};

    auto const inc = [&](auto const step) {
      tdo_ride.valid_from_ += step;
      tdo_off.valid_from_ += step;
    };

    for (; tdo_ride.valid_from_ < sys_days{2024_y / June / 19} + 9h;
         inc(5min)) {
      td_offsets[A].emplace_back(tdo_ride);
      td_offsets[A].emplace_back(tdo_off);
    }

    return td_offsets;
  }();

  auto const run_search = [&]() {
    return raptor_search(
        tt, nullptr,
        routing::query{
            .start_time_ = unixtime_t{sys_days{2024_y / June / 19}} + 7h,
            .start_match_mode_ = routing::location_match_mode::kIntermodal,
            .dest_match_mode_ = routing::location_match_mode::kIntermodal,
            .use_start_footpaths_ = false,
            .destination_ = {{B, duration_t{0}, transport_mode_id_t{0}}},
            .td_start_ = td_start,
            .prf_idx_ = 0U},
        direction::kForward);
  };

  std::cout << "\n" << to_string(tt, run_search()) << "\n";

  // EXPECT_EQ(kExpStartFwd, to_string(tt, run_search()));
}