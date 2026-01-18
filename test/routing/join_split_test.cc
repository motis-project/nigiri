#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/loader/load.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/gtfsrt_resolve_run.h"
#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable_metrics.h"

#include "../raptor_search.h"
#include "../rt/util.h"
#include "results_to_string.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace std::chrono_literals;
using namespace nigiri::test;
using nigiri::test::raptor_search;

namespace {

// Days are relative to i, k.
// That means that x|x1, j, l are +1 day.
//
// Mo-Th    We-Fr      Mo,Th-Sa
//
// A --i-\            /-j-- E
//       C1          D1
//        C --x|x1-- D
//       C2          D2
// B --k-/            \-l-- F
//
//
// Columns: i,k x|x1 j,l
//
// Mo: X_X
// Tu: X__
// We: XX_
// Th: XXX
// Fr: _XX
// Sa: __X
// Su: ___
//
// Th 12.06.2025: XXX
// Fr 13.06.2025: _XX
// Sa 14.06.2025: __X
// Su 15.06.2025: ___
// Mo 16.06.2025: X_X
// Tu 17.06.2025: X__
// We 18.06.2025: XX_
// Th 19.06.2025: XXX   x1 instead of x
// Fr 20.06.2025: _XX
// Sa 21.06.2025: __X
// Su 22.06.2025: _X_   x active alone (exception)

struct test_case {
  std::string_view from_, to_;
  date::sys_days date_;
  bool expected_;
};
constexpr auto const kTests =
    std::initializer_list<test_case>{{"A", "E", 2025_y / June / 12, true},
                                     {"A", "F", 2025_y / June / 12, true},
                                     {"B", "E", 2025_y / June / 12, true},
                                     {"B", "F", 2025_y / June / 12, true},
                                     // still 12.06.2025 in table; +1 day
                                     {"C", "E", 2025_y / June / 13, true},
                                     {"C", "F", 2025_y / June / 13, true},

                                     // 13.06.2025 in table
                                     {"C", "E", 2025_y / June / 14, true},
                                     {"C", "F", 2025_y / June / 14, true},

                                     {"A", "C", 2025_y / June / 17, true},
                                     {"B", "C", 2025_y / June / 17, true},
                                     // 16.06.2025 in table
                                     {"D1", "E", 2025_y / June / 17, true},
                                     {"D2", "F", 2025_y / June / 17, true},

                                     {"A", "D", 2025_y / June / 18, true},
                                     {"B", "D", 2025_y / June / 18, true}};

template <typename... Args>
mem_dir test_files(Args... order) {
  constexpr auto const kTrips =
      std::array{"x,x,x,x,",  "x1,x1,x1,x1,", "k,ik,k,k,",
                 "l,jl,l,l,", "i,ik,i,i,",    "j,jl,j,j,"};
  return mem_dir::read(fmt::format(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,2.0,3.0,,
C,C,,4.0,5.0,,
C1,C1,,4.0,5.0,,
C2,C2,,4.0,5.0,,
D,D,,6.0,7.0,,
D1,D1,,6.0,7.0,,
D2,D2,,6.0,7.0,,
E,E,,8.0,9.0,,
F,F,,10.0,11.0,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
ik,1,1,1,1,0,0,0,20250101,20251231
x,0,0,0,1,1,1,0,20250101,20251231
jl,0,1,0,0,1,1,1,20250101,20251231

# calendar_dates.txt
service_id,date,exception_type
x,20250620,2
x1,20250618,1
x,20250623,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
i,DB,i,,,2
j,DB,j,,,2
k,DB,k,,,2
l,DB,l,,,2
x,DB,x,,,2
x1,DB,x1,,,2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
{}
{}
{}
{}
{}
{}

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
i,25:00:00,25:00:00,A,1,0,0
i,25:59:00,25:59:00,C1,2,0,0
k,25:00:00,25:00:00:00,B,1,0,0
k,26:00:00,26:00:00:00,C2,2,0,0
x,02:00:00,02:00:00,C,1,0,0
x,03:00:00,03:00:00,D,2,0,0
x1,50:00:00,50:00:00,C,1,0,0
x1,51:00:00,51:00:00,D,2,0,0
j,03:01:00,03:01:00,D1,1,0,0
j,04:01:00,04:01:00,E,2,0,0
l,03:00:00,03:00:00,D2,1,0,0
l,04:00:00,04:00:00,F,2,0,0

# transfers.txt
transfer_type,from_trip_id,to_trip_id
4,i,x
4,k,x
4,x,j
4,x,l
4,i,x1
4,k,x1
4,x1,j
4,x1,l
)",
                                   kTrips[order]...));
}

}  // namespace

TEST(join_split, complex) {
  auto const run = [](auto&&... trips) {
    timetable tt;
    tt.date_range_ = {date::sys_days{2025_y / June / 12},
                      date::sys_days{2025_y / June / 22}};
    auto const src = source_idx_t{0};
    load_timetable({}, src, test_files(trips...), tt);
    finalize(tt);

    auto rtt = rt::create_rt_timetable(tt, 2025_y / June / 12);

    auto const run_test = [&]() {
      for (auto const [from, to, date, expected] : kTests) {
        auto const results = raptor_search(tt, &rtt, from, to, date);
        EXPECT_EQ(expected, !results.empty())
            << "from=" << from << ", to=" << to << ", on " << date
            << ", expected " << expected;
        EXPECT_TRUE(utl::all_of(results, [](routing::journey const& j) {
          return j.transfers_ == 0U;
        }));
      }
    };
    run_test();

    auto const msg1 = test::to_feed_msg(
        {trip{.trip_id_ = "i",
              .delays_ = {{.seq_ = 1,
                           .ev_type_ = nigiri::event_type::kArr,
                           .delay_minutes_ = 60U}}},
         trip{.trip_id_ = "k",
              .delays_ = {{.seq_ = 1,
                           .ev_type_ = event_type::kDep,
                           .delay_minutes_ = 0U}}},
         trip{.trip_id_ = "x",
              .delays_ = {{.seq_ = 1,
                           .ev_type_ = event_type::kDep,
                           .delay_minutes_ = 5U}}}},
        date::sys_days{2025_y / June / 12});

    rt::gtfsrt_update_msg(tt, rtt, source_idx_t{0}, "tag", msg1);

    auto td = transit_realtime::TripDescriptor{};
    td.set_trip_id("x");
    td.set_start_date("20250613");
    td.set_start_time("02:00:00");

    auto ss = std::stringstream{};
    rt::resolve_static(
        2025_y / June / 13, tt, source_idx_t{0U}, td,
        [&](rt::run r, trip_idx_t const) {
          r.stop_range_.from_ = 0U;
          r.stop_range_.to_ = static_cast<stop_idx_t>(
              tt.route_location_seq_[tt.transport_route_[r.t_.t_idx_]].size());
          ss << rt::frun{tt, &rtt, r} << "\n";
          return utl::continue_t::kContinue;
        });

    constexpr auto const kExpected =
        R"(   0: A       A...............................................                                                             d: 12.06 23:00 [13.06 01:00]  RT 13.06 00:00 [13.06 02:00]  [{name=i, day=2025-06-12, id=i, src=0}]
   1: C1      C1.............................................. a: 12.06 23:59 [13.06 01:59]  RT 13.06 00:59 [13.06 02:59]  d: 13.06 00:00 [13.06 02:00]  RT 13.06 00:59 [13.06 02:59]  [{name=x, day=2025-06-12, id=x, src=0}]
   2: D       D............................................... a: 13.06 01:00 [13.06 03:00]  RT 13.06 01:05 [13.06 03:05]  d: 13.06 01:00 [13.06 03:00]  RT 13.06 01:05 [13.06 03:05]  [{name=l, day=2025-06-12, id=l, src=0}]
   3: F       F............................................... a: 13.06 02:00 [13.06 04:00]  RT 13.06 02:00 [13.06 04:00]

   0: A       A...............................................                                                             d: 12.06 23:00 [13.06 01:00]  RT 13.06 00:00 [13.06 02:00]  [{name=i, day=2025-06-12, id=i, src=0}]
   1: C1      C1.............................................. a: 12.06 23:59 [13.06 01:59]  RT 13.06 00:59 [13.06 02:59]  d: 13.06 00:00 [13.06 02:00]  RT 13.06 00:59 [13.06 02:59]  [{name=x, day=2025-06-12, id=x, src=0}]
   2: D       D............................................... a: 13.06 01:00 [13.06 03:00]  RT 13.06 01:05 [13.06 03:05]  d: 13.06 01:01 [13.06 03:01]  RT 13.06 01:06 [13.06 03:06]  [{name=j, day=2025-06-12, id=j, src=0}]
   3: E       E............................................... a: 13.06 02:01 [13.06 04:01]  RT 13.06 02:01 [13.06 04:01]

   0: B       B...............................................                                                             d: 12.06 23:00 [13.06 01:00]  RT 12.06 23:00 [13.06 01:00]  [{name=k, day=2025-06-12, id=k, src=0}]
   1: C2      C2.............................................. a: 13.06 00:00 [13.06 02:00]  RT 13.06 00:00 [13.06 02:00]  d: 13.06 00:00 [13.06 02:00]  RT 13.06 00:05 [13.06 02:05]  [{name=x, day=2025-06-12, id=x, src=0}]
   2: D       D............................................... a: 13.06 01:00 [13.06 03:00]  RT 13.06 01:05 [13.06 03:05]  d: 13.06 01:00 [13.06 03:00]  RT 13.06 01:05 [13.06 03:05]  [{name=l, day=2025-06-12, id=l, src=0}]
   3: F       F............................................... a: 13.06 02:00 [13.06 04:00]  RT 13.06 02:00 [13.06 04:00]

   0: B       B...............................................                                                             d: 12.06 23:00 [13.06 01:00]  RT 12.06 23:00 [13.06 01:00]  [{name=k, day=2025-06-12, id=k, src=0}]
   1: C2      C2.............................................. a: 13.06 00:00 [13.06 02:00]  RT 13.06 00:00 [13.06 02:00]  d: 13.06 00:00 [13.06 02:00]  RT 13.06 00:05 [13.06 02:05]  [{name=x, day=2025-06-12, id=x, src=0}]
   2: D       D............................................... a: 13.06 01:00 [13.06 03:00]  RT 13.06 01:05 [13.06 03:05]  d: 13.06 01:01 [13.06 03:01]  RT 13.06 01:06 [13.06 03:06]  [{name=j, day=2025-06-12, id=j, src=0}]
   3: E       E............................................... a: 13.06 02:01 [13.06 04:01]  RT 13.06 02:01 [13.06 04:01]

)";

    EXPECT_EQ(kExpected, ss.str());

    run_test();

    // Service days per trip for [12th, 22nd[:
    // ik: 11, 12, 16, 17, 18, 19  // > 24:00
    // x: 12, 13, 14, 19, 21
    // x1: 18  // > 48:00
    // jl: 13, 14, 15, 17, 20, 21
    EXPECT_EQ(
        R"([{"idx":0,"firstDay":"2025-06-11","lastDay":"2025-06-21","noLocations":10,"noTrips":6,"transportsXDays":30}])",
        to_str(get_metrics(tt), tt));
  };

  run(0U, 1U, 2U, 3U, 4U, 5U);
  run(5U, 4U, 3U, 2U, 1U, 0U);
}

namespace {

template <typename... Args>
std::string simple_test_files(Args... order) {
  constexpr auto const kTrips = std::array{"a,a,a,a,", "b,b,b,b,"};
  return fmt::format(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,2.0,3.0,,
C,C,,4.0,5.0,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
a,1,1,1,1,1,1,1,20260327,20260331
b,1,1,1,1,1,1,1,20260327,20260331

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
a,DB,a,,,2
b,DB,b,,,2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
{}
{}

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
a,07:00:00,07:00:00,A,1,0,0
a,08:00:00,08:00:00,B,2,0,0
b,08:00:00,08:00:00,B,1,0,0
b,09:00:00,09:00:00,C,2,0,0

# transfers.txt
transfer_type,from_trip_id,to_trip_id
4,a,b
)",
                     kTrips[order]...);
}

}  // namespace

TEST(join_split, simple) {
  auto const run = [](auto&&... trips) {
    auto tt =
        loader::load({{.tag_ = "test",
                       .path_ = simple_test_files(trips...),
                       .loader_config_ = {.default_tz_ = "Europe/Berlin"}}},
                     {},
                     {date::sys_days{2026_y / March / 27},
                      date::sys_days{2026_y / April / 1}});
    std::cout << tt << "\n";
  };
  run(0U, 1U);
  run(1U, 0U);
}

namespace {

template <typename... Args>
std::string over_night_test_files(Args... order) {
  constexpr auto const kTrips = std::array{"a,a,a,a,", "b,b,b,b,"};
  return fmt::format(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,2.0,3.0,,
C,C,,4.0,5.0,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
a,1,1,1,1,1,1,1,20260327,20260331
b,1,1,1,1,1,1,1,20260327,20260331

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
a,DB,a,,,2
b,DB,b,,,2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
{}
{}

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
a,24:00:00,24:00:00,A,1,0,0
a,25:00:00,25:00:00,B,2,0,0
b,25:00:00,25:00:00,B,1,0,0
b,26:00:00,26:00:00,C,2,0,0

# transfers.txt
transfer_type,from_trip_id,to_trip_id
4,a,b
)",
                     kTrips[order]...);
}

}  // namespace

TEST(join_split, over_night) {
  auto const run = [](auto&&... trips) {
    auto tt =
        loader::load({{.tag_ = "test",
                       .path_ = over_night_test_files(trips...),
                       .loader_config_ = {.default_tz_ = "Europe/Berlin"}}},
                     {},
                     {date::sys_days{2026_y / March / 27},
                      date::sys_days{2026_y / April / 1}});
    std::cout << tt << "\n";
  };
  run(0U, 1U);
  run(1U, 0U);
}