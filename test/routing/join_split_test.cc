#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"

#include "../raptor_search.h"
#include "results_to_string.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace std::chrono_literals;
using nigiri::test::raptor_search;

namespace {

// Days are relative to i, j.
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

                                     {"C", "E", 2025_y / June / 13, true},
                                     {"C", "F", 2025_y / June / 13, true},

                                     {"C", "E", 2025_y / June / 14, false},
                                     {"C", "F", 2025_y / June / 14, false},

                                     {"A", "C", 2025_y / June / 16, true},
                                     {"B", "C", 2025_y / June / 16, true},
                                     {"D", "E", 2025_y / June / 16, true},
                                     {"D", "F", 2025_y / June / 16, true},

                                     {"A", "D", 2025_y / June / 18, true},
                                     {"B", "D", 2025_y / June / 18, true}};

mem_dir test_files() {
  return mem_dir::read(R"(
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
D,D,,5.0,7.0,,
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
i,ik,i,i,
j,jl,j,j,
k,ik,k,k,
l,jl,l,l,
x,x,x,x,
x1,x1,x1,x1,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
i,23:00:00,23:00:00,A,1,0,0
i,23:59:00,23:59:00,C1,2,0,0
k,23:00:00,23:00:00:00,B,1,0,0
k,24:00:00,24:00:00:00,C2,2,0,0
x,00:00:00,00:00:00,C,1,0,0
x,01:00:00,01:00:00,D,2,0,0
x1,48:00:00,48:00:00,C,1,0,0
x1,49:00:00,49:00:00,D,2,0,0
j,01:01:00,01:01:00,D1,1,0,0
j,02:01:00,02:01:00,E,2,0,0
l,01:00:00,01:00:00,D2,1,0,0
l,02:00:00,02:00:00,F,2,0,0

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
)");
}

}  // namespace

TEST(routing, join_split) {
  timetable tt;
  tt.date_range_ = {date::sys_days{2025_y / June / 12},
                    date::sys_days{2025_y / June / 22}};
  auto const src = source_idx_t{0};
  load_timetable({}, src, test_files(), tt);
  finalize(tt);

  for (auto const [from, to, date, expected] : kTests) {
    EXPECT_EQ(expected, !raptor_search(tt, nullptr, from, to,
                                       interval{date + 10h, date + 16h})
                             .empty())
        << "from=" << from << ", to=" << to << ", on " << date << ", expected "
        << expected;
  }
}
