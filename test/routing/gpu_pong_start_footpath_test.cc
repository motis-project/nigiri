#include "gtest/gtest.h"

#include "nigiri/loader/load.h"
#include "nigiri/timetable.h"

#include "../raptor_search.h"

using namespace date;
using namespace nigiri;
using namespace std::chrono_literals;
using namespace std::string_view_literals;
using nigiri::test::raptor_search;

namespace {

// X <-> P: 20 minute walks. T: P 10:30 -> Q 10:40, T2: Q 11:00 -> P 11:10.
constexpr auto kFeed = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
MTA,MTA,https://motis-project.de/,Etc/UTC

# calendar_dates.txt
service_id,date,exception_type
D,20190503,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R,MTA,R,,,2
R2,MTA,R2,,,2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R,D,T,,
R2,D,T2,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T,10:30:00,10:30:00,P,1,0,0
T,10:40:00,10:40:00,Q,2,0,0
T2,11:00:00,11:00:00,Q,1,0,0
T2,11:10:00,11:10:00,P,2,0,0

# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time
X,P,2,1200
P,X,2,1200

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
X,X,,,,,,
P,P,,,,,,
Q,Q,,,,,,
)"sv;

constexpr auto kFwdJourney = R"(
[2019-05-03 11:00, 2019-05-03 11:30]
TRANSFERS: 0
     FROM: (Q, Q) [2019-05-03 11:00]
       TO: (X, X) [2019-05-03 11:30]
leg 0: (Q, Q) [2019-05-03 11:00] -> (P, P) [2019-05-03 11:10]
   0: Q       Q...............................................                               d: 03.05 11:00 [03.05 11:00]  [{name=R2, day=2019-05-03, id=T2, src=0}]
   1: P       P............................................... a: 03.05 11:10 [03.05 11:10]
leg 1: (P, P) [2019-05-03 11:10] -> (X, X) [2019-05-03 11:30]
  FOOTPATH (duration=20)


)"sv;

constexpr auto kBwdJourney = R"(
[2019-05-03 10:10, 2019-05-03 10:40]
TRANSFERS: 0
     FROM: (X, X) [2019-05-03 10:10]
       TO: (Q, Q) [2019-05-03 10:40]
leg 0: (X, X) [2019-05-03 10:10] -> (P, P) [2019-05-03 10:30]
  FOOTPATH (duration=20)
leg 1: (P, P) [2019-05-03 10:30] -> (Q, Q) [2019-05-03 10:40]
   0: P       P...............................................                               d: 03.05 10:30 [03.05 10:30]  [{name=R, day=2019-05-03, id=T, src=0}]
   1: Q       Q............................................... a: 03.05 10:40 [03.05 10:40]


)"sv;

}  // namespace

// The walk is at the search end, where pong's reverse search starts: its
// reconstruction has to find the start walk of the reverse search.
// raptor_search compares CPU and GPU pong for interval queries.
TEST(routing, pong_walk_at_search_end) {
  auto const tt = loader::load(loader::mem_dir::read(kFeed),
                               {2019_y / May / 2, 2019_y / May / 4});
  auto const day = unixtime_t{sys_days{2019_y / May / 3}};

  // Q -> X: T2, then the walk P -> X.
  EXPECT_EQ(
      kFwdJourney,
      test::print_results(tt, nullptr,
                          raptor_search(tt, nullptr, "Q", "X",
                                        interval{day + 9h, day + 11h + 30min},
                                        direction::kForward)));

  // arriveBy, the search starts at Q. X -> Q: the walk X -> P, then T.
  EXPECT_EQ(kBwdJourney,
            test::print_results(tt, nullptr,
                                raptor_search(tt, nullptr, "Q", "X",
                                              interval{day + 10h, day + 13h},
                                              direction::kBackward)));
}
