#include "gtest/gtest.h"

#include "nigiri/common/parse_time.h"
#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/timetable.h"

#include "../../raptor_search.h"

using namespace nigiri;
using namespace date;
using namespace std::string_view_literals;
using nigiri::test::raptor_search;

namespace {

// Scenarios (all on 2019-05-01, Europe/Berlin):
//
// (1) station-level min transfer time cascade:
//     XS (station) with platforms X1, X2; transfers.txt XS->XS type=2 600s.
//     T1: A 10:00 -> X1 10:30
//     T2: X2 10:35 -> B 11:00 (5 min gap < 10 -> not reachable)
//     T3: X2 10:42 -> B 11:10 (12 min gap -> ok)
//
// (2) route-qualified same-stop rule:
//     transfers.txt Y->Y type=2 0s from_route=R10 to_route=R11.
//     U1 (R10): A2 12:00 -> Y 12:30
//     U2 (R11): Y 12:31 -> C 13:00 (rule 0 min -> reachable)
//     U3 (R12): Y 12:31 -> D 13:00 (default 2 min -> not reachable)
//
// (3) directed forbidden transfer:
//     F1, F2 platforms of FS (walking distance); transfers.txt F1->F2 type=3.
//     V1: A3 14:00 -> F1 14:30
//     V2: F2 14:40 -> E 15:00 (would be reachable by foot -> banned)
//
// (4) trip-qualified rule beats route-qualified rule:
//     Z->Z type=2 0s from_route=RZ1 to_route=RZ2 (allows tight transfers),
//     Z->Z type=3 from_trip=W1 to_trip=W2 (bans this one pair).
//     W1 (RZ1): A4 16:00 -> Z 16:30
//     W2 (RZ2): Z 16:31 -> BZ 17:00 (banned by trip rule)
//     W3 (RZ2): Z 16:31 -> CZ 17:10 (allowed via route rule)
constexpr auto const test_files = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
AG,Agency,https://example.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,50.0,6.0,,,
XS,XS,,50.0,6.5,,1,
X1,X1,,50.0001,6.5,,,XS
X2,X2,,50.0002,6.5,,,XS
B,B,,50.0,7.0,,,
A2,A2,,51.0,6.0,,,
Y,Y,,51.0,6.5,,,
C,C,,51.0,7.0,,,
D,D,,51.0,7.5,,,
A3,A3,,52.0,6.0,,,
FS,FS,,52.0,6.5,,1,
F1,F1,,52.0001,6.5,,,FS
F2,F2,,52.0002,6.5,,,FS
E,E,,52.0,7.0,,,
A4,A4,,53.0,6.0,,,
Z,Z,,53.0,6.5,,,
BZ,BZ,,53.0,7.0,,,
CZ,CZ,,53.0,7.5,,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,AG,1,,,3
R2,AG,2,,,3
R3,AG,3,,,3
R10,AG,10,,,3
R11,AG,11,,,3
R12,AG,12,,,3
R20,AG,20,,,3
R21,AG,21,,,3
RZ1,AG,z1,,,3
RZ2,AG,z2,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R1,S1,T1,,
R2,S1,T2,,
R2,S1,T3,,
R10,S1,U1,,
R11,S1,U2,,
R12,S1,U3,,
R20,S1,V1,,
R21,S1,V2,,
RZ1,S1,W1,,
RZ2,S1,W2,,
RZ2,S1,W3,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
T1,10:00:00,10:00:00,A,0
T1,10:30:00,10:30:00,X1,1
T2,10:35:00,10:35:00,X2,0
T2,11:00:00,11:00:00,B,1
T3,10:42:00,10:42:00,X2,0
T3,11:10:00,11:10:00,B,1
U1,12:00:00,12:00:00,A2,0
U1,12:30:00,12:30:00,Y,1
U2,12:31:00,12:31:00,Y,0
U2,13:00:00,13:00:00,C,1
U3,12:31:00,12:31:00,Y,0
U3,13:00:00,13:00:00,D,1
V1,14:00:00,14:00:00,A3,0
V1,14:30:00,14:30:00,F1,1
V2,14:40:00,14:40:00,F2,0
V2,15:00:00,15:00:00,E,1
W1,16:00:00,16:00:00,A4,0
W1,16:30:00,16:30:00,Z,1
W2,16:31:00,16:31:00,Z,0
W2,17:00:00,17:00:00,BZ,1
W3,16:31:00,16:31:00,Z,0
W3,17:10:00,17:10:00,CZ,1

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1

# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time,from_route_id,to_route_id,from_trip_id,to_trip_id
XS,XS,2,600,,,,
Y,Y,2,0,R10,R11,,
F1,F2,3,,,,,
Z,Z,2,0,RZ1,RZ2,,
Z,Z,3,,,,W1,W2
)"sv;

timetable load() {
  auto tt = timetable{};
  tt.date_range_ = {date::sys_days{2019_y / May / 1},
                    date::sys_days{2019_y / May / 2}};
  loader::register_special_stations(tt);
  loader::gtfs::load_timetable({}, source_idx_t{0},
                               loader::mem_dir::read(test_files), tt);
  loader::finalize(tt);
  return tt;
}

unixtime_t t(char const* s) { return parse_time_tz(s, "%Y-%m-%d %H:%M %Z"); }

}  // namespace

TEST(gtfs, transfer_rules_station_level_min_time) {
  auto const tt = load();

  // 10 min required between X1/X2 (cascaded from station rule)
  // -> the 10:35 departure is not reachable, the 10:42 departure is
  auto const res = raptor_search(tt, nullptr, "A", "B",
                                 "2019-05-01 10:00 Europe/Berlin");
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(t("2019-05-01 11:10 Europe/Berlin"), begin(res)->dest_time_);
}

TEST(gtfs, transfer_rules_route_qualified) {
  auto const tt = load();

  // R10 -> R11: 0 min rule -> tight transfer works
  auto const res_c = raptor_search(tt, nullptr, "A2", "C",
                                   "2019-05-01 12:00 Europe/Berlin");
  ASSERT_EQ(1U, res_c.size());
  EXPECT_EQ(t("2019-05-01 13:00 Europe/Berlin"), begin(res_c)->dest_time_);

  // R10 -> R12: default 2 min transfer time -> 12:31 not reachable
  auto const res_d = raptor_search(tt, nullptr, "A2", "D",
                                   "2019-05-01 12:00 Europe/Berlin");
  EXPECT_EQ(0U, res_d.size());
}

TEST(gtfs, transfer_rules_forbidden) {
  auto const tt = load();

  // F1 -> F2 would be walkable (same station) but is forbidden
  auto const res = raptor_search(tt, nullptr, "A3", "E",
                                 "2019-05-01 14:00 Europe/Berlin");
  EXPECT_EQ(0U, res.size());
}

TEST(gtfs, transfer_rules_trip_beats_route) {
  auto const tt = load();

  // trip pair W1 -> W2 banned although the route rule would allow it
  auto const res_bz = raptor_search(tt, nullptr, "A4", "BZ",
                                    "2019-05-01 16:00 Europe/Berlin");
  EXPECT_EQ(0U, res_bz.size());

  // W1 -> W3 allowed via the 0 min route rule (default would need 2 min)
  auto const res_cz = raptor_search(tt, nullptr, "A4", "CZ",
                                    "2019-05-01 16:00 Europe/Berlin");
  ASSERT_EQ(1U, res_cz.size());
  EXPECT_EQ(t("2019-05-01 17:10 Europe/Berlin"), begin(res_cz)->dest_time_);
}
