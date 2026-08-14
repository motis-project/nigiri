#include "gtest/gtest.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/common/parse_time.h"
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
//     transfers.txt Y->Y type=2 0s from_route=R10 to_route=R11,
//     plus an unqualified Y->Y type=2 120s row (the pair default -> the
//     qualified rule stays a scoped exception, see majority fold).
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
//     plus an unqualified Z->Z type=2 120s row (pair default).
//     W2 (RZ2): Z 16:31 -> BZ 17:00 (banned by trip rule)
//     W3 (RZ2): Z 16:31 -> CZ 17:10 (allowed via route rule)
//
// (5) majority fold, uniform pair (MetroNorth pattern):
//     M,M type=1 trip pair M1->M2, no min_transfer_time -> folds into
//     transfer_time_[M] = 0, no virtual locations.
//     M1: A5 18:00 -> M 18:30
//     M2: M 18:30 -> G 19:00 (0 min -> reachable)
//     M3: M 18:30 -> H 19:00 (unnamed, gets the folded default too)
//
// (6) majority fold with exception:
//     N,N type=2 60s for trip pairs N1->N2 and N4->N2 (majority ->
//     transfer_time_[N] = 1 min), N,N type=2 600s for N1->N3 (exception,
//     kept as virtual location split).
//     N1: A6 20:00 -> N 20:30
//     N4: A7 20:05 -> N 20:29
//     N2: N 20:32 -> I 21:00 (1 min majority -> reachable from N1)
//     N3: N 20:32 -> J 21:00 (10 min exception -> not reachable)
//
// (7) self-pair rule (both sides match the same trips):
//     P,P type=2 600s from_route=RP to_route=RP -> all RP trips share one
//     virtual location, the value becomes its transfer time; explicit
//     P,P type=2 120s row keeps the rule an exception (majority fold).
//     P1 (RP): A8 22:00 -> P 22:30
//     P2 (RP): P 22:35 -> K 23:00 (5 min < 10 -> not reachable)
//     P3 (RQ): P 22:35 -> L 23:00 (unnamed, pair default 2 min -> ok)
//
// (8) mutually restricting trip pairs:
//     Q,Q type=2 600s for Q1->Q2 and for Q2->Q1, plus an unqualified
//     Q,Q type=2 120s row. Both virtual locations are the source AND the
//     target of a slower-than-default transfer, so neither is unrestricted.
//     Q1:  A9 08:00 -> Q 08:30
//     Q2:  Q 08:32 -> QA 09:00 (2 min gap < 10 -> not reachable)
//     Q2L: Q 08:41 -> QA 09:10 (11 min gap -> ok)
//     Q3:  Q 08:32 -> QB 09:00 (unnamed, pair default 2 min -> ok)
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
A5,A5,,54.0,6.0,,,
M,M,,54.0,6.5,,,
G,G,,54.0,7.0,,,
H,H,,54.0,7.5,,,
A6,A6,,55.0,6.0,,,
A7,A7,,55.0,6.2,,,
N,N,,55.0,6.5,,,
I,I,,55.0,7.0,,,
J,J,,55.0,7.5,,,
A8,A8,,56.0,6.0,,,
P,P,,56.0,6.5,,,
K,K,,56.0,7.0,,,
L,L,,56.0,7.5,,,
A9,A9,,57.0,6.0,,,
Q,Q,,57.0,6.5,,,
QA,QA,,57.0,7.0,,,
QB,QB,,57.0,7.5,,,

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
R30,AG,30,,,3
R31,AG,31,,,3
R32,AG,32,,,3
R40,AG,40,,,3
R41,AG,41,,,3
R42,AG,42,,,3
R43,AG,43,,,3
RP,AG,p,,,3
RQ,AG,q,,,3
RQ1,AG,q1,,,3
RQ2,AG,q2,,,3
RQ3,AG,q3,,,3

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
R30,S1,M1,,
R31,S1,M2,,
R32,S1,M3,,
R40,S1,N1,,
R41,S1,N2,,
R42,S1,N3,,
R43,S1,N4,,
RP,S1,P1,,
RP,S1,P2,,
RQ,S1,P3,,
RQ1,S1,Q1,,
RQ2,S1,Q2,,
RQ2,S1,Q2L,,
RQ3,S1,Q3,,

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
M1,18:00:00,18:00:00,A5,0
M1,18:30:00,18:30:00,M,1
M2,18:30:00,18:30:00,M,0
M2,19:00:00,19:00:00,G,1
M3,18:30:00,18:30:00,M,0
M3,19:00:00,19:00:00,H,1
N1,20:00:00,20:00:00,A6,0
N1,20:30:00,20:30:00,N,1
N4,20:05:00,20:05:00,A7,0
N4,20:29:00,20:29:00,N,1
N2,20:32:00,20:32:00,N,0
N2,21:00:00,21:00:00,I,1
N3,20:32:00,20:32:00,N,0
N3,21:00:00,21:00:00,J,1
P1,22:00:00,22:00:00,A8,0
P1,22:30:00,22:30:00,P,1
P2,22:35:00,22:35:00,P,0
P2,23:00:00,23:00:00,K,1
P3,22:35:00,22:35:00,P,0
P3,23:00:00,23:00:00,L,1
Q1,08:00:00,08:00:00,A9,0
Q1,08:30:00,08:30:00,Q,1
Q2,08:32:00,08:32:00,Q,0
Q2,09:00:00,09:00:00,QA,1
Q2L,08:41:00,08:41:00,Q,0
Q2L,09:10:00,09:10:00,QA,1
Q3,08:32:00,08:32:00,Q,0
Q3,09:00:00,09:00:00,QB,1

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1

# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time,from_route_id,to_route_id,from_trip_id,to_trip_id
XS,XS,2,600,,,,
Y,Y,2,120,,,,
Y,Y,2,0,R10,R11,,
F1,F2,3,,,,,
Z,Z,2,120,,,,
Z,Z,2,0,RZ1,RZ2,,
Z,Z,3,,,,W1,W2
M,M,1,,,,M1,M2
N,N,2,60,,,N1,N2
N,N,2,60,,,N4,N2
N,N,2,600,,,N1,N3
P,P,2,120,,,,
P,P,2,600,RP,RP,,
Q,Q,2,120,,,,
Q,Q,2,600,,,Q1,Q2
Q,Q,2,600,,,Q2,Q1
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
  auto const res =
      raptor_search(tt, nullptr, "A", "B", "2019-05-01 10:00 Europe/Berlin");
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(t("2019-05-01 11:10 Europe/Berlin"), begin(res)->dest_time_);
}

TEST(gtfs, transfer_rules_route_qualified) {
  auto const tt = load();

  // R10 -> R11: 0 min rule -> tight transfer works
  auto const res_c =
      raptor_search(tt, nullptr, "A2", "C", "2019-05-01 12:00 Europe/Berlin");
  ASSERT_EQ(1U, res_c.size());
  EXPECT_EQ(t("2019-05-01 13:00 Europe/Berlin"), begin(res_c)->dest_time_);

  // R10 -> R12: default 2 min transfer time -> 12:31 not reachable
  auto const res_d =
      raptor_search(tt, nullptr, "A2", "D", "2019-05-01 12:00 Europe/Berlin");
  EXPECT_EQ(0U, res_d.size());
}

TEST(gtfs, transfer_rules_forbidden) {
  auto const tt = load();

  // F1 -> F2 would be walkable (same station) but is forbidden
  auto const res =
      raptor_search(tt, nullptr, "A3", "E", "2019-05-01 14:00 Europe/Berlin");
  EXPECT_EQ(0U, res.size());
}

TEST(gtfs, transfer_rules_trip_beats_route) {
  auto const tt = load();

  // trip pair W1 -> W2 banned although the route rule would allow it
  auto const res_bz =
      raptor_search(tt, nullptr, "A4", "BZ", "2019-05-01 16:00 Europe/Berlin");
  EXPECT_EQ(0U, res_bz.size());

  // W1 -> W3 allowed via the 0 min route rule (default would need 2 min)
  auto const res_cz =
      raptor_search(tt, nullptr, "A4", "CZ", "2019-05-01 16:00 Europe/Berlin");
  ASSERT_EQ(1U, res_cz.size());
  EXPECT_EQ(t("2019-05-01 17:10 Europe/Berlin"), begin(res_cz)->dest_time_);
}

TEST(gtfs, transfer_rules_majority_fold_uniform) {
  auto const tt = load();

  // the uniform trip pair rule folds into the stop's transfer time
  auto const m = tt.locations_.location_id_to_idx_.at({"M", source_idx_t{0}});
  EXPECT_EQ(duration_t{0}, tt.locations_.transfer_time_[m]);

  // named trip pair: 0 min timed transfer works
  auto const res_g =
      raptor_search(tt, nullptr, "A5", "G", "2019-05-01 18:00 Europe/Berlin");
  ASSERT_EQ(1U, res_g.size());
  EXPECT_EQ(t("2019-05-01 19:00 Europe/Berlin"), begin(res_g)->dest_time_);

  // unnamed trip gets the folded default as well
  auto const res_h =
      raptor_search(tt, nullptr, "A5", "H", "2019-05-01 18:00 Europe/Berlin");
  ASSERT_EQ(1U, res_h.size());
  EXPECT_EQ(t("2019-05-01 19:00 Europe/Berlin"), begin(res_h)->dest_time_);
}

TEST(gtfs, transfer_rules_majority_fold_exception) {
  auto const tt = load();

  // majority 60s becomes the stop default
  auto const n = tt.locations_.location_id_to_idx_.at({"N", source_idx_t{0}});
  EXPECT_EQ(duration_t{1}, tt.locations_.transfer_time_[n]);

  // N1 -> N2 uses the folded 1 min default (via base stop)
  auto const res_i =
      raptor_search(tt, nullptr, "A6", "I", "2019-05-01 20:00 Europe/Berlin");
  ASSERT_EQ(1U, res_i.size());
  EXPECT_EQ(t("2019-05-01 21:00 Europe/Berlin"), begin(res_i)->dest_time_);

  // N1 -> N3 is the 10 min exception -> not reachable
  auto const res_j =
      raptor_search(tt, nullptr, "A6", "J", "2019-05-01 20:00 Europe/Berlin");
  EXPECT_EQ(0U, res_j.size());
}

TEST(gtfs, transfer_rules_self_pair) {
  auto const tt = load();

  // RP -> RP: both rule sides match the same trips -> one shared virtual
  // location; the 10 min value is its transfer time, 5 min gap fails
  auto const res_k =
      raptor_search(tt, nullptr, "A8", "K", "2019-05-01 22:00 Europe/Berlin");
  EXPECT_EQ(0U, res_k.size());

  // unnamed RQ departure uses the explicit 2 min pair default
  auto const res_l =
      raptor_search(tt, nullptr, "A8", "L", "2019-05-01 22:00 Europe/Berlin");
  ASSERT_EQ(1U, res_l.size());
  EXPECT_EQ(t("2019-05-01 23:00 Europe/Berlin"), begin(res_l)->dest_time_);
}

TEST(gtfs, transfer_rules_mutual_restriction) {
  auto const tt = load();

  // Q1 -> Q2 needs 10 min: the 08:32 departure is out, the 08:41 one works
  auto const res_qa = raptor_search(tt, nullptr, "A9", "QA",
                                    "2019-05-01 08:00 Europe/Berlin");
  ASSERT_EQ(1U, res_qa.size());
  EXPECT_EQ(t("2019-05-01 09:10 Europe/Berlin"), begin(res_qa)->dest_time_);

  // the unnamed trip still departs from the base stop at the 2 min default
  auto const res_qb = raptor_search(tt, nullptr, "A9", "QB",
                                    "2019-05-01 08:00 Europe/Berlin");
  ASSERT_EQ(1U, res_qb.size());
  EXPECT_EQ(t("2019-05-01 09:00 Europe/Berlin"), begin(res_qb)->dest_time_);
}
