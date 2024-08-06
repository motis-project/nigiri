#include "gtest/gtest.h"

#include <algorithm>
#include <regex>

#include "utl/erase_if.h"

#include "nigiri/loader/gtfs/agency.h"
#include "nigiri/loader/gtfs/calendar.h"
#include "nigiri/loader/gtfs/calendar_date.h"
#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/gtfs/local_to_utc.h"
#include "nigiri/loader/gtfs/noon_offsets.h"
#include "nigiri/loader/init_finish.h"

#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

#include "../raptor_search.h"
#include "../rt/util.h"

using namespace nigiri;
using namespace date;
using namespace std::chrono_literals;
using namespace std::string_view_literals;
using nigiri::test::parse_time;
using nigiri::test::raptor_search;

namespace {

constexpr auto const test_files_1 = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,2.0,3.0,,
C,C,,4.0,5.0,,
D,D,,6.0,7.0,,
E,E,,8.0,9.0,,
F,F,,10.0,11.0,,
G,G,,12.0,13.0,,
H,H,,14.0,15.0,,
I,I,,16.0,17.0,1,
I1,I1,,16.0,17.0,,I
I2,I2,,16.0,17.0,,I
J,J,,18.0,19.0,,
K,K,,20.0,21.0,1,
K1,K1,,20.0,21.0,,K
K2,K2,,20.0,21.0,,K
L,L,,22.0,23.0,1,
L1,L1,,22.0,23.0,,L
L2,L2,,22.0,23.0,,L
M,M,,24.0,25.0,,
N,N,,26.0,27.0,,
O,O,,28.0,29.0,,
P,P,,30.0,31.0,,
Q,Q,,32.0,33.0,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DB,0,,,3
R1,DB,1,,,3
R2,DB,2,,,3
R3,DB,3,,,3
R4,DB,4,,,3
R5,DB,5,,,3
R6,DB,6,,,3
R7,DB,7,,,3
R8,DB,8,,,3
R9,DB,9,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,S1,T0,,
R1,S1,T1,,
R1,S1,T2,,
R2,S1,T3,,
R2,S1,T4,,
R3,S1,T5,,
R5,S1,T6,,
R4,S1,T7,,
R6,S1,T8,,
R7,S1,T9,,
R7,S1,T10,,
R8,S1,T11,,
R9,S1,T12,,
R2,S1,T13,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
T0,10:00:00,10:00:00,A,0
T0,10:10:00,10:10:00,B,1
T1,10:15:00,10:15:00,B,0
T1,10:30:00,10:30:00,D,1
T2,10:30:00,10:30:00,B,0
T2,10:45:00,10:45:00,D,1
T3,10:00:00,10:00:00,A,0
T3,10:20:00,10:22:00,C,1
T3,10:40:00,10:40:00,D,2
T4,10:15:00,10:15:00,A,0
T4,10:35:00,10:37:00,C,1
T4,10:55:00,10:55:00,D,2
T5,15:00:00,15:00:00,A,0
T5,15:15:00,15:17:00,I1,1
T5,15:30:00,15:30:00,J,2
T6,16:00:00,16:00:00,A,0
T6,16:15:00,16:17:00,I1,1
T7,16:20:00,16:20:00,I2,0
T7,16:40:00,16:40:00,J,1
T8,15:30:00,15:30:00,K1,0
T8,16:00:00,16:00:00,J,1
T9,11:00:00,11:00:00,M,0
T9,11:13:00,11:15:00,N,1
T9,11:28:00,11:30:00,O,2
T9,11:43:00,11:45:00,P,3
T9,12:00:00,12:00:00,Q,4
T10,12:00:00,12:00:00,M,0
T10,12:13:00,12:15:00,N,1
T10,12:28:00,12:30:00,O,2
T10,12:43:00,12:45:00,P,3
T10,13:00:00,13:00:00,Q,4
T11,11:00:00,11:00:00,H,0
T11,11:42:00,11:42:00,M,1
T12,11:00:00,11:00:00,H,0
T12,11:20:00,11:20:00,O,1
T13,10:30:00,10:30:00,A,0
T13,10:45:00,10:55:00,C,1
T13,11:30:00,11:30:00,D,2

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1

# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time
E,A,2,300
D,H,2,180
I1,I2,2,240
I2,I1,2,240
I,K,2,300
K,I,2,300
K,L,2,360
L,K,2,360
)"sv;

constexpr auto const expected_A_D_no_via =
    R"(
[2019-05-01 08:00, 2019-05-01 08:40]
TRANSFERS: 0
     FROM: (A, A) [2019-05-01 08:00]
       TO: (D, D) [2019-05-01 08:40]
leg 0: (A, A) [2019-05-01 08:00] -> (D, D) [2019-05-01 08:40]
   0: A       A...............................................                               d: 01.05 08:00 [01.05 10:00]  [{name=Bus 2, day=2019-05-01, id=T3, src=0}]
   1: C       C............................................... a: 01.05 08:20 [01.05 10:20]  d: 01.05 08:22 [01.05 10:22]  [{name=Bus 2, day=2019-05-01, id=T3, src=0}]
   2: D       D............................................... a: 01.05 08:40 [01.05 10:40]


[2019-05-01 08:00, 2019-05-01 08:30]
TRANSFERS: 1
     FROM: (A, A) [2019-05-01 08:00]
       TO: (D, D) [2019-05-01 08:30]
leg 0: (A, A) [2019-05-01 08:00] -> (B, B) [2019-05-01 08:10]
   0: A       A...............................................                               d: 01.05 08:00 [01.05 10:00]  [{name=Bus 0, day=2019-05-01, id=T0, src=0}]
   1: B       B............................................... a: 01.05 08:10 [01.05 10:10]
leg 1: (B, B) [2019-05-01 08:10] -> (B, B) [2019-05-01 08:12]
  FOOTPATH (duration=2)
leg 2: (B, B) [2019-05-01 08:15] -> (D, D) [2019-05-01 08:30]
   0: B       B...............................................                               d: 01.05 08:15 [01.05 10:15]  [{name=Bus 1, day=2019-05-01, id=T1, src=0}]
   1: D       D............................................... a: 01.05 08:30 [01.05 10:30]


)"sv;

constexpr auto const expected_A_D_via_B_0min =
    R"(
[2019-05-01 08:00, 2019-05-01 08:30]
TRANSFERS: 1
     FROM: (A, A) [2019-05-01 08:00]
       TO: (D, D) [2019-05-01 08:30]
leg 0: (A, A) [2019-05-01 08:00] -> (B, B) [2019-05-01 08:10]
   0: A       A...............................................                               d: 01.05 08:00 [01.05 10:00]  [{name=Bus 0, day=2019-05-01, id=T0, src=0}]
   1: B       B............................................... a: 01.05 08:10 [01.05 10:10]
leg 1: (B, B) [2019-05-01 08:10] -> (B, B) [2019-05-01 08:12]
  FOOTPATH (duration=2)
leg 2: (B, B) [2019-05-01 08:15] -> (D, D) [2019-05-01 08:30]
   0: B       B...............................................                               d: 01.05 08:15 [01.05 10:15]  [{name=Bus 1, day=2019-05-01, id=T1, src=0}]
   1: D       D............................................... a: 01.05 08:30 [01.05 10:30]


)"sv;

constexpr auto const expected_A_D_via_B_3min =
    R"(
[2019-05-01 08:00, 2019-05-01 08:30]
TRANSFERS: 1
     FROM: (A, A) [2019-05-01 08:00]
       TO: (D, D) [2019-05-01 08:30]
leg 0: (A, A) [2019-05-01 08:00] -> (B, B) [2019-05-01 08:10]
   0: A       A...............................................                               d: 01.05 08:00 [01.05 10:00]  [{name=Bus 0, day=2019-05-01, id=T0, src=0}]
   1: B       B............................................... a: 01.05 08:10 [01.05 10:10]
leg 1: (B, B) [2019-05-01 08:13] -> (B, B) [2019-05-01 08:15]
  FOOTPATH (duration=2)
leg 2: (B, B) [2019-05-01 08:15] -> (D, D) [2019-05-01 08:30]
   0: B       B...............................................                               d: 01.05 08:15 [01.05 10:15]  [{name=Bus 1, day=2019-05-01, id=T1, src=0}]
   1: D       D............................................... a: 01.05 08:30 [01.05 10:30]


)"sv;

constexpr auto const expected_A_D_via_B_10min =
    R"(
[2019-05-01 08:00, 2019-05-01 08:45]
TRANSFERS: 1
     FROM: (A, A) [2019-05-01 08:00]
       TO: (D, D) [2019-05-01 08:45]
leg 0: (A, A) [2019-05-01 08:00] -> (B, B) [2019-05-01 08:10]
   0: A       A...............................................                               d: 01.05 08:00 [01.05 10:00]  [{name=Bus 0, day=2019-05-01, id=T0, src=0}]
   1: B       B............................................... a: 01.05 08:10 [01.05 10:10]
leg 1: (B, B) [2019-05-01 08:20] -> (B, B) [2019-05-01 08:22]
  FOOTPATH (duration=2)
leg 2: (B, B) [2019-05-01 08:30] -> (D, D) [2019-05-01 08:45]
   0: B       B...............................................                               d: 01.05 08:30 [01.05 10:30]  [{name=Bus 1, day=2019-05-01, id=T2, src=0}]
   1: D       D............................................... a: 01.05 08:45 [01.05 10:45]


)"sv;

constexpr auto const expected_A_D_via_C_0min =
    R"(
[2019-05-01 08:00, 2019-05-01 08:40]
TRANSFERS: 0
     FROM: (A, A) [2019-05-01 08:00]
       TO: (D, D) [2019-05-01 08:40]
leg 0: (A, A) [2019-05-01 08:00] -> (D, D) [2019-05-01 08:40]
   0: A       A...............................................                               d: 01.05 08:00 [01.05 10:00]  [{name=Bus 2, day=2019-05-01, id=T3, src=0}]
   1: C       C............................................... a: 01.05 08:20 [01.05 10:20]  d: 01.05 08:22 [01.05 10:22]  [{name=Bus 2, day=2019-05-01, id=T3, src=0}]
   2: D       D............................................... a: 01.05 08:40 [01.05 10:40]


)"sv;

constexpr auto const expected_A_D_via_C_10min =
    R"(
[2019-05-01 08:00, 2019-05-01 08:55]
TRANSFERS: 1
     FROM: (A, A) [2019-05-01 08:00]
       TO: (D, D) [2019-05-01 08:55]
leg 0: (A, A) [2019-05-01 08:00] -> (C, C) [2019-05-01 08:20]
   0: A       A...............................................                               d: 01.05 08:00 [01.05 10:00]  [{name=Bus 2, day=2019-05-01, id=T3, src=0}]
   1: C       C............................................... a: 01.05 08:20 [01.05 10:20]
leg 1: (C, C) [2019-05-01 08:30] -> (C, C) [2019-05-01 08:32]
  FOOTPATH (duration=2)
leg 2: (C, C) [2019-05-01 08:37] -> (D, D) [2019-05-01 08:55]
   1: C       C...............................................                               d: 01.05 08:37 [01.05 10:37]  [{name=Bus 2, day=2019-05-01, id=T4, src=0}]
   2: D       D............................................... a: 01.05 08:55 [01.05 10:55]


)"sv;

constexpr auto const expected_A_J_via_I_1500_0min =
    R"(
[2019-05-01 13:00, 2019-05-01 13:30]
TRANSFERS: 0
     FROM: (A, A) [2019-05-01 13:00]
       TO: (J, J) [2019-05-01 13:30]
leg 0: (A, A) [2019-05-01 13:00] -> (J, J) [2019-05-01 13:30]
   0: A       A...............................................                               d: 01.05 13:00 [01.05 15:00]  [{name=Bus 3, day=2019-05-01, id=T5, src=0}]
   1: I1      I1.............................................. a: 01.05 13:15 [01.05 15:15]  d: 01.05 13:17 [01.05 15:17]  [{name=Bus 3, day=2019-05-01, id=T5, src=0}]
   2: J       J............................................... a: 01.05 13:30 [01.05 15:30]


)"sv;

constexpr auto const expected_A_J_via_I_1600_0min =
    R"(
[2019-05-01 13:00, 2019-05-01 13:30]
TRANSFERS: 0
     FROM: (A, A) [2019-05-01 13:00]
       TO: (J, J) [2019-05-01 13:30]
leg 0: (A, A) [2019-05-01 13:00] -> (J, J) [2019-05-01 13:30]
   0: A       A...............................................                               d: 01.05 13:00 [01.05 15:00]  [{name=Bus 3, day=2019-05-01, id=T5, src=0}]
   1: I1      I1.............................................. a: 01.05 13:15 [01.05 15:15]  d: 01.05 13:17 [01.05 15:17]  [{name=Bus 3, day=2019-05-01, id=T5, src=0}]
   2: J       J............................................... a: 01.05 13:30 [01.05 15:30]


[2019-05-01 14:00, 2019-05-01 14:40]
TRANSFERS: 1
     FROM: (A, A) [2019-05-01 14:00]
       TO: (J, J) [2019-05-01 14:40]
leg 0: (A, A) [2019-05-01 14:00] -> (I1, I1) [2019-05-01 14:15]
   0: A       A...............................................                               d: 01.05 14:00 [01.05 16:00]  [{name=Bus 5, day=2019-05-01, id=T6, src=0}]
   1: I1      I1.............................................. a: 01.05 14:15 [01.05 16:15]
leg 1: (I1, I1) [2019-05-01 14:15] -> (I2, I2) [2019-05-01 14:19]
  FOOTPATH (duration=4)
leg 2: (I2, I2) [2019-05-01 14:20] -> (J, J) [2019-05-01 14:40]
   0: I2      I2..............................................                               d: 01.05 14:20 [01.05 16:20]  [{name=Bus 4, day=2019-05-01, id=T7, src=0}]
   1: J       J............................................... a: 01.05 14:40 [01.05 16:40]


)"sv;

constexpr auto const expected_A_J_via_I_1500_10min =
    R"(
[2019-05-01 13:00, 2019-05-01 14:40]
TRANSFERS: 1
     FROM: (A, A) [2019-05-01 13:00]
       TO: (J, J) [2019-05-01 14:40]
leg 0: (A, A) [2019-05-01 13:00] -> (I1, I1) [2019-05-01 13:15]
   0: A       A...............................................                               d: 01.05 13:00 [01.05 15:00]  [{name=Bus 3, day=2019-05-01, id=T5, src=0}]
   1: I1      I1.............................................. a: 01.05 13:15 [01.05 15:15]
leg 1: (I1, I1) [2019-05-01 13:25] -> (I2, I2) [2019-05-01 13:29]
  FOOTPATH (duration=4)
leg 2: (I2, I2) [2019-05-01 14:20] -> (J, J) [2019-05-01 14:40]
   0: I2      I2..............................................                               d: 01.05 14:20 [01.05 16:20]  [{name=Bus 4, day=2019-05-01, id=T7, src=0}]
   1: J       J............................................... a: 01.05 14:40 [01.05 16:40]


)"sv;

constexpr auto const expected_A_J_via_K_0min =
    R"(
[2019-05-01 13:00, 2019-05-01 14:00]
TRANSFERS: 1
     FROM: (A, A) [2019-05-01 13:00]
       TO: (J, J) [2019-05-01 14:00]
leg 0: (A, A) [2019-05-01 13:00] -> (I1, I1) [2019-05-01 13:15]
   0: A       A...............................................                               d: 01.05 13:00 [01.05 15:00]  [{name=Bus 3, day=2019-05-01, id=T5, src=0}]
   1: I1      I1.............................................. a: 01.05 13:15 [01.05 15:15]
leg 1: (I1, I1) [2019-05-01 13:15] -> (K1, K1) [2019-05-01 13:24]
  FOOTPATH (duration=9)
leg 2: (K1, K1) [2019-05-01 13:30] -> (J, J) [2019-05-01 14:00]
   0: K1      K1..............................................                               d: 01.05 13:30 [01.05 15:30]  [{name=Bus 6, day=2019-05-01, id=T8, src=0}]
   1: J       J............................................... a: 01.05 14:00 [01.05 16:00]


)"sv;

constexpr auto const expected_H_Q_via_N_0min =
    R"(
[2019-05-01 09:00, 2019-05-01 11:00]
TRANSFERS: 1
     FROM: (H, H) [2019-05-01 09:00]
       TO: (Q, Q) [2019-05-01 11:00]
leg 0: (H, H) [2019-05-01 09:00] -> (M, M) [2019-05-01 09:42]
   0: H       H...............................................                               d: 01.05 09:00 [01.05 11:00]  [{name=Bus 8, day=2019-05-01, id=T11, src=0}]
   1: M       M............................................... a: 01.05 09:42 [01.05 11:42]
leg 1: (M, M) [2019-05-01 09:42] -> (M, M) [2019-05-01 09:44]
  FOOTPATH (duration=2)
leg 2: (M, M) [2019-05-01 10:00] -> (Q, Q) [2019-05-01 11:00]
   0: M       M...............................................                               d: 01.05 10:00 [01.05 12:00]  [{name=Bus 7, day=2019-05-01, id=T10, src=0}]
   1: N       N............................................... a: 01.05 10:13 [01.05 12:13]  d: 01.05 10:15 [01.05 12:15]  [{name=Bus 7, day=2019-05-01, id=T10, src=0}]
   2: O       O............................................... a: 01.05 10:28 [01.05 12:28]  d: 01.05 10:30 [01.05 12:30]  [{name=Bus 7, day=2019-05-01, id=T10, src=0}]
   3: P       P............................................... a: 01.05 10:43 [01.05 12:43]  d: 01.05 10:45 [01.05 12:45]  [{name=Bus 7, day=2019-05-01, id=T10, src=0}]
   4: Q       Q............................................... a: 01.05 11:00 [01.05 13:00]


)"sv;

constexpr auto const expected_A_D_via_C_5min =
    R"(
[2019-05-01 08:30, 2019-05-01 09:30]
TRANSFERS: 1
     FROM: (A, A) [2019-05-01 08:30]
       TO: (D, D) [2019-05-01 09:30]
leg 0: (A, A) [2019-05-01 08:30] -> (C, C) [2019-05-01 08:45]
   0: A       A...............................................                               d: 01.05 08:30 [01.05 10:30]  [{name=Bus 2, day=2019-05-01, id=T13, src=0}]
   1: C       C............................................... a: 01.05 08:45 [01.05 10:45]
leg 1: (C, C) [2019-05-01 08:50] -> (C, C) [2019-05-01 08:52]
  FOOTPATH (duration=2)
leg 2: (C, C) [2019-05-01 08:55] -> (D, D) [2019-05-01 09:30]
   1: C       C...............................................                               d: 01.05 08:55 [01.05 10:55]  [{name=Bus 2, day=2019-05-01, id=T13, src=0}]
   2: D       D............................................... a: 01.05 09:30 [01.05 11:30]


)"sv;

constexpr auto const expected_intermodal_HO_Q_via_P_0min =
    R"(
[2019-05-01 09:00, 2019-05-01 11:00]
TRANSFERS: 0
     FROM: (START, START) [2019-05-01 09:00]
       TO: (Q, Q) [2019-05-01 11:00]
leg 0: (START, START) [2019-05-01 09:00] -> (O, O) [2019-05-01 09:40]
  MUMO (id=1, duration=40)
leg 1: (O, O) [2019-05-01 10:30] -> (Q, Q) [2019-05-01 11:00]
   2: O       O...............................................                               d: 01.05 10:30 [01.05 12:30]  [{name=Bus 7, day=2019-05-01, id=T10, src=0}]
   3: P       P............................................... a: 01.05 10:43 [01.05 12:43]  d: 01.05 10:45 [01.05 12:45]  [{name=Bus 7, day=2019-05-01, id=T10, src=0}]
   4: Q       Q............................................... a: 01.05 11:00 [01.05 13:00]


[2019-05-01 09:00, 2019-05-01 10:00]
TRANSFERS: 1
     FROM: (START, START) [2019-05-01 09:00]
       TO: (Q, Q) [2019-05-01 10:00]
leg 0: (START, START) [2019-05-01 09:00] -> (H, H) [2019-05-01 09:00]
  MUMO (id=0, duration=0)
leg 1: (H, H) [2019-05-01 09:00] -> (O, O) [2019-05-01 09:20]
   0: H       H...............................................                               d: 01.05 09:00 [01.05 11:00]  [{name=Bus 9, day=2019-05-01, id=T12, src=0}]
   1: O       O............................................... a: 01.05 09:20 [01.05 11:20]
leg 2: (O, O) [2019-05-01 09:20] -> (O, O) [2019-05-01 09:22]
  FOOTPATH (duration=2)
leg 3: (O, O) [2019-05-01 09:30] -> (Q, Q) [2019-05-01 10:00]
   2: O       O...............................................                               d: 01.05 09:30 [01.05 11:30]  [{name=Bus 7, day=2019-05-01, id=T9, src=0}]
   3: P       P............................................... a: 01.05 09:43 [01.05 11:43]  d: 01.05 09:45 [01.05 11:45]  [{name=Bus 7, day=2019-05-01, id=T9, src=0}]
   4: Q       Q............................................... a: 01.05 10:00 [01.05 12:00]


)"sv;

constexpr auto const expected_intermodal_N_P_via_P_0min =
    R"(
[2019-05-01 09:15, 2019-05-01 09:53]
TRANSFERS: 0
     FROM: (N, N) [2019-05-01 09:15]
       TO: (END, END) [2019-05-01 09:53]
leg 0: (N, N) [2019-05-01 09:15] -> (P, P) [2019-05-01 09:43]
   1: N       N...............................................                               d: 01.05 09:15 [01.05 11:15]  [{name=Bus 7, day=2019-05-01, id=T9, src=0}]
   2: O       O............................................... a: 01.05 09:28 [01.05 11:28]  d: 01.05 09:30 [01.05 11:30]  [{name=Bus 7, day=2019-05-01, id=T9, src=0}]
   3: P       P............................................... a: 01.05 09:43 [01.05 11:43]
leg 1: (P, P) [2019-05-01 09:43] -> (END, END) [2019-05-01 09:53]
  MUMO (id=0, duration=10)


)"sv;

std::string results_to_str(pareto_set<routing::journey> const& results,
                           timetable const& tt,
                           rt_timetable const* rtt = nullptr) {
  std::stringstream ss;
  ss << "\n";
  for (auto const& j : results) {
    j.print(ss, tt, rtt);
    ss << "\n\n";
  }
  return ss.str();
}

location_idx_t loc(timetable const& tt, std::string_view const id) {
  return tt.locations_.location_id_to_idx_.at({id, source_idx_t{0}});
}

unixtime_t time(std::string_view const time) {
  return parse_time(time, "%Y-%m-%d %H:%M %Z");
}

timetable load_timetable(std::string_view s) {
  auto tt = timetable{};

  tt.date_range_ = {date::sys_days{2019_y / May / 1},
                    date::sys_days{2019_y / May / 2}};
  loader::register_special_stations(tt);
  loader::gtfs::load_timetable({}, source_idx_t{0}, loader::mem_dir::read(s),
                               tt);
  loader::finalize(tt);
  return tt;
}

pareto_set<routing::journey> search(timetable const& tt,
                                    rt_timetable const* rtt,
                                    routing::query q,
                                    direction const dir) {
  if (dir == direction::kBackward) {
    std::swap(q.start_, q.destination_);
    std::swap(q.start_match_mode_, q.dest_match_mode_);
    std::reverse(begin(q.via_stops_), end(q.via_stops_));
  }
  return raptor_search(tt, rtt, q, dir);
}

}  // namespace

TEST(routing, via_test_1) {
  auto const tt = load_timetable(test_files_1);

  // A -> D, no via
  auto const results = search(
      tt, nullptr,
      routing::query{.start_time_ = time("2019-05-01 10:00 Europe/Berlin"),
                     .start_ = {{loc(tt, "A"), 0_minutes, 0U}},
                     .destination_ = {{loc(tt, "D"), 0_minutes, 0U}},
                     .via_stops_ = {}},
      direction::kForward);

  EXPECT_EQ(expected_A_D_no_via, results_to_str(results, tt));
}

TEST(routing, via_test_2) {
  auto const tt = load_timetable(test_files_1);

  test::with_rt_trips(
      tt, sys_days{2019_y / May / 1}, {"T0", "T1"},
      [&](rt_timetable const* rtt, std::string_view rt_trips) {
        for (auto const& [dir, start_time] :
             {std::pair{direction::kForward,
                        time("2019-05-01 10:00 Europe/Berlin")},
              std::pair{direction::kBackward,
                        time("2019-05-01 10:30 Europe/Berlin")}}) {
          // A -> D, via B (0 min)
          auto const results = search(
              tt, rtt,
              routing::query{.start_time_ = start_time,
                             .start_ = {{loc(tt, "A"), 0_minutes, 0U}},
                             .destination_ = {{loc(tt, "D"), 0_minutes, 0U}},
                             .via_stops_ = {{loc(tt, "B"), 0_minutes}}},
              dir);

          EXPECT_EQ(expected_A_D_via_B_0min, results_to_str(results, tt))
              << " rt trips: " << rt_trips;
        }
      });
}

TEST(routing, via_test_3) {
  auto tt = load_timetable(test_files_1);

  for (auto const& [dir, start_time] :
       {std::pair{direction::kForward, time("2019-05-01 10:00 Europe/Berlin")},
        std::pair{direction::kBackward,
                  time("2019-05-01 10:30 Europe/Berlin")}}) {
    // A -> D, via B (3 min)
    auto const results =
        search(tt, nullptr,
               routing::query{.start_time_ = start_time,
                              .start_ = {{loc(tt, "A"), 0_minutes, 0U}},
                              .destination_ = {{loc(tt, "D"), 0_minutes, 0U}},
                              .via_stops_ = {{loc(tt, "B"), 3_minutes}}},
               dir);

    EXPECT_EQ(expected_A_D_via_B_3min, results_to_str(results, tt));
  }
}

TEST(routing, via_test_4) {
  auto tt = load_timetable(test_files_1);

  test::with_rt_trips(
      tt, sys_days{2019_y / May / 1}, {"T0", "T2"},
      [&](rt_timetable const* rtt, std::string_view rt_trips) {
        for (auto const& [dir, start_time] :
             {std::pair{direction::kForward,
                        time("2019-05-01 10:00 Europe/Berlin")},
              std::pair{direction::kBackward,
                        time("2019-05-01 10:45 Europe/Berlin")}}) {
          // A -> D, via B (10 min)
          auto const results = search(
              tt, rtt,
              routing::query{.start_time_ = start_time,
                             .start_ = {{loc(tt, "A"), 0_minutes, 0U}},
                             .destination_ = {{loc(tt, "D"), 0_minutes, 0U}},
                             .via_stops_ = {{loc(tt, "B"), 10_minutes}}},
              dir);

          EXPECT_EQ(expected_A_D_via_B_10min, results_to_str(results, tt))
              << " rt trips: " << rt_trips;
        }
      });
}

TEST(routing, via_test_5) {
  auto tt = load_timetable(test_files_1);

  for (auto const& [dir, start_time] :
       {std::pair{direction::kForward, time("2019-05-01 10:00 Europe/Berlin")},
        std::pair{direction::kBackward,
                  time("2019-05-01 10:40 Europe/Berlin")}}) {
    // A -> D, via C (0 min)
    auto const results =
        search(tt, nullptr,
               routing::query{.start_time_ = start_time,
                              .start_ = {{loc(tt, "A"), 0_minutes, 0U}},
                              .destination_ = {{loc(tt, "D"), 0_minutes, 0U}},
                              .via_stops_ = {{loc(tt, "C"), 0_minutes}}},
               dir);

    EXPECT_EQ(expected_A_D_via_C_0min, results_to_str(results, tt));
  }
}

TEST(routing, via_test_6) {
  auto tt = load_timetable(test_files_1);

  for (auto const& [dir, start_time] :
       {std::pair{direction::kForward, time("2019-05-01 10:00 Europe/Berlin")},
        std::pair{direction::kBackward,
                  time("2019-05-01 10:55 Europe/Berlin")}}) {
    // A -> D, via C (10 min)
    auto const results =
        search(tt, nullptr,
               routing::query{.start_time_ = start_time,
                              .start_ = {{loc(tt, "A"), 0_minutes, 0U}},
                              .destination_ = {{loc(tt, "D"), 0_minutes, 0U}},
                              .via_stops_ = {{loc(tt, "C"), 10_minutes}}},
               dir);

    EXPECT_EQ(expected_A_D_via_C_10min, results_to_str(results, tt));
  }
}

TEST(routing, via_test_7) {
  auto tt = load_timetable(test_files_1);

  for (auto const& [dir, start_time] :
       {std::pair{direction::kForward, time("2019-05-01 15:00 Europe/Berlin")},
        std::pair{direction::kBackward,
                  time("2019-05-01 15:30 Europe/Berlin")}}) {
    // A -> J, via I (0 min)
    auto const results =
        search(tt, nullptr,
               routing::query{.start_time_ = start_time,
                              .start_ = {{loc(tt, "A"), 0_minutes, 0U}},
                              .destination_ = {{loc(tt, "J"), 0_minutes, 0U}},
                              .via_stops_ = {{loc(tt, "I"), 0_minutes}}},
               dir);

    EXPECT_EQ(expected_A_J_via_I_1500_0min, results_to_str(results, tt));
  }
}

TEST(routing, via_test_8) {
  auto tt = load_timetable(test_files_1);

  for (auto const& [dir, start_time] :
       {std::pair{direction::kForward, time("2019-05-01 15:00 Europe/Berlin")},
        std::pair{direction::kBackward,
                  time("2019-05-01 15:30 Europe/Berlin")}}) {
    // A -> J, via I1 (0 min)
    auto const results =
        search(tt, nullptr,
               routing::query{.start_time_ = start_time,
                              .start_ = {{loc(tt, "A"), 0_minutes, 0U}},
                              .destination_ = {{loc(tt, "J"), 0_minutes, 0U}},
                              .via_stops_ = {{loc(tt, "I1"), 0_minutes}}},
               dir);

    EXPECT_EQ(expected_A_J_via_I_1500_0min, results_to_str(results, tt));
  }
}

TEST(routing, via_test_9) {
  auto tt = load_timetable(test_files_1);

  for (auto const& [dir, start_time] :
       {std::pair{direction::kForward, time("2019-05-01 15:00 Europe/Berlin")},
        std::pair{direction::kBackward,
                  time("2019-05-01 15:30 Europe/Berlin")}}) {
    // A -> J, via I2 (0 min)
    auto const results =
        search(tt, nullptr,
               routing::query{.start_time_ = start_time,
                              .start_ = {{loc(tt, "A"), 0_minutes, 0U}},
                              .destination_ = {{loc(tt, "J"), 0_minutes, 0U}},
                              .via_stops_ = {{loc(tt, "I2"), 0_minutes}}},
               dir);

    EXPECT_EQ(expected_A_J_via_I_1500_0min, results_to_str(results, tt));
  }
}

TEST(routing, via_test_10) {
  auto tt = load_timetable(test_files_1);

  for (auto const& dir : {direction::kForward, direction::kBackward}) {
    // A -> J, via I (0 min)
    auto const results =
        search(tt, nullptr,
               routing::query{.start_time_ = tt.date_range_,
                              .start_ = {{loc(tt, "A"), 0_minutes, 0U}},
                              .destination_ = {{loc(tt, "J"), 0_minutes, 0U}},
                              .via_stops_ = {{loc(tt, "I"), 0_minutes}}},
               dir);

    EXPECT_EQ(expected_A_J_via_I_1600_0min, results_to_str(results, tt));
  }
}

TEST(routing, via_test_11) {
  auto tt = load_timetable(test_files_1);

  for (auto const& [dir, start_time] :
       {std::pair{direction::kForward, time("2019-05-01 15:00 Europe/Berlin")},
        std::pair{direction::kBackward,
                  time("2019-05-01 16:40 Europe/Berlin")}}) {
    // A -> J, via I (10 min)
    auto const results =
        search(tt, nullptr,
               routing::query{.start_time_ = start_time,
                              .start_ = {{loc(tt, "A"), 0_minutes, 0U}},
                              .destination_ = {{loc(tt, "J"), 0_minutes, 0U}},
                              .via_stops_ = {{loc(tt, "I"), 10_minutes}}},
               dir);

    EXPECT_EQ(expected_A_J_via_I_1500_10min, results_to_str(results, tt));
  }
}

TEST(routing, via_test_12) {
  auto tt = load_timetable(test_files_1);

  for (auto const& [dir, start_time] :
       {std::pair{direction::kForward, time("2019-05-01 15:00 Europe/Berlin")},
        std::pair{direction::kBackward,
                  time("2019-05-01 16:00 Europe/Berlin")}}) {
    // A -> J, via K (0 min)
    auto const results =
        search(tt, nullptr,
               routing::query{.start_time_ = start_time,
                              .start_ = {{loc(tt, "A"), 0_minutes, 0U}},
                              .destination_ = {{loc(tt, "J"), 0_minutes, 0U}},
                              .via_stops_ = {{loc(tt, "K"), 0_minutes}}},
               dir);

    EXPECT_EQ(expected_A_J_via_K_0min, results_to_str(results, tt));
  }
}

TEST(routing, via_test_13) {
  auto tt = load_timetable(test_files_1);

  for (auto const& [dir, start_time] :
       {std::pair{direction::kForward, time("2019-05-01 15:00 Europe/Berlin")},
        std::pair{direction::kBackward,
                  time("2019-05-01 16:00 Europe/Berlin")}}) {
    // A -> J, via K (5 min)
    auto const results =
        search(tt, nullptr,
               routing::query{.start_time_ = start_time,
                              .start_ = {{loc(tt, "A"), 0_minutes, 0U}},
                              .destination_ = {{loc(tt, "J"), 0_minutes, 0U}},
                              .via_stops_ = {{loc(tt, "K"), 5_minutes}}},
               dir);

    EXPECT_EQ(expected_A_J_via_K_0min, results_to_str(results, tt));
  }
}

TEST(routing, via_test_14) {
  auto tt = load_timetable(test_files_1);

  for (auto const& [dir, start_time] :
       {std::pair{direction::kForward, time("2019-05-01 15:00 Europe/Berlin")},
        std::pair{direction::kBackward,
                  time("2019-05-01 16:00 Europe/Berlin")}}) {
    // A -> J, via I (0 min), K (5 min)
    auto const results =
        search(tt, nullptr,
               routing::query{.start_time_ = start_time,
                              .start_ = {{loc(tt, "A"), 0_minutes, 0U}},
                              .destination_ = {{loc(tt, "J"), 0_minutes, 0U}},
                              .via_stops_ = {{loc(tt, "I"), 0_minutes},
                                             {loc(tt, "K"), 5_minutes}}},
               dir);

    EXPECT_EQ(expected_A_J_via_K_0min, results_to_str(results, tt));
  }
}

TEST(routing, via_test_15) {
  auto tt = load_timetable(test_files_1);

  for (auto const& [dir, start_time] :
       {std::pair{direction::kForward, time("2019-05-01 11:00 Europe/Berlin")},
        std::pair{direction::kBackward,
                  time("2019-05-01 13:00 Europe/Berlin")}}) {
    // H -> Q, via N (0 min)
    auto const results =
        search(tt, nullptr,
               routing::query{.start_time_ = start_time,
                              .start_ = {{loc(tt, "H"), 0_minutes, 0U}},
                              .destination_ = {{loc(tt, "Q"), 0_minutes, 0U}},
                              .via_stops_ = {{loc(tt, "N"), 0_minutes}}},
               dir);

    EXPECT_EQ(expected_H_Q_via_N_0min, results_to_str(results, tt));
  }
}

TEST(routing, via_test_16) {
  auto tt = load_timetable(test_files_1);

  for (auto const& [dir, start_time] :
       {std::pair{direction::kForward, time("2019-05-01 10:30 Europe/Berlin")},
        std::pair{direction::kBackward,
                  time("2019-05-01 11:30 Europe/Berlin")}}) {
    // A -> D, via C (5 min)
    auto const results =
        search(tt, nullptr,
               routing::query{.start_time_ = start_time,
                              .start_ = {{loc(tt, "A"), 0_minutes, 0U}},
                              .destination_ = {{loc(tt, "D"), 0_minutes, 0U}},
                              .via_stops_ = {{loc(tt, "C"), 5_minutes}}},
               dir);

    EXPECT_EQ(expected_A_D_via_C_5min, results_to_str(results, tt));
  }
}

TEST(routing, via_test_17) {
  auto tt = load_timetable(test_files_1);

  test::with_rt_trips(
      tt, sys_days{2019_y / May / 1}, {"T11", "T10"},
      [&](rt_timetable const* rtt, std::string_view rt_trips) {
        for (auto const& [dir, start_time] :
             {std::pair{direction::kForward,
                        time("2019-05-01 11:00 Europe/Berlin")},
              std::pair{direction::kBackward,
                        time("2019-05-01 13:00 Europe/Berlin")}}) {
          // H -> Q, via N (0 min), P (0 min)
          auto const results = search(
              tt, rtt,
              routing::query{.start_time_ = start_time,
                             .start_ = {{loc(tt, "H"), 0_minutes, 0U}},
                             .destination_ = {{loc(tt, "Q"), 0_minutes, 0U}},
                             .via_stops_ = {{loc(tt, "N"), 0_minutes},
                                            {loc(tt, "P"), 0_minutes}}},
              dir);

          EXPECT_EQ(expected_H_Q_via_N_0min, results_to_str(results, tt))
              << " rt trips: " << rt_trips;
        }
      });
}

TEST(routing, via_test_18) {
  auto tt = load_timetable(test_files_1);

  test::with_rt_trips(
      tt, sys_days{2019_y / May / 1}, {"T11", "T10"},
      [&](rt_timetable const* rtt, std::string_view rt_trips) {
        // intermodal start: H / O -> Q, via P (0 min)
        auto const results = search(
            tt, rtt,
            routing::query{
                .start_time_ = time("2019-05-01 11:00 Europe/Berlin"),
                .start_match_mode_ = routing::location_match_mode::kIntermodal,
                .start_ = {{loc(tt, "H"), 0_minutes, 0U},
                           {loc(tt, "O"), 40_minutes, 1U}},
                .destination_ = {{loc(tt, "Q"), 0_minutes, 0U}},
                .via_stops_ = {{loc(tt, "P"), 0_minutes}}},
            direction::kForward);

        EXPECT_EQ(expected_intermodal_HO_Q_via_P_0min,
                  results_to_str(results, tt))
            << " rt trips: " << rt_trips;
      });
}

TEST(routing, via_test_19) {
  auto tt = load_timetable(test_files_1);

  test::with_rt_trips(
      tt, sys_days{2019_y / May / 1}, {"T11", "T10"},
      [&](rt_timetable const* rtt, std::string_view rt_trips) {
        // intermodal start: H / O -> Q, via O (0 min), P (0 min)
        auto const results = search(
            tt, rtt,
            routing::query{
                .start_time_ = time("2019-05-01 11:00 Europe/Berlin"),
                .start_match_mode_ = routing::location_match_mode::kIntermodal,
                .start_ = {{loc(tt, "H"), 0_minutes, 0U},
                           {loc(tt, "O"), 40_minutes, 1U}},
                .destination_ = {{loc(tt, "Q"), 0_minutes, 0U}},
                .via_stops_ = {{loc(tt, "O"), 0_minutes},
                               {loc(tt, "P"), 0_minutes}}},
            direction::kForward);

        EXPECT_EQ(expected_intermodal_HO_Q_via_P_0min,
                  results_to_str(results, tt))
            << " rt trips: " << rt_trips;
      });
}

TEST(routing, via_test_20) {
  auto tt = load_timetable(test_files_1);

  test::with_rt_trips(
      tt, sys_days{2019_y / May / 1}, {"T11", "T10"},
      [&](rt_timetable const* rtt, std::string_view rt_trips) {
        for (auto const& [dir, start_time] :
             {std::pair{direction::kForward,
                        time("2019-05-01 11:15 Europe/Berlin")},
              std::pair{direction::kBackward,
                        time("2019-05-01 11:53 Europe/Berlin")}}) {
          // intermodal dest: N -> P, via P (0 min)
          auto const results = search(
              tt, rtt,
              routing::query{
                  .start_time_ = start_time,
                  .dest_match_mode_ = routing::location_match_mode::kIntermodal,
                  .start_ =
                      {
                          {loc(tt, "N"), 0_minutes, 0U},
                      },
                  .destination_ = {{loc(tt, "P"), 10_minutes, 0U}},
                  .via_stops_ = {{loc(tt, "P"), 0_minutes}}},
              dir);

          auto results_str = results_to_str(results, tt);
          if (dir == direction::kBackward) {
            results_str =
                std::regex_replace(results_str, std::regex("START"), "END");
          }

          EXPECT_EQ(expected_intermodal_N_P_via_P_0min, results_str)
              << " rt trips: " << rt_trips;
        }
      });
}

TEST(routing, via_test_21) {
  auto tt = load_timetable(test_files_1);

  for (auto const& [dir, start_time] :
       {std::pair{direction::kForward, time("2019-05-01 11:00 Europe/Berlin")},
        std::pair{direction::kBackward,
                  time("2019-05-01 13:00 Europe/Berlin")}}) {
    // H -> Q, via H (0 min), N (0 min)
    auto const results =
        search(tt, nullptr,
               routing::query{.start_time_ = start_time,
                              .start_ = {{loc(tt, "H"), 0_minutes, 0U}},
                              .destination_ = {{loc(tt, "Q"), 0_minutes, 0U}},
                              .via_stops_ = {{loc(tt, "H"), 0_minutes},
                                             {loc(tt, "N"), 0_minutes}}},
               dir);

    EXPECT_EQ(expected_H_Q_via_N_0min, results_to_str(results, tt));
  }
}

TEST(routing, via_test_22) {
  auto tt = load_timetable(test_files_1);

  for (auto const& [dir, start_time] :
       {std::pair{direction::kForward, time("2019-05-01 11:00 Europe/Berlin")},
        std::pair{direction::kBackward,
                  time("2019-05-01 13:00 Europe/Berlin")}}) {
    // H -> Q, via H (0 min), N (0 min), Q (0 min)
    auto const results =
        search(tt, nullptr,
               routing::query{.start_time_ = start_time,
                              .start_ = {{loc(tt, "H"), 0_minutes, 0U}},
                              .destination_ = {{loc(tt, "Q"), 0_minutes, 0U}},
                              .via_stops_ = {{loc(tt, "H"), 0_minutes},
                                             {loc(tt, "N"), 0_minutes},
                                             {loc(tt, "Q"), 0_minutes}}},
               dir);

    EXPECT_EQ(expected_H_Q_via_N_0min, results_to_str(results, tt));
  }
}
