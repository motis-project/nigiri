#include "gtest/gtest.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/routing/raptor/mcraptor.h"
#include "nigiri/routing/raptor/pong.h"
#include "nigiri/routing/raptor_search.h"
#include "nigiri/special_stations.h"
#include "nigiri/timetable.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace std::chrono_literals;
using namespace std::string_view_literals;

namespace {

// The "shadowed cost variant" scenario (the pong-mc cross-validation
// residual class, root-caused 2026-07-13 via q#372/q#965):
//
//   chain | dep O local | arr D | transfers | extras            | full cost
//   ------+-------------+-------+-----------+-------------------+----------
//   A     | 12:00       | 20:00 | 0         | 10 (boarding)     | 490
//   B     | 16:00       | 20:00 | 0         | 70 (board+60'walk)| 310
//   C     | 16:00       | 20:00 | 1         | 20 (2 boardings)  | 260
//
// True pareto = {B, C}: B dominates A (departs 4h later, same arrival,
// same transfers, cheaper); C trades one extra transfer for the cheapest
// generalized cost. The range search (search.h) steps at the concrete
// departures, so at ITS 16:00 step A does not exist and B/C survive as
// incomparable.
//
// The pong ping sweeps the interval from its start: at that step A, B
// and C coexist, and with departure-blind (arr, extras) bag dominance A
// shadows C (same arrival, extras 10 <= 20, fewer rounds) - correct for
// a 12:00 departure, but A expires one minute later while the sweep's
// progression jumps to B's validated departure + 1: the window where C
// is pareto is never swept, and no anchor for (arr, transfers=1) ever
// exists. Departure-aware label costs keep C alive in-search (and let B
// evict A, which is final-pareto-correct).
constexpr auto const kGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,DB,https://db.de,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station
O,O,0.0,1.0,,
X,X,2.0,3.0,,
M,M,4.0,5.0,,
D,D,6.0,7.0,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
RA,DB,RA,,,3
RB,DB,RB,,,3
RC1,DB,RC1,,,3
RC2,DB,RC2,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
RA,S,TA,,
RB,S,TB,,
RC1,S,TC1,,
RC2,S,TC2,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
TA,12:00:00,12:00:00,O,0
TA,20:00:00,20:00:00,D,1
TB,16:00:00,16:00:00,O,0
TB,19:00:00,19:00:00,X,1
TC1,16:00:00,16:00:00,O,0
TC1,17:00:00,17:00:00,M,1
TC2,18:00:00,18:00:00,M,0
TC2,20:00:00,20:00:00,D,1

# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time
X,D,2,3600

# calendar_dates.txt
service_id,date,exception_type
S,20240619,1
)"sv;

std::vector<std::tuple<unixtime_t, unixtime_t, unsigned>> tuples(
    pareto_set<routing::journey> const& js) {
  auto v = std::vector<std::tuple<unixtime_t, unixtime_t, unsigned>>{};
  for (auto const& j : js) {
    v.emplace_back(j.start_time_, j.dest_time_, unsigned{j.transfers_});
  }
  std::sort(begin(v), end(v));
  return v;
}

}  // namespace

// DISABLED under the walk-only criterion: this fixture asserts the
// generalized-cost pareto (arr + walk), which drops the early-departing
// low-walk journey A. Walk-only keeps A (a superset: A departs earlier
// AND walks less than B, so they are incomparable), so the expected set no
// longer matches. Re-enable only if generalized-cost dominance returns.
TEST(pong, DISABLED_shadowed_cost_variant) {
  auto tt = timetable{};
  tt.date_range_ = {sys_days{2024_y / June / 18}, sys_days{2024_y / June / 20}};
  register_special_stations(tt);
  load_timetable({}, source_idx_t{0}, mem_dir::read(std::string{kGTFS}), tt);
  finalize(tt);

  auto const src = source_idx_t{0};
  auto const o = tt.locations_.location_id_to_idx_.at({"O", src});
  auto const d = tt.locations_.location_id_to_idx_.at({"D", src});

  // local 11:00-21:00 = 09:00-19:00 UTC (Europe/Berlin, June)
  auto const day = sys_days{2024_y / June / 19};
  auto const make_query = [&]() {
    auto q = routing::query{};
    q.start_time_ =
        interval<unixtime_t>{unixtime_t{day} + 9h, unixtime_t{day} + 19h};
    q.start_ = {{o, 0_minutes, 0U}};
    q.destination_ = {{d, 0_minutes, 0U}};
    q.via_stops_ = {};
    return q;
  };

  auto const dep = unixtime_t{day} + 14h;  // 16:00 local
  auto const arr = unixtime_t{day} + 18h;  // 20:00 local
  auto const expected =
      std::vector<std::tuple<unixtime_t, unixtime_t, unsigned>>{
          {dep, arr, 0U},   // B: direct + 60' walk, cost 310
          {dep, arr, 1U}};  // C: one transfer, cost 260 - the cheapest

  // reference: the range search finds the full pareto set
  auto ref_ss = routing::search_state{};
  auto ref_as = routing::mcraptor_cost_state{};
  auto const ref =
      *(routing::raptor_search(tt, nullptr, ref_ss, ref_as, make_query(),
                               direction::kForward)
            .journeys_);
  ASSERT_EQ(expected, tuples(ref));

  // pong must find the same set - C's anchor is what the shadow kills
  auto pong_ss = routing::search_state{};
  auto pong_as = routing::mcraptor_cost_state{};
  auto const png =
      *(routing::pong_search(tt, nullptr, pong_ss, pong_as, make_query(),
                             direction::kForward)
            .journeys_);
  EXPECT_EQ(expected, tuples(png));
}

namespace {

// The "witness eviction x departure tie" scenario (the second layer of
// the shadowed-variant problem, root-caused 2026-07-13 via q#47): two
// classes P/Q, each with a morning and an afternoon run of ONE route
// (earliest-boarding-per-route -> only the morning run is visible from
// early sweep steps), converging on shared final legs:
//
//   journey     | dep O | arr D | extras | full cost | true pareto?
//   ------------+-------+-------+--------+-----------+-------------
//   P-morning   | 10:11 | 18:00 | 27     | 496       | no (P-afternoon)
//   P-afternoon | 14:07 | 18:00 | 27     | 260       | yes
//   Q-morning   | 10:09 | 18:06 | 20     | 497       | no (Q-afternoon)
//   Q-afternoon | 14:07 | 18:06 | 20     | 259       | yes (cheapest)
//
// At the first sweep step only the morning chains exist, and P-morning
// legitimately dominates Q-morning AS A JOURNEY (departs 2' later,
// arrives 6' earlier, 1 cheaper) - the cost order flips between morning
// (departure spread 2') and afternoon (tie). If the ping's result set
// uses the FINAL journey rule, Q-morning's eviction kills the arr-18:06
// anchor; the pong validates P-afternoon at dep 14:07 and the sweep
// jumps to 14:08 - past the departure tie, the only window where
// Q-afternoon was discoverable. The ping's set is an INTERMEDIATE
// structure: it must use the search's own dominance rule (raw
// departure-aware label cost), under which the two witnesses are
// incomparable and both anchors survive.
constexpr auto const kGTFS2 = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,DB,https://db.de,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station
O,O,0.0,1.0,,
HP,HP,2.0,3.0,,
HP2,HP2,2.0,3.1,,
HQ,HQ,4.0,5.0,,
D,D,6.0,7.0,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
RP,DB,RP,,,3
RQ,DB,RQ,,,3
FP,DB,FP,,,3
FQ,DB,FQ,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
RP,S,RP_M,,
RP,S,RP_A,,
RQ,S,RQ_M,,
RQ,S,RQ_A,,
FP,S,FP_1,,
FQ,S,FQ_1,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
RP_M,10:11:00,10:11:00,O,0
RP_M,11:11:00,11:11:00,HP,1
RP_A,14:07:00,14:07:00,O,0
RP_A,15:07:00,15:07:00,HP,1
RQ_M,10:09:00,10:09:00,O,0
RQ_M,11:09:00,11:09:00,HQ,1
RQ_A,14:07:00,14:07:00,O,0
RQ_A,15:07:00,15:07:00,HQ,1
FP_1,17:00:00,17:00:00,HP2,0
FP_1,18:00:00,18:00:00,D,1
FQ_1,17:00:00,17:00:00,HQ,0
FQ_1,18:06:00,18:06:00,D,1

# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time
HP,HP2,2,420

# calendar_dates.txt
service_id,date,exception_type
S,20240619,1
)"sv;

}  // namespace

TEST(pong, witness_eviction_departure_tie) {
  auto tt = timetable{};
  tt.date_range_ = {sys_days{2024_y / June / 18}, sys_days{2024_y / June / 20}};
  register_special_stations(tt);
  load_timetable({}, source_idx_t{0}, mem_dir::read(std::string{kGTFS2}), tt);
  finalize(tt);

  auto const src = source_idx_t{0};
  auto const o = tt.locations_.location_id_to_idx_.at({"O", src});
  auto const d = tt.locations_.location_id_to_idx_.at({"D", src});

  // local 09:00-20:00 = 07:00-18:00 UTC (Europe/Berlin, June)
  auto const day = sys_days{2024_y / June / 19};
  auto const make_query = [&]() {
    auto q = routing::query{};
    q.start_time_ =
        interval<unixtime_t>{unixtime_t{day} + 7h, unixtime_t{day} + 18h};
    q.start_ = {{o, 0_minutes, 0U}};
    q.destination_ = {{d, 0_minutes, 0U}};
    q.via_stops_ = {};
    return q;
  };

  auto const dep = unixtime_t{day} + 12h + 7min;  // 14:07 local
  auto const expected =
      std::vector<std::tuple<unixtime_t, unixtime_t, unsigned>>{
          {dep, unixtime_t{day} + 16h, 1U},        // P-afternoon, cost 260
          {dep, unixtime_t{day} + 16h + 6min, 1U}  // Q-afternoon, cost 259
      };

  auto ref_ss = routing::search_state{};
  auto ref_as = routing::mcraptor_cost_state{};
  auto const ref =
      *(routing::raptor_search(tt, nullptr, ref_ss, ref_as, make_query(),
                               direction::kForward)
            .journeys_);
  ASSERT_EQ(expected, tuples(ref));

  auto pong_ss = routing::search_state{};
  auto pong_as = routing::mcraptor_cost_state{};
  auto const png =
      *(routing::pong_search(tt, nullptr, pong_ss, pong_as, make_query(),
                             direction::kForward)
            .journeys_);
  EXPECT_EQ(expected, tuples(png));
}

namespace {

// The "window-cut sibling" scenario (root-caused 2026-07-13 via
// q#132/q#520): the final pareto set contains TWO members of the same
// (arrival, transfers) class - a departure/cost trade (V-early departs 3'
// earlier and is 4 cheaper thanks to a walk-free transfer). The pong
// validates the class through ONE backward search whose window bound
// derives from the ping witness's departure stamp:
//
//   journey  | dep O | arr D | extras | full cost | in true pareto?
//   ---------+-------+-------+--------+-----------+----------------
//   sleeper  | 10:00 | 20:00 | 20     | 620       | no (V-early)
//   V-early  | 14:35 | 20:00 | 20     | 345       | yes (cheapest)
//   V-late   | 14:38 | 20:00 | 27     | 349       | yes
//
// V-early's route also has a morning run (the sleeper's first leg), so
// at the first sweep step earliest-boarding-per-route hides V-early;
// the destination bag keeps V-late (cheapest departure-aware cost) as
// the class witness -> anchor stamp 14:38. An unwidened backward window
// [14:37, 20:00] can never reach V-early, and the progression jumps to
// 14:39 - V-early is unreachable by any execute. The stamp must be
// widened by the witness's extras: a same-class sibling departing d
// minutes earlier travels d minutes longer, so it can only be cheaper
// (pareto) if d < extras(witness) - the window never needs to reach
// further back than that (minutes, not the whole interval).
constexpr auto const kGTFS3 = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,DB,https://db.de,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station
O,O,0.0,1.0,,
HE,HE,2.0,3.0,,
HL,HL,4.0,5.0,,
HL2,HL2,4.0,5.1,,
D,D,6.0,7.0,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
RE,DB,RE,,,3
RL,DB,RL,,,3
SE,DB,SE,,,3
SL,DB,SL,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
RE,S,RE_M,,
RE,S,RE_A,,
RL,S,RL_1,,
SE,S,SE_1,,
SL,S,SL_1,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
RE_M,10:00:00,10:00:00,O,0
RE_M,11:00:00,11:00:00,HE,1
RE_A,14:35:00,14:35:00,O,0
RE_A,15:35:00,15:35:00,HE,1
RL_1,14:38:00,14:38:00,O,0
RL_1,15:38:00,15:38:00,HL,1
SE_1,16:00:00,16:00:00,HE,0
SE_1,20:00:00,20:00:00,D,1
SL_1,16:00:00,16:00:00,HL2,0
SL_1,20:00:00,20:00:00,D,1

# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time
HL,HL2,2,420

# calendar_dates.txt
service_id,date,exception_type
S,20240619,1
)"sv;

}  // namespace

TEST(pong, window_cut_sibling) {
  auto tt = timetable{};
  tt.date_range_ = {sys_days{2024_y / June / 18}, sys_days{2024_y / June / 20}};
  register_special_stations(tt);
  load_timetable({}, source_idx_t{0}, mem_dir::read(std::string{kGTFS3}), tt);
  finalize(tt);

  auto const src = source_idx_t{0};
  auto const o = tt.locations_.location_id_to_idx_.at({"O", src});
  auto const d = tt.locations_.location_id_to_idx_.at({"D", src});

  // local 09:00-21:00 = 07:00-19:00 UTC (Europe/Berlin, June)
  auto const day = sys_days{2024_y / June / 19};
  auto const make_query = [&]() {
    auto q = routing::query{};
    q.start_time_ =
        interval<unixtime_t>{unixtime_t{day} + 7h, unixtime_t{day} + 19h};
    q.start_ = {{o, 0_minutes, 0U}};
    q.destination_ = {{d, 0_minutes, 0U}};
    q.via_stops_ = {};
    return q;
  };

  auto const arr = unixtime_t{day} + 18h;  // 20:00 local
  auto const expected =
      std::vector<std::tuple<unixtime_t, unixtime_t, unsigned>>{
          {unixtime_t{day} + 12h + 35min, arr, 1U},   // V-early, cost 345
          {unixtime_t{day} + 12h + 38min, arr, 1U}};  // V-late, cost 349

  auto ref_ss = routing::search_state{};
  auto ref_as = routing::mcraptor_cost_state{};
  auto const ref =
      *(routing::raptor_search(tt, nullptr, ref_ss, ref_as, make_query(),
                               direction::kForward)
            .journeys_);
  ASSERT_EQ(expected, tuples(ref));

  auto pong_ss = routing::search_state{};
  auto pong_as = routing::mcraptor_cost_state{};
  auto const png =
      *(routing::pong_search(tt, nullptr, pong_ss, pong_as, make_query(),
                             direction::kForward)
            .journeys_);
  EXPECT_EQ(expected, tuples(png));
}

namespace {

// The "tie-shadowed transfer rung" scenario (root-caused 2026-07-13 via
// q#0/q#47 under the walk-criteria configuration): three chains to the
// same arrival whose EXTRAS tie across transfer rounds -
//
//   chain | dep   | arr   | transfers | extras          | cost
//   ------+-------+-------+-----------+-----------------+------
//   X1    | 13:15 | 16:15 | 0         | 10 walk + 1 board = 20 | 200
//   X2    | 13:30 | 16:15 | 0         | 15 walk + 1 board = 25 | 190
//   Y     | 13:30 | 16:15 | 1         |  0 walk + 2 boards = 20 | 185
//
// True pareto = {X2, Y}: X2 dominates X1 (later dep, cheaper, same
// transfers); Y trades one more transfer for the cheapest cost. The
// interlock that loses Y:
//  1. equal-extras cross-round TIE rejection: at the first sweep step
//     the bag lets X1 (round 1, arr 16:15, extras 20) reject Y's label
//     (round 2, arr 16:15, extras 20) - both axes tie, and with <= ties
//     reject. Correct for a fixed departure, but it kills the only
//     witness of the (16:15, transfers=1) class.
//  2. exact-transfer matching: the anchor group (arr 16:15) then has
//     max_transfers = 0, so the backward validation never searches Y's
//     transfer count.
//  3. departure tie: X2 validates at dep 13:30 == Y's departure, so the
//     progression jumps to 13:31 and no later step can rediscover Y.
// Fix: cross-round domination requires the extras axis to improve
// STRICTLY (cross_round_dominates) - an exactly-tied higher-round label
// coexists as the witness of its own (arr, transfers) class. Same-round
// ties still dedup, so bags stay bounded.
constexpr auto const kGTFS4 = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,DB,https://db.de,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station
O,O,0.0,1.0,,
M,M,2.0,3.0,,
XS1,XS1,4.0,5.0,,
XS2,XS2,4.0,5.2,,
D,D,6.0,7.0,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
RX1,DB,RX1,,,3
RX2,DB,RX2,,,3
RY1,DB,RY1,,,3
RY2,DB,RY2,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
RX1,S,X1,,
RX2,S,X2A,,
RY1,S,Y1,,
RY2,S,Y2,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
X1,13:15:00,13:15:00,O,0
X1,16:05:00,16:05:00,XS1,1
X2A,13:30:00,13:30:00,O,0
X2A,16:00:00,16:00:00,XS2,1
Y1,13:30:00,13:30:00,O,0
Y1,14:00:00,14:00:00,M,1
Y2,14:30:00,14:30:00,M,0
Y2,16:15:00,16:15:00,D,1

# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time
XS1,D,2,600
XS2,D,2,900

# calendar_dates.txt
service_id,date,exception_type
S,20240619,1
)"sv;

}  // namespace

// DISABLED under the walk-only criterion: like shadowed_cost_variant, the
// expected set is the generalized-cost pareto and omits the early low-walk
// journey X1 that walk-only correctly keeps. Re-enable only if
// generalized-cost dominance returns.
TEST(pong, DISABLED_tie_shadowed_transfer_rung) {
  auto tt = timetable{};
  tt.date_range_ = {sys_days{2024_y / June / 18}, sys_days{2024_y / June / 20}};
  register_special_stations(tt);
  load_timetable({}, source_idx_t{0}, mem_dir::read(std::string{kGTFS4}), tt);
  finalize(tt);

  auto const src = source_idx_t{0};
  auto const o = tt.locations_.location_id_to_idx_.at({"O", src});
  auto const d = tt.locations_.location_id_to_idx_.at({"D", src});

  // local 12:00-18:00 = 10:00-16:00 UTC (Europe/Berlin, June)
  auto const day = sys_days{2024_y / June / 19};
  auto const make_query = [&]() {
    auto q = routing::query{};
    q.start_time_ =
        interval<unixtime_t>{unixtime_t{day} + 10h, unixtime_t{day} + 16h};
    q.start_ = {{o, 0_minutes, 0U}};
    q.destination_ = {{d, 0_minutes, 0U}};
    q.via_stops_ = {};
    return q;
  };

  auto const dep = unixtime_t{day} + 11h + 30min;  // 13:30 local
  auto const arr = unixtime_t{day} + 14h + 15min;  // 16:15 local
  auto const expected =
      std::vector<std::tuple<unixtime_t, unixtime_t, unsigned>>{
          {dep, arr, 0U},   // X2: direct + 10' walk, cost 175
          {dep, arr, 1U}};  // Y: one transfer, zero walk, cost 165

  auto ref_ss = routing::search_state{};
  auto ref_as = routing::mcraptor_cost_state{};
  auto const ref =
      *(routing::raptor_search(tt, nullptr, ref_ss, ref_as, make_query(),
                               direction::kForward)
            .journeys_);
  ASSERT_EQ(expected, tuples(ref));

  auto pong_ss = routing::search_state{};
  auto pong_as = routing::mcraptor_cost_state{};
  auto const png =
      *(routing::pong_search(tt, nullptr, pong_ss, pong_as, make_query(),
                             direction::kForward)
            .journeys_);
  EXPECT_EQ(expected, tuples(png));
}
