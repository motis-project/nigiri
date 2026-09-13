#include "gtest/gtest.h"

#include <algorithm>
#include <set>
#include <tuple>
#include <vector>

#include "nigiri/footpath.h"
#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/routing/raptor/bmrap_filters.h"
#include "nigiri/routing/raptor/bmraptor.h"
#include "nigiri/routing/raptor/mcraptor.h"
#include "nigiri/routing/raptor/pong.h"
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/routing/raptor_search.h"
#include "nigiri/rt/run.h"
#include "nigiri/special_stations.h"
#include "nigiri/timetable.h"

#if defined(NIGIRI_CUDA)
#include "nigiri/routing/gpu/mcraptor.h"
#include "nigiri/routing/gpu/raptor.h"
#endif

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace std::chrono_literals;
using namespace std::string_view_literals;

namespace {

// Three ways from O to D, forming a proper (arrival, transfers) pareto set
// so the restricted set has something to restrict, and several departures
// so the profile driver takes more than one step:
//
//   route(s)        | dep O | arr D | transfers
//   ----------------+-------+-------+----------
//   RD (direct)     | 10:00 | 14:00 | 0
//   RA1 + RA2       | 10:10 | 12:30 | 1
//   RB1 + RB2 + RB3 | 10:20 | 12:00 | 2
//   RD (direct)     | 11:00 | 15:00 | 0
//   RA1 + RA2       | 11:10 | 13:30 | 1
//
// All pareto-optimal on (departure, arrival, transfers): later departures
// never dominate earlier arrivals, and each extra transfer buys a strictly
// earlier one. Times below are LOCAL (Europe/Berlin, June => UTC+2).
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
RD,DB,RD,,,3
RA1,DB,RA1,,,3
RA2,DB,RA2,,,3
RB1,DB,RB1,,,3
RB2,DB,RB2,,,3
RB3,DB,RB3,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
RD,S,TD1,,
RD,S,TD2,,
RA1,S,TA1a,,
RA2,S,TA2a,,
RA1,S,TA1b,,
RA2,S,TA2b,,
RB1,S,TB1,,
RB2,S,TB2,,
RB3,S,TB3,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
TD1,10:05:00,10:05:00,O,0,0,0
TD1,14:00:00,14:00:00,D,1,0,0
TD2,11:00:00,11:00:00,O,0,0,0
TD2,15:00:00,15:00:00,D,1,0,0
TA1a,10:10:00,10:10:00,O,0,0,0
TA1a,11:00:00,11:00:00,M,1,0,0
TA2a,11:10:00,11:10:00,M,0,0,0
TA2a,12:30:00,12:30:00,D,1,0,0
TA1b,11:10:00,11:10:00,O,0,0,0
TA1b,12:00:00,12:00:00,M,1,0,0
TA2b,12:10:00,12:10:00,M,0,0,0
TA2b,13:30:00,13:30:00,D,1,0,0
TB1,10:20:00,10:20:00,O,0,0,0
TB1,10:50:00,10:50:00,X,1,0,0
TB2,11:00:00,11:00:00,X,0,0,0
TB2,11:30:00,11:30:00,M,1,0,0
TB3,11:40:00,11:40:00,M,0,0,0
TB3,12:00:00,12:00:00,D,1,0,0

# calendar_dates.txt
service_id,date,exception_type
S,20240618,1
S,20240619,1
S,20240620,1
)"sv;

using tuple_t = std::tuple<unixtime_t, unixtime_t, unsigned>;

// (departure, arrival, transfers), NORMALISED: a backward search reports
// start_time_ as the ARRIVAL, and departure_time()/arrival_time() undo that
// so forward and backward results compare directly.
std::vector<tuple_t> tuples(pareto_set<routing::journey> const& js) {
  auto v = std::vector<tuple_t>{};
  for (auto const& j : js) {
    v.emplace_back(j.departure_time(), j.arrival_time(),
                   static_cast<unsigned>(j.transfers_));
  }
  std::sort(begin(v), end(v));
  return v;
}

struct fixture {
  fixture() {
    tt_.date_range_ = {sys_days{2024_y / June / 18},
                       sys_days{2024_y / June / 20}};
    register_special_stations(tt_);
    load_timetable({}, source_idx_t{0}, mem_dir::read(std::string{kGTFS}),
                   tt_);
    finalize(tt_);
    o_ = tt_.locations_.location_id_to_idx_.at({"O", source_idx_t{0}});
    d_ = tt_.locations_.location_id_to_idx_.at({"D", source_idx_t{0}});
  }

  // Forward: start_ is O, window is a DEPARTURE window. Backward (arriveBy):
  // the search starts at the destination, so start_ is D and the window is
  // an ARRIVAL window - query::flip_dir()'s convention, and the one the
  // routing endpoint builds for arriveBy=true.
  routing::query make_query(direction const dir = direction::kForward) const {
    auto q = routing::query{};
    auto const day = sys_days{2024_y / June / 19};
    if (dir == direction::kForward) {
      // local 10:00-12:00 = 08:00-10:00 UTC, covering all five departures
      q.start_time_ =
          interval<unixtime_t>{unixtime_t{day} + 8h, unixtime_t{day} + 10h};
      q.start_ = {{o_, 0_minutes, 0U}};
      q.destination_ = {{d_, 0_minutes, 0U}};
    } else {
      // local 12:00-15:00 = 10:00-13:00 UTC, covering every arrival those
      // departures reach except the last one (15:00 local)
      q.start_time_ =
          interval<unixtime_t>{unixtime_t{day} + 10h, unixtime_t{day} + 13h};
      q.start_ = {{d_, 0_minutes, 0U}};
      q.destination_ = {{o_, 0_minutes, 0U}};
    }
    q.via_stops_ = {};
    q.min_connection_count_ = 1U;
    return q;
  }

  timetable tt_;
  location_idx_t o_, d_;
};

struct run_result {
  std::vector<tuple_t> js_;
  interval<unixtime_t> scanned_{};
};

// BMRAPP on the CPU scalar engine. The state type is what selects the
// scalar engine (see bmrap_algo_for), exactly as it does for pong_search,
// so the GPU variant below runs the identical assertions.
template <typename Criteria>
run_result bmrapp(fixture const& f, routing::query q, direction const dir) {
  auto ss = routing::search_state{};
  auto as = routing::raptor_state{};
  auto const r = routing::bmrap_profile_search<Criteria>(f.tt_, nullptr, ss, as,
                                                         std::move(q), dir);
  return {tuples(*r.journeys_), r.interval_};
}

// Plain range search over the same state type: with raptor_state the
// two-criteria search, i.e. the anchor set J_A itself; with an mcraptor
// state the unrestricted multicriteria set.
template <typename AlgoState>
run_result reference(fixture const& f, routing::query q, direction const dir) {
  auto ss = routing::search_state{};
  auto as = AlgoState{};
  auto const r =
      routing::raptor_search(f.tt_, nullptr, ss, as, std::move(q), dir);
  return {tuples(*r.journeys_), r.interval_};
}

// "normal PONG operation": the engine that actually serves a window whose
// extension side matches the search direction.
run_result pong_reference(fixture const& f,
                          routing::query q,
                          direction const dir) {
  auto ss = routing::search_state{};
  auto as = routing::raptor_state{};
  auto const r =
      routing::pong_search(f.tt_, nullptr, ss, as, std::move(q), dir);
  return {tuples(*r.journeys_), r.interval_};
}

template <typename Criteria>
std::vector<tuple_t> run_bmrapp(fixture const& f) {
  return bmrapp<Criteria>(f, f.make_query(), direction::kForward).js_;
}

template <typename McState>
std::vector<tuple_t> run_mcraptor(fixture const& f) {
  return reference<McState>(f, f.make_query(), direction::kForward).js_;
}

std::vector<tuple_t> run_bicriteria(fixture const& f) {
  return reference<routing::raptor_state>(f, f.make_query(),
                                          direction::kForward)
      .js_;
}

void expect_subset(std::vector<tuple_t> const& super,
                   std::vector<tuple_t> const& sub, char const* const what) {
  ASSERT_FALSE(sub.empty()) << what << ": nothing to check";
  auto const have = std::set<tuple_t>{begin(super), end(super)};
  for (auto const& j : sub) {
    EXPECT_TRUE(have.contains(j))
        << what << ": journey the reference does not have: dep="
        << std::get<0>(j).time_since_epoch().count()
        << " arr=" << std::get<1>(j).time_since_epoch().count()
        << " transfers=" << std::get<2>(j);
  }
}

// Every element of `sub` inside the range `super` actually scanned must be
// in `super`. Restricting to the covered range is what makes this
// meaningful: the two engines stop scanning at different points, so
// journeys past the shorter scan are legitimately absent. The scan key is
// the DEPARTURE going forward and the ARRIVAL going backward, and the
// covered range runs the way the scan does.
void expect_contains_prefix(std::vector<tuple_t> const& super,
                            std::vector<tuple_t> const& sub,
                            char const* const what,
                            direction const dir = direction::kForward) {
  ASSERT_FALSE(super.empty()) << what << ": nothing to compare against";
  auto const fwd = dir == direction::kForward;
  auto const key = [&](tuple_t const& t) {
    return fwd ? std::get<0>(t) : std::get<1>(t);
  };
  auto cut = key(super.front());
  for (auto const& t : super) {
    if (fwd ? key(t) > cut : key(t) < cut) {
      cut = key(t);
    }
  }
  auto const have = std::set<tuple_t>{begin(super), end(super)};
  auto checked = 0U;
  for (auto const& j : sub) {
    if (fwd ? key(j) < cut : key(j) > cut) {
      ++checked;
      EXPECT_TRUE(have.contains(j))
          << what << ": missing journey dep="
          << std::get<0>(j).time_since_epoch().count()
          << " arr=" << std::get<1>(j).time_since_epoch().count()
          << " transfers=" << std::get<2>(j);
    }
  }
  // guard against a vacuous pass: the two scans must actually overlap
  EXPECT_GT(checked, 0U) << what << ": no journey fell inside the range";
}

}  // namespace

// The restricted set is a SUBSET of the full multicriteria set: BM-RAPTOR
// may drop journeys the restriction rules out, but must never invent one.
TEST(bmrap, subset_of_mcraptor) {
  auto const f = fixture{};
  auto const full = run_mcraptor<routing::mcraptor_state>(f);
  auto const restricted = run_bmrapp<routing::arr_criteria>(f);

  ASSERT_FALSE(restricted.empty());
  expect_subset(full, restricted, "BMRAPP vs McRAPTOR");
}

// Anchors are their own A(J), so no two-criteria journey may be restricted
// away. Regressed twice during development (anchor pareto-domination, and
// the filter/bounds reference-point mismatch).
TEST(bmrap, contains_bicriteria_journeys) {
  auto const f = fixture{};
  auto const bm = run_bmrapp<routing::arr_criteria>(f);
  auto const bi = run_bicriteria(f);
  expect_contains_prefix(bm, bi, "BMRAPP vs bicriteria RAPTOR");
}

// Same for a COMPOSED criteria, which also covers arr_with<>'s machinery:
// dominance, the carried state and apply_to must fold over two dimensions.
TEST(bmrap, composed_subset_of_mcraptor) {
  auto const f = fixture{};
  auto const full =
      run_mcraptor<routing::mcraptor_non_transit_mode_switches_state>(f);
  auto const restricted =
      run_bmrapp<routing::arr_non_transit_mode_switches_criteria>(f);

  ASSERT_FALSE(restricted.empty());
  expect_subset(full, restricted, "BMRAPP(walk+clasz) vs McRAPTOR");
}

// Adding a pareto dimension can only split classes apart, never merge them,
// so a composed criteria must never lose what a SUBSET of its dimensions
// finds. This is what breaks if a dimension's dominance folds the wrong way.
TEST(bmrap, more_dimensions_never_lose_journeys) {
  auto const f = fixture{};
  auto const walk = run_bmrapp<routing::arr_non_transit_criteria>(f);
  auto const walk_clasz =
      run_bmrapp<routing::arr_non_transit_mode_switches_criteria>(f);

  ASSERT_FALSE(walk.empty());
  expect_subset(walk_clasz, walk,
                "adding the clasz dimension dropped a journey");
}

// ---------------------------------------------------------------------------
// arriveBy, i.e. SearchDir == kBackward
// ---------------------------------------------------------------------------
// Everything mirrors: the scan steps from the LATEST arrival downwards, the
// anchors re-anchor to their earliest arrival, and tau_dep^<- becomes a
// forward reach bound. Same invariants, so these are the forward tests with
// the direction and the window flipped.

TEST(bmrap, backward_subset_of_mcraptor) {
  auto const f = fixture{};
  auto const q = f.make_query(direction::kBackward);
  auto const full =
      reference<routing::mcraptor_state>(f, q, direction::kBackward).js_;
  auto const restricted =
      bmrapp<routing::arr_criteria>(f, q, direction::kBackward).js_;

  ASSERT_FALSE(restricted.empty());
  expect_subset(full, restricted, "backward BMRAPP vs McRAPTOR");
}

TEST(bmrap, backward_contains_bicriteria_journeys) {
  auto const f = fixture{};
  auto const q = f.make_query(direction::kBackward);
  auto const bm = bmrapp<routing::arr_criteria>(f, q, direction::kBackward).js_;
  auto const bi =
      reference<routing::raptor_state>(f, q, direction::kBackward).js_;
  expect_contains_prefix(bm, bi, "backward BMRAPP vs bicriteria RAPTOR",
                         direction::kBackward);
}

TEST(bmrap, backward_composed_subset_of_mcraptor) {
  auto const f = fixture{};
  auto const q = f.make_query(direction::kBackward);
  auto const full =
      reference<routing::mcraptor_non_transit_mode_switches_state>(
          f, q, direction::kBackward)
          .js_;
  auto const restricted =
      bmrapp<routing::arr_non_transit_mode_switches_criteria>(
          f, q, direction::kBackward)
          .js_;

  ASSERT_FALSE(restricted.empty());
  expect_subset(full, restricted,
                "backward BMRAPP(walk+clasz) vs McRAPTOR");
}

// The same journeys whichever end the search starts from, compared on the
// box BOTH scans cover (departure in the forward window, arrival in the
// backward one) since outside it either scan is legitimately blind.
TEST(bmrap, backward_matches_forward) {
  auto const f = fixture{};
  auto const qf = f.make_query(direction::kForward);
  auto const qb = f.make_query(direction::kBackward);
  auto const dep_win = std::get<interval<unixtime_t>>(qf.start_time_);
  auto const arr_win = std::get<interval<unixtime_t>>(qb.start_time_);

  auto const box = [&](std::vector<tuple_t> const& v) {
    auto out = std::vector<tuple_t>{};
    for (auto const& j : v) {
      if (dep_win.contains(std::get<0>(j)) &&
          arr_win.contains(std::get<1>(j))) {
        out.emplace_back(j);
      }
    }
    return out;
  };

  auto const fwd =
      box(bmrapp<routing::arr_criteria>(f, qf, direction::kForward).js_);
  auto const bwd =
      box(bmrapp<routing::arr_criteria>(f, qb, direction::kBackward).js_);

  ASSERT_FALSE(fwd.empty()) << "the shared box is empty - test says nothing";
  EXPECT_EQ(fwd, bwd);
}

// ---------------------------------------------------------------------------
// interval extension
// ---------------------------------------------------------------------------
// BMRAPP extends the window itself, stepping past it until
// min_connection_count_ is met. The side it grows on is fixed by the SEARCH
// direction, and only matches the side the query asked for in the two
// PONG-applicable combinations; for the opposed pair the driver hands the
// window to a bicriteria range search up front instead. See "WHERE THE
// WINDOW COMES FROM" in bmrap_profile.cc. Both halves are covered below.

namespace {

struct extend_case {
  direction dir_;
  bool earlier_, later_;
  interval<unixtime_t> win_;
  char const* name_;
};

unixtime_t at(std::chrono::hours const h, std::chrono::minutes const m) {
  return unixtime_t{sys_days{2024_y / June / 19}} + h + m;
}

// the two combinations whose requested side is the side the scan grows on
std::vector<extend_case> aligned_cases() {
  return {
      // one departure (10:05 local) in the window, more available later
      {direction::kForward, false, true, {at(8h, 0min), at(8h, 6min)},
       "departAfter, extend later"},
      // one arrival (15:00 local) in the window, more available earlier
      {direction::kBackward, true, false, {at(12h, 55min), at(13h, 1min)},
       "arriveBy, extend earlier"},
  };
}

// ...and the two where they are opposed
std::vector<extend_case> opposed_cases() {
  return {
      // one departure (11:00 local), the others are EARLIER
      {direction::kForward, true, false, {at(9h, 0min), at(9h, 6min)},
       "departAfter, extend earlier"},
      // one arrival (12:00 local), the others are LATER
      {direction::kBackward, false, true, {at(10h, 0min), at(10h, 6min)},
       "arriveBy, extend later"},
  };
}

routing::query extend_query(fixture const& f, extend_case const& c) {
  auto q = f.make_query(c.dir_);
  q.start_time_ = c.win_;
  q.extend_interval_earlier_ = c.earlier_;
  q.extend_interval_later_ = c.later_;
  q.min_connection_count_ = 3U;
  return q;
}

// A query that forbids growing on one side must not grow on it: journeys
// outside the requested window are not what the caller asked for, and on a
// paging request they are the ones already shown.
void expect_stays_inside(fixture const& f, extend_case const& c) {
  auto const r = bmrapp<routing::arr_criteria>(f, extend_query(f, c), c.dir_);
  if (!c.earlier_) {
    EXPECT_GE(r.scanned_.from_, c.win_.from_)
        << c.name_ << ": extended EARLIER although the query forbids it";
  }
  if (!c.later_) {
    EXPECT_LE(r.scanned_.to_, c.win_.to_)
        << c.name_ << ": extended LATER although the query forbids it";
  }
}

// ...and it must grow on the side it is allowed to, until
// min_connection_count_ is met. Each window below holds exactly one journey,
// so without extension the result would be that single journey.
void expect_grows_on_allowed_side(fixture const& f, extend_case const& c) {
  auto const q = extend_query(f, c);
  auto const r = bmrapp<routing::arr_criteria>(f, q, c.dir_);
  // the reference engine on the same query, so the expectation is the
  // behaviour of the rest of nigiri and not a hard-coded count
  auto const ref = reference<routing::raptor_state>(f, q, c.dir_);
  EXPECT_GE(ref.js_.size(), q.min_connection_count_)
      << c.name_ << ": the reference engine did not extend either - the "
                    "fixture cannot support this case";
  EXPECT_GE(r.js_.size(), q.min_connection_count_)
      << c.name_ << ": stopped at " << r.js_.size() << " journeys";
  if (c.earlier_) {
    EXPECT_LT(r.scanned_.from_, c.win_.from_)
        << c.name_ << ": did not extend earlier";
  }
  if (c.later_) {
    EXPECT_GT(r.scanned_.to_, c.win_.to_)
        << c.name_ << ": did not extend later";
  }
}

}  // namespace

TEST(bmrap, extends_towards_the_search_direction) {
  auto const f = fixture{};
  for (auto const& c : aligned_cases()) {
    expect_stays_inside(f, c);
    expect_grows_on_allowed_side(f, c);
  }
}

// The opposed pair: the window comes from the range search instead, and
// must still be right on both counts, exactly as in the aligned case.
TEST(bmrap, extends_against_the_search_direction) {
  auto const f = fixture{};
  for (auto const& c : opposed_cases()) {
    expect_stays_inside(f, c);
    expect_grows_on_allowed_side(f, c);
  }
}

// The fallback must not LOSE anything. numItineraries is satisfied on the
// BICRITERIA journeys there rather than on the multicriteria ones, so the
// window comes out wider than strictly needed and the result is a strict
// SUPERSET of what normal PONG operation returns for that same window -
// which is the harmless direction, and the property worth pinning down.
TEST(bmrap, opposed_extension_is_a_superset_of_pong) {
  auto const f = fixture{};
  for (auto const& c : opposed_cases()) {
    auto const r = bmrapp<routing::arr_criteria>(f, extend_query(f, c), c.dir_);
    ASSERT_FALSE(r.js_.empty()) << c.name_ << ": no result to compare";

    // the same window, requested the normal way round: interval pinned to
    // what the fallback settled on, extension side aligned with the search
    // direction so PONG applies, and no numItineraries growth on top
    auto q = f.make_query(c.dir_);
    q.start_time_ = r.scanned_;
    q.extend_interval_earlier_ = c.dir_ == direction::kBackward;
    q.extend_interval_later_ = c.dir_ == direction::kForward;
    q.min_connection_count_ = 0U;
    auto const ref = pong_reference(f, q, c.dir_);

    expect_subset(r.js_, ref.js_, c.name_);
  }
}

// ---------------------------------------------------------------------------
// non_transit_dim::with_walk's is_egress parameter (kNonTransitCountsInter-
// changeWalks toggle)
// ---------------------------------------------------------------------------

// Pins the CURRENT default (kNonTransitCountsInterchangeWalks = true): every
// footpath relaxation counts towards non_transit_, egress or not. If that
// constant is ever flipped, the is_egress=false expectation below flips too.
TEST(bmrap, non_transit_dim_with_walk_is_egress) {
  static_assert(routing::kNonTransitCountsInterchangeWalks,
                "flip this test's is_egress=false expectation along with it");
  auto const before = routing::non_transit_dim{10U};
  EXPECT_EQ(15U, before.with_walk(0, 5U, /*is_egress=*/true).non_transit_);
  EXPECT_EQ(15U, before.with_walk(0, 5U, /*is_egress=*/false).non_transit_)
      << "kNonTransitCountsInterchangeWalks=true: a mid-journey interchange "
         "walk must count exactly like an egress one";
}

// mode_filter_dim / mode_switches_dim never look at non-transit time at all,
// with_walk is a no-op regardless of is_egress - pins that the parameter did
// not leak into dimensions it has no business affecting.
TEST(bmrap, non_egress_dims_ignore_is_egress) {
  auto const mf = routing::mode_filter_dim{true};
  EXPECT_EQ(mf, mf.with_walk(0, 100U, true));
  EXPECT_EQ(mf, mf.with_walk(0, 100U, false));

  auto const ms = routing::mode_switches_dim{clasz::kAir, 3U};
  EXPECT_EQ(ms, ms.with_walk(0, 100U, true));
  EXPECT_EQ(ms, ms.with_walk(0, 100U, false));
}

// arr_with<Dims...>::with_walk must forward is_egress to every dimension
// unchanged, not just default it away - the one thing composing dimensions
// could get wrong.
TEST(bmrap, arr_with_with_walk_forwards_is_egress) {
  using routing::non_transit_dim;
  auto const c = routing::arr_non_transit_criteria{
      delta_t{0}, {non_transit_dim{0U}}};
  EXPECT_EQ(7U, c.with_walk(0, 7U, true).get<non_transit_dim>().non_transit_);
  EXPECT_EQ(7U, c.with_walk(0, 7U, false).get<non_transit_dim>().non_transit_)
      << "is_egress must reach the wrapped dimension, not get lost in "
         "arr_with's own forwarding";
}

// ---------------------------------------------------------------------------
// bmrap_filters.h: the two additional non-transit trade-off filters
// ---------------------------------------------------------------------------
// Both operate on already-reconstructed journeys, so these are direct unit
// tests against hand-built journeys, no timetable needed.

namespace {

routing::journey tradeoff_journey(unixtime_t const dest,
                                  std::uint8_t const transfers,
                                  unixtime_t const start,
                                  std::uint16_t const non_transit) {
  auto j = routing::journey{};
  j.dest_time_ = dest;
  j.transfers_ = transfers;
  j.start_time_ = start;
  j.criteria_cost_ = non_transit;
  return j;
}

routing::journey::leg make_walk_leg(unixtime_t const dep,
                                    unixtime_t const arr) {
  auto l = routing::journey::leg{};
  l.dep_time_ = dep;
  l.arr_time_ = arr;
  l.uses_ = footpath{};
  return l;
}

routing::journey::leg make_transit_leg(rt::run const& r) {
  auto l = routing::journey::leg{};
  l.uses_ = routing::journey::run_enter_exit{r, stop_idx_t{0U}, stop_idx_t{1U}};
  return l;
}

rt::run run_with_range(stop_idx_t const from) {
  auto r = rt::run{};
  r.stop_range_ = {from, static_cast<stop_idx_t>(from + 1U)};
  return r;
}

// A journey with a single transit leg (riding `run`) and, unless `walk` is
// zero, one walk leg of exactly `walk` duration on the bookend side `B`.
// `fixed` is the invariant across bookend variants: the physical arrival
// for kEntry, the physical departure for kExit.
template <routing::bookend B>
routing::journey bookend_journey(std::uint8_t const transfers,
                                 unixtime_t const fixed,
                                 duration_t const walk,
                                 rt::run const& run) {
  auto j = routing::journey{};
  j.transfers_ = transfers;
  if constexpr (B == routing::bookend::kEntry) {
    j.dest_time_ = fixed;  // arrival_time() == fixed
    j.start_time_ = fixed - walk - 30min;
    if (walk.count() > 0) {
      j.legs_.push_back(make_walk_leg(j.start_time_, j.start_time_ + walk));
    }
    j.legs_.push_back(make_transit_leg(run));
  } else {
    j.start_time_ = fixed;  // departure_time() == fixed
    j.dest_time_ = fixed + walk + 30min;
    j.legs_.push_back(make_transit_leg(run));
    if (walk.count() > 0) {
      j.legs_.push_back(make_walk_leg(j.dest_time_ - walk, j.dest_time_));
    }
  }
  return j;
}

}  // namespace

// The anchor is the group member with the shortest travel_time(); a
// shorter-walk candidate is rejected once its extra travel time exceeds
// kNonTransitTradeoffMinutesPerMinute per minute saved. A different
// (arrival, transfers) tuple is its own, untouched group.
TEST(bmrap, filter_non_transit_tradeoff_keeps_within_budget) {
  auto const d = unixtime_t{sys_days{2024_y / June / 19}} + 12h;
  auto const t = std::uint8_t{2U};
  auto results = std::vector<routing::journey>{
      tradeoff_journey(d, t, d - 60min, 20U),   // anchor: 20 min walk, 60 min trip
      tradeoff_journey(d, t, d - 80min, 15U),   // saves 5 min walk, +20 min trip: 4:1, within budget
      tradeoff_journey(d, t, d - 120min, 10U),  // saves 10 min walk, +60 min trip: 6:1, over budget
      tradeoff_journey(d + 5min, t, d - 60min, 1U),  // different arrival: separate group
  };
  auto rejected = std::vector<bool>(results.size(), false);
  filter_non_transit_tradeoff(results, rejected);

  EXPECT_FALSE(rejected[0]) << "the anchor itself must always survive";
  EXPECT_FALSE(rejected[1]) << "4 extra minutes per minute saved is within "
                               "the 5:1 budget";
  EXPECT_TRUE(rejected[2]) << "6 extra minutes per minute saved exceeds the "
                              "5:1 budget";
  EXPECT_FALSE(rejected[3]) << "a different (arrival, transfers) group "
                               "must not be touched";
}

// transfers_ must be part of the join key, not just arrival: merging the two
// transfers_ groups would move the anchor and change every other verdict.
TEST(bmrap, filter_non_transit_tradeoff_respects_transfers) {
  auto const d = unixtime_t{sys_days{2024_y / June / 19}} + 12h;
  auto results = std::vector<routing::journey>{
      tradeoff_journey(d, 1U, d - 200min, 30U),  // T=1 anchor: 30 min walk, 200 min trip
      tradeoff_journey(d, 1U, d - 202min, 29U),  // saves 1 min, +2 min: 2:1, within budget
      tradeoff_journey(d, 2U, d - 60min, 20U),   // T=2 anchor: 20 min walk, 60 min trip
      tradeoff_journey(d, 2U, d - 120min, 10U),  // saves 10 min, +60 min: 6:1, over budget
  };
  auto rejected = std::vector<bool>(results.size(), false);
  filter_non_transit_tradeoff(results, rejected);

  EXPECT_FALSE(rejected[0]);
  EXPECT_FALSE(rejected[1]);
  EXPECT_FALSE(rejected[2]);
  EXPECT_TRUE(rejected[3]) << "must be judged against its own (T=2) anchor, "
                              "not the T=1 group's cheaper one";
}

// Same scenario as filter_non_transit_tradeoff_keeps_within_budget, but with
// start_time_/dest_time_ swapped to the mid-scan convention bmrap_profile.cc
// actually calls this filter under (dest_time_ = departure, start_time_ =
// arrival). Grouping on the raw dest_time_ field would group by departure
// instead and never form a group at all; arrival_time() must still find the
// same group and the same verdicts.
TEST(bmrap, filter_non_transit_tradeoff_mid_scan_convention) {
  auto const d = unixtime_t{sys_days{2024_y / June / 19}} + 12h;
  auto const t = std::uint8_t{2U};
  auto results = std::vector<routing::journey>{
      tradeoff_journey(d - 60min, t, d, 20U),   // anchor: 20 min walk, 60 min trip
      tradeoff_journey(d - 80min, t, d, 15U),   // saves 5 min walk, +20 min trip: 4:1, within budget
      tradeoff_journey(d - 120min, t, d, 10U),  // saves 10 min walk, +60 min trip: 6:1, over budget
      tradeoff_journey(d - 60min, t, d + 5min, 1U),  // different arrival: separate group
  };
  auto rejected = std::vector<bool>(results.size(), false);
  filter_non_transit_tradeoff(results, rejected);

  EXPECT_FALSE(rejected[0]) << "the anchor itself must always survive";
  EXPECT_FALSE(rejected[1]) << "4 extra minutes per minute saved is within "
                               "the 5:1 budget";
  EXPECT_TRUE(rejected[2]) << "6 extra minutes per minute saved exceeds the "
                              "5:1 budget";
  EXPECT_FALSE(rejected[3]) << "a different arrival_time() group must not be "
                               "touched";
}

// Keeps only the longest (anchor) and shortest bookend walk in a group of
// three sharing the same run, transfers_ and anchor field; the middle one is
// rejected. Run for both bookend sides and both forward/backward-style
// journeys (start_time_/dest_time_ swapped) to pin bookend_anchor_field's
// direction-agnostic arrival_time()/departure_time() use.
template <routing::bookend B>
void expect_keeps_longest_and_shortest(bool const backward_style) {
  auto const fixed = unixtime_t{sys_days{2024_y / June / 19}} + 12h;
  auto const run = run_with_range(stop_idx_t{0U});
  auto results = std::vector<routing::journey>{
      bookend_journey<B>(2U, fixed, 20min, run),  // longest: the anchor
      bookend_journey<B>(2U, fixed, 10min, run),  // middle: must be rejected
      bookend_journey<B>(2U, fixed, 2min, run),   // shortest
  };
  if (backward_style) {
    for (auto& j : results) {
      std::swap(j.start_time_, j.dest_time_);
    }
  }
  auto rejected = std::vector<bool>(results.size(), false);
  filter_bookend_variants<B>(results, rejected);

  EXPECT_FALSE(rejected[0]) << "longest bookend walk (the anchor) must survive";
  EXPECT_TRUE(rejected[1]) << "strictly-between walk must be rejected";
  EXPECT_FALSE(rejected[2]) << "shortest bookend walk must survive";
}

TEST(bmrap, filter_bookend_variants_entry) {
  expect_keeps_longest_and_shortest<routing::bookend::kEntry>(false);
  expect_keeps_longest_and_shortest<routing::bookend::kEntry>(true);
}

TEST(bmrap, filter_bookend_variants_exit) {
  expect_keeps_longest_and_shortest<routing::bookend::kExit>(false);
  expect_keeps_longest_and_shortest<routing::bookend::kExit>(true);
}

// A different run (a different first/last transit leg entirely) must never
// be merged into the same group, however close its bookend walk or anchor
// field.
TEST(bmrap, filter_bookend_variants_separates_different_runs) {
  auto const fixed = unixtime_t{sys_days{2024_y / June / 19}} + 12h;
  auto results = std::vector<routing::journey>{
      bookend_journey<routing::bookend::kEntry>(2U, fixed, 20min,
                                                run_with_range(stop_idx_t{0U})),
      bookend_journey<routing::bookend::kEntry>(2U, fixed, 2min,
                                                run_with_range(stop_idx_t{5U})),
  };
  auto rejected = std::vector<bool>(results.size(), false);
  filter_bookend_variants<routing::bookend::kEntry>(results, rejected);
  EXPECT_FALSE(rejected[0]);
  EXPECT_FALSE(rejected[1]) << "different runs are different groups - "
                               "neither is 'the middle one'";
}

// Sharing a run is not enough to group two journeys - the anchor field
// (arrival, for kEntry variants) must also match, or unrelated journeys that
// merely happen to board/alight the same run get collapsed.
TEST(bmrap, filter_bookend_variants_requires_matching_anchor_field) {
  auto const run = run_with_range(stop_idx_t{0U});
  auto const arr_a = unixtime_t{sys_days{2024_y / June / 19}} + 12h;
  auto const arr_b = arr_a + 3h;  // same run, different continuation
  auto results = std::vector<routing::journey>{
      bookend_journey<routing::bookend::kEntry>(2U, arr_a, 20min, run),
      bookend_journey<routing::bookend::kEntry>(2U, arr_a, 2min, run),
      bookend_journey<routing::bookend::kEntry>(2U, arr_b, 10min, run),
  };
  auto rejected = std::vector<bool>(results.size(), false);
  filter_bookend_variants<routing::bookend::kEntry>(results, rejected);
  EXPECT_FALSE(rejected[0]);
  EXPECT_FALSE(rejected[1]);
  EXPECT_FALSE(rejected[2]) << "different downstream arrival: its own group "
                               "of one, not the middle of the other group";
}

// preview_non_transit_filters<Criteria> must be a strict no-op for a
// Criteria without a non_transit dimension, however the toggles are set.
TEST(bmrap, preview_non_transit_filters_gated_on_criteria) {
  auto const d = unixtime_t{sys_days{2024_y / June / 19}} + 12h;
  auto const results = std::vector<routing::journey>{
      tradeoff_journey(d, 2U, d - 120min, 10U),
      tradeoff_journey(d, 2U, d - 60min, 20U),
  };
  auto const rejected =
      routing::preview_non_transit_filters<routing::arr_criteria>(results);
  EXPECT_TRUE(std::none_of(rejected.begin(), rejected.end(),
                          [](bool const b) { return b; }))
      << "arr_criteria has no non_transit dimension - kHasNonTransitDim "
         "must gate the filters off regardless of their own toggles";
}

// The same group, run through a Criteria that DOES have the dimension,
// applies the trade-off filter.
TEST(bmrap, preview_non_transit_filters_applies_when_supported) {
  static_assert(routing::kFilterNonTransitTradeoff);
  auto const d = unixtime_t{sys_days{2024_y / June / 19}} + 12h;
  auto const results = std::vector<routing::journey>{
      tradeoff_journey(d, 2U, d - 60min, 20U),   // anchor
      tradeoff_journey(d, 2U, d - 120min, 10U),  // 6:1, over budget
  };
  auto const rejected =
      routing::preview_non_transit_filters<
          routing::arr_non_transit_criteria>(results);
  EXPECT_FALSE(rejected[0]);
  EXPECT_TRUE(rejected[1]);
}

// ---------------------------------------------------------------------------
// bounded_needs_lb<Algo>(): the lower-bound-dijkstra skip trait
// ---------------------------------------------------------------------------

namespace {

// Minimal stand-ins for the three shapes bounded_needs_lb<Algo>() branches
// on: never uses lb (gpu_raptor), uses lb but declares it unneeded once
// bounded (raptor/basic_mcraptor), and uses lb with no declared opinion
// (gpu_mcraptor, the conservative default).
struct mock_no_lb_algo {
  static constexpr bool kUseLowerBounds = false;
};
struct mock_bounded_skips_lb_algo {
  static constexpr bool kUseLowerBounds = true;
  static constexpr bool kNeedsLbWhenBounded = false;
};
struct mock_bounded_still_needs_lb_algo {
  static constexpr bool kUseLowerBounds = true;
  static constexpr bool kNeedsLbWhenBounded = true;
};
struct mock_undeclared_algo {
  static constexpr bool kUseLowerBounds = true;
};

}  // namespace

TEST(bmrap, bounded_needs_lb_trait) {
  EXPECT_FALSE(routing::bounded_needs_lb<mock_no_lb_algo>());
  EXPECT_FALSE(routing::bounded_needs_lb<mock_bounded_skips_lb_algo>());
  EXPECT_TRUE(routing::bounded_needs_lb<mock_bounded_still_needs_lb_algo>());
  EXPECT_TRUE(routing::bounded_needs_lb<mock_undeclared_algo>())
      << "an engine that declares no opinion must default to needing lb";
}

// Pins the actual values on the two engines bmrap_profile.cc bounds
// unconditionally: both must resolve to "does not need lb once bounded",
// which is what makes skipping the bwd_lb dijkstra there safe.
TEST(bmrap, bounded_needs_lb_matches_real_engines) {
  using cpu_raptor =
      routing::raptor<direction::kForward, false, via_offset_t{0U},
                      routing::search_mode::kOneToOne>;
  using cpu_mcraptor =
      routing::basic_mcraptor<direction::kForward, routing::arr_criteria>;
  EXPECT_FALSE(routing::bounded_needs_lb<cpu_raptor>());
  EXPECT_FALSE(routing::bounded_needs_lb<cpu_mcraptor>());
}

#if defined(NIGIRI_CUDA)
// The GPU scalar engine must produce the same restricted set as the CPU one.
// Only the ping / pong / backward-pruning searches move to the device; the
// multicriteria phases run on the CPU either way (see bmrap_algo_for), so
// every criteria configuration is comparable, not just the two the GPU
// mcraptor implements.
template <typename Criteria>
std::vector<tuple_t> run_bmrapp_gpu(fixture const& f,
                                    routing::query q,
                                    direction const dir,
                                    int const gpu_mc_mode = 1) {
  auto ss = routing::search_state{};
  auto gtt = routing::gpu::gpu_timetable{f.tt_};
  auto as = routing::gpu::gpu_raptor_state{gtt};
  return tuples(*(routing::bmrap_profile_search<Criteria>(
                      f.tt_, nullptr, ss, as, std::move(q), dir, std::nullopt,
                      gpu_mc_mode)
                      .journeys_));
}

// The device mcraptor on its own, outside BMRAPP: phases 4/5 can only move
// to the GPU (NIGIRI_BMRAPP_GPU_MC) if this holds, and nothing else in the
// tree exercises gpu_mcraptor, so this is where a divergence in it shows up
// rather than as a BMRAPP failure.
std::vector<tuple_t> run_gpu_mcraptor(fixture const& f,
                                      routing::query q,
                                      direction const dir) {
  auto ss = routing::search_state{};
  auto gtt = routing::gpu::gpu_timetable{f.tt_};
  auto as = routing::gpu::gpu_mcraptor_state{gtt};
  return tuples(
      *(routing::raptor_search(f.tt_, nullptr, ss, as, std::move(q), dir)
            .journeys_));
}

TEST(bmrap, gpu_mcraptor_matches_cpu) {
  if (!routing::gpu::gpu_available()) {
    GTEST_SKIP() << "no CUDA device";
  }
  auto const f = fixture{};
  auto const q = f.make_query();
  EXPECT_EQ(reference<routing::mcraptor_state>(f, q, direction::kForward).js_,
            run_gpu_mcraptor(f, q, direction::kForward));
}

TEST(bmrap, gpu_mcraptor_matches_cpu_backward) {
  if (!routing::gpu::gpu_available()) {
    GTEST_SKIP() << "no CUDA device";
  }
  auto const f = fixture{};
  auto const q = f.make_query(direction::kBackward);
  EXPECT_EQ(reference<routing::mcraptor_state>(f, q, direction::kBackward).js_,
            run_gpu_mcraptor(f, q, direction::kBackward));
}

TEST(bmrap, gpu_matches_cpu) {
  if (!routing::gpu::gpu_available()) {
    GTEST_SKIP() << "no CUDA device";
  }
  auto const f = fixture{};
  auto const q = f.make_query();
  EXPECT_EQ(bmrapp<routing::arr_criteria>(f, q, direction::kForward).js_,
            run_bmrapp_gpu<routing::arr_criteria>(f, q, direction::kForward));
}

TEST(bmrap, gpu_matches_cpu_walk) {
  if (!routing::gpu::gpu_available()) {
    GTEST_SKIP() << "no CUDA device";
  }
  auto const f = fixture{};
  auto const q = f.make_query();
  EXPECT_EQ(
      bmrapp<routing::arr_non_transit_criteria>(f, q, direction::kForward).js_,
      run_bmrapp_gpu<routing::arr_non_transit_criteria>(f, q,
                                                        direction::kForward));
}

TEST(bmrap, gpu_matches_cpu_walk_backward) {
  if (!routing::gpu::gpu_available()) {
    GTEST_SKIP() << "no CUDA device";
  }
  auto const f = fixture{};
  auto const q = f.make_query(direction::kBackward);
  EXPECT_EQ(bmrapp<routing::arr_non_transit_criteria>(f, q,
                                                      direction::kBackward)
                .js_,
            run_bmrapp_gpu<routing::arr_non_transit_criteria>(
                f, q, direction::kBackward));
}

// mode_filter (avoid AIR) on the device. The test timetable has no flights,
// so the criterion is all-zeros here - this pins that the extra label field
// does not perturb the arrival-only result; real-data validation covers the
// bit itself.
TEST(bmrap, gpu_matches_cpu_mode_filter) {
  if (!routing::gpu::gpu_available()) {
    GTEST_SKIP() << "no CUDA device";
  }
  auto const f = fixture{};
  for (auto const dir : {direction::kForward, direction::kBackward}) {
    auto const q = f.make_query(dir);
    EXPECT_EQ(bmrapp<routing::arr_mode_filter_criteria>(f, q, dir).js_,
              run_bmrapp_gpu<routing::arr_mode_filter_criteria>(f, q, dir))
        << (dir == direction::kForward ? "fwd" : "bwd");
  }
}

// gpu_mc_mode 2 also runs the mc PONG (phase 5) on the device; the journeys
// must not change.
TEST(bmrap, gpu_mc_pong_matches_mc_ping) {
  if (!routing::gpu::gpu_available()) {
    GTEST_SKIP() << "no CUDA device";
  }
  auto const f = fixture{};
  auto const q = f.make_query();
  EXPECT_EQ(run_bmrapp_gpu<routing::arr_criteria>(f, q, direction::kForward, 1),
            run_bmrapp_gpu<routing::arr_criteria>(f, q, direction::kForward, 2));
  EXPECT_EQ(run_bmrapp_gpu<routing::arr_non_transit_criteria>(
                f, q, direction::kForward, 1),
            run_bmrapp_gpu<routing::arr_non_transit_criteria>(
                f, q, direction::kForward, 2));
}

// arriveBy on the device: the per-query buffers are direction-indexed, so
// the backward scan exercises a different half of the GPU state.
TEST(bmrap, gpu_matches_cpu_backward) {
  if (!routing::gpu::gpu_available()) {
    GTEST_SKIP() << "no CUDA device";
  }
  auto const f = fixture{};
  auto const q = f.make_query(direction::kBackward);
  EXPECT_EQ(bmrapp<routing::arr_criteria>(f, q, direction::kBackward).js_,
            run_bmrapp_gpu<routing::arr_criteria>(f, q, direction::kBackward));
}
#endif
