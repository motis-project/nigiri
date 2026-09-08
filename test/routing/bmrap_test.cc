#include "gtest/gtest.h"

#include <algorithm>
#include <set>
#include <tuple>
#include <vector>

#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/routing/raptor/bmraptor.h"
#include "nigiri/routing/raptor/mcraptor.h"
#include "nigiri/routing/raptor/pong.h"
#include "nigiri/routing/raptor_search.h"
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
// Every one of these is pareto-optimal on (departure, arrival, transfers):
// later departures never dominate earlier arrivals, and each extra
// transfer buys a strictly earlier arrival. Times below are LOCAL
// (Europe/Berlin, June => UTC+2).
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

// (departure, arrival, transfers), NORMALISED. A backward search reports its
// journeys in the backward convention - start_time_ is the ARRIVAL and
// dest_time_ the departure - and departure_time()/arrival_time() undo that,
// so forward and backward result sets are directly comparable.
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

  // Forward: start_ is O and the window is a DEPARTURE window.
  // Backward (arriveBy): the search starts at the destination, so start_ is
  // D and the window is an ARRIVAL window - the same convention
  // query::flip_dir() uses and the one the routing endpoint builds for
  // arriveBy=true.
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

// Reference engine: plain range search over the same state type. With
// raptor_state that is the two-criteria (arrival, transfers) search - by
// definition the anchor set J_A that BM-RAPTOR restricts around - and with
// an mcraptor state it is the unrestricted multicriteria set.
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
// away: everything the plain two-criteria search finds inside the scanned
// range must come back.
// This is the invariant that regressed twice during development (anchor
// pareto-domination, and the filter/bounds reference-point mismatch).
TEST(bmrap, contains_bicriteria_journeys) {
  auto const f = fixture{};
  auto const bm = run_bmrapp<routing::arr_criteria>(f);
  auto const bi = run_bicriteria(f);
  expect_contains_prefix(bm, bi, "BMRAPP vs bicriteria RAPTOR");
}

// Same, with a COMPOSED criteria (walking + vehicle-class switches). The
// dimensions are combined by arr_with<>, so this also covers the
// composition machinery: dominance, the carried state and apply_to all
// have to fold correctly over more than one dimension.
TEST(bmrap, composed_subset_of_mcraptor) {
  auto const f = fixture{};
  auto const full = run_mcraptor<routing::mcraptor_walk_clasz_state>(f);
  auto const restricted = run_bmrapp<routing::arr_walk_clasz_criteria>(f);

  ASSERT_FALSE(restricted.empty());
  expect_subset(full, restricted, "BMRAPP(walk+clasz) vs McRAPTOR");
}

// A composed criteria must never lose a journey the same search finds with
// a SUBSET of its dimensions: adding a pareto dimension can only split
// classes apart, never merge them. This is the property that would break
// if a dimension's dominance folded the wrong way.
TEST(bmrap, more_dimensions_never_lose_journeys) {
  auto const f = fixture{};
  auto const walk = run_bmrapp<routing::arr_walk_criteria>(f);
  auto const walk_clasz = run_bmrapp<routing::arr_walk_clasz_criteria>(f);

  ASSERT_FALSE(walk.empty());
  expect_subset(walk_clasz, walk,
                "adding the clasz dimension dropped a journey");
}

// ---------------------------------------------------------------------------
// arriveBy, i.e. SearchDir == kBackward
// ---------------------------------------------------------------------------
// Backward, everything mirrors: the scan steps from the LATEST arrival
// towards earlier ones, the anchors are re-anchored to their earliest
// arrival, and tau_dep^<- becomes a forward reach bound. The invariants are
// the same ones, so the tests are the forward ones with the direction and
// the window flipped.

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
      reference<routing::mcraptor_walk_clasz_state>(f, q, direction::kBackward)
          .js_;
  auto const restricted =
      bmrapp<routing::arr_walk_clasz_criteria>(f, q, direction::kBackward).js_;

  ASSERT_FALSE(restricted.empty());
  expect_subset(full, restricted,
                "backward BMRAPP(walk+clasz) vs McRAPTOR");
}

// The same journeys must come back whichever end the search starts from.
// Compared on the box BOTH scans cover - departure inside the forward
// window, arrival inside the backward one - because outside it either scan
// is legitimately blind.
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
// BMRAPP extends the search window itself: the scan keeps stepping past the
// nominal window until min_connection_count_ is met, the way PONG does. The
// side it grows on is fixed by the SEARCH direction - a forward scan
// enumerates departures upwards, a backward one arrivals downwards - and
// that coincides with the side the query asks for only in the two
// PONG-applicable combinations (same table as in motis' routing endpoint):
//
//   arriveBy | extend_later | BMRAPP grows | requested
//   ---------+--------------+--------------+-----------
//   false    | true         | later        | later
//   true     | false        | earlier      | earlier
//   false    | false        | later        | earlier   <- mismatch
//   true     | true         | earlier      | later     <- mismatch
//
// The mismatched half is reachable through paging - cursor_to_query() takes
// the extension side from the cursor, independently of arriveBy - and is
// covered by the DISABLED_ test below.

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

// The scan cannot grow the window on the side these two ask for, so the
// driver hands the job to a plain bicriteria RANGE search up front and steps
// over the window that one settles on (see "WHERE THE WINDOW COMES FROM" in
// bmrap_profile.cc). The window must then be right on both counts, exactly
// as in the aligned case.
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

#if defined(NIGIRI_CUDA)
// The GPU scalar engine must produce the same restricted set as the CPU one.
// Only the ping / pong / backward-pruning searches move to the device; the
// multicriteria phases run on the CPU either way (see bmrap_algo_for), so
// every criteria configuration is comparable, not just the two the GPU
// mcraptor implements.
template <typename Criteria>
std::vector<tuple_t> run_bmrapp_gpu(fixture const& f,
                                    routing::query q,
                                    direction const dir) {
  auto ss = routing::search_state{};
  auto gtt = routing::gpu::gpu_timetable{f.tt_};
  auto as = routing::gpu::gpu_raptor_state{gtt};
  return tuples(*(routing::bmrap_profile_search<Criteria>(
                      f.tt_, nullptr, ss, as, std::move(q), dir)
                      .journeys_));
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
      bmrapp<routing::arr_walk_criteria>(f, q, direction::kForward).js_,
      run_bmrapp_gpu<routing::arr_walk_criteria>(f, q, direction::kForward));
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
