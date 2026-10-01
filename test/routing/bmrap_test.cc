#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "fmt/format.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/load.h"
#include "nigiri/routing/get_fastest_direct.h"
#include "nigiri/routing/raptor/bmrap_common.h"
#include "nigiri/routing/raptor/bmrap_filters.h"
#include "nigiri/routing/raptor/bmraptor.h"
#include "nigiri/routing/raptor/mcraptor.h"
#include "nigiri/routing/raptor/pong.h"
#include "nigiri/routing/raptor/raptor.h"
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
using namespace std::chrono_literals;
using namespace std::string_view_literals;

namespace {

// Three ways from O to D forming a proper (arrival, transfers) pareto set, so
// the restricted set has something to restrict, over several departures, so
// the profile driver takes more than one step:
//
//   route(s)        | dep O | arr D | transfers
//   ----------------+-------+-------+----------
//   RD (direct)     | 10:00 | 14:00 | 0
//   RA1 + RA2       | 10:10 | 12:30 | 1
//   RB1 + RB2 + RB3 | 10:20 | 12:00 | 2
//   RD (direct)     | 11:00 | 15:00 | 0
//   RA1 + RA2       | 11:10 | 13:30 | 1
//
// All pareto-optimal on (departure, arrival, transfers). Times below are local
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

// (departure, arrival, transfers), normalised: a backward search reports
// start_time_ as the arrival, departure_time()/arrival_time() undo that.
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
  fixture()
      : tt_{loader::load(
            mem_dir::read(std::string{kGTFS}),
            {sys_days{2024_y / June / 18}, sys_days{2024_y / June / 20}})} {
    o_ = tt_.locations_.location_id_to_idx_.at({"O", source_idx_t{0}});
    d_ = tt_.locations_.location_id_to_idx_.at({"D", source_idx_t{0}});
  }

  // Backward (arriveBy) starts at the destination, so start_ is D and the
  // window is an arrival window, as the routing endpoint builds it.
  routing::query make_query(direction const dir = direction::kForward) const {
    auto q = routing::query{};
    auto const day = sys_days{2024_y / June / 19};
    if (dir == direction::kForward) {
      // local 10:00-12:00, covering all five departures
      q.start_time_ =
          interval<unixtime_t>{unixtime_t{day} + 8h, unixtime_t{day} + 10h};
      q.start_ = {{o_, 0_minutes, 0U}};
      q.destination_ = {{d_, 0_minutes, 0U}};
    } else {
      // local 12:00-15:00, covering every arrival except the last (15:00)
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

// BMRAPP on the CPU scalar engine; the state type selects the scalar engine
// (see bmrap_algo_for), so the GPU variant below runs the same assertions.
template <typename Criteria>
run_result bmrapp(fixture const& f, routing::query q, direction const dir) {
  auto ss = routing::search_state{};
  auto as = routing::raptor_state{};
  auto const r = routing::bmrap_profile_search<Criteria>(f.tt_, nullptr, ss, as,
                                                         std::move(q), dir);
  return {tuples(*r.journeys_), r.interval_};
}

// Plain range search: with raptor_state the two-criteria search, i.e. the
// anchor set J_A itself; with an mcraptor state the unrestricted set.
template <typename AlgoState>
run_result reference(fixture const& f, routing::query q, direction const dir) {
  auto ss = routing::search_state{};
  auto as = AlgoState{};
  auto const r =
      routing::raptor_search(f.tt_, nullptr, ss, as, std::move(q), dir);
  return {tuples(*r.journeys_), r.interval_};
}

// "normal PONG operation": the engine serving a window whose extension side
// matches the search direction.
run_result pong_reference(fixture const& f,
                          routing::query q,
                          direction const dir) {
  auto ss = routing::search_state{};
  auto as = routing::raptor_state{};
  auto const r =
      routing::pong_search(f.tt_, nullptr, ss, as, std::move(q), dir);
  return {tuples(*r.journeys_), r.interval_};
}

std::string describe(tuple_t const& j) {
  return fmt::format("dep={} arr={} transfers={}",
                     std::get<0>(j).time_since_epoch().count(),
                     std::get<1>(j).time_since_epoch().count(), std::get<2>(j));
}

void expect_subset(std::vector<tuple_t> const& super,
                   std::vector<tuple_t> const& sub,
                   char const* const what) {
  ASSERT_FALSE(sub.empty()) << what << ": nothing to check";
  auto const have = std::set<tuple_t>{begin(super), end(super)};
  for (auto const& j : sub) {
    EXPECT_TRUE(have.contains(j))
        << what << ": journey the reference does not have: " << describe(j);
  }
}

// Every element of `sub` inside the range `super` actually scanned must be in
// `super`; the engines stop scanning at different points, so journeys beyond
// the shorter scan are legitimately absent. The scan key is the departure
// forward and the arrival backward.
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
          << what << ": missing journey " << describe(j);
    }
  }
  EXPECT_GT(checked, 0U) << what << ": no journey fell inside the range";
}

constexpr auto const kDirs =
    std::array{direction::kForward, direction::kBackward};

char const* name(direction const dir) {
  return dir == direction::kForward ? "forward" : "backward";
}

// The restricted set is a subset of the full multicriteria set: BM-RAPTOR may
// drop journeys the restriction rules out, but must never invent one.
template <typename Criteria, typename McState>
void expect_restricted_subset_of_mcraptor(direction const dir) {
  auto const f = fixture{};
  auto const q = f.make_query(dir);
  auto const restricted = bmrapp<Criteria>(f, q, dir).js_;
  ASSERT_FALSE(restricted.empty());
  expect_subset(reference<McState>(f, q, dir).js_, restricted, name(dir));
}

// The unrestricted range mcraptor set cut down by the restricted-set
// definition (bmrap_common.h): the anchors are its own journeys that are
// optimal on (departure, arrival, transfers) alone, and a journey survives if
// it is an anchor or stays within its A(J)'s trip budget and deadline. This is
// what bmrap_profile_search computes, only the slow way.
template <typename McState>
std::vector<tuple_t> restricted_reference(fixture const& f,
                                          routing::query const& q,
                                          direction const dir) {
  auto const fwd = dir == direction::kForward;
  auto js = reference<McState>(f, q, dir).js_;
  // the tuple is (dep, arr, transfers) in both directions
  auto const dominates = [](tuple_t const& a, tuple_t const& b) {
    return std::get<2>(a) <= std::get<2>(b) &&
           std::get<0>(a) >= std::get<0>(b) && std::get<1>(a) <= std::get<1>(b);
  };
  auto anchors = std::vector<routing::bmrap_detail::anchor>{};
  for (auto const& j : js) {
    if (std::none_of(begin(js), end(js), [&](tuple_t const& o) {
          return dominates(o, j) && !dominates(j, o);
        })) {
      auto const& [dep, arr, transfers] = j;
      anchors.push_back({fwd ? dep : arr, fwd ? arr : dep,
                         static_cast<std::uint8_t>(transfers + 1U)});
    }
  }
  std::erase_if(js, [&](tuple_t const& j) {
    auto const& [dep, arr, transfers] = j;
    return fwd ? routing::bmrap_detail::outside_restriction<
                     direction::kForward>(anchors, dep, arr, transfers + 1U)
               : routing::bmrap_detail::outside_restriction<
                     direction::kBackward>(anchors, arr, dep, transfers + 1U);
  });
  return js;
}

// bmrap_profile_search must return exactly the restricted set of the range
// mcraptor, not just a subset of it.
template <typename Criteria, typename McState>
void expect_bmrap_equals_restricted_mcraptor(direction const dir) {
  auto const f = fixture{};
  auto const q = f.make_query(dir);
  auto const bm = bmrapp<Criteria>(f, q, dir).js_;
  ASSERT_FALSE(bm.empty());
  EXPECT_EQ(restricted_reference<McState>(f, q, dir), bm) << name(dir);
}

}  // namespace

TEST(bmrap, subset_of_mcraptor) {
  for (auto const dir : kDirs) {
    expect_restricted_subset_of_mcraptor<routing::arr_criteria,
                                         routing::mcraptor_state>(dir);
  }
}

TEST(bmrap, equals_restricted_mcraptor) {
  for (auto const dir : kDirs) {
    expect_bmrap_equals_restricted_mcraptor<routing::arr_criteria,
                                            routing::mcraptor_state>(dir);
  }
}

TEST(bmrap, composed_equals_restricted_mcraptor) {
  for (auto const dir : kDirs) {
    expect_bmrap_equals_restricted_mcraptor<
        routing::arr_non_transit_mode_switches_criteria,
        routing::mcraptor_non_transit_mode_switches_state>(dir);
  }
}

// Same for composed criteria, covering arr_with<>'s machinery: dominance, the
// carried state and apply_to must fold over two dimensions.
TEST(bmrap, composed_subset_of_mcraptor) {
  for (auto const dir : kDirs) {
    expect_restricted_subset_of_mcraptor<
        routing::arr_non_transit_mode_switches_criteria,
        routing::mcraptor_non_transit_mode_switches_state>(dir);
  }
}

// Anchors are their own A(J), so no two-criteria journey may be restricted
// away. Regressed twice (anchor pareto-domination, filter/bounds reference
// mismatch).
TEST(bmrap, contains_bicriteria_journeys) {
  auto const f = fixture{};
  for (auto const dir : kDirs) {
    auto const q = f.make_query(dir);
    expect_contains_prefix(bmrapp<routing::arr_criteria>(f, q, dir).js_,
                           reference<routing::raptor_state>(f, q, dir).js_,
                           name(dir), dir);
  }
}

// Adding a pareto dimension can only split classes apart, so a composed
// criteria must never lose what a subset of its dimensions finds. This breaks
// if a dimension's dominance folds the wrong way.
TEST(bmrap, more_dimensions_never_lose_journeys) {
  auto const f = fixture{};
  auto const walk = bmrapp<routing::arr_non_transit_criteria>(
                        f, f.make_query(), direction::kForward)
                        .js_;
  auto const walk_clasz =
      bmrapp<routing::arr_non_transit_mode_switches_criteria>(
          f, f.make_query(), direction::kForward)
          .js_;
  ASSERT_FALSE(walk.empty());
  expect_subset(walk_clasz, walk,
                "adding the clasz dimension dropped a journey");
}

// The same journeys whichever end the search starts from, compared on the box
// both scans cover (departure in the forward window, arrival in the backward
// one); outside it either scan is legitimately blind.
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
// min_connection_count_ is met. The side it grows on is fixed by the search
// direction and only matches the requested side in the two PONG-applicable
// ("aligned") combinations; for the opposed pair a bicriteria range search
// settles the window up front (see bmrap_profile.cc).

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

// requested side == the side the scan grows on
std::vector<extend_case> aligned_cases() {
  return {
      // one departure (10:05 local) in the window, more available later
      {direction::kForward,
       false,
       true,
       {at(8h, 0min), at(8h, 6min)},
       "departAfter, extend later"},
      // one arrival (15:00 local) in the window, more available earlier
      {direction::kBackward,
       true,
       false,
       {at(12h, 55min), at(13h, 1min)},
       "arriveBy, extend earlier"},
  };
}

std::vector<extend_case> opposed_cases() {
  return {
      // one departure (11:00 local), the others are earlier
      {direction::kForward,
       true,
       false,
       {at(9h, 0min), at(9h, 6min)},
       "departAfter, extend earlier"},
      // one arrival (12:00 local), the others are later
      {direction::kBackward,
       false,
       true,
       {at(10h, 0min), at(10h, 6min)},
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
// outside the requested window are not what was asked for, and on a paging
// request they are the ones already shown.
void expect_stays_inside(fixture const& f, extend_case const& c) {
  auto const r = bmrapp<routing::arr_criteria>(f, extend_query(f, c), c.dir_);
  if (!c.earlier_) {
    EXPECT_GE(r.scanned_.from_, c.win_.from_)
        << c.name_ << ": extended earlier although the query forbids it";
  }
  if (!c.later_) {
    EXPECT_LE(r.scanned_.to_, c.win_.to_)
        << c.name_ << ": extended later although the query forbids it";
  }
}

// ...and it must grow on the allowed side until min_connection_count_ is met.
// Each window holds exactly one journey, so without extension that is all
// there would be.
void expect_grows_on_allowed_side(fixture const& f, extend_case const& c) {
  auto const q = extend_query(f, c);
  auto const r = bmrapp<routing::arr_criteria>(f, q, c.dir_);
  auto const ref = reference<routing::raptor_state>(f, q, c.dir_);
  EXPECT_GE(ref.js_.size(), q.min_connection_count_)
      << c.name_
      << ": the reference did not extend either - the fixture "
         "cannot support this case";
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

// The opposed pair: the window comes from the range search instead and must be
// right on both counts as well.
TEST(bmrap, extends_against_the_search_direction) {
  auto const f = fixture{};
  for (auto const& c : opposed_cases()) {
    expect_stays_inside(f, c);
    expect_grows_on_allowed_side(f, c);
  }
}

// The fallback must not lose anything. numItineraries is satisfied on the
// bicriteria journeys there, so the window comes out wider than needed and the
// result is a superset of what normal PONG operation returns for that window -
// the harmless direction, and the property worth pinning down.
TEST(bmrap, opposed_extension_is_a_superset_of_pong) {
  auto const f = fixture{};
  for (auto const& c : opposed_cases()) {
    auto const r = bmrapp<routing::arr_criteria>(f, extend_query(f, c), c.dir_);
    ASSERT_FALSE(r.js_.empty()) << c.name_ << ": no result to compare";

    // the window the fallback settled on, requested the normal way round: the
    // extension side aligned with the search direction so PONG applies, and no
    // numItineraries growth
    auto q = f.make_query(c.dir_);
    q.start_time_ = r.scanned_;
    q.extend_interval_earlier_ = c.dir_ == direction::kBackward;
    q.extend_interval_later_ = c.dir_ == direction::kForward;
    q.min_connection_count_ = 0U;

    expect_subset(r.js_, pong_reference(f, q, c.dir_).js_, c.name_);
  }
}

// The next departure worth taking can lie far beyond fastest_direct: the scan
// bound limits the arrival relative to the step's departure, not the travel
// time. Queried after the last departure of the 18th, the next one leaves ~18h
// later - a ping bounded by fastest_direct (6h) found nothing and the scan
// ended empty, while PONG kept looking.
TEST(bmrap, finds_departures_beyond_fastest_direct) {
  auto const f = fixture{};
  auto q = f.make_query();
  auto const t = unixtime_t{sys_days{2024_y / June / 18}} + 14h;
  q.start_time_ = interval<unixtime_t>{t, t + 1min};
  q.extend_interval_later_ = true;
  q.fastest_direct_ = 6h;
  auto const pong = pong_reference(f, q, direction::kForward);
  ASSERT_FALSE(pong.js_.empty()) << "the fixture cannot support this case";
  EXPECT_EQ(pong.js_,
            bmrapp<routing::arr_criteria>(f, q, direction::kForward).js_);
}

// The restriction references a SET of anchors: A(J) (the earliest-arriving
// anchor available at J with at most J's trips) plus every available anchor
// arriving before A(J), whatever its trip count. J stays if any of them has
// its slack satisfied. Default slack: deadline = departure + travel +
// clamp(travel / 2, 20 min, 180 min).
TEST(bmrap, restriction_uses_earlier_arriving_anchors) {
  using routing::bmrap_detail::anchor;
  using routing::bmrap_detail::outside_restriction;
  auto const at = [](int const h, int const m) {
    return unixtime_t{sys_days{2024_y / June / 19}} + std::chrono::hours{h} +
           std::chrono::minutes{m};
  };
  auto const anchors = std::vector<anchor>{
      {at(10, 30), at(11, 0), 1U},  // A(J) for J with 2 trips, deadline 11:20
      {at(10, 0), at(10, 55), 3U},  // arrives before A(J), deadline 11:23
      {at(10, 0), at(11, 10), 4U},  // arrives after A(J), deadline 11:45
  };
  auto const outside = [&](unixtime_t const found) {
    return outside_restriction<direction::kForward>(anchors, at(10, 0), found,
                                                    2U);
  };
  EXPECT_FALSE(outside(at(11, 20))) << "within A(J)'s deadline";
  EXPECT_FALSE(outside(at(11, 22)))
      << "past A(J)'s deadline, within the earlier-arriving anchor's";
  EXPECT_TRUE(outside(at(11, 24))) << "past both deadlines";
  EXPECT_TRUE(outside(at(11, 30)))
      << "only the later-arriving anchor's deadline covers it, and that one "
         "is not in the set";
}

// get_fastest_direct() scales the footpath distance by fastest_direct_factor_
// in 16 bit minutes: benchmarks disable the cap with a large factor, which
// wrapped around to a tiny bound, and every engine filtered all journeys.
namespace {

constexpr auto const kFootpathGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,DB,https://db.de,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station
A,A,0.0,1.0,,
B,B,0.01,1.0,,

# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time
A,B,2,2400

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R,DB,R,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R,S,T,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T,10:00:00,10:00:00,A,0,0,0
T,10:30:00,10:30:00,B,1,0,0

# calendar_dates.txt
service_id,date,exception_type
S,20240619,1
)"sv;

}  // namespace

TEST(bmrap, fastest_direct_factor_saturates) {
  auto const tt = loader::load(
      mem_dir::read(std::string{kFootpathGTFS}),
      {sys_days{2024_y / June / 18}, sys_days{2024_y / June / 20}});
  auto const loc = [&](char const* id) {
    return tt.locations_.location_id_to_idx_.at({id, source_idx_t{0}});
  };

  auto q = routing::query{};
  q.start_ = {{loc("A"), 0_minutes, 0U}};
  q.destination_ = {{loc("B"), 0_minutes, 0U}};
  q.start_match_mode_ = routing::location_match_mode::kEquivalent;
  q.dest_match_mode_ = routing::location_match_mode::kEquivalent;

  auto const fp = routing::get_fastest_direct(tt, q, direction::kForward);
  ASSERT_GE(fp, 33min) << "the footpath must be long enough to overflow";
  ASSERT_LT(fp, routing::kUnreachableDirect);

  q.fastest_direct_factor_ = 1000.0;
  EXPECT_EQ(routing::kUnreachableDirect,
            routing::get_fastest_direct(tt, q, direction::kForward));
}

// The mc ping's arrival bound. A step's departure can lie days before the
// journeys it finds (the scan jumps to the first anchor), so "step + 5 days"
// cut off mc journeys that depart right away and arrive after it, though an
// anchor's deadline admits them. Here the query is 5 days before the only
// departures; A (walk to O2, 08:55 -> 09:50) is the anchor, J (08:00 ->
// 10:10 from O itself) is optimal by walking less and within A's deadline
// (10:18), but arrives after query + 5 days (10:00).
namespace {

constexpr auto const kFarGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,DB,https://db.de,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station
O,O,0.0,1.0,,
O2,O2,0.001,1.0,,
D,D,5.0,1.0,,

# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time
O,O2,2,300

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
RA,DB,RA,,,3
RJ,DB,RJ,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
RA,S,TA,,
RJ,S,TJ,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
TA,09:00:00,09:00:00,O2,0,0,0
TA,09:50:00,09:50:00,D,1,0,0
TJ,08:00:00,08:00:00,O,0,0,0
TJ,10:10:00,10:10:00,D,1,0,0

# calendar_dates.txt
service_id,date,exception_type
S,20240616,1
)"sv;

}  // namespace

TEST(bmrap, mc_ping_bound_follows_anchor_deadlines) {
  auto const tt = loader::load(
      mem_dir::read(std::string{kFarGTFS}),
      {sys_days{2024_y / June / 10}, sys_days{2024_y / June / 20}});
  auto const loc = [&](char const* id) {
    return tt.locations_.location_id_to_idx_.at({id, source_idx_t{0}});
  };

  auto q = routing::query{};
  // local 2024-06-11 10:00, five days before the only departures
  auto const t = unixtime_t{sys_days{2024_y / June / 11}} + 8h;
  q.start_time_ = interval<unixtime_t>{t, t + 1min};
  q.start_ = {{loc("O"), 0_minutes, 0U}};
  q.destination_ = {{loc("D"), 0_minutes, 0U}};
  q.use_start_footpaths_ = true;
  q.via_stops_ = {};
  q.min_connection_count_ = 1U;
  q.extend_interval_later_ = true;

  auto const day = unixtime_t{sys_days{2024_y / June / 16}};
  auto const j = tuple_t{day + 6h, day + 8h + 10min, 0U};  // 08:00 -> 10:10
  auto ss = routing::search_state{};
  auto as = routing::raptor_state{};
  auto const bm =
      tuples(*routing::bmrap_profile_search<routing::arr_non_transit_criteria>(
                  tt, nullptr, ss, as, q, direction::kForward)
                  .journeys_);
  ASSERT_FALSE(bm.empty()) << "the anchor itself was not found";
  EXPECT_TRUE(std::find(begin(bm), end(bm), j) != end(bm))
      << "the journey walking less, arriving after query + 5 days, is lost";
}

// close_anchor_profile() has to supply every trip count's A(J) past the
// window: a low-trip A(J) arrives later than the fastest journey from the
// boundary. With the anchors only reaching the fastest one, L (1 trip, far
// past the window) was missing, so there was no A(J) for J and it was
// dropped - by range BMRAP, and by BMRAPP for a window it may not extend
// (there its anchors come from the same range search).
namespace {

constexpr auto const kMarginGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,DB,https://db.de,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station
O,O,0.0,1.0,,
O2,O2,0.001,1.0,,
X1,X1,1.0,1.0,,
X2,X2,2.0,1.0,,
Y1,Y1,1.0,2.0,,
Y2,Y2,2.0,2.0,,
D,D,5.0,1.0,,

# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time
O,O2,2,300

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
H1,DB,H1,,,3
H2,DB,H2,,,3
H3,DB,H3,,,3
E1,DB,E1,,,3
E2,DB,E2,,,3
E3,DB,E3,,,3
RJ,DB,RJ,,,3
RL,DB,RL,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
H1,S,TH1,,
H2,S,TH2,,
H3,S,TH3,,
E1,S,TE1,,
E2,S,TE2,,
E3,S,TE3,,
RJ,S,TJ,,
RL,S,TL,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
TH1,10:10:00,10:10:00,O,0,0,0
TH1,10:15:00,10:15:00,Y1,1,0,0
TH2,10:20:00,10:20:00,Y1,0,0,0
TH2,10:25:00,10:25:00,Y2,1,0,0
TH3,10:30:00,10:30:00,Y2,0,0,0
TH3,10:40:00,10:40:00,D,1,0,0
TE1,11:00:00,11:00:00,O,0,0,0
TE1,11:10:00,11:10:00,X1,1,0,0
TE2,11:15:00,11:15:00,X1,0,0,0
TE2,11:20:00,11:20:00,X2,1,0,0
TE3,11:25:00,11:25:00,X2,0,0,0
TE3,11:30:00,11:30:00,D,1,0,0
TJ,10:20:00,10:20:00,O,0,0,0
TJ,15:00:00,15:00:00,D,1,0,0
TL,12:00:00,12:00:00,O2,0,0,0
TL,14:00:00,14:00:00,D,1,0,0

# calendar_dates.txt
service_id,date,exception_type
S,20240619,1
)"sv;

}  // namespace

TEST(bmrap, anchor_margin_covers_every_trip_count) {
  auto const tt = loader::load(
      mem_dir::read(std::string{kMarginGTFS}),
      {sys_days{2024_y / June / 18}, sys_days{2024_y / June / 20}});
  auto const loc = [&](char const* id) {
    return tt.locations_.location_id_to_idx_.at({id, source_idx_t{0}});
  };

  auto const day = unixtime_t{sys_days{2024_y / June / 19}};
  auto q = routing::query{};
  // local 10:00 - 10:30, fixed
  q.start_time_ = interval<unixtime_t>{day + 8h, day + 8h + 30min};
  q.start_ = {{loc("O"), 0_minutes, 0U}};
  q.destination_ = {{loc("D"), 0_minutes, 0U}};
  q.use_start_footpaths_ = true;
  q.via_stops_ = {};
  q.min_connection_count_ = 0U;

  auto const j = tuple_t{day + 8h + 20min, day + 13h, 0U};  // 10:20 -> 15:00
  auto ss = routing::search_state{};
  auto as = routing::raptor_state{};
  auto const profile =
      tuples(*routing::bmrap_profile_search<routing::arr_non_transit_criteria>(
                  tt, nullptr, ss, as, q, direction::kForward)
                  .journeys_);
  // a fixed window is BMRAPP's "opposed" case: its anchors come from the same
  // range search and close_anchor_profile()
  EXPECT_TRUE(std::find(begin(profile), end(profile), j) != end(profile))
      << "BMRAPP: J restricted away, its A(J) lies past the margin";
}

// The forward re-realization (bmrap_profile.cc 5b) must still replace the mc
// pong's LATEST connections with the EARLIEST ones under its own bounds (the
// mc pong's labels). O 10:00 -> M 10:20, then M -> X by 10:25 -> 10:35 or
// 10:45 -> 10:55, then X 11:00 -> D 11:30: the backward reconstruction takes
// the later M -> X trip, the realization the earlier one.
namespace {

constexpr auto const kRealizeGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,DB,https://db.de,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station
O,O,0.0,1.0,,
M,M,1.0,1.0,,
X,X,2.0,1.0,,
D,D,3.0,1.0,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,R1,,,3
R2,DB,R2,,,3
R3,DB,R3,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R1,S,T1,,
R2,S,T2,,
R2,S,T3,,
R3,S,T4,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T1,10:00:00,10:00:00,O,0,0,0
T1,10:20:00,10:20:00,M,1,0,0
T2,10:25:00,10:25:00,M,0,0,0
T2,10:35:00,10:35:00,X,1,0,0
T3,10:45:00,10:45:00,M,0,0,0
T3,10:55:00,10:55:00,X,1,0,0
T4,11:00:00,11:00:00,X,0,0,0
T4,11:30:00,11:30:00,D,1,0,0

# calendar_dates.txt
service_id,date,exception_type
S,20240619,1
)"sv;

}  // namespace

TEST(bmrap, realization_takes_earliest_connections) {
  auto const tt = loader::load(
      mem_dir::read(std::string{kRealizeGTFS}),
      {sys_days{2024_y / June / 18}, sys_days{2024_y / June / 20}});
  auto const loc = [&](char const* id) {
    return tt.locations_.location_id_to_idx_.at({id, source_idx_t{0}});
  };

  auto const day = unixtime_t{sys_days{2024_y / June / 19}};
  auto q = routing::query{};
  q.start_time_ = interval<unixtime_t>{day + 8h, day + 8h + 1min};
  q.start_ = {{loc("O"), 0_minutes, 0U}};
  q.destination_ = {{loc("D"), 0_minutes, 0U}};
  q.via_stops_ = {};
  q.min_connection_count_ = 1U;
  q.extend_interval_later_ = true;

  auto ss = routing::search_state{};
  auto as = routing::raptor_state{};
  auto const r =
      routing::bmrap_profile_search<routing::arr_non_transit_criteria>(
          tt, nullptr, ss, as, q, direction::kForward);
  auto const j = std::find_if(begin(*r.journeys_), end(*r.journeys_),
                              [&](routing::journey const& x) {
                                return x.departure_time() == day + 8h &&
                                       x.arrival_time() == day + 9h + 30min;
                              });
  ASSERT_NE(j, end(*r.journeys_)) << "O 10:00 -> D 11:30 not found";
  auto rides = std::vector<routing::journey::leg>{};
  for (auto const& l : j->legs_) {
    if (std::holds_alternative<routing::journey::run_enter_exit>(l.uses_)) {
      rides.push_back(l);
    }
  }
  ASSERT_EQ(3U, rides.size());
  EXPECT_EQ(day + 8h + 35min, rides[1].arr_time_)
      << "M -> X is not the earliest connection (10:35 local)";
  EXPECT_GT(r.algo_stats_.at("bmrapp_realized"), 0U);
}

// Two destination offsets reached in the same round: D1 at 10:30 (2
// transfers), then D2 at 10:35. The ping's target pruning is relaxed by the
// arrival slack, so 10:35 still beats time_at_dest_ - it must not overwrite the
// round's 10:30, or the journey turns into one a direct trip dominates (10:32
// from the 10:00 step, 10:33 from the 10:01 step), the anchors keep only one
// trip, and the trip budget cuts the mc phases off from the 2-transfer
// journey.
namespace {

constexpr auto const kTwoDestGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,DB,https://db.de,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station
O,O,0.0,1.0,,
A,A,2.0,3.0,,
B,B,4.0,5.0,,
D1,D1,6.0,7.0,,
D2,D2,8.0,9.0,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DB,R0,,,3
R1,DB,R1,,,3
R2,DB,R2,,,3
R3,DB,R3,,,3
R4,DB,R4,,,3
R5,DB,R5,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,S,T0,,
R1,S,T1,,
R2,S,T2,,
R3,S,T3,,
R4,S,T4,,
R5,S,T5,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T0,10:00:00,10:00:00,O,0,0,0
T0,10:32:00,10:32:00,D2,1,0,0
T1,10:01:00,10:01:00,O,0,0,0
T1,10:08:00,10:08:00,A,1,0,0
T2,10:12:00,10:12:00,A,0,0,0
T2,10:18:00,10:18:00,B,1,0,0
T3,10:22:00,10:22:00,B,0,0,0
T3,10:30:00,10:30:00,D1,1,0,0
T4,10:22:00,10:22:00,B,0,0,0
T4,10:35:00,10:35:00,D2,1,0,0
T5,10:02:00,10:02:00,O,0,0,0
T5,10:33:00,10:33:00,D2,1,0,0

# calendar_dates.txt
service_id,date,exception_type
S,20240619,1
)"sv;

}  // namespace

TEST(bmrap, relaxed_ping_keeps_best_intermodal_arrival) {
  auto const tt = loader::load(
      mem_dir::read(std::string{kTwoDestGTFS}),
      {sys_days{2024_y / June / 18}, sys_days{2024_y / June / 20}});
  auto const loc = [&](char const* id) {
    return tt.locations_.location_id_to_idx_.at({id, source_idx_t{0}});
  };
  ASSERT_LT(loc("D1"), loc("D2")) << "D2 must be written after D1";

  auto q = routing::query{};
  auto const day = sys_days{2024_y / June / 19};
  q.start_time_ =
      interval<unixtime_t>{unixtime_t{day} + 8h, unixtime_t{day} + 8h + 2min};
  q.start_ = {{loc("O"), 0_minutes, 0U}};
  q.destination_ = {{loc("D1"), 0_minutes, 0U}, {loc("D2"), 0_minutes, 0U}};
  q.dest_match_mode_ = routing::location_match_mode::kIntermodal;
  q.via_stops_ = {};
  q.min_connection_count_ = 1U;
  q.extend_interval_later_ = true;  // the per-step ping ("aligned") path

  auto ss = routing::search_state{};
  auto as = routing::raptor_state{};
  auto const pong = tuples(*routing::pong_search(tt, nullptr, ss, as, q,
                                                 direction::kForward)
                                .journeys_);
  auto const two_transfers =
      tuple_t{unixtime_t{day} + 8h + 1min, unixtime_t{day} + 8h + 30min, 2U};
  ASSERT_TRUE(std::find(begin(pong), end(pong), two_transfers) != end(pong))
      << "the fixture cannot support this case";

  auto ss2 = routing::search_state{};
  auto as2 = routing::raptor_state{};
  expect_subset(tuples(*routing::bmrap_profile_search<routing::arr_criteria>(
                            tt, nullptr, ss2, as2, q, direction::kForward)
                            .journeys_),
                pong, "two destination offsets in one round");
}

// ---------------------------------------------------------------------------
// bounded_needs_lb<Algo>(): the lower-bound-dijkstra skip trait
// ---------------------------------------------------------------------------

namespace {

// the three shapes it branches on: never uses lb (gpu_raptor), uses lb but does
// not need it once bounded (raptor, basic_mcraptor), no declared opinion
// (gpu_mcraptor, the conservative default)
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

// The two engines bmrap_profile.cc bounds unconditionally must not need lb once
// bounded, which is what makes skipping the bwd_lb dijkstra there safe.
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
namespace {

// The multicriteria phases run on the CPU unless gpu_mc_mode says otherwise
// (see bmrap_algo_for), so every criteria configuration is comparable here.
template <typename Criteria>
std::vector<tuple_t> run_bmrapp_gpu(fixture const& f,
                                    routing::query q,
                                    direction const dir,
                                    int const gpu_mc_mode = 1) {
  auto ss = routing::search_state{};
  auto gtt = routing::gpu::gpu_timetable{f.tt_};
  auto as = routing::gpu::gpu_raptor_state{gtt};
  return tuples(*(
      routing::bmrap_profile_search<Criteria>(
          f.tt_, nullptr, ss, as, std::move(q), dir, std::nullopt, gpu_mc_mode)
          .journeys_));
}

// The device mcraptor on its own: nothing else exercises gpu_mcraptor, so a
// divergence shows up here rather than as a BMRAPP failure.
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

template <typename Criteria>
void expect_gpu_matches_cpu(direction const dir) {
  auto const f = fixture{};
  auto const q = f.make_query(dir);
  EXPECT_EQ(bmrapp<Criteria>(f, q, dir).js_,
            run_bmrapp_gpu<Criteria>(f, q, dir))
      << name(dir);
}

}  // namespace

#define SKIP_WITHOUT_GPU()              \
  if (!routing::gpu::gpu_available()) { \
    GTEST_SKIP() << "no CUDA device";   \
  }

TEST(bmrap, gpu_mcraptor_matches_cpu) {
  SKIP_WITHOUT_GPU();
  auto const f = fixture{};
  for (auto const dir : kDirs) {
    auto const q = f.make_query(dir);
    EXPECT_EQ(reference<routing::mcraptor_state>(f, q, dir).js_,
              run_gpu_mcraptor(f, q, dir))
        << name(dir);
  }
}

// Only the ping, pong and backward pruning searches move to the device, and
// arriveBy exercises the other, direction-indexed half of the GPU state.
TEST(bmrap, gpu_matches_cpu) {
  SKIP_WITHOUT_GPU();
  for (auto const dir : kDirs) {
    expect_gpu_matches_cpu<routing::arr_criteria>(dir);
    expect_gpu_matches_cpu<routing::arr_non_transit_criteria>(dir);
    // the timetable has no flights, so this only pins that the extra label
    // field does not perturb the arrival-only result
    expect_gpu_matches_cpu<routing::arr_mode_filter_criteria>(dir);
  }
}

// gpu_mc_mode 2 also runs the mc pong on the device; the journeys must not
// change.
TEST(bmrap, gpu_mc_pong_matches_mc_ping) {
  SKIP_WITHOUT_GPU();
  auto const f = fixture{};
  auto const q = f.make_query();
  EXPECT_EQ(
      run_bmrapp_gpu<routing::arr_criteria>(f, q, direction::kForward, 1),
      run_bmrapp_gpu<routing::arr_criteria>(f, q, direction::kForward, 2));
  EXPECT_EQ(run_bmrapp_gpu<routing::arr_non_transit_criteria>(
                f, q, direction::kForward, 1),
            run_bmrapp_gpu<routing::arr_non_transit_criteria>(
                f, q, direction::kForward, 2));
}

#undef SKIP_WITHOUT_GPU
#endif
