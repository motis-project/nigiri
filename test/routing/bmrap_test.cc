#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "fmt/format.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
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
using namespace nigiri::loader::gtfs;
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
  fixture() {
    tt_.date_range_ = {sys_days{2024_y / June / 18},
                       sys_days{2024_y / June / 20}};
    register_special_stations(tt_);
    load_timetable({}, source_idx_t{0}, mem_dir::read(std::string{kGTFS}), tt_);
    finalize(tt_);
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

}  // namespace

TEST(bmrap, subset_of_mcraptor) {
  for (auto const dir : kDirs) {
    expect_restricted_subset_of_mcraptor<routing::arr_criteria,
                                         routing::mcraptor_state>(dir);
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
