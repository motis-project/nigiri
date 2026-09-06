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

std::vector<tuple_t> tuples(pareto_set<routing::journey> const& js) {
  auto v = std::vector<tuple_t>{};
  for (auto const& j : js) {
    v.emplace_back(j.start_time_, j.dest_time_,
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

  routing::query make_query() const {
    auto q = routing::query{};
    // local 10:00-12:00 = 08:00-10:00 UTC
    auto const day = sys_days{2024_y / June / 19};
    q.start_time_ =
        interval<unixtime_t>{unixtime_t{day} + 8h, unixtime_t{day} + 10h};
    q.start_ = {{o_, 0_minutes, 0U}};
    q.destination_ = {{d_, 0_minutes, 0U}};
    q.via_stops_ = {};
    q.min_connection_count_ = 1U;
    return q;
  }

  timetable tt_;
  location_idx_t o_, d_;
};

// BMRAPP on the CPU scalar engine. The state type is what selects the
// scalar engine (see bmrap_algo_for), exactly as it does for pong_search,
// so the GPU variant below runs the identical assertions.
template <typename Criteria>
std::vector<tuple_t> run_bmrapp(fixture const& f) {
  auto ss = routing::search_state{};
  auto as = routing::raptor_state{};
  return tuples(*(routing::bmrap_profile_search<Criteria>(
                      f.tt_, nullptr, ss, as, f.make_query(),
                      direction::kForward)
                      .journeys_));
}

template <typename McState>
std::vector<tuple_t> run_mcraptor(fixture const& f) {
  auto ss = routing::search_state{};
  auto as = McState{};
  return tuples(*(routing::raptor_search(f.tt_, nullptr, ss, as,
                                         f.make_query(), direction::kForward)
                      .journeys_));
}

// The two-criteria (arrival, transfers) range search - by definition the
// anchor set J_A that BM-RAPTOR restricts around.
std::vector<tuple_t> run_bicriteria(fixture const& f) {
  auto ss = routing::search_state{};
  auto as = routing::raptor_state{};
  return tuples(*(routing::raptor_search(f.tt_, nullptr, ss, as,
                                         f.make_query(), direction::kForward)
                      .journeys_));
}

// Every element of `sub` that departs strictly before `super`'s last
// departure must be in `super`. Restricting to the covered range is what
// makes this meaningful: the two engines stop scanning at different
// points, so journeys past the shorter scan are legitimately absent.
void expect_contains_prefix(std::vector<tuple_t> const& super,
                            std::vector<tuple_t> const& sub,
                            char const* const what) {
  ASSERT_FALSE(super.empty()) << what << ": nothing to compare against";
  auto const cut = std::get<0>(*std::max_element(
      begin(super), end(super), [](tuple_t const& a, tuple_t const& b) {
        return std::get<0>(a) < std::get<0>(b);
      }));
  auto const have = std::set<tuple_t>{begin(super), end(super)};
  auto checked = 0U;
  for (auto const& j : sub) {
    if (std::get<0>(j) < cut) {
      ++checked;
      EXPECT_TRUE(have.contains(j))
          << what << ": missing journey dep=" << std::get<0>(j).time_since_epoch().count()
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
  auto const have = std::set<tuple_t>{begin(full), end(full)};
  for (auto const& j : restricted) {
    EXPECT_TRUE(have.contains(j))
        << "BMRAPP returned a journey McRAPTOR does not have: dep="
        << std::get<0>(j).time_since_epoch().count()
        << " arr=" << std::get<1>(j).time_since_epoch().count()
        << " transfers=" << std::get<2>(j);
  }
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
  auto const have = std::set<tuple_t>{begin(full), end(full)};
  for (auto const& j : restricted) {
    EXPECT_TRUE(have.contains(j))
        << "BMRAPP(walk+clasz) returned a journey McRAPTOR does not have";
  }
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
  auto const have = std::set<tuple_t>{begin(walk_clasz), end(walk_clasz)};
  for (auto const& j : walk) {
    EXPECT_TRUE(have.contains(j))
        << "adding the clasz dimension dropped a journey walk-only found";
  }
}

#if defined(NIGIRI_CUDA)
// The GPU scalar engine must produce the same restricted set as the CPU one.
// Only the ping / pong / backward-pruning searches move to the device; the
// multicriteria phases run on the CPU either way (see bmrap_algo_for), so
// every criteria configuration is comparable, not just the two the GPU
// mcraptor implements.
template <typename Criteria>
std::vector<tuple_t> run_bmrapp_gpu(fixture const& f) {
  auto ss = routing::search_state{};
  auto gtt = routing::gpu::gpu_timetable{f.tt_};
  auto as = routing::gpu::gpu_raptor_state{gtt};
  return tuples(*(routing::bmrap_profile_search<Criteria>(
                      f.tt_, nullptr, ss, as, f.make_query(),
                      direction::kForward)
                      .journeys_));
}

TEST(bmrap, gpu_matches_cpu) {
  if (!routing::gpu::gpu_available()) {
    GTEST_SKIP() << "no CUDA device";
  }
  auto const f = fixture{};
  EXPECT_EQ(run_bmrapp<routing::arr_criteria>(f),
            run_bmrapp_gpu<routing::arr_criteria>(f));
}

TEST(bmrap, gpu_matches_cpu_walk) {
  if (!routing::gpu::gpu_available()) {
    GTEST_SKIP() << "no CUDA device";
  }
  auto const f = fixture{};
  EXPECT_EQ(run_bmrapp<routing::arr_walk_criteria>(f),
            run_bmrapp_gpu<routing::arr_walk_criteria>(f));
}
#endif
