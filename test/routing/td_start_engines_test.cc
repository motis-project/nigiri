#include "gtest/gtest.h"

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

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace std::chrono_literals;

// Time-dependent START offsets across every engine that can serve them.
//
// The window engines (PONG, BM-RAPTOR) do not report the departure their
// forward ping was seeded with: the ping runs its window as one step and the
// backward pong re-derives each journey's latest feasible departure. With a
// td offset the ingress duration is itself a function of the departure, so
// this is exactly where a re-anchoring that assumes a constant offset
// produces a departure that does not exist - see pong.cc's tight-start
// guard, which the `slower_later` case below pins down.
namespace {

mem_dir test_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,2.0,3.0,,

# calendar_dates.txt
service_id,date,exception_type
S,20240619,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,RE 1,,,2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R1,S,T1,RE 1,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T1,10:00:00,10:00:00,A,1,0,0
T1,11:00:00,11:00:00,B,2,0,0
)");
}

using tuple_t = std::tuple<unixtime_t, unixtime_t, std::uint8_t>;

std::vector<tuple_t> tuples(pareto_set<routing::journey> const& js) {
  auto v = std::vector<tuple_t>{};
  for (auto const& j : js) {
    v.emplace_back(j.start_time_, j.dest_time_, j.transfers_);
  }
  std::sort(begin(v), end(v));
  return v;
}

struct fixture {
  fixture() {
    tt_.date_range_ = {sys_days{2024_y / June / 18}, sys_days{2024_y / June / 20}};
    register_special_stations(tt_);
    load_timetable({}, source_idx_t{0}, test_files(), tt_);
    finalize(tt_);
    a_ = tt_.find({"A", {}}).value();
    b_ = tt_.find({"B", {}}).value();
  }

  // intermodal start whose only access to A is time-dependent
  routing::query query(std::vector<routing::td_offset> offsets,
                       interval<unixtime_t> const window) const {
    auto q = routing::query{};
    q.start_time_ = window;
    q.start_match_mode_ = routing::location_match_mode::kIntermodal;
    q.dest_match_mode_ = routing::location_match_mode::kIntermodal;
    q.use_start_footpaths_ = false;
    q.destination_ = {{b_, duration_t{0}, transport_mode_id_t{0}}};
    q.td_start_ = {{a_, std::move(offsets)}};
    return q;
  }

  timetable tt_;
  location_idx_t a_, b_;
};

template <typename AlgoState>
std::vector<tuple_t> range_search(fixture const& f, routing::query q) {
  auto ss = routing::search_state{};
  auto as = AlgoState{};
  return tuples(*routing::raptor_search(f.tt_, nullptr, ss, as, std::move(q),
                                        direction::kForward)
                     .journeys_);
}

template <typename AlgoState>
std::vector<tuple_t> pong(fixture const& f, routing::query q) {
  auto ss = routing::search_state{};
  auto as = AlgoState{};
  return tuples(*routing::pong_search(f.tt_, nullptr, ss, as, std::move(q),
                                      direction::kForward)
                     .journeys_);
}

std::vector<tuple_t> bmrapp(fixture const& f, routing::query q) {
  auto ss = routing::search_state{};
  auto as = routing::raptor_state{};
  return tuples(*routing::bmrap_profile_search<routing::arr_criteria>(
                     f.tt_, nullptr, ss, as, std::move(q), direction::kForward)
                     .journeys_);
}

// every engine must agree with the plain rRAPTOR reference
void expect_all_engines_agree(fixture const& f, routing::query const& q) {
  auto const ref = range_search<routing::raptor_state>(f, q);
  ASSERT_FALSE(ref.empty()) << "reference found no journey";
  EXPECT_EQ(ref, range_search<routing::mcraptor_state>(f, q)) << "mcraptor";
  EXPECT_EQ(ref, pong<routing::raptor_state>(f, q)) << "pong (scalar)";
  EXPECT_EQ(ref, pong<routing::mcraptor_state>(f, q)) << "pong (mcraptor)";
  EXPECT_EQ(ref, pong<routing::mcraptor_cost_state>(f, q)) << "pong (mc cost)";
  EXPECT_EQ(ref, bmrapp(f, q)) << "bmrapp";
}

}  // namespace

// The offset is valid for a single minute, so exactly one departure works
// and the journey waits 20 minutes at A.
TEST(routing, td_start_engines_narrow_validity) {
  auto const f = fixture{};
  expect_all_engines_agree(
      f, f.query({{.valid_from_ = unixtime_t{0h},
                   .duration_ = footpath::kMaxDuration,
                   .transport_mode_id_ = 5},
                  {.valid_from_ = sys_days{2024_y / June / 19} + 7h + 30min,
                   .duration_ = 10min,
                   .transport_mode_id_ = 5},
                  {.valid_from_ = sys_days{2024_y / June / 19} + 7h + 31min,
                   .duration_ = footpath::kMaxDuration,
                   .transport_mode_id_ = 5}},
                 {sys_days{2024_y / June / 19},
                  sys_days{2024_y / June / 20}}));
}

// The offset gets SLOWER later in the day (5min before 06:00, 60min after).
// Re-anchoring a ping journey by the wait at its first boarding assumes the
// ingress duration is unchanged at the shifted departure; here that lands
// past every feasible departure and the window engines used to find no
// journey for their own anchor at all.
TEST(routing, td_start_engines_slower_later) {
  auto const f = fixture{};
  expect_all_engines_agree(
      f, f.query({{.valid_from_ = unixtime_t{0h},
                   .duration_ = 5min,
                   .transport_mode_id_ = 5},
                  {.valid_from_ = sys_days{2024_y / June / 19} + 6h,
                   .duration_ = 60min,
                   .transport_mode_id_ = 5}},
                 {sys_days{2024_y / June / 19} + 5h,
                  sys_days{2024_y / June / 19} + 9h}));
}
