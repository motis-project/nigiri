#include "gtest/gtest.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"

#include "nigiri/routing/raptor/pong.h"
#include "nigiri/special_stations.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace std::chrono_literals;

namespace {

// R1: A -> B, on Monday (T1) and Friday (T4).
// R2: B -> C, Friday only (T2).
mem_dir test_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Etc/UTC

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,2.0,3.0,,
C,C,,4.0,5.0,,

# calendar_dates.txt
service_id,date,exception_type
MON,20240617,1
FRI,20240621,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,RE 1,,,2
R2,DB,RE 2,,,2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R1,MON,T1,RE 1,
R1,FRI,T4,RE 1,
R2,FRI,T2,RE 2,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T1,10:00:00,10:00:00,A,1,0,0
T1,10:30:00,10:30:00,B,2,0,0
T4,09:00:00,09:00:00,A,1,0,0
T4,09:30:00,09:30:00,B,2,0,0
T2,11:00:00,11:00:00,B,1,0,0
T2,11:30:00,11:30:00,C,2,0,0
)");
}

}  // namespace

// td offsets break RAPTOR's FIFO assumption: a td (flex) access/egress
// window is only usable from labels within kMaxTransferTime of it. The
// backward pong reaches A both via T4 (Friday, later = "better" backward)
// and via T1 (Monday). Per-stop domination keeps only Friday - but the
// only flex access window at A is Monday morning, unreachable from the
// Friday label. The ping finds the journey regardless (td starts are
// expanded into per-window start events, which sidestep domination), so
// the pong probe is unmatchable: "no pong for transfers=1" is thrown and
// production falls back to plain RAPTOR for the whole query.
//
// Fixed by bounding td waits with kMaxTravelTime (like transit waits)
// instead of kMaxTransferTime: horizon-bounded waits make window
// reachability monotone in the label time, so per-stop domination is
// sound again. The expected departure is 09:59 (flex at the window's
// last minute + wait at A + Friday trip) - later than the Monday-trip
// shape the ping finds first.
TEST(routing, pong_td_egress_dominated_label) {
  timetable tt;
  tt.date_range_ = {sys_days{2024_y / June / 16}, sys_days{2024_y / June / 23}};
  register_special_stations(tt);
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  auto const A = tt.find(location_id{"A", source_idx_t{0}}).value();
  auto const C = tt.find(location_id{"C", source_idx_t{0}}).value();

  auto const monday = sys_days{2024_y / June / 17};
  auto q = routing::query{
      .start_time_ = interval<unixtime_t>{monday + 9h, monday + 10h},
      .start_match_mode_ = routing::location_match_mode::kIntermodal,
      .dest_match_mode_ = routing::location_match_mode::kIntermodal,
      .use_start_footpaths_ = false,
      .destination_ = {{C, 0min, 0U}},
      .td_start_ = {{A,
                     {{.valid_from_ = monday + 9h + 50min,
                       .duration_ = 5min,
                       .transport_mode_id_ = 5},
                      {.valid_from_ = monday + 10h,
                       .duration_ = footpath::kMaxDuration,
                       .transport_mode_id_ = 5}}}},
      .min_connection_count_ = 1U};

  auto search_state = routing::search_state{};
  auto raptor_state = routing::raptor_state{};

  // without horizon-bounded td waits this throws "no pong for
  // transfers=1": the journey exists (the ping finds it via the Monday
  // start window) but the pong cannot reproduce it from its dominated
  // Friday label at A.
  auto result = routing::pong_search(tt, nullptr, search_state, raptor_state,
                                     std::move(q), direction::kForward);

  auto found = false;
  auto got = std::string{};
  for (auto const& j : *result.journeys_) {
    found |= j.start_time_ == monday + 9h + 59min &&
             j.dest_time_ == sys_days{2024_y / June / 21} + 11h + 30min &&
             j.transfers_ == 1U;
    got += fmt::format("({}, {}, {})\n", j.start_time_, j.dest_time_,
                       j.transfers_);
  }
  EXPECT_TRUE(found) << "multi-day flex journey lost, got:\n" << got;
}
