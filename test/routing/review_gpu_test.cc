#include "gtest/gtest.h"

#include <string>

#include "nigiri/loader/build_lb_graph.h"
#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/common/parse_time.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor_search.h"
#include "nigiri/routing/search.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

#include "nigiri/routing/gpu/raptor.h"

#include "../raptor_search.h"

// Regression tests for GPU defects found in the t2t-rt review. They only exist
// in CUDA builds. Each compares the device search with the host search on the
// same query: a difference is the defect.

#if defined(NIGIRI_CUDA)

using namespace nigiri;
using namespace date;
using namespace std::chrono_literals;

namespace {

constexpr auto const kProfile = profile_idx_t{1U};

timetable load(std::string const& transfers,
               std::string const& extra_stops,
               std::string const& trips,
               std::string const& stop_times) {
  auto const gtfs =
      std::string{R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
AG,Agency,https://example.com,Europe/Berlin

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_type
R1,AG,R1,,3
R2,AG,R2,,3
R5,AG,R5,,3
R9,AG,R9,,3

# stops.txt
stop_id,stop_name,stop_lat,stop_lon
)"} + extra_stops +
      "\n# trips.txt\nroute_id,service_id,trip_id\n" + trips +
      "\n# stop_times.txt\n"
      "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n" +
      stop_times +
      "\n# transfers.txt\n"
      "from_stop_id,to_stop_id,transfer_type,min_transfer_time,"
      "from_route_id,to_route_id,from_trip_id,to_trip_id\n" +
      transfers;
  auto tt = timetable{};
  tt.date_range_ = {date::sys_days{2019_y / May / 1},
                    date::sys_days{2019_y / May / 2}};
  loader::register_special_stations(tt);
  loader::gtfs::load_timetable({}, source_idx_t{0}, loader::mem_dir::read(gtfs),
                               tt);
  loader::finalize(tt);

  // a second profile without walks, as a routed profile would be
  tt.locations_.footpaths_out_[kProfile].resize(tt.n_locations());
  tt.locations_.footpaths_in_[kProfile].resize(tt.n_locations());
  loader::build_lb_graph<direction::kForward>(tt, kProfile);
  loader::build_lb_graph<direction::kBackward>(tt, kProfile);
  return tt;
}

unixtime_t at(char const* s) {
  return parse_time_tz(std::string{"2019-05-01 "} + s + " Europe/Berlin",
                       "%Y-%m-%d %H:%M %Z");
}

location_idx_t lidx(timetable const& tt, char const* id) {
  return tt.locations_.location_id_to_idx_.at({id, source_idx_t{0}});
}

// host and device result of the same query, printed
std::pair<std::string, std::string> cpu_and_gpu(timetable const& tt,
                                                rt_timetable const* rtt,
                                                routing::query const& q) {
  auto cpu_search_state = routing::search_state{};
  auto cpu_state = routing::raptor_state{};
  auto const cpu = *routing::raptor_search(tt, rtt, cpu_search_state, cpu_state,
                                           q, direction::kForward)
                        .journeys_;

  auto gpu_search_state = routing::search_state{};
  auto gpu_tt = routing::gpu::gpu_timetable{tt};
  auto gpu_state = routing::gpu::gpu_raptor_state{gpu_tt};
  auto const gpu = *routing::raptor_search(tt, rtt, gpu_search_state, gpu_state,
                                           q, direction::kForward)
                        .journeys_;
  return {test::print_results(tt, rtt, cpu), test::print_results(tt, rtt, gpu)};
}

}  // namespace

// Changing vehicles at GX is banned by an unqualified same-stop row, and no
// rule is qualified, so the timetable has no virtual location. A profile that
// ignores transfers.txt changes at GX after the stop's base change time - on
// the host as on the device.
TEST(t2t_review_gpu, other_profile_without_virtual_locations) {
  auto const tt = load("GX,GX,3,,,,,\n",
                       "GX,GX,65.0,24.0\nGA,GA,65.1,24.0\nGB,GB,65.2,24.0\n",
                       "R1,S1,GT1\nR2,S1,GT2\n",
                       "GT1,10:00:00,10:00:00,GA,0\n"
                       "GT1,10:30:00,10:30:00,GX,1\n"
                       "GT2,10:40:00,10:40:00,GX,0\n"
                       "GT2,11:00:00,11:00:00,GB,1\n");
  auto const [cpu, gpu] = cpu_and_gpu(
      tt, nullptr,
      routing::query{.start_time_ = at("10:00"),
                     .start_ = {{lidx(tt, "GA"), 0_minutes, 0U}},
                     .destination_ = {{lidx(tt, "GB"), 0_minutes, 0U}},
                     .prf_idx_ = kProfile});
  EXPECT_EQ(cpu, gpu);
}

// W2's own change time is 0 (timed row), and the R9 -> R5 rule splits R5's
// departures off, so the start W2 reaches GD's virtual location through W2's
// 0 min hub. With a 5 min minimum transfer time the host prices that hub at
// 0 min in the search and in the start leg; the device's start leg must too.
TEST(t2t_review_gpu, start_leg_through_zero_min_hub) {
  auto const tt = load("W2,W2,1,,,,,\nW2,W2,2,300,R9,R5,,\n",
                       "W2,W2,65.5,24.5\nWD,WD,65.6,24.5\n", "R5,S1,GD\n",
                       "GD,10:02:00,10:02:00,W2,0\n"
                       "GD,10:30:00,10:30:00,WD,1\n");
  auto const [cpu, gpu] = cpu_and_gpu(
      tt, nullptr,
      routing::query{.start_time_ = at("10:00"),
                     .use_start_footpaths_ = true,
                     .start_ = {{lidx(tt, "W2"), 0_minutes, 0U}},
                     .destination_ = {{lidx(tt, "WD"), 0_minutes, 0U}},
                     .transfer_time_settings_ = {
                         .default_ = false, .min_transfer_time_ = 5_minutes}});
  EXPECT_EQ(cpu, gpu);
}

// A non-default profile with real-time time-dependent footpaths (an elevator
// outage at GX) makes the device pong fill its bounds from the td bit vector.
// The kernel walks the real-time label slots too, past the end of that bit
// vector. Run under compute-sanitizer to see the invalid read; the results
// themselves are expected to match.
TEST(t2t_review_gpu, fill_bounds_with_td_footpaths) {
  auto const tt =
      load("", "GX,GX,65.0,24.0\nGA,GA,65.1,24.0\nGB,GB,65.2,24.0\n",
           "R1,S1,GT1\nR2,S1,GT2\n",
           "GT1,10:00:00,10:00:00,GA,0\n"
           "GT1,10:30:00,10:30:00,GX,1\n"
           "GT2,10:40:00,10:40:00,GX,0\n"
           "GT2,11:00:00,11:00:00,GB,1\n");
  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2019_y / May / 1});
  auto const gx = lidx(tt, "GX");
  rtt.has_td_footpaths_out_[kProfile].set(gx, true);
  rtt.has_td_footpaths_in_[kProfile].set(gx, true);
  rtt.td_footpaths_out_[kProfile].resize(tt.n_locations());
  rtt.td_footpaths_in_[kProfile].resize(tt.n_locations());

  auto const res = test::raptor_search(
      tt, &rtt,
      routing::query{.start_time_ = interval{at("09:30"), at("10:30")},
                     .start_ = {{lidx(tt, "GA"), 0_minutes, 0U}},
                     .destination_ = {{lidx(tt, "GB"), 0_minutes, 0U}},
                     .prf_idx_ = kProfile},
      direction::kForward);
  EXPECT_EQ(1U, res.size());
}

#endif
