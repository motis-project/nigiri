#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/routing/raptor/pong.h"
#include "nigiri/routing/raptor/raptor_state.h"
#include "nigiri/routing/search.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/rt/rt_timetable.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace std::chrono_literals;

namespace {

mem_dir test_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,0.02,1.03,,

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,RE 1,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R1,S1,T1,RE 1,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T1,10:00:00,10:00:00,A,1,0,0
T1,11:00:00,11:00:00,B,2,0,0
)");
}

}  // namespace

TEST(rt, delay_test) {
  auto const to_unix = [](auto&& x) {
    return std::chrono::time_point_cast<std::chrono::seconds>(x)
        .time_since_epoch()
        .count();
  };

  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2019_y / March / 25},
                    date::sys_days{2019_y / November / 1}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2019_y / May / 1});

  transit_realtime::FeedMessage msg;
  auto const hdr = msg.mutable_header();
  hdr->set_gtfs_realtime_version("2.0");
  hdr->set_incrementality(
      transit_realtime::FeedHeader_Incrementality_FULL_DATASET);
  hdr->set_timestamp(to_unix(date::sys_days{2019_y / May / 1} + 8h));

  auto const e = msg.add_entity();
  e->set_id("1");
  e->set_is_deleted(false);

  auto const td = e->mutable_trip_update()->mutable_trip();
  td->set_trip_id("T1");
  td->set_start_date("20190501");
  td->set_start_time("10:00:00");

  auto const stop_update_a = e->mutable_trip_update()->add_stop_time_update();
  stop_update_a->set_stop_sequence(1U);
  stop_update_a->mutable_departure()->set_delay(10 * 60);  // 10 minutes delay

  auto const stop_update_b = e->mutable_trip_update()->add_stop_time_update();
  stop_update_b->set_stop_sequence(2U);
  stop_update_b->mutable_arrival()->set_delay(0);  // on time

  auto const stats =
      rt::gtfsrt_update_msg(tt, rtt, source_idx_t{0}, "tag", msg);
  EXPECT_EQ(stats.total_entities_success_, 1U);
  rtt.update_lbs(tt);

  auto search_state = routing::search_state{};
  auto raptor_state = routing::raptor_state{};

  auto q = routing::query{
      .start_time_ = sys_days{2019_y / May / 1} + 7h,
      .start_match_mode_ = routing::location_match_mode::kEquivalent,
      .dest_match_mode_ = routing::location_match_mode::kEquivalent,
      .use_start_footpaths_ = true,
      .start_ = {{tt.locations_.location_id_to_idx_.at({"A", source_idx_t{0}}),
                  0_minutes, 0U}},
      .destination_ = {{tt.locations_.location_id_to_idx_.at(
                            {"B", source_idx_t{0}}),
                        0_minutes, 0U}},
      .max_transfers_ = 1,
      .min_connection_count_ = 1,
      .extend_interval_earlier_ = false,
      .extend_interval_later_ = false,
      .prf_idx_ = 0,
      .allowed_claszes_ = routing::all_clasz_allowed(),
      .require_bike_transport_ = false,
      .require_car_transport_ = false,
      .via_stops_ = {}};

  auto const result =
      routing::pong_search(tt, &rtt, search_state, raptor_state, std::move(q),
                           nigiri::direction::kForward);

  ASSERT_FALSE(result.journeys_->empty());
  auto const& journey = result.journeys_->begin();

  auto const expected_arrival = sys_days{2019_y / May / 1} + 9h;
  EXPECT_EQ(to_unix(journey->dest_time_), to_unix(expected_arrival));
}
