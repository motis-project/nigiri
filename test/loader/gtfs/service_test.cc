#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/rt/frun.h"
#include "nigiri/timetable.h"

#include "nigiri/lookup/get_transport.h"

using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace date;

namespace {

mem_dir test_files() {
  using std::filesystem::path;
  return {
      {{path{kAgencyFile},
        std::string{
            R"(agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin
)"}},
       {path{kStopFile},
        std::string{
            R"(stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,2.0,3.0,,
C,C,,4.0,5.0,,
D,D,,6.0,7.0,,
E,E,,8.0,9.0,,
F,F,,10.0,11.0,,
G,G,,12.0,13.0,,
H,H,,14.0,15.0,,
I,I,,16.0,17.0,,
)"}},
       {path{kCalendarDatesFile}, std::string{R"(service_id,date,exception_type
X,20190331,1
X,20191027,1
)"}},
       {path{kRoutesFile},
        std::string{
            R"(route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
A,DB,1337,Long Name,Route Description,3
B,DB,99,MisMatch,A route containing a stop without an entry in stops.txt,3
)"}},
       {path{kTripsFile},
        std::string{R"(route_id,service_id,trip_id,trip_headsign,block_id
A,X,X1,Trip X,
B,X,X2,Stop ID missing,
)"}},
       {path{kStopTimesFile},
        std::string{
            R"(trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
X1,00:00:00,00:00:00,A,1,0,0
X1,00:59:00,01:59:00,B,2,0,0
X1,02:00:00,02:01:00,C,3,0,0
X1,02:59:00,03:00:00,D,4,0,0
X1,03:30:00,03:30:00,E,5,0,0
X2,00:00:00,00:00:00,A,1,0,0
X2,00:30:00,00:30:00,missing,2,0,0
X2,01:00:00,01:00:00,D,3,0,0
)"}}}};
}

}  // namespace

TEST(gtfs, local_to_unix_trip_test) {
  timetable tt;
  tt.date_range_ = {date::sys_days{2019_y / March / 25},
                    date::sys_days{2019_y / November / 1}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);

  auto const unixtime = [&](transport const t, stop_idx_t const stop_idx,
                            event_type const ev_type) {
    return std::chrono::time_point_cast<std::chrono::seconds>(
               tt.event_time(t, stop_idx, ev_type))
        .time_since_epoch()
        .count();
  };

  auto const get_tz = [&](transport const t, stop_idx_t const stop_idx) {
    auto const fr = rt::frun::from_t(tt, nullptr, t);
    auto const tz_idx = fr[stop_idx].get_tz(nigiri::event_type::kDep);
    auto const tz = tt.timezones_[tz_idx];
    utl::verify(holds_alternative<pair<string, void const*>>(tz), "bad tz");
    return reinterpret_cast<date::time_zone const*>(
        tz.as<pair<string, void const*>>().second);
  };

  auto const iso = [&](transport const t, stop_idx_t const stop_idx,
                       event_type const ev_type) {
    return date::format(
        "%FT%R%Ez",
        zoned_time{get_tz(t, stop_idx), tt.event_time(t, stop_idx, ev_type)});
  };

  auto const t_oct = get_ref_transport(tt, trip_id{"X1", source_idx_t{0}},
                                       2019_y / October / 27, true);
  ASSERT_TRUE(t_oct.has_value());
  EXPECT_EQ(1572130800, unixtime(t_oct->first, 0, event_type::kDep));
  EXPECT_EQ(1572134340, unixtime(t_oct->first, 1, event_type::kArr));
  EXPECT_EQ(1572137940, unixtime(t_oct->first, 1, event_type::kDep));
  EXPECT_EQ(1572138000, unixtime(t_oct->first, 2, event_type::kArr));
  EXPECT_EQ(1572141540, unixtime(t_oct->first, 3, event_type::kArr));
  EXPECT_EQ(1572141600, unixtime(t_oct->first, 3, event_type::kDep));
  EXPECT_EQ("2019-10-27T01:00+02:00", iso(t_oct->first, 0, event_type::kDep));
  EXPECT_EQ("2019-10-27T01:59+02:00", iso(t_oct->first, 1, event_type::kArr));
  EXPECT_EQ("2019-10-27T02:59+02:00", iso(t_oct->first, 1, event_type::kDep));
  EXPECT_EQ("2019-10-27T02:00+01:00", iso(t_oct->first, 2, event_type::kArr));
  EXPECT_EQ("2019-10-27T02:59+01:00", iso(t_oct->first, 3, event_type::kArr));
  EXPECT_EQ("2019-10-27T03:00+01:00", iso(t_oct->first, 3, event_type::kDep));

  auto const t_march = get_ref_transport(tt, trip_id{"X1", source_idx_t{0}},
                                         2019_y / March / 30, true);
  ASSERT_TRUE(t_march.has_value());
  EXPECT_EQ(1553990400, unixtime(t_march->first, 2, event_type::kArr));
  EXPECT_EQ(1553993940, unixtime(t_march->first, 3, event_type::kArr));
  EXPECT_EQ(1553994000, unixtime(t_march->first, 3, event_type::kDep));
  EXPECT_EQ("2019-03-31T01:00+01:00", iso(t_march->first, 2, event_type::kArr));
  EXPECT_EQ("2019-03-31T01:59+01:00", iso(t_march->first, 3, event_type::kArr));
  EXPECT_EQ("2019-03-31T03:00+02:00", iso(t_march->first, 3, event_type::kDep));

  // Route with stop_id not contained in 'stops.txt'
  {
    auto const t_mm = get_ref_transport(tt, trip_id{"X2", source_idx_t{0}},
                                        2019_y / March / 30, true);
    ASSERT_TRUE(t_mm.has_value());
    constexpr auto kExpectedStops = 2U;
    auto const r = tt.transport_route_[t_mm->first.t_idx_];
    auto trip_count = 0U;

    EXPECT_EQ(kExpectedStops, tt.route_location_seq_[r].size());
    for (auto const transport_idx : tt.route_transport_ranges_[r]) {
      for (auto const& merged_trip_idx :
           tt.transport_to_trip_section_[transport_idx]) {
        for (auto const& trip_idx : tt.merged_trips_[merged_trip_idx]) {
          EXPECT_EQ(kExpectedStops, tt.trip_stop_seq_numbers_[trip_idx].size());
          ++trip_count;
        }
      }
    }
    EXPECT_TRUE(trip_count > 0U);
  }
}
