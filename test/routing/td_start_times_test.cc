#include "gtest/gtest.h"

#include "utl/equal_ranges_linear.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/routing/start_times.h"
#include "nigiri/timetable.h"

using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace nigiri::routing;
using namespace date;
using namespace std::chrono_literals;

namespace {

mem_dir test_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
X,X,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,2.0,3.0,,

# calendar_dates.txt
service_id,date,exception_type
X,20200330,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
X,X,X,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
X,X,X,X,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
X,00:00:00,00:00:00,A,1,0,0
X,00:10:00,00:10:00,B,2,0,0

# frequencies.txt
trip_id,start_time,end_time,headway_secs
X,00:00:00,24:00:00,3600
)");
}

constexpr auto expected = R"(
start_time=2020-03-30 11:59
      {time_at_start=2020-03-30 11:59, time_at_stop=2020-03-30 13:00, stop=A}
start_time=2020-03-30 11:50
      {time_at_start=2020-03-30 11:50, time_at_stop=2020-03-30 12:00, stop=A}
start_time=2020-03-30 10:50
      {time_at_start=2020-03-30 10:50, time_at_stop=2020-03-30 11:00, stop=A}
)";

}  // namespace

TEST(routing, td_start_times) {
  auto const src = source_idx_t{0U};
  auto tt = timetable{};
  tt.date_range_ =
      interval{sys_days{2020_y / March / 30}, sys_days{2020_y / March / 32}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  using namespace date;
  auto const A = tt.locations_.location_id_to_idx_.at(
      location_id{.id_ = "A", .src_ = src});
  auto starts = std::vector<start>{};
  get_starts(
      direction::kForward, tt, nullptr,
      interval<unixtime_t>{sys_days{2020_y / March / 30},
                           sys_days{2020_y / March / 31}},
      {},
      hash_map<location_idx_t, std::vector<td_offset>>{
          {std::pair{A,
                     std::vector<td_offset>{
                         {.valid_from_ = sys_days{1970_y / January / 1},
                          .duration_ = footpath::kMaxDuration,
                          .transport_mode_id_ = 0U},
                         {.valid_from_ = sys_days{2020_y / March / 30} + 10h,
                          .duration_ = 10min,
                          .transport_mode_id_ = 0U},
                         {.valid_from_ = sys_days{2020_y / March / 30} + 12h,
                          .duration_ = footpath::kMaxDuration,
                          .transport_mode_id_ = 0U}}}}},
      kMaxTravelTime, location_match_mode::kExact, false, starts, true, 0U, {});
  std::sort(begin(starts), end(starts),
            [](auto&& a, auto&& b) { return a > b; });
  starts.erase(std::unique(begin(starts), end(starts)), end(starts));

  std::stringstream ss;
  ss << "\n";
  utl::equal_ranges_linear(
      starts,
      [](start const& a, start const& b) {
        return a.time_at_start_ == b.time_at_start_;
      },
      [&](std::vector<start>::const_iterator const& from_it,
          std::vector<start>::const_iterator const& to_it) {
        ss << "start_time=" << from_it->time_at_start_ << "\n";
        for (auto const& s : it_range{from_it, to_it}) {
          ss << "      {time_at_start=" << s.time_at_start_
             << ", time_at_stop=" << s.time_at_stop_
             << ", stop=" << tt.locations_.names_[s.stop_].view() << "}\n";
        }
      });

  EXPECT_EQ(std::string_view{expected}, ss.str());
}
