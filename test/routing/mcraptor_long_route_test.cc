#include "gtest/gtest.h"

#include <string>
#include <utility>
#include <variant>

#include "fmt/format.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/load.h"
#include "nigiri/routing/raptor/mcraptor.h"
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/routing/raptor_search.h"
#include "nigiri/timetable.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace std::chrono_literals;

// A route longer than 2047 stops (the limit of 11-bit stop positions).
namespace {

constexpr auto kStops = 2100U;

mem_dir test_files() {
  auto stops = std::string{
      "stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,"
      "parent_station\n"};
  auto stop_times = std::string{
      "trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,"
      "drop_off_type\n"};
  for (auto i = 0U; i != kStops; ++i) {
    stops += fmt::format("S{},S{},,{:.2f},0.0,,\n", i, i, i * 0.01);
    auto const t = fmt::format("{:02}:{:02}:00", 6U + i / 600U, i / 10U % 60U);
    stop_times += fmt::format("T,{},{},S{},{},0,0\n", t, t, i, i + 1U);
  }
  return mem_dir::read(fmt::format(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
{}
# calendar_dates.txt
service_id,date,exception_type
S,20240619,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R,DB,RE 1,,,2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R,S,T,RE 1,

# stop_times.txt
{})",
                                   stops, stop_times));
}

template <typename AlgoState>
pareto_set<routing::journey> search(timetable const& tt,
                                    stop_idx_t const from,
                                    stop_idx_t const to) {
  auto const loc = [&](stop_idx_t const i) {
    return tt.find({fmt::format("S{}", i), {}}).value();
  };
  auto q = routing::query{};
  q.start_time_ = unixtime_t{sys_days{2024_y / June / 19} + 3h};
  q.start_ = {{loc(from), 0min, 0U}};
  q.destination_ = {{loc(to), 0min, 0U}};
  auto ss = routing::search_state{};
  auto as = AlgoState{};
  return *routing::raptor_search(tt, nullptr, ss, as, std::move(q),
                                 direction::kForward)
              .journeys_;
}

void expect_one_ride(timetable const& tt,
                     pareto_set<routing::journey> const& js,
                     stop_idx_t const enter,
                     stop_idx_t const exit) {
  ASSERT_EQ(1U, js.size());
  auto const& j = *begin(js);
  ASSERT_EQ(1U, j.legs_.size());
  auto const* ree =
      std::get_if<routing::journey::run_enter_exit>(&j.legs_.front().uses_);
  ASSERT_NE(nullptr, ree);
  EXPECT_EQ(enter, ree->stop_range_.from_);
  EXPECT_EQ(exit + 1U, ree->stop_range_.to_);
  EXPECT_EQ(tt.find({fmt::format("S{}", exit), {}}).value(), j.dest_);
}

}  // namespace

TEST(mcraptor, long_route) {
  auto const tt =
      loader::load(test_files(), {2024_y / June / 18, 2024_y / June / 20});

  // alight beyond, then board and alight beyond the 11-bit range
  using stops_t = std::pair<stop_idx_t, stop_idx_t>;
  for (auto const& [from, to] :
       {stops_t{0U, kStops - 1U}, stops_t{2050U, kStops - 1U}}) {
    auto const raptor = search<routing::raptor_state>(tt, from, to);
    expect_one_ride(tt, raptor, from, to);
    expect_one_ride(tt, search<routing::mcraptor_state>(tt, from, to), from,
                    to);
    expect_one_ride(
        tt,
        search<routing::mcraptor_non_transit_mode_switches_state>(tt, from, to),
        from, to);
  }
}
