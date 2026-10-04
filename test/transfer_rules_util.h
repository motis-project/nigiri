#pragma once

#include <algorithm>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "gtest/gtest.h"

#include "date/date.h"

#include "fmt/format.h"

#include "utl/erase_duplicates.h"
#include "utl/helpers/algorithm.h"

#include "nigiri/loader/build_footpaths.h"
#include "nigiri/loader/build_lb_graph.h"
#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/common/parse_time.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/query.h"
#include "nigiri/timetable.h"

#include "./raptor_search.h"

// Helpers of the transfers.txt rule tests. Their feeds run on 2019-05-01 in
// Europe/Berlin.

namespace nigiri::test {

inline constexpr auto const kDay =
    date::sys_days{date::year{2019} / date::May / 1};

// "2019-05-01 10:30 Europe/Berlin"
inline unixtime_t t(char const* s) {
  return parse_time_tz(s, "%Y-%m-%d %H:%M %Z");
}

// "10:30" on kDay in Europe/Berlin.
inline unixtime_t at(std::string_view const hhmm) {
  return parse_time_tz(fmt::format("2019-05-01 {} Europe/Berlin", hhmm),
                       "%Y-%m-%d %H:%M %Z");
}

inline location_idx_t lidx(timetable const& tt,
                           std::string_view const id,
                           source_idx_t const src = source_idx_t{0U}) {
  return tt.locations_.location_id_to_idx_.at({id, src});
}

// Loads the feeds (one source each), valid on kDay.
inline timetable load_feeds(std::vector<std::string> const& feeds,
                            bool const adjust_footpaths = false) {
  auto tt = timetable{};
  tt.date_range_ = {kDay, kDay + date::days{1}};
  loader::register_special_stations(tt);
  for (auto i = 0U; i != feeds.size(); ++i) {
    loader::gtfs::load_timetable(
        {.adjust_footpaths_ = adjust_footpaths},
        source_idx_t{static_cast<source_idx_t::value_t>(i)},
        loader::mem_dir::read(feeds[i]), tt);
  }
  loader::finalize(tt);
  return tt;
}

// S is a station with the child stops S1..S4 a few meters apart (mutually
// equivalent -> valid track changes). Default transfer time at a stop: 2 min,
// S1 <-> S2 walk: a few minutes.
//
//   F  (RF): A  10:00 -> S1 10:30         feeder
//   F0 (RF): A2 09:00 -> S3 09:30         a second RF trip, scheduled at S3
//   G  (RG): S2 10:40 -> B 11:00          10 min after F: fine by default
//   GL (RG): S2 11:10 -> B 11:30          fallback if G cannot be reached
//   H  (RH): S1 10:31 -> C 11:00          1 min after F: only with a rule
//   HL (RH): S1 11:01 -> C 11:30          fallback if H cannot be reached
//   F2 (RF): S1 10:36 -> D 11:06          a second RF trip meeting F at S1
//
// P and Q meet twice, at S and at the station X (child stops X1, X2):
//   P  (RP): A  12:00 -> S1 12:30 -> X1 12:50
//   Q  (RQ): S2 12:40 -> X2 13:00 -> B 13:20
//
// {0}..{3}: further stops, routes, trips and stop times (network_rows), {4}:
// the transfers.txt rows.
inline constexpr auto const kNetwork = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
AG,Agency,https://example.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,50.0,6.0,,,
A2,A2,,50.0,6.1,,,
S,S,,50.0,6.5,,1,
S1,S1,,50.0001,6.5,,,S
S2,S2,,50.0002,6.5,,,S
S3,S3,,50.0003,6.5,,,S
S4,S4,,50.0004,6.5,,,S
B,B,,50.0,7.0,,,
C,C,,50.0,7.5,,,
D,D,,50.0,8.0,,,
X,X,,50.0,6.8,,1,
X1,X1,,50.0001,6.8,,,X
X2,X2,,50.0002,6.8,,,X
{0}
# calendar_dates.txt
service_id,date,exception_type
X,20190501,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
RF,AG,RF,,,3
RG,AG,RG,,,3
RH,AG,RH,,,3
RP,AG,RP,,,3
RQ,AG,RQ,,,3
{1}
# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
RF,X,F,,
RF,X,F0,,
RF,X,F2,,
RG,X,G,,
RG,X,GL,,
RH,X,H,,
RH,X,HL,,
RP,X,P,,
RQ,X,Q,,
{2}
# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
F,10:00:00,10:00:00,A,1,0,0
F,10:30:00,10:30:00,S1,2,0,0
F0,09:00:00,09:00:00,A2,1,0,0
F0,09:30:00,09:30:00,S3,2,0,0
F2,10:36:00,10:36:00,S1,1,0,0
F2,11:06:00,11:06:00,D,2,0,0
G,10:40:00,10:40:00,S2,1,0,0
G,11:00:00,11:00:00,B,2,0,0
GL,11:10:00,11:10:00,S2,1,0,0
GL,11:30:00,11:30:00,B,2,0,0
H,10:31:00,10:31:00,S1,1,0,0
H,11:00:00,11:00:00,C,2,0,0
HL,11:01:00,11:01:00,S1,1,0,0
HL,11:30:00,11:30:00,C,2,0,0
P,12:00:00,12:00:00,A,1,0,0
P,12:30:00,12:30:00,S1,2,0,0
P,12:50:00,12:50:00,X1,3,0,0
Q,12:40:00,12:40:00,S2,1,0,0
Q,13:00:00,13:00:00,X2,2,0,0
Q,13:20:00,13:20:00,B,3,0,0
{3}
# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time,from_route_id,to_route_id,from_trip_id,to_trip_id
{4}
)";

// Rows added to kNetwork, in its column layout, each ending in a newline.
struct network_rows {
  std::string_view stops_{};
  std::string_view routes_{};
  std::string_view trips_{};
  std::string_view stop_times_{};
};

// kNetwork with the given transfers.txt rows (and further rows).
inline std::string network(std::string_view const transfers,
                           network_rows const& rows = {}) {
  return fmt::format(fmt::runtime(kNetwork), rows.stops_, rows.routes_,
                     rows.trips_, rows.stop_times_, transfers);
}

inline timetable load_network(std::string_view const transfers,
                              network_rows const& rows = {}) {
  return load_feeds({network(transfers, rows)});
}

// Stop-to-stop query the way a server asks it: a stop stands for its station,
// so everything below it (child stops, virtual locations) is a start / target.
inline routing::query station_query(timetable const& tt,
                                    std::string_view const from,
                                    std::string_view const to,
                                    routing::start_time_t const time) {
  return routing::query{
      .start_time_ = time,
      .start_match_mode_ = routing::location_match_mode::kEquivalent,
      .dest_match_mode_ = routing::location_match_mode::kEquivalent,
      .start_ = {{lidx(tt, from), 0_minutes, 0U}},
      .destination_ = {{lidx(tt, to), 0_minutes, 0U}}};
}

// Sets up the profile prf without any walks, next to the loaded ones: a
// profile that ignores transfers.txt.
inline void add_empty_profile(timetable& tt, profile_idx_t const prf) {
  tt.locations_.footpaths_out_[prf].resize(tt.n_locations());
  tt.locations_.footpaths_in_[prf].resize(tt.n_locations());
  loader::build_lb_graph<direction::kForward>(tt, prf);
  loader::build_lb_graph<direction::kBackward>(tt, prf);
}

// A feed built from its stops, trips and transfers.txt rows.
struct stop_def {
  std::string_view id_;
  double lat_;
  double lng_;
  std::string_view parent_{};
  bool is_station_{false};
};

struct trip_def {
  std::string_view id_;
  std::string_view route_;
  std::vector<std::pair<std::string_view, std::string_view>> times_;
  std::string_view block_{};
};

// Routes are taken from the trips, plus the ones rules name without a trip.
inline std::string feed(
    std::vector<stop_def> const& stops,
    std::vector<trip_def> const& trips,
    std::string_view const transfers,
    std::vector<std::string_view> const& extra_routes = {}) {
  auto s = std::string{
      "# agency.txt\n"
      "agency_id,agency_name,agency_url,agency_timezone\n"
      "AG,Agency,https://example.com,Europe/Berlin\n\n"
      "# calendar_dates.txt\n"
      "service_id,date,exception_type\n"
      "S1,20190501,1\n\n"
      "# stops.txt\n"
      "stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station\n"};
  for (auto const& x : stops) {
    s += fmt::format("{},{} stop,{},{},{},{}\n", x.id_, x.id_, x.lat_, x.lng_,
                     x.is_station_ ? 1 : 0, x.parent_);
  }

  auto routes = extra_routes;
  for (auto const& x : trips) {
    routes.push_back(x.route_);
  }
  utl::erase_duplicates(routes);
  s += "\n# routes.txt\n"
       "route_id,agency_id,route_short_name,route_long_name,route_type\n";
  for (auto const r : routes) {
    s += fmt::format("{},AG,{},,3\n", r, r);
  }

  s += "\n# trips.txt\nroute_id,service_id,trip_id,block_id\n";
  for (auto const& x : trips) {
    s += fmt::format("{},S1,{},{}\n", x.route_, x.id_, x.block_);
  }

  s += "\n# stop_times.txt\n"
       "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n";
  for (auto const& x : trips) {
    auto seq = 0U;
    for (auto const& [sid, hhmm] : x.times_) {
      s += fmt::format("{},{}:00,{}:00,{},{}\n", x.id_, hhmm, hhmm, sid, seq++);
    }
  }

  s += "\n# transfers.txt\n"
       "from_stop_id,to_stop_id,transfer_type,min_transfer_time,"
       "from_route_id,to_route_id,from_trip_id,to_trip_id\n";
  s += transfers;
  return s;
}

// Stop-to-stop search, departing at hhmm on kDay.
inline pareto_set<routing::journey> search_at(timetable const& tt,
                                              std::string_view const from,
                                              std::string_view const to,
                                              std::string_view const hhmm) {
  return raptor_search(tt, nullptr, from, to,
                       fmt::format("2019-05-01 {} Europe/Berlin", hhmm));
}

inline std::size_t n_transit_legs(routing::journey const& j) {
  return static_cast<std::size_t>(
      utl::count_if(j.legs_, [](routing::journey::leg const& l) {
        return std::holds_alternative<routing::journey::run_enter_exit>(
            l.uses_);
      }));
}

// The arrival of the journey of a station query from od.first to od.second
// (at most one expected), nullopt if there is none.
inline std::optional<unixtime_t> arrival(
    timetable const& tt,
    rt_timetable const* rtt,
    std::pair<char const*, char const*> const& od,
    char const* start = "2019-05-01 10:00 Europe/Berlin") {
  auto const res =
      raptor_search(tt, rtt, station_query(tt, od.first, od.second, t(start)),
                    direction::kForward);
  EXPECT_LE(res.size(), 1U);
  return res.size() == 0U ? std::nullopt
                          : std::optional{begin(res)->dest_time_};
}

inline std::size_t n_virts(timetable const& tt) {
  return static_cast<std::size_t>(
      std::ranges::count(tt.locations_.types_, location_type::kVirt));
}

// A walk as a street router hands it over: between two stops, both ways.
struct walk {
  std::string_view a_;
  std::string_view b_;
  int minutes_;
};

// Replaces the walks of the default profile like motis does with the routed
// walks (osr_footpath): never shorter than the transfer time at either end,
// then the profile is written again.
inline void rebuild_default_profile(timetable& tt,
                                    std::vector<walk> const& walks) {
  auto& pending = tt.locations_.preprocessing_footpaths_out_;
  pending.clear();
  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    pending.emplace_back();
  }
  for (auto const& w : walks) {
    auto const a = lidx(tt, w.a_);
    auto const b = lidx(tt, w.b_);
    auto const d = duration_t{w.minutes_};
    pending[a].push_back(
        footpath{b, loader::max_with_transfer_times(tt, a, b, d)});
    pending[b].push_back(
        footpath{a, loader::max_with_transfer_times(tt, b, a, d)});
  }
  loader::write_default_profile(tt);
}

}  // namespace nigiri::test
