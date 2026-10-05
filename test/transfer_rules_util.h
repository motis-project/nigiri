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
#include "nigiri/routing/transfers.h"
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
    s += fmt::format("{},{},{},{},{},{}\n", x.id_, x.id_, x.lat_, x.lng_,
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
    auto seq = 1U;
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
inline std::vector<stop_def> network_stops() {
  return {{"A", 50.0, 6.0},           {"A2", 50.0, 6.1},
          {"S", 50.0, 6.5, "", true}, {"S1", 50.0001, 6.5, "S"},
          {"S2", 50.0002, 6.5, "S"},  {"S3", 50.0003, 6.5, "S"},
          {"S4", 50.0004, 6.5, "S"},  {"B", 50.0, 7.0},
          {"C", 50.0, 7.5},           {"D", 50.0, 8.0},
          {"X", 50.0, 6.8, "", true}, {"X1", 50.0001, 6.8, "X"},
          {"X2", 50.0002, 6.8, "X"}};
}

inline std::vector<trip_def> network_trips() {
  return {{"F", "RF", {{"A", "10:00"}, {"S1", "10:30"}}},
          {"F0", "RF", {{"A2", "09:00"}, {"S3", "09:30"}}},
          {"F2", "RF", {{"S1", "10:36"}, {"D", "11:06"}}},
          {"G", "RG", {{"S2", "10:40"}, {"B", "11:00"}}},
          {"GL", "RG", {{"S2", "11:10"}, {"B", "11:30"}}},
          {"H", "RH", {{"S1", "10:31"}, {"C", "11:00"}}},
          {"HL", "RH", {{"S1", "11:01"}, {"C", "11:30"}}},
          {"P", "RP", {{"A", "12:00"}, {"S1", "12:30"}, {"X1", "12:50"}}},
          {"Q", "RQ", {{"S2", "12:40"}, {"X2", "13:00"}, {"B", "13:20"}}}};
}

// Stops and trips added to the network.
struct network_rows {
  std::vector<stop_def> stops_{};
  std::vector<trip_def> trips_{};
};

// The network with the given transfers.txt rows (and further stops and trips).
inline std::string network(std::string_view const transfers,
                           network_rows const& rows = {}) {
  auto stops = network_stops();
  stops.insert(end(stops), begin(rows.stops_), end(rows.stops_));
  auto trips = network_trips();
  trips.insert(end(trips), begin(rows.trips_), end(rows.trips_));
  return feed(stops, trips, transfers);
}

inline timetable load_network(std::string_view const transfers,
                              network_rows const& rows = {}) {
  return load_feeds({network(transfers, rows)});
}

// The network's arrivals of A -F-> S -G-> B, A -F-> S -GL-> B, A -F-> S -H-> C
// and A -F-> S -HL-> C. Functions, not constants: at static init time the time
// zone database is not loaded yet.
inline unixtime_t g_from_f() { return at("11:00"); }
inline unixtime_t gl_from_f() { return at("11:30"); }
inline unixtime_t h_from_f() { return at("11:00"); }
inline unixtime_t hl_from_f() { return at("11:30"); }

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

// The arrival of the only journey in res (at most one expected), nullopt if
// there is none.
inline std::optional<unixtime_t> arrival(
    pareto_set<routing::journey> const& res) {
  EXPECT_LE(res.size(), 1U);
  return res.size() == 0U ? std::nullopt
                          : std::optional{begin(res)->dest_time_};
}

// The arrival of the stop-to-stop search from -> to, departing at hhmm.
inline std::optional<unixtime_t> arrival_at(timetable const& tt,
                                            std::string_view const from,
                                            std::string_view const to,
                                            std::string_view const hhmm) {
  return arrival(search_at(tt, from, to, hhmm));
}

// The arrival of the station query from od.first to od.second, departing at
// hhmm.
inline std::optional<unixtime_t> arrival(
    timetable const& tt,
    rt_timetable const* rtt,
    std::pair<char const*, char const*> const& od,
    std::string_view const hhmm = "10:00") {
  return arrival(raptor_search(tt, rtt,
                               station_query(tt, od.first, od.second, at(hhmm)),
                               direction::kForward));
}

// The fastest transfer from -> to in the default profile (footpaths and hubs),
// nullopt if there is none.
inline std::optional<duration_t> transfer_duration(timetable const& tt,
                                                   location_idx_t const from,
                                                   location_idx_t const to) {
  auto d = std::optional<duration_t>{};
  routing::for_each_transfer<direction::kForward>(
      tt, nullptr, kDefaultProfile, from, [&](footpath const fp) {
        if (fp.target() == to && (!d.has_value() || fp.duration() < *d)) {
          d = fp.duration();
        }
      });
  return d;
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
