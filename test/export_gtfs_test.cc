#include "gtest/gtest.h"

#include <filesystem>
#include <format>
#include <fstream>
#include <set>
#include <sstream>
#include <string>

#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/export_gtfs.h"
#include "nigiri/special_stations.h"
#include "nigiri/timetable.h"

using namespace nigiri;
using namespace date;
using namespace std::string_view_literals;

namespace {

// - P is a parent station with platforms P1, P2
// - T2 serves the parent station P directly (-> needs a generated station)
// - names/headsigns/descriptions contain quotes, commas
// - T1 runs past midnight, has pickup/drop-off restrictions
constexpr auto const kFeed = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,"Deutsche Bahn, ""DB""",https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,location_type,parent_station,platform_code
P,Hauptbahnhof,,49.8728253,8.6511929,1,,
P1,"Hauptbahnhof, Gleis 1","Desc, with ""quotes""",49.8729123,8.6512456,0,P,1
P2,Hauptbahnhof Gleis 2,,49.8727001,8.6510002,0,P,2
A,"Stop ""A""",,49.9123456,8.7123456,0,,
B,B,,50.0,8.8,0,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_type,route_color,route_text_color
R1,DB,"S1, Express","Long ""Name"", with comma",109,FF0000,FFFFFF
R2,DB,Bus 2,,3,,

# trips.txt
route_id,service_id,trip_id,trip_headsign,trip_short_name
R1,WD,T1,"To ""B""",123
R2,ALL,T2,A,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type,stop_headsign
T1,10:00:00,10:00:00,P1,0,0,1,
T1,10:30:00,10:31:00,A,1,1,0,
T1,26:30:00,26:30:00,B,2,0,0,
T2,12:00:00,12:00:00,P,0,0,0,Via A
T2,12:10:00,12:10:00,A,1,0,0,Final
T2,12:20:00,12:20:00,B,2,0,0,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
WD,1,1,1,1,1,0,0,20190501,20190531
ALL,1,1,1,1,1,1,1,20190501,20190531

# calendar_dates.txt
service_id,date,exception_type
WD,20190530,2
WD,20190504,1
)"sv;

timetable load(loader::dir const& d) {
  auto tt = timetable{};
  tt.date_range_ = {sys_days{2019_y / May / 1}, sys_days{2019_y / June / 1}};
  loader::register_special_stations(tt);
  loader::gtfs::load_timetable({}, source_idx_t{0}, d, tt);
  loader::finalize(tt);
  return tt;
}

std::string_view root_name(timetable const& tt, location_idx_t const l) {
  return tt.get_default_name(tt.locations_.get_root_idx(l));
}

// Everything observable about the scheduled trips, independent of IDs.
std::set<std::string> trip_events(timetable const& tt) {
  auto ret = std::set<std::string>{};
  for (auto t = transport_idx_t{0}; t < tt.transport_traffic_days_.size();
       ++t) {
    auto const& bf = tt.bitfields_[tt.transport_traffic_days_[t]];
    auto const r = tt.transport_route_[t];
    auto const stops = tt.route_location_seq_[r];
    auto const directions = tt.transport_section_directions_[t];
    auto const trip = tt.merged_trips_[tt.transport_to_trip_section_[t][0]][0];
    auto const src = tt.trip_id_src_[tt.trip_ids_[trip][0]];
    auto const& route_ids = tt.route_ids_[src];
    auto const route = tt.trip_route_id_[trip];
    auto const provider = route_ids.route_id_provider_[route];
    for (auto d = 0U; d < bf.size(); ++d) {
      if (!bf.test(d)) {
        continue;
      }
      auto s = std::stringstream{};
      s << tt.get_default_translation(tt.providers_[provider].name_) << " | "
        << tt.get_default_translation(route_ids.route_id_short_names_[route])
        << " | "
        << tt.get_default_translation(route_ids.route_id_long_names_[route])
        << " | " << route_ids.route_id_type_[route] << " | "
        << to_str(route_ids.route_id_colors_[route].color_).value_or("-")
        << " | " << tt.get_default_translation(tt.trip_short_names_[trip])
        << "\n";
      for (auto i = stop_idx_t{0}; i != stops.size(); ++i) {
        auto const stp = stop{stops[i]};
        auto const l = stp.location_idx();
        auto const tr = transport{t, day_idx_t{d}};
        s << "  " << tt.get_default_name(l) << " [" << root_name(tt, l)
          << "] in=" << stp.in_allowed() << " out=" << stp.out_allowed();
        if (i != 0U) {
          s << " arr=" << tt.event_time(tr, i, event_type::kArr);
        }
        if (i != stops.size() - 1U) {
          s << " dep=" << tt.event_time(tr, i, event_type::kDep) << " dir="
            << (directions.empty()
                    ? ""
                    : tt.get_default_translation(
                          directions[directions.size() == 1U ? 0U : i]));
        }
        s << "\n";
      }
      ret.emplace(s.str());
    }
  }
  return ret;
}

std::set<std::string> stops(timetable const& tt) {
  auto ret = std::set<std::string>{};
  for (auto l = location_idx_t{kNSpecialStations}; l < tt.n_locations(); ++l) {
    auto const c = tt.locations_.coordinates_[l];
    ret.emplace(std::format(
        "{} [{}] ({}, {}) desc={} platform={}", tt.get_default_name(l),
        root_name(tt, l), c.lat_, c.lng_,
        tt.get_default_translation(tt.locations_.descriptions_[l]),
        tt.get_default_translation(tt.locations_.platform_codes_[l])));
  }
  return ret;
}

}  // namespace

TEST(export_gtfs, round_trip) {
  auto const tt = load(loader::mem_dir::read(kFeed));

  // loader: quoted fields are unescaped
  auto const p1 = tt.find(location_id{"P1", source_idx_t{0}});
  ASSERT_TRUE(p1.has_value());
  EXPECT_EQ(R"(Desc, with "quotes")",
            tt.get_default_translation(tt.locations_.descriptions_[*p1]));
  EXPECT_EQ(
      R"(Long "Name", with comma)",
      tt.get_default_translation(tt.route_ids_[source_idx_t{0}]
                                     .route_id_long_names_[route_id_idx_t{0}]));

  auto const out_dir =
      std::filesystem::temp_directory_path() / "nigiri_export_gtfs_test";
  std::filesystem::remove_all(out_dir);
  export_gtfs(tt, out_dir);

  auto const tt2 = load(loader::fs_dir{out_dir});

  auto const expected_events = trip_events(tt);
  EXPECT_FALSE(expected_events.empty());
  EXPECT_EQ(expected_events, trip_events(tt2));

  EXPECT_EQ(stops(tt), stops(tt2));

  // The served parent P gets an additional generated station entry.
  EXPECT_EQ(tt.n_locations() + 1U, tt2.n_locations());

  // Served locations must never be stations (location_type=1).
  for (auto r = route_idx_t{0}; r < tt2.n_routes(); ++r) {
    for (auto const s : tt2.route_location_seq_[r]) {
      EXPECT_TRUE(tt2.locations_.children_[stop{s}.location_idx()].empty());
    }
  }

  std::filesystem::remove_all(out_dir);
}
