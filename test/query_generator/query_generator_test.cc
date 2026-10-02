#include "gtest/gtest.h"

#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/query_generator/generator.h"
#include "nigiri/query_generator/generator_settings.h"
#include "nigiri/query_generator/transport_mode.h"
#include "nigiri/routing/query.h"

#include "geo/box.h"
#include "geo/latlng.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/load_timetable.h"

#include "../loader/hrd/hrd_timetable.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::test_data::hrd_timetable;
using namespace nigiri::query_generation;
using namespace std::string_view_literals;

TEST(query_generation, pretrip_station) {
  constexpr auto const src = source_idx_t{0U};
  timetable tt;
  tt.date_range_ = full_period();
  register_special_stations(tt);
  load_timetable(src, loader::hrd::hrd_5_20_26, files_abc(), tt);
  finalize(tt);

  generator_settings gs;
  gs.start_match_mode_ = routing::location_match_mode::kEquivalent;
  gs.dest_match_mode_ = routing::location_match_mode::kEquivalent;

  auto qg = generator{tt, gs};

  auto const sdq = qg.random_query();
  ASSERT_TRUE(sdq.has_value());
}

TEST(query_generation, pretrip_intermodal) {
  constexpr auto const src = source_idx_t{0U};
  timetable tt;
  tt.date_range_ = full_period();
  register_special_stations(tt);
  load_timetable(src, loader::hrd::hrd_5_20_26, files_abc(), tt);
  finalize(tt);

  generator_settings gs;
  gs.start_mode_ = kCar;

  auto qg = generator{tt, gs};

  auto const sdq = qg.random_query();
  ASSERT_TRUE(sdq.has_value());
}

TEST(query_generation, reproducibility) {
  constexpr auto const src = source_idx_t{0U};
  timetable tt;
  tt.date_range_ = full_period();
  register_special_stations(tt);
  load_timetable(src, loader::hrd::hrd_5_20_26, files_abc(), tt);
  finalize(tt);

  generator_settings const gs;
  auto const seed = 2342;
  auto const num_queries = 3U;

  auto qg0 = generator{tt, gs, seed};
  auto result_qg0 =
      std::vector<std::optional<query_generation::start_dest_query>>{};
  result_qg0.reserve(num_queries);
  for (auto i = 0U; i < num_queries; ++i) {
    result_qg0.emplace_back(qg0.random_query());
  }

  auto qg1 = generator{tt, gs, seed};
  for (auto i = 0U; i < num_queries; ++i) {
    auto const result_qg1 = qg1.random_query();
    ASSERT_EQ(result_qg0[i].has_value(), result_qg1.has_value());
    if (result_qg0[i].has_value()) {
      EXPECT_EQ(result_qg0[i].value().q_, result_qg1.value().q_);
    }
  }
}

// Feed 0: a route-qualified same-stop rule next to the unqualified pair
// default gives the trip stops at BY virtual locations.
constexpr auto const kFeedRules = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
AG0,Agency0,https://example.com,Europe/Berlin

# calendar_dates.txt
service_id,date,exception_type
S0,20190501,1

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
BA,BA,,52.50,13.30,,,
BY,BY,,52.50,13.40,,,
BC,BC,,52.50,13.50,,,
BD,BD,,52.50,13.60,,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_type
R10,AG0,R10,,3
R11,AG0,R11,,3
R12,AG0,R12,,3

# trips.txt
route_id,service_id,trip_id
R10,S0,U1
R11,S0,U2
R12,S0,U3

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
U1,12:00:00,12:00:00,BA,0
U1,12:30:00,12:30:00,BY,1
U2,12:31:00,12:31:00,BY,0
U2,13:00:00,13:00:00,BC,1
U3,12:31:00,12:31:00,BY,0
U3,13:00:00,13:00:00,BD,1

# transfers.txt
from_stop_id,to_stop_id,from_route_id,to_route_id,transfer_type,min_transfer_time
BY,BY,,,2,120
BY,BY,R10,R11,2,0
)"sv;

// Feed 1 is loaded after feed 0, so its stops come after feed 0's virtual
// locations. They are far enough apart that the generator does not discard
// every query for having a short direct walk.
constexpr auto const kFeedPlain = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
AG1,Agency1,https://example.com,Europe/Paris

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
PA,PA,,48.8500,2.3500,,,
PB,PB,,48.8500,2.4200,,,
PC,PC,,48.8500,2.4900,,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_type
RP,AG1,RP,,3

# trips.txt
route_id,service_id,trip_id
RP,S1,P1
RP,S1,P2

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
P1,09:00:00,09:00:00,PA,0
P1,09:10:00,09:10:00,PB,1
P1,09:20:00,09:20:00,PC,2
P2,10:00:00,10:00:00,PA,0
P2,10:10:00,10:10:00,PB,1
P2,10:20:00,10:20:00,PC,2
)"sv;

// The location r-tree only holds the non-virtual locations and yields
// positions in that pool. Taken as location indices, they would point the
// intermodal offsets at unrelated, far away stops.
TEST(query_generation, intermodal_offsets_skip_virtual_locations) {
  auto tt = timetable{};
  tt.date_range_ = {date::sys_days{2019_y / May / 1},
                    date::sys_days{2019_y / May / 2}};
  loader::register_special_stations(tt);
  loader::gtfs::load_timetable({}, source_idx_t{0},
                               loader::mem_dir::read(kFeedRules), tt);
  loader::gtfs::load_timetable({}, source_idx_t{1},
                               loader::mem_dir::read(kFeedPlain), tt);
  loader::finalize(tt);

  // Pool positions and location indices only differ if virtual locations
  // exist and real locations follow them.
  auto first_virt = std::optional<location_idx_t>{};
  auto last_real = location_idx_t{0U};
  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    if (tt.locations_.types_[l] == location_type::kVirt) {
      if (!first_virt.has_value()) {
        first_virt = l;
      }
    } else {
      last_real = l;
    }
  }
  ASSERT_TRUE(first_virt.has_value())
      << "precondition: feed 0 has virtual locations";
  ASSERT_GT(last_real, *first_virt)
      << "precondition: a real location follows a virtual one, so pool "
         "positions and location indices differ";

  auto gs = generator_settings{};
  gs.bbox_ =
      geo::make_box({geo::latlng{48.83, 2.33}, geo::latlng{48.87, 2.51}});

  auto qg = generator{tt, gs, 42U};

  auto checked = 0U;
  for (auto i = 0U; i != 50U; ++i) {
    auto const sdq = qg.random_query();
    if (!sdq.has_value()) {
      continue;
    }
    auto const check = [&](std::variant<location_idx_t, geo::latlng> const& p,
                           std::vector<routing::offset> const& offsets,
                           transport_mode const& mode) {
      auto const* pos = std::get_if<geo::latlng>(&p);
      if (pos == nullptr) {
        return;
      }
      for (auto const& o : offsets) {
        EXPECT_NE(tt.locations_.types_[o.target()], location_type::kVirt);
        EXPECT_LE(geo::distance(*pos, tt.locations_.coordinates_[o.target()]),
                  static_cast<double>(mode.range()) + 1.0)
            << "offset points outside the search radius";
        ++checked;
      }
    };
    check(sdq->start_, sdq->q_.start_, gs.start_mode_);
    check(sdq->dest_, sdq->q_.destination_, gs.dest_mode_);
  }
  EXPECT_GT(checked, 0U) << "no intermodal offset was generated";
}

// Stops are drawn by the number of their events, the ones at their virtual
// locations included: all trips at X stop at virtual locations of X (the rule
// R1 -> R2), yet X is a start and a destination. Z has no events and virtual
// locations are no stops: neither is ever drawn.
TEST(query_generation, draws_stops_by_events) {
  constexpr auto const kFeed = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
AG0,Agency0,https://example.com,Europe/Berlin

# calendar_dates.txt
service_id,date,exception_type
S0,20190501,1

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
XA,XA,,52.50,13.30,,,
X,X,,52.50,13.40,,,
XB,XB,,52.50,13.50,,,
Z,Z,,52.50,13.60,,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_type
R1,AG0,R1,,3
R2,AG0,R2,,3

# trips.txt
route_id,service_id,trip_id
R1,S0,T1
R2,S0,T2

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
T1,12:00:00,12:00:00,XA,0
T1,12:30:00,12:30:00,X,1
T2,12:31:00,12:31:00,X,0
T2,13:00:00,13:00:00,XB,1

# transfers.txt
from_stop_id,to_stop_id,from_route_id,to_route_id,transfer_type,min_transfer_time
X,X,,,2,120
X,X,R1,R2,2,0
)"sv;

  auto tt = timetable{};
  tt.date_range_ = {date::sys_days{2019_y / May / 1},
                    date::sys_days{2019_y / May / 2}};
  loader::register_special_stations(tt);
  loader::gtfs::load_timetable({}, source_idx_t{0},
                               loader::mem_dir::read(kFeed), tt);
  loader::finalize(tt);
  auto const x = tt.locations_.location_id_to_idx_.at({"X", source_idx_t{0}});
  auto const z = tt.locations_.location_id_to_idx_.at({"Z", source_idx_t{0}});
  ASSERT_TRUE(tt.location_routes_[x].empty())
      << "precondition: X is served at its virtual locations only";

  auto gs = generator_settings{};
  gs.start_match_mode_ = routing::location_match_mode::kEquivalent;
  gs.dest_match_mode_ = routing::location_match_mode::kEquivalent;
  auto qg = generator{tt, gs, 42U};

  auto has_x_start = false;
  auto has_x_dest = false;
  for (auto i = 0U; i != 100U; ++i) {
    auto const sdq = qg.random_query();
    ASSERT_TRUE(sdq.has_value());
    for (auto const& o :
         {sdq->q_.start_.front(), sdq->q_.destination_.front()}) {
      EXPECT_NE(location_type::kVirt, tt.locations_.types_[o.target()]);
      EXPECT_NE(z, o.target());
    }
    has_x_start |= sdq->q_.start_.front().target() == x;
    has_x_dest |= sdq->q_.destination_.front().target() == x;
  }
  EXPECT_TRUE(has_x_start);
  EXPECT_TRUE(has_x_dest);
}
