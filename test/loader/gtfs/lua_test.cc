#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/loader/load.h"

#include "nigiri/rt/frun.h"
#include "nigiri/rt/gtfsrt_resolve_run.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace std::chrono_literals;

namespace {

constexpr auto kTimetable = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Paris
TT,Tischtennis,https://deutschebahn.com,Europe/Paris

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station,platform_code
A,A Berlin,i,0.0,1.0,,,,1
B,B Berlin,j,2.0,3.0,,,,1
C,C Berlin,k,4.0,5.0,,,,1
D,D Berlin,l,6.0,7.0,,,,1

# calendar_dates.txt
service_id,date,exception_type
S_RE1,20190501,1
S_RE2,20190503,1
S_RE3,20190504,1
S_RE3,20190505,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R_RE1,DB,1,,,3
R_RE2,DB,2,,,3
R_RE3,TT,2,,,3
R_RE4,DB,2,,,3

# trips.txt
route_id,service_id,trip_id,trip_short_name,trip_headsign,block_id
R_RE1,S_RE1,T_RE1,00123,RE 1,1
R_RE2,S_RE2,T_RE2,00456,RE 2,1
R_RE3,S_RE3,T_RE3,00789,RE 3,1
R_RE4,S_RE4,T_RE4,00555,RE 4,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T_RE1,49:00:00,49:00:00,A,1,0,0
T_RE1,50:00:00,50:00:00,B,2,0,0
T_RE2,00:30:00,00:30:00,B,1,0,0
T_RE2,00:45:00,00:45:00,C,2,0,0
T_RE2,01:00:00,01:00:00,D,3,0,0
T_RE3,00:30:00,00:30:00,B,1,0,0
T_RE3,00:45:00,00:45:00,C,2,0,0
T_RE4,00:30:00,00:30:00,B,1,0,0
T_RE4,00:45:00,00:45:00,C,2,0,0
)";

constexpr auto kUserScript = R"(
function process_location(stop)
  local name = stop:get_name()
  if string.sub(name, -7) == ' Berlin' then
    stop:set_name(string.sub(name, 1, -8))
  end

  local pos = stop:get_pos()
  pos:set_lat(stop:get_pos():get_lat() + 2.0)
  pos:set_lng(stop:get_pos():get_lng() - 2.0)
  stop:set_pos(pos)

  stop:set_description(stop:get_description() .. ' ' .. stop:get_id() .. ' YEAH')
  stop:set_timezone('Europe/Berlin')
  stop:set_transfer_time(stop:get_transfer_time() + 98)
  stop:set_platform_code(stop:get_platform_code() .. 'A')

  return true
end

function process_route(route)
  if route:get_id() == 'R_RE4' then
    return false
  end

  if route:get_route_type() == 3 then
    route:set_clasz(7)
    route:set_route_type(101)
  elseif route:get_route_type() == 1 then
    route:set_clasz(8)
    route:set_route_type(400)
  end

  if route:get_agency():get_name() == 'Deutsche Bahn' and route:get_route_type() == 101 then
    route:set_short_name('RE ' .. route:get_short_name())
  end

  return true
end

function process_agency(agency)
  if agency:get_id() == 'TT' then
    return false
  end

  if agency:get_name() == 'Deutsche Bahn' and agency:get_id() == 'DB' then
    agency:set_url(agency:get_timezone())
    agency:set_timezone('Europe/Berlin')
    agency:set_name('SNCF')
    return true
  end
  return false
end

function process_trip(trip)
  if trip:get_route():get_route_type() == 101 then
    -- Prepend category and eliminate leading zeros (e.g. '00123' -> 'ICE 123')
    trip:set_short_name('ICE ' .. string.format("%d", tonumber(trip:get_short_name())))
    trip:set_display_name(trip:get_short_name())
  end
  return trip:get_id() == 'T_RE1'
end
)";

}  // namespace

TEST(gtfs, lua_test) {
  auto tt = loader::load({{.tag_ = "test",
                           .path_ = kTimetable,
                           .loader_config_ = {.default_tz_ = "Europe/Berlin",
                                              .user_script_ = kUserScript}}},
                         {},
                         {date::sys_days{2019_y / March / 25},
                          date::sys_days{2019_y / November / 1}});

  auto const get_tz_name = [&](timezone_idx_t const tz) {
    return tt.timezones_[tz].apply(utl::overloaded{
        [](pair<string, void const*> const& x) -> std::string_view {
          return x.first;
        },
        [](tz_offsets) -> std::string_view { return ""; }});
  };

  auto const p = tt.get_provider_idx("DB", {});
  ASSERT_NE(provider_idx_t::invalid(), p);
  auto const agency = tt.providers_[p];
  EXPECT_EQ("Europe/Berlin", get_tz_name(agency.tz_));
  EXPECT_EQ("SNCF", tt.get_default_translation(agency.name_));
  EXPECT_EQ("Europe/Paris", tt.get_default_translation(agency.url_));

  auto const a = tt.find(location_id{"A", source_idx_t{}});
  ASSERT_TRUE(a.has_value());
  EXPECT_EQ("A", tt.locations_.ids_[*a].view());

  auto const b = tt.find(location_id{"B", source_idx_t{}});
  ASSERT_TRUE(b.has_value());
  EXPECT_EQ((geo::latlng{4.0, 1.0}), tt.locations_.coordinates_[*b]);
  EXPECT_EQ("j B YEAH",
            tt.get_default_translation(tt.locations_.descriptions_[*b]));
  EXPECT_EQ(100min, tt.locations_.transfer_time_[*b]);
  EXPECT_EQ("1A",
            tt.get_default_translation(tt.locations_.platform_codes_[*b]));
  EXPECT_EQ("Europe/Berlin",
            get_tz_name(tt.locations_.location_timezones_[*b]));

  {  // Renamed to "ICE 123".
    auto td = transit_realtime::TripDescriptor();
    td.set_trip_id("T_RE1");
    td.set_start_date("20190501");
    auto const [r, _] = rt::gtfsrt_resolve_run(date::sys_days{2019_y / May / 3},
                                               tt, nullptr, {}, td);
    ASSERT_TRUE(r.valid());

    constexpr auto const kExpected =
        R"(   0: A       A...............................................                               d: 02.05 23:00 [03.05 01:00]  [{name=ICE 123, day=2019-05-02, id=T_RE1, src=0}]
   1: B       B............................................... a: 03.05 00:00 [03.05 02:00]
)";
    EXPECT_EQ(kExpected,
              (std::stringstream{} << rt::frun{tt, nullptr, r}).view());
  }

  {  // Filtered by trip.
    auto td = transit_realtime::TripDescriptor();
    td.set_trip_id("T_RE2");
    td.set_start_date("20190503");
    auto const [r, _] = rt::gtfsrt_resolve_run(date::sys_days{2019_y / May / 3},
                                               tt, nullptr, {}, td);
    EXPECT_FALSE(r.valid());
  }

  {  // Filtered by agency.
    auto td = transit_realtime::TripDescriptor();
    td.set_trip_id("T_RE3");
    td.set_start_date("20190504");
    auto const [r, _] = rt::gtfsrt_resolve_run(date::sys_days{2019_y / May / 4},
                                               tt, nullptr, {}, td);
    EXPECT_FALSE(r.valid());
  }

  {  // Filtered by route.
    auto td = transit_realtime::TripDescriptor();
    td.set_trip_id("T_RE4");
    td.set_start_date("20190505");
    auto const [r, _] = rt::gtfsrt_resolve_run(date::sys_days{2019_y / May / 5},
                                               tt, nullptr, {}, td);
    EXPECT_FALSE(r.valid());
  }
}
