#include "gtest/gtest.h"

#include "geo/box.h"

#include "utl/enumerate.h"
#include "utl/equal_ranges_linear.h"
#include "utl/parser/split.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/routing/hmetis.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/routing/reach.h"
#include "nigiri/routing/search.h"
#include "nigiri/routing/start_times.h"
#include "nigiri/rt/frun.h"
#include "nigiri/timetable.h"

#include "../raptor_search.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace std::string_view_literals;
using namespace std::chrono_literals;
using nigiri::test::run_search;

constexpr auto const test_files = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
H1,H1,,49.874446125541,8.649950760650633,,
H2,H2,,50.118039690771,8.669541533394973,,
H3,H3,,50.586661845548,8.681833607424945,,
L11,L11,,49.8938671974,8.650568450613406,,
L12,L12,,49.8606820772,8.636774999023730,,
L21,L21,,50.1567601578,8.657005394688590,,
L22,L22,,50.0883693880,8.627579364630613,,
L31,L31,,50.5662229997,8.682753171380188,,
L32,L32,,50.5948349022,8.734248723981645,,
HK1,HK1,,50.4410610968,8.657005394805864,,
HK2,HK2,,50.4404754493,8.704822693650074,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
RHL1,DB,HL1,,,3
RHL2,DB,HL2,,,3
RHR1,DB,HR1,,,3
RHR2,DB,HR2,,,3
RL11,DB,L11,,,3
RL12,DB,L12,,,3
RL21,DB,L21,,,3
RL22,DB,L22,,,3
RL31,DB,L31,,,3
RL32,DB,L32,,,3
RK1,DB,K1,,,3
RK2,DB,K2,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
RHL1,S,HL1,,
RHL2,S,HL2,,
RHR1,S,HR1,,
RHR2,S,HR2,,
RL11,S,L11,,
RL12,S,L12,,
RL21,S,L21,,
RL22,S,L22,,
RL31,S,L31,,
RL32,S,L32,,
RK1,S,K1,,
RK2,S,K2,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
HL1,00:00:00,00:00:00,H1,1,0,0
HL1,02:00:00,02:00:00,H2,2,0,0
HL1,03:00:00,03:00:00,HK1,3,0,0
HR1,00:00:00,00:00:00,HK2,1,0,0
HR1,02:00:00,02:00:00,H3,2,0,0
HL2,00:00:00,00:00:00,HK1,1,0,0
HL2,02:00:00,02:00:00,H2,2,0,0
HL2,03:00:00,03:00:00,H1,3,0,0
HR2,00:00:00,00:00:00,H3,1,0,0
HR2,01:00:00,01:00:00,HK2,2,0,0
K1,00:00:00,00:00:00,HK1,1,0,0
K1,00:10:00,00:10:00,HK2,2,0,0
K2,00:00:00,00:00:00,HK2,1,0,0
K2,00:10:00,00:10:00,HK1,2,0,0
L11,00:00:00,00:00:00,L11,1,0,0
L11,00:10:00,00:10:00,H1,2,0,0
L11,00:20:00,00:20:00,L12,3,0,0
L12,00:00:00,00:00:00,L12,1,0,0
L12,00:10:00,00:10:00,H1,2,0,0
L12,00:20:00,00:20:00,L11,3,0,0
L21,00:00:00,00:00:00,L21,1,0,0
L21,00:10:00,00:10:00,H2,2,0,0
L21,00:20:00,00:20:00,L22,3,0,0
L22,00:00:00,00:00:00,L22,1,0,0
L22,00:10:00,00:10:00,H2,2,0,0
L22,00:20:00,00:20:00,L21,3,0,0
L31,00:00:00,00:00:00,L31,1,0,0
L31,00:10:00,00:10:00,H3,2,0,0
L31,00:20:00,00:20:00,L32,3,0,0
L32,00:00:00,00:00:00,L32,1,0,0
L32,00:10:00,00:10:00,H3,2,0,0
L32,00:20:00,00:20:00,L31,3,0,0

# frequencies.txt
trip_id,start_time,end_time,headway_secs
HL1,00:00:00,24:00:00,7200
HR1,00:00:00,24:00:00,7200
HL2,00:00:00,24:00:00,7200
HR2,00:00:00,24:00:00,7200
K1,00:00:00,24:00:00,600
K2,00:00:00,24:00:00,600
L11,00:00:00,24:00:00,600
L12,00:00:00,24:00:00,600
L21,00:00:00,24:00:00,600
L22,00:00:00,24:00:00,600
L31,00:00:00,24:00:00,600
L32,00:00:00,24:00:00,600

# calendar_dates.txt
service_id,date,exception_type
S,20190502,1
S,20190501,1
S,20190430,1

# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time
C1,D,0,2
)"sv;

TEST(routing, arcflags_test) {
  constexpr auto const src = source_idx_t{0U};

  /*
  timetable tt;
  tt.date_range_ = {sys_days{2019_y / April / 30}, sys_days{2019_y / May / 3}};
  gtfs::load_timetable(
      {.link_stop_distance_ = 0, .default_tz_ = "Europe/Berlin"}, src,
      mem_dir::read(test_files), tt);
  finalize(tt);

  routing::compute_arc_flags(tt);

  tt.write("/tmp/arcflag_tt");
  */

  auto tt_wrapped = cista::wrapped<timetable>{timetable::read(
      cista::memory_holder{cista::file{"/tmp/arcflag_tt", "r"}.content()})};
  tt_wrapped->locations_.resolve_timezones();
  auto& tt = *tt_wrapped;

  for (auto const [r, flags] : utl::enumerate(tt.arc_flags_)) {
    std::cout << "name=" << std::setw(22)
              << tt.transport_name(
                     tt.route_transport_ranges_[route_idx_t{r}].from_)
              << "  -->   ";
    for (auto i = 0U; i != 8U; ++i) {
      std::cout << std::boolalpha << std::setw(6) << flags[i] << "  ";
    }
    std::cout << "\n";
  }

  auto const file =
      cista::mmap{"hmetis.txt.part.8", cista::mmap::protection::READ};

  //  routing::hmetis_out_to_geojson(file.view(), std::cout, tt);

  for (auto l = location_idx_t{0}; l != tt.n_locations(); ++l) {
    std::cout << location{tt, l} << ": " << tt.locations_.components_[l]
              << " => ";
    for (auto const p :
         tt.component_partitions_[tt.locations_.components_[l]]) {
      std::cout << static_cast<int>(to_idx(p)) << " ";
    }
    std::cout << "\n";
  }

  auto const result = run_search<direction::kForward>(
      tt, nullptr,
      routing::query{
          .start_time_ = unixtime_t{date::sys_days{2019_y / May / 1} + 8h},
          .start_match_mode_ = routing::location_match_mode::kEquivalent,
          .dest_match_mode_ = routing::location_match_mode::kEquivalent,
          .start_ = {{tt.locations_.location_id_to_idx_.at({"L22", src}),
                      0_minutes, 0U}},
          .destination_ = {{tt.locations_.location_id_to_idx_.at({"L32", src}),
                            0_minutes, 0U}}});
  std::cout << "stats:\n" << result.algo_stats_ << "\n";
  std::cout << "journeys\n";
  for (auto const& j : *result.journeys_) {
    j.print(std::cout, tt);
  }
}