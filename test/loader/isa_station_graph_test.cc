#include "gtest/gtest.h"

#include <span>

#include "nigiri/loader/load.h"
#include "nigiri/footpath_restriction.h"
#include "nigiri/routing/transfers.h"
#include "nigiri/timetable.h"

using namespace nigiri;
using namespace date;
using namespace std::string_view_literals;
using fr = nigiri::footpath_restriction;

constexpr auto const kIsaGtfs = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station
de:1:1,S,48.0,9.0,1,
de:1:1::1,S 1,48.0,9.0,0,de:1:1
de:1:1::2,S 2,48.001,9.0,0,de:1:1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,1,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R1,S1,T1,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
T1,10:00:00,10:00:00,de:1:1::1,0
T1,10:10:00,10:10:00,de:1:1::2,1

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)"sv;

// Areas 1, 2 (platforms, linked via halteste.asc), 3 (between, level -1).
// Paths: 1 -> 3 stairs 60 s, 3 -> 2 level 30 s, 1 -> 2 elevator 90 s
// (banned for walkers), all with reverse rows. 7 = self loop (transfer time).
// Speed profile 300 is ignored (500 is closer to "mittel").
loader::mem_dir isa_station_graph_test_files() {
  return loader::mem_dir{loader::mem_dir::dir_t{
      {"halteste.asc",
       "%Haltestellennummer#Lieferant#...\n"
       "101#SUP#x#x#x#x#9.0#48.0#x#x#x#x#x#x#x#x#x#x#x#de:1:1::1#\n"
       "102#SUP#x#x#x#x#9.0#48.001#x#x#x#x#x#x#x#x#x#x#x#de:1:1::2#\n"},
      {"umbwbereichtyp.asc",
       "%Nummer#OEV#Wegenetz#Name#POI#\n"
       "1#1#1#Zugang und \xC3\x96V#0#\n"
       "4#0#0#Zwischenpunkt#0#\n"},
      {"umbwbereich.asc",
       "%Bauwerk#Version#Lieferant#Bereich#...\n"
       "7#1#MASTER#1#x#0#1###0#SUP#101#9.0#48.0#1#Steig 1#\n"
       "#SUP#101#1\n"
       "7#1#MASTER#2#x#0#1###0#SUP#102#9.0#48.001#1#Steig 2#\n"
       "#SUP#102#2\n"
       "7#1#MASTER#3#x#0#4###-1#####1#Gang#\n"},
      {"umbwwegtyp.asc",
       "%Nummer#gleich#unterschiedlich#Name#\n"
       "1#1#0#niveaugleich\n"
       "2#0#1#Treppe\n"
       "4#0#1#Aufzug\n"},
      {"attribut.asc",
       "%Nummer#Kuerzel#...\n"
       "19376#20166#0#deu#Gesperrt f\xC3\xBCr Rollator-/Kinderwagennutzer#\n"
       "18508#2021#0#deu#Breite des niveaugleichen Wegs: w cm#\n"
       "18528#2093#0#deu#Grundflaechenlaenge: w cm#\n"
       "18529#2094#0#deu#Grundflaechenbreite: w cm#\n"},
      {"umbwwegattr.asc",
       "%Weg#Attribut#Wert#\n5#19376#\n6#19376#\n1#18508#150#\n"
       "3#18508#120#\n5#18528#100#\n5#18529#100#\n"},
      {"umbwweg.asc",
       "%Geschwindigkeit#Profil#\n"
       "##\n"
       "#1#7#1#MASTER#1#7#1#MASTER#3#2#60###\n"
       "#2#7#1#MASTER#3#7#1#MASTER#1#2#60###\n"
       "#3#7#1#MASTER#3#7#1#MASTER#2#1#30###\n"
       "#4#7#1#MASTER#2#7#1#MASTER#3#1#30###\n"
       "#5#7#1#MASTER#1#7#1#MASTER#2#4#90###\n"
       "#6#7#1#MASTER#2#7#1#MASTER#1#4#90###\n"
       "#7#7#1#MASTER#1#7#1#MASTER#1#1#120###\n"
       "300#langsam\n"
       "#8#7#1#MASTER#1#7#1#MASTER#2#1#1###\n"
       "500#mittel\n"}}};
}

// Restricted tail of l's footpath list (default profile).
std::span<footpath const> isa_restricted(timetable const& tt,
                                         bool const is_out,
                                         location_idx_t const l) {
  auto const b = (is_out ? tt.locations_.footpaths_out_
                         : tt.locations_.footpaths_in_)[kDefaultProfile][l];
  auto const split = tt.locations_.restricted_begin(kDefaultProfile, is_out, l);
  return std::span{b.data() + split, b.size() - split};
}

TEST(loader, isa_station_graph) {
  auto const gtfs = loader::mem_dir::read(kIsaGtfs);
  auto const isa = isa_station_graph_test_files();
  auto const tt =
      loader::load({loader::dir_source{.dir_ = gtfs, .loader_config_ = {}},
                    loader::dir_source{.dir_ = isa, .loader_config_ = {}}},
                   {2019_y / May / 1, 2019_y / May / 2});
  auto const l = [&](std::string_view const id) {
    return tt.locations_.location_id_to_idx_.at({id, source_idx_t{0U}});
  };

  auto const fps = isa_restricted(tt, true, l("de:1:1::1"));
  auto const rs =
      tt.locations_.footpath_restrictions_out_[kDefaultProfile][l("de:1:1::1")];
  ASSERT_EQ(2U, fps.size());

  // stairs + level: 90 s walking, narrowest 120 cm; elevator: 90 s fixed,
  // walkers banned, cabin 1 x 1 m too small for wheelchairs
  auto n_stairs = 0U;
  auto n_elevator = 0U;
  for (auto i = 0U; i != fps.size(); ++i) {
    EXPECT_EQ(l("de:1:1::2"), fps[i].target());
    EXPECT_EQ(duration_t{2}, fps[i].duration());
    EXPECT_FALSE(rs[i].has_unknown_slope());  // no ramp: level
    if (rs[i].flags_ == fr::kStairs) {
      ++n_stairs;
      EXPECT_EQ(0U, rs[i].fixed_seconds());
      EXPECT_FALSE(rs[i].has_unknown_width());
      EXPECT_EQ(fr::to_width(1.2), rs[i].width());
    } else if (rs[i].flags_ ==
               (fr::kElevator | fr::kBanWalker | fr::kBanWheelchair |
                fr::kBanWheelchairAssisted)) {
      ++n_elevator;
      EXPECT_EQ(90U, rs[i].fixed_seconds());
      EXPECT_TRUE(rs[i].has_unknown_width());
    }
  }
  EXPECT_EQ(1U, n_stairs);
  EXPECT_EQ(1U, n_elevator);

  // walker: no stairs, banned from the elevator -> no footpath
  auto const walker = routing::transfer_time_settings{
      .restrictions_ = {.forbidden_ = static_cast<std::uint8_t>(
                            fr::kStairs | fr::kBanWalker)}};
  EXPECT_FALSE(routing::shortest_transfer(tt, nullptr, kDefaultProfile, walker,
                                          l("de:1:1::1"), l("de:1:1::2"))
                   .has_value());
}
