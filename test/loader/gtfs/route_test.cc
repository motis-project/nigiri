#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/agency.h"
#include "nigiri/loader/gtfs/route.h"
#include "nigiri/timetable.h"

using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;

auto const example_agency_file_content = std::string_view{
    R"(agency_id,agency_name,agency_url,agency_timezone,agency_phone,agency_lang
FunBus,The Fun Bus,http://www.thefunbus.org,America/Los_Angeles,(310) 555-0222,en
)"};

auto const example_routes_file_content = std::string_view{
    R"(route_id,route_short_name,route_long_name,route_desc,route_type
A,17,Mission,"The ""A"" route travels from lower Mission to Downtown.",3)"};

TEST(gtfs, read_routes_example_data) {
  timetable tt;

  auto const agencies = parse_agencies(tt, example_agency_file_content);
  auto const routes = read_routes(agencies, example_routes_file_content);

  EXPECT_EQ(1, routes.size());
  EXPECT_NE(end(routes), routes.find("A"));
  EXPECT_EQ(provider_idx_t::invalid(), routes.at("A")->agency_);
  EXPECT_EQ("17", routes.at("A")->short_name_);
  EXPECT_EQ("Mission", routes.at("A")->long_name_);
  EXPECT_EQ(clasz::kBus, routes.at("A")->clasz_);
}

constexpr auto const berlin_agencies_file_content = std::string_view{
    R"(agency_id,agency_name,agency_url,agency_timezone,agency_lang,agency_phone
ANG---,Günter Anger Güterverkehrs GmbH & Co. Omnibusvermietung KG,http://www.anger-busvermietung.de,Europe/Berlin,de,033208 22010
BMO---,Busverkehr Märkisch-Oderland GmbH,http://www.busmol.de,Europe/Berlin,de,03341 478383
N04---,DB Regio AG,http://www.bahn.de/brandenburg,Europe/Berlin,de,0331 2356881
BON---,Busverkehr Oder-Spree GmbH,http://www.bos-fw.de,Europe/Berlin,de,03361 556133
)"};

constexpr auto const berlin_routes_file_content = std::string_view{
    R"(route_id,agency_id,route_short_name,route_long_name,route_desc,route_type,route_url,route_color,route_text_color
1,ANG---,SXF2,,,700,http://www.vbb.de,,
10,BMO---,927,,,700,http://www.vbb.de,,
2,BEH---,548,,,700,http://www.vbb.de,,
809,N04---,,"Leisnig -- Leipzig, Hauptbahnhof",,100,http://www.vbb.de,,
81,BON---,2/412,,,700,http://www.vbb.de,,
810,N04---,,"S+U Lichtenberg Bhf (Berlin) -- Senftenberg, Bahnhof",,100,http://www.vbb.de,,
811,N04---,,"S+U Lichtenberg Bhf (Berlin) -- Altdöbern, Bahnhof",,100,http://www.vbb.de,,
812,N04---,RB14,,,100,http://www.vbb.de,,
)"};

TEST(gtfs, read_routes_berlin_data) {
  timetable tt;

  auto const agencies = parse_agencies(tt, berlin_agencies_file_content);
  auto const routes = read_routes(agencies, berlin_routes_file_content);

  EXPECT_EQ(8, routes.size());

  ASSERT_NE(end(routes), routes.find("1"));
  EXPECT_EQ("ANG---", tt.providers_[routes.at("1")->agency_].short_name_);
  EXPECT_EQ("SXF2", routes.at("1")->short_name_);
  EXPECT_EQ("", routes.at("1")->long_name_);
  EXPECT_EQ(clasz::kBus, routes.at("1")->clasz_);

  ASSERT_NE(end(routes), routes.find("809"));
  EXPECT_EQ("N04---", tt.providers_[routes.at("809")->agency_].short_name_);
  EXPECT_EQ("", routes.at("809")->short_name_);
  EXPECT_EQ("Leisnig -- Leipzig, Hauptbahnhof", routes.at("809")->long_name_);
  EXPECT_EQ(clasz::kRegional, routes.at("809")->clasz_);

  ASSERT_NE(end(routes), routes.find("812"));
  EXPECT_EQ("N04---", tt.providers_[routes.at("812")->agency_].short_name_);
  EXPECT_EQ("RB14", routes.at("812")->short_name_);
  EXPECT_EQ("", routes.at("812")->long_name_);
  EXPECT_EQ(clasz::kRegional, routes.at("812")->clasz_);
}
