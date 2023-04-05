#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/agency.h"
#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/route.h"
#include "nigiri/timetable.h"

#include "./test_data.h"

using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;

TEST(gtfs, read_routes_example_data) {
  timetable tt;

  auto const agencies =
      parse_agencies(tt, example_files().get_file(kAgencyFile).data());
  auto const routes =
      read_routes(agencies, example_files().get_file(kRoutesFile).data());

  EXPECT_EQ(1, routes.size());
  EXPECT_NE(end(routes), routes.find("A"));
  EXPECT_EQ(provider_idx_t::invalid(), routes.at("A")->agency_);
  EXPECT_EQ("17", routes.at("A")->short_name_);
  EXPECT_EQ("Mission", routes.at("A")->long_name_);
  EXPECT_EQ(clasz::kBus, routes.at("A")->clasz_);
}

TEST(gtfs, read_routes_berlin_data) {
  timetable tt;

  auto const agencies =
      parse_agencies(tt, berlin_files().get_file(kAgencyFile).data());
  auto const routes =
      read_routes(agencies, berlin_files().get_file(kRoutesFile).data());

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
