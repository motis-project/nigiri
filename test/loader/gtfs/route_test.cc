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
  auto tt = timetable{};
  auto timezones = tz_map{};
  auto i18n = translator{.tt_ = tt};

  auto const src = source_idx_t{};
  auto const r = route_id_idx_t{0};

  auto agencies = read_agencies(source_idx_t{0}, tt, i18n, timezones,
                                example_files().get_file(kAgencyFile).data(),
                                "Europe/Berlin");
  auto const routes =
      read_routes(src, tt, i18n, timezones, agencies,
                  example_files().get_file(kRoutesFile).data(), "CET");

  EXPECT_EQ(1, routes.size());
  EXPECT_NE(end(routes), routes.find("A"));
  EXPECT_EQ(1, tt.route_ids_.size());
  EXPECT_EQ("A", tt.route_ids_[src].ids_.get(r));
  EXPECT_EQ("17", tt.get_default_translation(
                      tt.route_ids_[src].route_id_short_names_.at(r)));
  EXPECT_EQ("Mission", tt.get_default_translation(
                           tt.route_ids_[src].route_id_long_names_.at(r)));
  EXPECT_EQ(route_type_t{3}, tt.route_ids_[src].route_id_type_.at(r));
  EXPECT_EQ(
      "Demo Transit Authority",
      tt.get_default_translation(
          tt.providers_[tt.route_ids_[src].route_id_provider_.at(r)].name_));
  EXPECT_EQ(
      "DTA",
      tt.strings_.get(
          tt.providers_[tt.route_ids_[src].route_id_provider_.at(r)].id_));
}

TEST(gtfs, read_routes_berlin_data) {
  auto tt = timetable{};
  auto timezones = tz_map{};
  auto i18n = translator{.tt_ = tt};

  tt.register_translation(std::string_view{});
  auto const src = source_idx_t{};

  auto agencies = read_agencies(source_idx_t{0}, tt, i18n, timezones,
                                berlin_files().get_file(kAgencyFile).data(),
                                "Europe/Berlin");
  auto const routes =
      read_routes(src, tt, i18n, timezones, agencies,
                  berlin_files().get_file(kRoutesFile).data(), "CET");

  EXPECT_EQ(9, routes.size());
  auto const& src_routes = tt.route_ids_[src];

  ASSERT_NE(end(routes), routes.find("1"));
  auto const route_1_idx = routes.at("1")->route_id_idx_;
  EXPECT_EQ(
      "Günter Anger Güterverkehrs GmbH & Co. Omnibusvermietung KG",
      tt.get_default_translation(
          tt.providers_[src_routes.route_id_provider_[route_1_idx]].name_));
  EXPECT_EQ("SXF2", tt.get_default_translation(
                        src_routes.route_id_short_names_[route_1_idx]));
  EXPECT_EQ("", tt.get_default_translation(
                    src_routes.route_id_long_names_[route_1_idx]));
  EXPECT_EQ(clasz::kBus, to_clasz(src_routes.route_id_type_[route_1_idx]));
  EXPECT_EQ(color_t{0}, src_routes.route_id_colors_[route_1_idx].color_);
  EXPECT_EQ(color_t{0}, src_routes.route_id_colors_[route_1_idx].text_color_);

  ASSERT_NE(end(routes), routes.find("809"));
  auto const route_809_idx = routes.at("809")->route_id_idx_;
  EXPECT_EQ(
      "DB Regio AG",
      tt.get_default_translation(
          tt.providers_[src_routes.route_id_provider_[route_809_idx]].name_));
  EXPECT_EQ("", tt.get_default_translation(
                    src_routes.route_id_short_names_[route_809_idx]));
  EXPECT_EQ("Leisnig -- Leipzig, Hauptbahnhof",
            tt.get_default_translation(
                src_routes.route_id_long_names_[route_809_idx]));
  EXPECT_EQ(clasz::kRegionalFast,
            to_clasz(src_routes.route_id_type_[route_809_idx]));

  ASSERT_NE(end(routes), routes.find("812"));
  auto const route_812_idx = routes.at("812")->route_id_idx_;
  EXPECT_EQ(
      "DB Regio AG",
      tt.get_default_translation(
          tt.providers_[src_routes.route_id_provider_[route_812_idx]].name_));
  EXPECT_EQ("RB14", tt.get_default_translation(
                        src_routes.route_id_short_names_[route_812_idx]));
  EXPECT_EQ("", tt.get_default_translation(
                    src_routes.route_id_long_names_[route_812_idx]));
  EXPECT_EQ(clasz::kRegionalFast,
            to_clasz(src_routes.route_id_type_[route_812_idx]));
  EXPECT_EQ(color_t{0xFFB10093},
            src_routes.route_id_colors_[route_812_idx].color_);
  EXPECT_EQ(color_t{0xFFFFFFFF},
            src_routes.route_id_colors_[route_812_idx].text_color_);

  ASSERT_NE(end(routes), routes.find("F11"));
  EXPECT_EQ(
      clasz::kShip,
      to_clasz(src_routes.route_id_type_[routes.at("F11")->route_id_idx_]));
}
