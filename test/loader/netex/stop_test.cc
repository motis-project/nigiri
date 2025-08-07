#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"

#include <iostream>
#include <string_view>
#include <vector>

#include "utl/to_vec.h"
#include "utl/verify.h"

#include "nigiri/loader/netex/load_timetable.h"
#include "nigiri/timetable.h"

using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::netex;
using namespace std::string_view_literals;
using namespace testing;

namespace {

location_idx_t find_location_by_name(timetable const& tt,
                                     std::string_view const name) {
  for (auto i = location_idx_t{0U}; i != tt.n_locations(); ++i) {
    if (tt.locations_.names_[i].view() == name) {
      return i;
    }
  }
  return location_idx_t::invalid();
}

string get_tz_name(timetable const& tt, timezone_idx_t const tz_idx) {
  auto const& tz = tt.locations_.timezones_[tz_idx];
  utl::verify(holds_alternative<pair<string, void const*>>(tz), "bad tz");
  return tz.as<pair<string, void const*>>().first;
}

std::vector<std::string_view> get_alt_names(timetable const& tt,
                                            location_idx_t const loc_idx) {
  return utl::to_vec(tt.locations_.alt_names_[loc_idx],
                     [&](alt_name_idx_t const ani) {
                       return tt.locations_.alt_name_strings_[ani].view();
                     });
}

}  // namespace

TEST(netex, stops) {
  auto const fs = fs_dir{"test/test_data/netex"};
  auto tt = timetable{};
  load_timetable(loader_config{.default_tz_ = "Europe/Paris"}, source_idx_t{0},
                 fs, tt);

  // CH

  auto const zurich_idx = find_location_by_name(tt, "Zürich HB");
  ASSERT_NE(zurich_idx, location_idx_t::invalid());
  auto const zurich = tt.locations_.get(zurich_idx);
  EXPECT_THAT(zurich.name_, Eq("Zürich HB"));
  EXPECT_THAT(zurich.pos_.lat_, DoubleEq(47.378176));
  EXPECT_THAT(zurich.pos_.lng_, DoubleEq(8.540212));
  EXPECT_THAT(zurich.type_, Eq(location_type::kStation));
  EXPECT_THAT(zurich.parent_, Eq(location_idx_t::invalid()));
  EXPECT_THAT(get_alt_names(tt, zurich_idx),
              UnorderedElementsAre("Tsüri", "ZH", "Züri", "Zurich", "Zürich",
                                   "Zurigo"));
  EXPECT_THAT(get_tz_name(tt, zurich.timezone_idx_), Eq("CET"));

  auto const zurich_children =
      utl::to_vec(tt.locations_.children_[zurich_idx],
                  [&](location_idx_t const c) { return tt.locations_.get(c); });
  EXPECT_THAT(zurich_children, Each(Field(&location::parent_, Eq(zurich_idx))));
  EXPECT_THAT(zurich_children,
              Each(Field(&location::type_, Eq(location_type::kTrack))));
  EXPECT_THAT(zurich_children.size(), Eq(25));
  EXPECT_THAT(
      zurich_children,
      Each(ResultOf(
          [&](auto const& c) { return get_tz_name(tt, c.timezone_idx_); },
          Eq("CET"))));
  EXPECT_THAT(
      utl::to_vec(zurich_children,
                  [&](location const& l) { return l.platform_code_; }),
      UnorderedElementsAre("10", "11", "12", "13", "14", "15", "16", "17", "18",
                           "21", "22", "3", "31", "32", "33", "34", "34AB", "4",
                           "41/42", "43/44", "5", "6", "7", "8", "9"));

  auto const zurich_quay_10 = tt.locations_.get({"ch:2:Quay:8503000:10", {}});
  EXPECT_THAT(zurich_quay_10.platform_code_, Eq("10"));
  EXPECT_THAT(zurich_quay_10.pos_.lat_, DoubleEq(47.378833));
  EXPECT_THAT(zurich_quay_10.pos_.lng_, DoubleEq(8.536528));

  // DE

  auto const darmstadt_idx =
      find_location_by_name(tt, "Darmstadt Hauptbahnhof");
  ASSERT_NE(darmstadt_idx, location_idx_t::invalid());
  auto const darmstadt = tt.locations_.get(darmstadt_idx);
  EXPECT_THAT(darmstadt.name_, Eq("Darmstadt Hauptbahnhof"));
  EXPECT_THAT(darmstadt.pos_.lat_, DoubleEq(49.871941));
  EXPECT_THAT(darmstadt.pos_.lng_, DoubleEq(8.631148));
  EXPECT_THAT(darmstadt.type_, Eq(location_type::kStation));
  EXPECT_THAT(darmstadt.parent_, Eq(location_idx_t::invalid()));
  EXPECT_THAT(get_tz_name(tt, darmstadt.timezone_idx_), Eq("Europe/Berlin"));

  auto const nuernberg_1_parent =
      tt.locations_.get({"DE::StopPlace:6989_rbo_d::", {}});
  EXPECT_THAT(nuernberg_1_parent.name_, Eq("Nürnberg"));
  EXPECT_THAT(nuernberg_1_parent.pos_.lat_, DoubleEq(48.362493));
  EXPECT_THAT(nuernberg_1_parent.pos_.lng_, DoubleEq(12.915564));
  EXPECT_THAT(nuernberg_1_parent.type_, Eq(location_type::kStation));
  EXPECT_THAT(nuernberg_1_parent.parent_, Eq(location_idx_t::invalid()));

  auto const nuernberg_1_child =
      tt.locations_.get({"DE::StopPlace:1069890000_rbo_d::", {}});
  EXPECT_THAT(nuernberg_1_child.name_, Eq("Nürnberg"));
  // this stop doesn't have a centroid in the netex data
  // should be calculated as the centroid of its bounding box (from the quays)
  EXPECT_THAT(nuernberg_1_child.pos_.lat_, DoubleNear(48.362493, 0.00001));
  EXPECT_THAT(nuernberg_1_child.pos_.lng_, DoubleNear(12.915564, 0.00001));
  EXPECT_THAT(nuernberg_1_child.type_, Eq(location_type::kStation));
  EXPECT_THAT(nuernberg_1_child.parent_, Eq(nuernberg_1_parent.l_));
  auto const nuernberg_1_quay_1 =
      tt.locations_.get({"DE::Quay:10698902_rbo_d::", {}});
  EXPECT_THAT(nuernberg_1_quay_1.pos_.lat_, DoubleEq(48.362451));
  EXPECT_THAT(nuernberg_1_quay_1.pos_.lng_, DoubleEq(12.915393));
  auto const nuernberg_1_quay_2 =
      tt.locations_.get({"DE::Quay:10698901_rbo_d::", {}});
  EXPECT_THAT(nuernberg_1_quay_2.pos_.lat_, DoubleEq(48.362535));
  EXPECT_THAT(nuernberg_1_quay_2.pos_.lng_, DoubleEq(12.915735));

  auto airport_business_park =
      tt.locations_.get({"DE::StopPlace:4113_GVH::", {}});
  EXPECT_THAT(airport_business_park.name_, Eq("Airport Business Park West"));
  EXPECT_THAT(airport_business_park.pos_.lat_, DoubleEq(52.461497));
  EXPECT_THAT(airport_business_park.pos_.lng_, DoubleEq(9.680919));
  EXPECT_THAT(airport_business_park.type_, Eq(location_type::kStation));
  // this quay doesn't have a centroid in the netex data
  // should be loaded with the centroid of the stop place
  auto const airport_business_park_quay =
      tt.locations_.get({"DE::Quay:1041130001_GVH::", {}});
  EXPECT_THAT(airport_business_park_quay.type_, Eq(location_type::kTrack));
  EXPECT_THAT(airport_business_park_quay.parent_, Eq(airport_business_park.l_));
  EXPECT_THAT(airport_business_park_quay.pos_.lat_,
              DoubleEq(airport_business_park.pos_.lat_));
  EXPECT_THAT(airport_business_park_quay.pos_.lng_,
              DoubleEq(airport_business_park.pos_.lng_));

  // NO

  auto const oslo_s_parent = tt.locations_.get({"NSR:StopPlace:59872", {}});
  EXPECT_THAT(oslo_s_parent.name_, Eq("Oslo S"));
  EXPECT_THAT(oslo_s_parent.pos_.lat_, DoubleEq(59.910357));
  EXPECT_THAT(oslo_s_parent.pos_.lng_, DoubleEq(10.753051));
  EXPECT_THAT(oslo_s_parent.type_, Eq(location_type::kStation));
  EXPECT_THAT(oslo_s_parent.parent_, Eq(location_idx_t::invalid()));
  EXPECT_THAT(get_tz_name(tt, oslo_s_parent.timezone_idx_), Eq("Europe/Oslo"));
  EXPECT_THAT(
      get_alt_names(tt, oslo_s_parent.l_),
      UnorderedElementsAre("Oslo Sentralstasjon", "Oslo Central Station"));

  auto const oslo_s_rail = tt.locations_.get({"NSR:StopPlace:337", {}});
  EXPECT_THAT(oslo_s_rail.name_, Eq("Oslo S"));
  EXPECT_THAT(oslo_s_rail.pos_.lat_, DoubleEq(59.910925));
  EXPECT_THAT(oslo_s_rail.pos_.lng_, DoubleEq(10.753276));
  EXPECT_THAT(oslo_s_rail.type_, Eq(location_type::kStation));
  EXPECT_THAT(oslo_s_rail.parent_, Eq(oslo_s_parent.l_));
  EXPECT_THAT(get_tz_name(tt, oslo_s_rail.timezone_idx_), Eq("Europe/Oslo"));

  auto const oslo_s_rail_children =
      utl::to_vec(tt.locations_.children_[oslo_s_rail.l_],
                  [&](location_idx_t const c) { return tt.locations_.get(c); });
  EXPECT_THAT(oslo_s_rail_children,
              Each(Field(&location::parent_, Eq(oslo_s_rail.l_))));
  EXPECT_THAT(oslo_s_rail_children,
              Each(Field(&location::type_, Eq(location_type::kTrack))));
  EXPECT_THAT(
      oslo_s_rail_children,
      Each(ResultOf(
          [&](auto const& c) { return get_tz_name(tt, c.timezone_idx_); },
          Eq("Europe/Oslo"))));
  EXPECT_THAT(utl::to_vec(oslo_s_rail_children,
                          [&](location const& l) { return l.platform_code_; }),
              UnorderedElementsAre("15", "2", "5", "7", "6", "11", "16", "13",
                                   "14", "9", "8", "4", "19", "17", "12", "3",
                                   "18", "10", "1"));

  auto const oslo_s_bus_1 = tt.locations_.get({"NSR:StopPlace:2", {}});
  EXPECT_THAT(oslo_s_bus_1.name_, Eq("Oslo S Trelastgata"));
  EXPECT_THAT(oslo_s_bus_1.pos_.lat_, DoubleEq(59.909584));
  EXPECT_THAT(oslo_s_bus_1.pos_.lng_, DoubleEq(10.755165));
  EXPECT_THAT(oslo_s_bus_1.type_, Eq(location_type::kStation));
  EXPECT_THAT(oslo_s_bus_1.parent_, Eq(oslo_s_parent.l_));
  EXPECT_THAT(get_tz_name(tt, oslo_s_bus_1.timezone_idx_), Eq("Europe/Oslo"));

  auto const oslo_s_bus_1_children =
      utl::to_vec(tt.locations_.children_[oslo_s_bus_1.l_],
                  [&](location_idx_t const c) { return tt.locations_.get(c); });
  EXPECT_THAT(oslo_s_bus_1_children,
              Each(Field(&location::parent_, Eq(oslo_s_bus_1.l_))));
  EXPECT_THAT(oslo_s_bus_1_children,
              Each(Field(&location::type_, Eq(location_type::kTrack))));
  EXPECT_THAT(oslo_s_bus_1_children.size(), Eq(16));
  EXPECT_THAT(
      oslo_s_bus_1_children,
      Each(ResultOf(
          [&](auto const& c) { return get_tz_name(tt, c.timezone_idx_); },
          Eq("Europe/Oslo"))));

  // FR

  auto const tour_eiffel_parent =
      tt.locations_.get({"FR::multimodalStopPlace:73797:FR1", {}});
  EXPECT_THAT(tour_eiffel_parent.name_, Eq("Tour Eiffel"));
  // the netex data uses EPSG:2154
  EXPECT_THAT(tour_eiffel_parent.pos_.lat_, DoubleNear(48.8597962, 0.00001));
  EXPECT_THAT(tour_eiffel_parent.pos_.lng_, DoubleNear(2.2943612, 0.00001));
  EXPECT_THAT(tour_eiffel_parent.type_, Eq(location_type::kStation));
  EXPECT_THAT(tour_eiffel_parent.parent_, Eq(location_idx_t::invalid()));
  // no timezone listed in the netex file, using the default
  EXPECT_THAT(get_tz_name(tt, tour_eiffel_parent.timezone_idx_),
              Eq("Europe/Paris"));

  auto const tour_eiffel_bus =
      tt.locations_.get({"FR::monomodalStopPlace:45122:FR1", {}});
  EXPECT_THAT(tour_eiffel_bus.name_, Eq("Tour Eiffel"));
  EXPECT_THAT(tour_eiffel_bus.pos_.lat_, DoubleNear(48.8597962, 0.00001));
  EXPECT_THAT(tour_eiffel_bus.pos_.lng_, DoubleNear(2.2943612, 0.00001));
  EXPECT_THAT(tour_eiffel_bus.type_, Eq(location_type::kStation));
  EXPECT_THAT(tour_eiffel_bus.parent_, Eq(tour_eiffel_parent.l_));
  EXPECT_THAT(get_tz_name(tt, tour_eiffel_bus.timezone_idx_),
              Eq("Europe/Paris"));

  // quays that have a parent stop place
  auto const tour_eiffel_bus_children =
      utl::to_vec(tt.locations_.children_[tour_eiffel_bus.l_],
                  [&](location_idx_t const c) { return tt.locations_.get(c); });
  EXPECT_THAT(utl::to_vec(tour_eiffel_bus_children,
                          [&](location const& l) { return l.id_; }),
              UnorderedElementsAre("FR::Quay:25775:FR1", "FR::Quay:23475:FR1",
                                   "FR::Quay:23455:FR1", "FR::Quay:9107:FR1"));

  // quays that don't have a parent (loaded as stations)
  auto const tour_eiffel_standalone_quay_1 =
      tt.locations_.get({"FR::Quay:50114645:FR1", {}});
  EXPECT_THAT(tour_eiffel_standalone_quay_1.name_, Eq("Tour Eiffel"));
  EXPECT_THAT(tour_eiffel_standalone_quay_1.pos_.lat_,
              DoubleNear(48.8603442, 0.00001));
  EXPECT_THAT(tour_eiffel_standalone_quay_1.pos_.lng_,
              DoubleNear(2.295819, 0.00001));
  EXPECT_THAT(tour_eiffel_standalone_quay_1.type_, Eq(location_type::kStation));
  EXPECT_THAT(tour_eiffel_standalone_quay_1.parent_,
              Eq(location_idx_t::invalid()));
  EXPECT_THAT(get_tz_name(tt, tour_eiffel_standalone_quay_1.timezone_idx_),
              Eq("Europe/Paris"));
}
