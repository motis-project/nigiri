#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"

#include <iostream>
#include <string_view>

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

}  // namespace

TEST(netex, stops) {
  auto const fs = fs_dir{"test/test_data/netex"};
  auto tt = timetable{};
  load_timetable(loader_config{}, source_idx_t{0}, fs, tt);

  // CH

  auto const zurich_idx = find_location_by_name(tt, "Zürich HB");
  ASSERT_NE(zurich_idx, location_idx_t::invalid());
  auto const zurich = tt.locations_.get(zurich_idx);
  EXPECT_THAT(zurich.name_, Eq("Zürich HB"));
  EXPECT_THAT(zurich.pos_.lat_, DoubleEq(47.378176));
  EXPECT_THAT(zurich.pos_.lng_, DoubleEq(8.540212));
  EXPECT_THAT(zurich.type_, Eq(location_type::kStation));
  EXPECT_THAT(zurich.parent_, Eq(location_idx_t::invalid()));
  EXPECT_THAT(utl::to_vec(zurich.alt_names_,
                          [&](alt_name_idx_t const ani) {
                            return tt.locations_.alt_name_strings_[ani].view();
                          }),
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
}
