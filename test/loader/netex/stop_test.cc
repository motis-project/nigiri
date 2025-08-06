#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"

#include <iostream>
#include <string_view>

#include "utl/to_vec.h"

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

location_idx_t find_location_by_id(timetable const& tt,
                                   std::string_view const id) {
  for (auto i = location_idx_t{0U}; i != tt.n_locations(); ++i) {
    if (tt.locations_.ids_[i].view() == id) {
      return i;
    }
  }
  return location_idx_t::invalid();
}

}  // namespace

TEST(netex, read_stations_example_data) {
  auto const fs = fs_dir{"test/test_data/netex"};
  auto tt = timetable{};
  load_timetable(loader_config{}, source_idx_t{0}, fs, tt);

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
}
