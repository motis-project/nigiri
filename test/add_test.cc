#include "doctest/doctest.h"

#include "nigiri/timetable.h"

using nigiri::timetable;

TEST_CASE("tile::direct_children()") {
  SUBCASE("root") {
    timetable tt;
    CHECK(true);
  }

  SUBCASE("darmstadt") {}
}
