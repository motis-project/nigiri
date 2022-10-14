#include "doctest/doctest.h"

#include "nigiri/loader/hrd/load_timetable.h"

#include "./hrd_timetable.h"

using namespace nigiri::loader::hrd;

TEST_CASE("loader.hrd.hash") {
  REQUIRE(applicable(hrd_5_20_26, nigiri::test_data::hrd_timetable::files()));

  auto h = std::uint64_t{};
  CHECK_NOTHROW(
      h = hash(hrd_5_20_26, nigiri::test_data::hrd_timetable::files()));
  CHECK(h != 0U);
}
