#include "nigiri/common/interval.h"

#include "nigiri/types.h"

#include "doctest/doctest.h"

using namespace nigiri;

unixtime_t clamp(interval<unixtime_t> const& i, unixtime_t const t) {
  // i.to_ is the first invalid value
  // so (i.to_ - 1) is the last valid value
  return std::clamp(i.from_, i.to_ - 1_minutes, t);
}

TEST_CASE("interval clamp") {
  auto const t = [](auto&& x) { return unixtime_t{duration_t{x}}; };
  auto const i = interval{t(10), t(15)};

  CHECK(i.contains(t(10)));
  CHECK(i.contains(t(11)));
  CHECK(i.contains(t(12)));
  CHECK(i.contains(t(13)));
  CHECK(i.contains(t(14)));
  CHECK(!i.contains(t(15)));

  CHECK(clamp(i, t(20)) == t(14));
}
