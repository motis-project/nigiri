#include "doctest/doctest.h"

#include "nigiri/algo/algo.h"
#include "nigiri/algo/query.h"

namespace nigiri::algo {

TEST_CASE("query simple") {
  ontrip_query q(location_idx_t{0}, location_idx_t{0}, unixtime_t{});
}

TEST_CASE("algo journey") {
  ontrip_query q(location_idx_t{0}, location_idx_t{0}, unixtime_t{});
  timetable tt;
  auto const r = algo(q, tt);
}

TEST_CASE("algo data") {
  ontrip_query q(location_idx_t{0}, location_idx_t{0}, unixtime_t{});
  timetable tt;
  std::unique_ptr<int> data;
  auto const r = algo(q, tt, std::move(data));
}

}  // namespace nigiri::algo
