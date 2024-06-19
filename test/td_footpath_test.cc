#include "gtest/gtest.h"

#include "nigiri/td_footpath.h"
#include "nigiri/types.h"

using namespace nigiri;
using namespace date;
using namespace std::chrono_literals;

TEST(td_footpath, simple) {
  auto td_footpath_out = vecvec<location_idx_t, td_footpath>{};

  auto const a = location_idx_t{0};
  auto const b = location_idx_t{1};
  auto const c = location_idx_t{2};
  auto const d = location_idx_t{3};
  auto const e = location_idx_t{4};
  auto const f = location_idx_t{5};

  td_footpath_out.emplace_back({
      /*
       *                      |  2024/06/15 10:00 am   |   5 min
       * 2024/06/15 10:00 am  |  2024/06/15  2:00 pm   |  10 min
       * 2024/06/15  2:00 pm  |  2024/06/16 11:00 am   |   5 min
       * 2024/06/16 11:00 am  |  2024/06/17 12:00 am   |  not possible
       * 2024/06/17 12:00 am  |                        |   5 min
       */
      td_footpath{b, kNull, 5_minutes},
      td_footpath{b, sys_days{June / 15 / 2024_y} + 10_hours, 10_minutes},
      td_footpath{b, sys_days{June / 15 / 2024_y} + 14_hours, 5_minutes},
      td_footpath{b, sys_days{June / 16 / 2024_y} + 11_hours, kInfeasible},
      td_footpath{b, sys_days{June / 17 / 2024_y} + 12_hours, 5_minutes},

      /*
       *                      |  2024/06/21 12:00 am   |   not possible
       * 2024/06/17 12:00 am  |                        |   5 min
       */
      td_footpath{c, kNull, kInfeasible},
      td_footpath{c, sys_days{June / 15 / 2024_y} + 13_hours, 5_minutes},

      /*
       * not possible
       */
      td_footpath{d, kNull, kInfeasible},

      /*
       * start > t
       */
      td_footpath{e, sys_days{June / 15 / 2024_y} + 12_hours, 5_minutes},
      td_footpath{e, sys_days{June / 15 / 2024_y} + 13_hours, 10_minutes},

      /*
       * start > t
       */
      td_footpath{f, sys_days{June / 15 / 2024_y} + 9_hours, 5_minutes},
      td_footpath{f, sys_days{June / 15 / 2024_y} + 10_hours, 7_minutes},
  });

  auto count = 0U;
  auto const map = vector_map<location_idx_t, unixtime_t>{
      kNull,
      sys_days{June / 15 / 2024_y} + 11_hours + 10_minutes,
      sys_days{June / 15 / 2024_y} + 13_hours + 5_minutes,
      kNull,
      sys_days{June / 15 / 2024_y} + 12_hours + 5_minutes,
      sys_days{June / 15 / 2024_y} + 11_hours + 7_minutes,
  };
  auto const now = sys_days{June / 15 / 2024_y} + 11_hours;
  for_each_footpath<direction::kForward>(
      td_footpath_out[a], now, [&](footpath const fp) {
        EXPECT_EQ(map[fp.target()], now + fp.duration());
        ++count;
        return utl::cflow::kContinue;
      });
  EXPECT_EQ(4, count);
}
