#include "gtest/gtest.h"

#include "utl/equal_ranges_linear.h"

#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/routing/start_times.h"

#include "../loader/hrd/hrd_timetable.h"
#include "../raptor_search.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::routing;
using namespace nigiri::test_data::hrd_timetable;
using nigiri::test::raptor_intermodal_search;

//   +---15--- 30freq --> A
//  /
// O-----30--  15freq --> B
//
// A:         [15-30],          [45-60],          [75-90], ...
// B: [0-30], [15-45], [30-60], [45-75], [60-90], [75-115], ...
constexpr auto const expected = R"(
start_time=2020-03-31 00:00
|  {time_at_start=2020-03-31 00:00, time_at_stop=2020-03-31 00:30, stop=B}
|  {time_at_start=2020-03-31 00:00, time_at_stop=2020-03-31 00:15, stop=A}
start_time=2020-03-30 23:45
|  {time_at_start=2020-03-30 23:45, time_at_stop=2020-03-31 00:00, stop=A}
start_time=2020-03-30 23:15
|  {time_at_start=2020-03-30 23:15, time_at_stop=2020-03-30 23:30, stop=A}
start_time=2020-03-30 23:00
|  {time_at_start=2020-03-30 23:00, time_at_stop=2020-03-30 23:30, stop=B}
start_time=2020-03-30 22:45
|  {time_at_start=2020-03-30 22:45, time_at_stop=2020-03-30 23:15, stop=B}
|  {time_at_start=2020-03-30 22:45, time_at_stop=2020-03-30 23:00, stop=A}
start_time=2020-03-30 22:30
|  {time_at_start=2020-03-30 22:30, time_at_stop=2020-03-30 23:00, stop=B}
start_time=2020-03-30 22:15
|  {time_at_start=2020-03-30 22:15, time_at_stop=2020-03-30 22:45, stop=B}
|  {time_at_start=2020-03-30 22:15, time_at_stop=2020-03-30 22:30, stop=A}
start_time=2020-03-30 22:00
|  {time_at_start=2020-03-30 22:00, time_at_stop=2020-03-30 22:30, stop=B}
start_time=2020-03-30 21:45
|  {time_at_start=2020-03-30 21:45, time_at_stop=2020-03-30 22:15, stop=B}
|  {time_at_start=2020-03-30 21:45, time_at_stop=2020-03-30 22:00, stop=A}
start_time=2020-03-30 21:30
|  {time_at_start=2020-03-30 21:30, time_at_stop=2020-03-30 22:00, stop=B}
start_time=2020-03-30 21:15
|  {time_at_start=2020-03-30 21:15, time_at_stop=2020-03-30 21:45, stop=B}
|  {time_at_start=2020-03-30 21:15, time_at_stop=2020-03-30 21:30, stop=A}
start_time=2020-03-30 21:00
|  {time_at_start=2020-03-30 21:00, time_at_stop=2020-03-30 21:30, stop=B}
start_time=2020-03-30 20:45
|  {time_at_start=2020-03-30 20:45, time_at_stop=2020-03-30 21:15, stop=B}
|  {time_at_start=2020-03-30 20:45, time_at_stop=2020-03-30 21:00, stop=A}
start_time=2020-03-30 20:30
|  {time_at_start=2020-03-30 20:30, time_at_stop=2020-03-30 21:00, stop=B}
start_time=2020-03-30 20:15
|  {time_at_start=2020-03-30 20:15, time_at_stop=2020-03-30 20:45, stop=B}
|  {time_at_start=2020-03-30 20:15, time_at_stop=2020-03-30 20:30, stop=A}
start_time=2020-03-30 20:00
|  {time_at_start=2020-03-30 20:00, time_at_stop=2020-03-30 20:30, stop=B}
start_time=2020-03-30 19:45
|  {time_at_start=2020-03-30 19:45, time_at_stop=2020-03-30 20:15, stop=B}
|  {time_at_start=2020-03-30 19:45, time_at_stop=2020-03-30 20:00, stop=A}
start_time=2020-03-30 19:30
|  {time_at_start=2020-03-30 19:30, time_at_stop=2020-03-30 20:00, stop=B}
start_time=2020-03-30 19:15
|  {time_at_start=2020-03-30 19:15, time_at_stop=2020-03-30 19:45, stop=B}
|  {time_at_start=2020-03-30 19:15, time_at_stop=2020-03-30 19:30, stop=A}
start_time=2020-03-30 19:00
|  {time_at_start=2020-03-30 19:00, time_at_stop=2020-03-30 19:30, stop=B}
start_time=2020-03-30 18:45
|  {time_at_start=2020-03-30 18:45, time_at_stop=2020-03-30 19:15, stop=B}
|  {time_at_start=2020-03-30 18:45, time_at_stop=2020-03-30 19:00, stop=A}
start_time=2020-03-30 18:30
|  {time_at_start=2020-03-30 18:30, time_at_stop=2020-03-30 19:00, stop=B}
start_time=2020-03-30 18:15
|  {time_at_start=2020-03-30 18:15, time_at_stop=2020-03-30 18:45, stop=B}
|  {time_at_start=2020-03-30 18:15, time_at_stop=2020-03-30 18:30, stop=A}
start_time=2020-03-30 18:00
|  {time_at_start=2020-03-30 18:00, time_at_stop=2020-03-30 18:30, stop=B}
start_time=2020-03-30 17:45
|  {time_at_start=2020-03-30 17:45, time_at_stop=2020-03-30 18:15, stop=B}
|  {time_at_start=2020-03-30 17:45, time_at_stop=2020-03-30 18:00, stop=A}
start_time=2020-03-30 17:30
|  {time_at_start=2020-03-30 17:30, time_at_stop=2020-03-30 18:00, stop=B}
start_time=2020-03-30 17:15
|  {time_at_start=2020-03-30 17:15, time_at_stop=2020-03-30 17:45, stop=B}
|  {time_at_start=2020-03-30 17:15, time_at_stop=2020-03-30 17:30, stop=A}
start_time=2020-03-30 17:00
|  {time_at_start=2020-03-30 17:00, time_at_stop=2020-03-30 17:30, stop=B}
start_time=2020-03-30 16:45
|  {time_at_start=2020-03-30 16:45, time_at_stop=2020-03-30 17:15, stop=B}
|  {time_at_start=2020-03-30 16:45, time_at_stop=2020-03-30 17:00, stop=A}
start_time=2020-03-30 16:30
|  {time_at_start=2020-03-30 16:30, time_at_stop=2020-03-30 17:00, stop=B}
start_time=2020-03-30 16:15
|  {time_at_start=2020-03-30 16:15, time_at_stop=2020-03-30 16:45, stop=B}
|  {time_at_start=2020-03-30 16:15, time_at_stop=2020-03-30 16:30, stop=A}
start_time=2020-03-30 16:00
|  {time_at_start=2020-03-30 16:00, time_at_stop=2020-03-30 16:30, stop=B}
start_time=2020-03-30 15:45
|  {time_at_start=2020-03-30 15:45, time_at_stop=2020-03-30 16:15, stop=B}
|  {time_at_start=2020-03-30 15:45, time_at_stop=2020-03-30 16:00, stop=A}
start_time=2020-03-30 15:30
|  {time_at_start=2020-03-30 15:30, time_at_stop=2020-03-30 16:00, stop=B}
start_time=2020-03-30 15:15
|  {time_at_start=2020-03-30 15:15, time_at_stop=2020-03-30 15:45, stop=B}
|  {time_at_start=2020-03-30 15:15, time_at_stop=2020-03-30 15:30, stop=A}
start_time=2020-03-30 15:00
|  {time_at_start=2020-03-30 15:00, time_at_stop=2020-03-30 15:30, stop=B}
start_time=2020-03-30 14:45
|  {time_at_start=2020-03-30 14:45, time_at_stop=2020-03-30 15:15, stop=B}
|  {time_at_start=2020-03-30 14:45, time_at_stop=2020-03-30 15:00, stop=A}
start_time=2020-03-30 14:30
|  {time_at_start=2020-03-30 14:30, time_at_stop=2020-03-30 15:00, stop=B}
start_time=2020-03-30 14:15
|  {time_at_start=2020-03-30 14:15, time_at_stop=2020-03-30 14:45, stop=B}
|  {time_at_start=2020-03-30 14:15, time_at_stop=2020-03-30 14:30, stop=A}
start_time=2020-03-30 14:00
|  {time_at_start=2020-03-30 14:00, time_at_stop=2020-03-30 14:30, stop=B}
start_time=2020-03-30 13:45
|  {time_at_start=2020-03-30 13:45, time_at_stop=2020-03-30 14:15, stop=B}
|  {time_at_start=2020-03-30 13:45, time_at_stop=2020-03-30 14:00, stop=A}
start_time=2020-03-30 13:30
|  {time_at_start=2020-03-30 13:30, time_at_stop=2020-03-30 14:00, stop=B}
start_time=2020-03-30 13:15
|  {time_at_start=2020-03-30 13:15, time_at_stop=2020-03-30 13:45, stop=B}
|  {time_at_start=2020-03-30 13:15, time_at_stop=2020-03-30 13:30, stop=A}
start_time=2020-03-30 13:00
|  {time_at_start=2020-03-30 13:00, time_at_stop=2020-03-30 13:30, stop=B}
start_time=2020-03-30 12:45
|  {time_at_start=2020-03-30 12:45, time_at_stop=2020-03-30 13:15, stop=B}
|  {time_at_start=2020-03-30 12:45, time_at_stop=2020-03-30 13:00, stop=A}
start_time=2020-03-30 12:30
|  {time_at_start=2020-03-30 12:30, time_at_stop=2020-03-30 13:00, stop=B}
start_time=2020-03-30 12:15
|  {time_at_start=2020-03-30 12:15, time_at_stop=2020-03-30 12:45, stop=B}
|  {time_at_start=2020-03-30 12:15, time_at_stop=2020-03-30 12:30, stop=A}
start_time=2020-03-30 12:00
|  {time_at_start=2020-03-30 12:00, time_at_stop=2020-03-30 12:30, stop=B}
start_time=2020-03-30 11:45
|  {time_at_start=2020-03-30 11:45, time_at_stop=2020-03-30 12:15, stop=B}
|  {time_at_start=2020-03-30 11:45, time_at_stop=2020-03-30 12:00, stop=A}
start_time=2020-03-30 11:30
|  {time_at_start=2020-03-30 11:30, time_at_stop=2020-03-30 12:00, stop=B}
start_time=2020-03-30 11:15
|  {time_at_start=2020-03-30 11:15, time_at_stop=2020-03-30 11:45, stop=B}
|  {time_at_start=2020-03-30 11:15, time_at_stop=2020-03-30 11:30, stop=A}
start_time=2020-03-30 11:00
|  {time_at_start=2020-03-30 11:00, time_at_stop=2020-03-30 11:30, stop=B}
start_time=2020-03-30 10:45
|  {time_at_start=2020-03-30 10:45, time_at_stop=2020-03-30 11:15, stop=B}
|  {time_at_start=2020-03-30 10:45, time_at_stop=2020-03-30 11:00, stop=A}
start_time=2020-03-30 10:30
|  {time_at_start=2020-03-30 10:30, time_at_stop=2020-03-30 11:00, stop=B}
start_time=2020-03-30 10:15
|  {time_at_start=2020-03-30 10:15, time_at_stop=2020-03-30 10:45, stop=B}
|  {time_at_start=2020-03-30 10:15, time_at_stop=2020-03-30 10:30, stop=A}
start_time=2020-03-30 10:00
|  {time_at_start=2020-03-30 10:00, time_at_stop=2020-03-30 10:30, stop=B}
start_time=2020-03-30 09:45
|  {time_at_start=2020-03-30 09:45, time_at_stop=2020-03-30 10:15, stop=B}
|  {time_at_start=2020-03-30 09:45, time_at_stop=2020-03-30 10:00, stop=A}
start_time=2020-03-30 09:30
|  {time_at_start=2020-03-30 09:30, time_at_stop=2020-03-30 10:00, stop=B}
start_time=2020-03-30 09:15
|  {time_at_start=2020-03-30 09:15, time_at_stop=2020-03-30 09:45, stop=B}
|  {time_at_start=2020-03-30 09:15, time_at_stop=2020-03-30 09:30, stop=A}
start_time=2020-03-30 09:00
|  {time_at_start=2020-03-30 09:00, time_at_stop=2020-03-30 09:30, stop=B}
start_time=2020-03-30 08:45
|  {time_at_start=2020-03-30 08:45, time_at_stop=2020-03-30 09:15, stop=B}
|  {time_at_start=2020-03-30 08:45, time_at_stop=2020-03-30 09:00, stop=A}
start_time=2020-03-30 08:30
|  {time_at_start=2020-03-30 08:30, time_at_stop=2020-03-30 09:00, stop=B}
start_time=2020-03-30 08:15
|  {time_at_start=2020-03-30 08:15, time_at_stop=2020-03-30 08:45, stop=B}
|  {time_at_start=2020-03-30 08:15, time_at_stop=2020-03-30 08:30, stop=A}
start_time=2020-03-30 08:00
|  {time_at_start=2020-03-30 08:00, time_at_stop=2020-03-30 08:30, stop=B}
start_time=2020-03-30 07:45
|  {time_at_start=2020-03-30 07:45, time_at_stop=2020-03-30 08:15, stop=B}
|  {time_at_start=2020-03-30 07:45, time_at_stop=2020-03-30 08:00, stop=A}
start_time=2020-03-30 07:30
|  {time_at_start=2020-03-30 07:30, time_at_stop=2020-03-30 08:00, stop=B}
start_time=2020-03-30 07:15
|  {time_at_start=2020-03-30 07:15, time_at_stop=2020-03-30 07:45, stop=B}
|  {time_at_start=2020-03-30 07:15, time_at_stop=2020-03-30 07:30, stop=A}
start_time=2020-03-30 07:00
|  {time_at_start=2020-03-30 07:00, time_at_stop=2020-03-30 07:30, stop=B}
start_time=2020-03-30 06:45
|  {time_at_start=2020-03-30 06:45, time_at_stop=2020-03-30 07:15, stop=B}
|  {time_at_start=2020-03-30 06:45, time_at_stop=2020-03-30 07:00, stop=A}
start_time=2020-03-30 06:30
|  {time_at_start=2020-03-30 06:30, time_at_stop=2020-03-30 07:00, stop=B}
start_time=2020-03-30 06:15
|  {time_at_start=2020-03-30 06:15, time_at_stop=2020-03-30 06:45, stop=B}
|  {time_at_start=2020-03-30 06:15, time_at_stop=2020-03-30 06:30, stop=A}
start_time=2020-03-30 06:00
|  {time_at_start=2020-03-30 06:00, time_at_stop=2020-03-30 06:30, stop=B}
start_time=2020-03-30 05:45
|  {time_at_start=2020-03-30 05:45, time_at_stop=2020-03-30 06:15, stop=B}
|  {time_at_start=2020-03-30 05:45, time_at_stop=2020-03-30 06:00, stop=A}
start_time=2020-03-30 05:30
|  {time_at_start=2020-03-30 05:30, time_at_stop=2020-03-30 06:00, stop=B}
start_time=2020-03-30 05:15
|  {time_at_start=2020-03-30 05:15, time_at_stop=2020-03-30 05:45, stop=B}
|  {time_at_start=2020-03-30 05:15, time_at_stop=2020-03-30 05:30, stop=A}
start_time=2020-03-30 05:00
|  {time_at_start=2020-03-30 05:00, time_at_stop=2020-03-30 05:30, stop=B}
start_time=2020-03-30 04:45
|  {time_at_start=2020-03-30 04:45, time_at_stop=2020-03-30 05:15, stop=B}
|  {time_at_start=2020-03-30 04:45, time_at_stop=2020-03-30 05:00, stop=A}
start_time=2020-03-30 04:30
|  {time_at_start=2020-03-30 04:30, time_at_stop=2020-03-30 05:00, stop=B}
start_time=2020-03-30 04:15
|  {time_at_start=2020-03-30 04:15, time_at_stop=2020-03-30 04:45, stop=B}
|  {time_at_start=2020-03-30 04:15, time_at_stop=2020-03-30 04:30, stop=A}
start_time=2020-03-30 04:00
|  {time_at_start=2020-03-30 04:00, time_at_stop=2020-03-30 04:30, stop=B}
start_time=2020-03-30 03:45
|  {time_at_start=2020-03-30 03:45, time_at_stop=2020-03-30 04:15, stop=B}
|  {time_at_start=2020-03-30 03:45, time_at_stop=2020-03-30 04:00, stop=A}
start_time=2020-03-30 03:30
|  {time_at_start=2020-03-30 03:30, time_at_stop=2020-03-30 04:00, stop=B}
start_time=2020-03-30 03:15
|  {time_at_start=2020-03-30 03:15, time_at_stop=2020-03-30 03:45, stop=B}
|  {time_at_start=2020-03-30 03:15, time_at_stop=2020-03-30 03:30, stop=A}
start_time=2020-03-30 03:00
|  {time_at_start=2020-03-30 03:00, time_at_stop=2020-03-30 03:30, stop=B}
start_time=2020-03-30 02:45
|  {time_at_start=2020-03-30 02:45, time_at_stop=2020-03-30 03:15, stop=B}
|  {time_at_start=2020-03-30 02:45, time_at_stop=2020-03-30 03:00, stop=A}
start_time=2020-03-30 02:30
|  {time_at_start=2020-03-30 02:30, time_at_stop=2020-03-30 03:00, stop=B}
start_time=2020-03-30 02:15
|  {time_at_start=2020-03-30 02:15, time_at_stop=2020-03-30 02:45, stop=B}
|  {time_at_start=2020-03-30 02:15, time_at_stop=2020-03-30 02:30, stop=A}
start_time=2020-03-30 02:00
|  {time_at_start=2020-03-30 02:00, time_at_stop=2020-03-30 02:30, stop=B}
start_time=2020-03-30 01:45
|  {time_at_start=2020-03-30 01:45, time_at_stop=2020-03-30 02:15, stop=B}
|  {time_at_start=2020-03-30 01:45, time_at_stop=2020-03-30 02:00, stop=A}
start_time=2020-03-30 01:30
|  {time_at_start=2020-03-30 01:30, time_at_stop=2020-03-30 02:00, stop=B}
start_time=2020-03-30 01:15
|  {time_at_start=2020-03-30 01:15, time_at_stop=2020-03-30 01:45, stop=B}
|  {time_at_start=2020-03-30 01:15, time_at_stop=2020-03-30 01:30, stop=A}
start_time=2020-03-30 01:00
|  {time_at_start=2020-03-30 01:00, time_at_stop=2020-03-30 01:30, stop=B}
start_time=2020-03-30 00:45
|  {time_at_start=2020-03-30 00:45, time_at_stop=2020-03-30 01:15, stop=B}
|  {time_at_start=2020-03-30 00:45, time_at_stop=2020-03-30 01:00, stop=A}
start_time=2020-03-30 00:30
|  {time_at_start=2020-03-30 00:30, time_at_stop=2020-03-30 01:00, stop=B}
start_time=2020-03-30 00:15
|  {time_at_start=2020-03-30 00:15, time_at_stop=2020-03-30 00:45, stop=B}
|  {time_at_start=2020-03-30 00:15, time_at_stop=2020-03-30 00:30, stop=A}
start_time=2020-03-30 00:00
|  {time_at_start=2020-03-30 00:00, time_at_stop=2020-03-30 00:30, stop=B}
)";

TEST(routing, start_times) {
  auto const src = source_idx_t{0U};
  auto tt = timetable{};
  tt.date_range_ = full_period();
  load_timetable(src, loader::hrd::hrd_5_20_26, files_simple(), tt);
  finalize(tt);

  using namespace date;
  auto const A = tt.locations_.location_id_to_idx_.at(
      location_id{.id_ = "0000001", .src_ = src});
  auto const B = tt.locations_.location_id_to_idx_.at(
      location_id{.id_ = "0000002", .src_ = src});
  auto starts = std::vector<start>{};
  get_starts(direction::kForward, tt, nullptr,
             interval<unixtime_t>{sys_days{2020_y / March / 30},
                                  sys_days{2020_y / March / 31}},
             {{A, 15_minutes, 0}, {B, 30_minutes, 0}},
             location_match_mode::kExact, false, starts, true, 0, 1);
  std::sort(begin(starts), end(starts),
            [](auto&& a, auto&& b) { return a > b; });
  starts.erase(std::unique(begin(starts), end(starts)), end(starts));

  std::stringstream ss;
  ss << "\n";
  utl::equal_ranges_linear(
      starts,
      [](start const& a, start const& b) {
        return a.time_at_start_ == b.time_at_start_;
      },
      [&](std::vector<start>::const_iterator const& from_it,
          std::vector<start>::const_iterator const& to_it) {
        ss << "start_time=" << from_it->time_at_start_ << "\n";
        for (auto const& s : it_range{from_it, to_it}) {
          ss << "|  {time_at_start=" << s.time_at_start_
             << ", time_at_stop=" << s.time_at_stop_
             << ", stop=" << tt.locations_.names_[s.stop_].view() << "}\n";
        }
      });

  EXPECT_EQ(std::string_view{expected}, ss.str());
}

mem_dir start_times_files() {
  return mem_dir::read(R"__(
"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
MTA,MOTIS Transit Authority,https://motis-project.de/,Europe/Berlin

# calendar_dates.txt
service_id,date,exception_type
D,20240608,1

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A0,A0,start_offset0,,,,,,
A1,A1,start_offset1,,,,,,
A2,A2,start_offset2,,,,,,
A3,A3,,,,,,,
A4,A4,,,,,,,
A5,A5,final_stop_of_A,,,,,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
A,MTA,A,A,A0 -> A5,0

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
A,D,AWE,AWE,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
AWE,02:00,02:00,A0,0,0,0
AWE,02:01,02:02,A1,1,0,0
AWE,02:06,02:07,A2,2,0,0
AWE,02:15,02:16,A3,3,0,0
AWE,02:20,02:21,A4,4,0,0
AWE,02:25,02:26,A5,5,0,0
)__");
}

constexpr interval<std::chrono::sys_days> start_times_period() {
  using namespace date;
  constexpr auto const from = (2024_y / June / 7).operator sys_days();
  constexpr auto const to = (2024_y / June / 9).operator sys_days();
  return {from, to};
}

constexpr auto const exp_fwd_journey = R"(
[2024-06-07 23:57, 2024-06-08 00:25]
TRANSFERS: 0
     FROM: (START, START) [2024-06-07 23:57]
       TO: (END, END) [2024-06-08 00:25]
leg 0: (START, START) [2024-06-07 23:57] -> (A0, A0) [2024-06-08 00:00]
  MUMO (id=23, duration=3)
leg 1: (A0, A0) [2024-06-08 00:00] -> (A5, A5) [2024-06-08 00:25]
   0: A0      A0..............................................                               d: 08.06 00:00 [08.06 02:00]  [{name=Tram A, day=2024-06-08, id=AWE, src=0}]
   1: A1      A1.............................................. a: 08.06 00:01 [08.06 02:01]  d: 08.06 00:02 [08.06 02:02]  [{name=Tram A, day=2024-06-08, id=AWE, src=0}]
   2: A2      A2.............................................. a: 08.06 00:06 [08.06 02:06]  d: 08.06 00:07 [08.06 02:07]  [{name=Tram A, day=2024-06-08, id=AWE, src=0}]
   3: A3      A3.............................................. a: 08.06 00:15 [08.06 02:15]  d: 08.06 00:16 [08.06 02:16]  [{name=Tram A, day=2024-06-08, id=AWE, src=0}]
   4: A4      A4.............................................. a: 08.06 00:20 [08.06 02:20]  d: 08.06 00:21 [08.06 02:21]  [{name=Tram A, day=2024-06-08, id=AWE, src=0}]
   5: A5      A5.............................................. a: 08.06 00:25 [08.06 02:25]
leg 2: (A5, A5) [2024-06-08 00:25] -> (END, END) [2024-06-08 00:25]
  MUMO (id=0, duration=0)


)";

TEST(routing, no_round_out_of_interval) {
  constexpr auto const src = source_idx_t{0U};
  auto const config = loader_config{};

  timetable tt;
  tt.date_range_ = start_times_period();
  register_special_stations(tt);
  gtfs::load_timetable(config, src, start_times_files(), tt);
  finalize(tt);

  auto const results = raptor_intermodal_search(
      tt, nullptr,
      {{tt.locations_.location_id_to_idx_.at({.id_ = "A0", .src_ = src}),
        3_minutes, 23U},
       {tt.locations_.location_id_to_idx_.at({.id_ = "A1", .src_ = src}),
        5_minutes, 23U},
       {tt.locations_.location_id_to_idx_.at({.id_ = "A2", .src_ = src}),
        10_minutes, 23U}},
      {{tt.locations_.location_id_to_idx_.at({.id_ = "A5", .src_ = src}),
        0_minutes, 0U}},
      interval{unixtime_t{sys_days{2024_y / June / 7} + 23_hours + 57_minutes},
               unixtime_t{sys_days{2024_y / June / 8}}},
      direction::kForward);

  std::stringstream ss;
  ss << "\n";
  for (auto const& x : results) {
    x.print(ss, tt);
    ss << "\n\n";
  }

  EXPECT_EQ(std::string_view{exp_fwd_journey}, ss.str());
}