#include "gtest/gtest.h"

#include "utl/equal_ranges_linear.h"

#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/routing/start_times.h"

#include "../loader/hrd/hrd_timetable.h"

using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::routing;
using namespace nigiri::test_data::hrd_timetable;

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
             location_match_mode::kExact, false, starts, true);

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
