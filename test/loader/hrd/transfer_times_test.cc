#include "gtest/gtest.h"

#include <algorithm>
#include <string>

#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/timetable.h"

#include "../../raptor_search.h"
#include "./hrd_timetable.h"

using namespace date;
using namespace std::chrono_literals;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::test_data::hrd_timetable;

namespace {

// X (01001) arrives at B 10:30, Y (02002) leaves B 10:35 and Y2 (02003) 11:35,
// both to C. Only on 2020-03-30 (local time +02:00).
constexpr auto const kServices = R"(
*Z 01001 80____                                           %
*A VE 0000001 0000002 000005                              %
*G RE  0000001 0000002                                    %
0000001 A                            01000                %
0000002 B                     01030                       %
*Z 02002 80____                                           %
*A VE 0000002 0000003 000005                              %
*G RE  0000002 0000003                                    %
0000002 B                            01035                %
0000003 C                     01100                       %
*Z 02003 80____                                           %
*A VE 0000002 0000003 000005                              %
*G RE  0000002 0000003                                    %
0000002 B                            01135                %
0000003 C                     01200                       %
)";

// Default 10 min (B too, unless it has a line of its own).
constexpr auto const kStationTimes = "9999999 10 10 DEFAULT\n";
constexpr auto const kStationB = "0000002 10 10 B\n";

// X -> Y at B: 3 min.
constexpr auto const kTripRule =
    "0000002 001001 80____ 002002 80____ 003  000000 B\n";

// 80____ -> 80____ at every station without a transfer time of its own: 3 min.
constexpr auto const kGlobalAdminRule = "@@@@@@@ 80____ 80____ 03 GLOBAL\n";

timetable load(std::string const& station_times,
               std::string const& admin_rules,
               std::string const& trip_rules) {
  auto const& c = hrd::hrd_5_20_26;
  auto d = base();
  d.add({c.fplan_ / "services.101", kServices});
  d.add({c.core_data_ / c.transfers_.station_[0], station_times});
  d.add({c.core_data_ / c.transfers_.admin_[0], admin_rules});
  d.add({c.core_data_ / c.transfers_.trip_[0], trip_rules});

  auto tt = timetable{};
  tt.date_range_ = full_period();
  register_special_stations(tt);
  hrd::load_timetable(source_idx_t{0U}, c, d, tt);
  finalize(tt);
  return tt;
}

unixtime_t arrival(timetable const& tt) {
  auto const results = test::raptor_search(tt, nullptr, "0000001", "0000003",
                                           "2020-03-30 09:55 Europe/Berlin");
  return std::ranges::min(results, {}, &routing::journey::dest_time_)
      .dest_time_;
}

// local time 2020-03-30 (+02:00)
unixtime_t at(std::chrono::hours const h, std::chrono::minutes const m) {
  return unixtime_t{sys_days{2020_y / March / 30}} + h + m - 2h;
}

}  // namespace

TEST(hrd, transfer_times_station_time) {
  // X -> Y: 5 min, B needs 10.
  EXPECT_EQ(at(12h, 0min), arrival(load(kStationTimes, "", "")));
}

TEST(hrd, transfer_times_trip_rule) {
  EXPECT_EQ(at(11h, 0min), arrival(load(std::string{kStationTimes} + kStationB,
                                        "", kTripRule)));
}

TEST(hrd, transfer_times_global_rule_shadowed_by_station_time) {
  EXPECT_EQ(at(12h, 0min), arrival(load(std::string{kStationTimes} + kStationB,
                                        kGlobalAdminRule, "")));
}

TEST(hrd, transfer_times_global_rule) {
  EXPECT_EQ(at(11h, 0min), arrival(load(kStationTimes, kGlobalAdminRule, "")));
}
