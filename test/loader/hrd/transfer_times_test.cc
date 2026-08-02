#include "gtest/gtest.h"

#include "nigiri/common/parse_time.h"
#include "nigiri/loader/dir.h"
#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/timetable.h"

#include "../../raptor_search.h"

using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::hrd;
using namespace date;
using nigiri::test::raptor_search;

namespace {

// Stations: A (origin), X (transfer hub with rules), B/C/D (destinations)
//
// Transfer rules at X (eva 0000002), station transfer time 5 min:
// - umsteigv: 80____ -> 81____ = 10 min (longer than station time)
// - umsteigl: (80____, RE, L1) -> (80____, RE, L2) = 2 min (shorter)
// - umsteigz: trip 21 -> trip 22 = 0 min, only on 28.03. (bitfield 000001)
//
// Trips (traffic days 28.-30.03., bitfield 000003):
// feeder1 (80): A 10:00 -> X 10:30
// out1 (81):    X 10:35 -> B 11:00  (banned: needs 10 min, has 5)
// out2 (81):    X 10:42 -> B 11:10  (ok: 12 min > 10)
// feeder2 (80, L1): A 12:00 -> X 12:30
// out3 (80, L2):    X 12:33 -> C 13:00  (ok: rule 2 min < station 5)
// feeder3 (80, nr 21): A 14:00 -> X 14:30
// out4 (80, nr 22):    X 14:30 -> D 15:00  (ok on 28.03. only: 0 min rule)

constexpr auto const basic_info = R"(26.03.2020
02.04.2020
Fahrplan 2020$29.03.2020 03:15:02$5.20.39$INFO+
)";

constexpr auto const stations = R"(
0000001     A
0000002     X
0000003     B
0000004     C
0000005     D
)";

constexpr auto const coordinates = R"(
0000001  32.034466  54.798343 A
0000002  34.317551  55.197393 X
0000003  36.579810  56.376671 B
0000004  38.579810  57.276672 C
0000005  40.579810  58.176673 D
)";

// 000001 = C86 = 28.03, 000003 = CE6 = 28.03 - 30.03
constexpr auto const bitfields = R"(
000001 C86
000003 CE6
)";

constexpr auto const timezones = R"(
0000000 +0100 +0200 29032020 0200 25102020 0300 +0200 28032021 0200 31102021 0300
)";

constexpr auto const categories = R"(
RE   3 C 0  RE        0 N Regional-Express
)";

constexpr auto const providers = R"(
00001 K '---' L 'DB' V 'DB AG'
00001 : 80____
00002 K '---' L 'AG2' V 'Agency 2'
00002 : 81____
)";

constexpr auto const umsteigb = R"(9999999  3  3
0000002  5  5 X
)";

constexpr auto const umsteigv = R"(
0000002 80____ 81____ 10 X
)";

constexpr auto const umsteigl = R"(
0000002 80____ RE  L1       * 80____ RE  L2       * 002
)";

constexpr auto const umsteigz = R"(
0000002 000021 80____ 000022 80____ 000  000001
)";

constexpr auto const services = R"(
*Z 00001 80____                                           %
*A VE 0000001 0000002 000003                              %
*G RE  0000001 0000002                                    %
0000001 A                            01000                %
0000002 X                     01030                       %
*Z 00002 81____                                           %
*A VE 0000002 0000003 000003                              %
*G RE  0000002 0000003                                    %
0000002 X                            01035                %
0000003 B                     01100                       %
*Z 00003 81____                                           %
*A VE 0000002 0000003 000003                              %
*G RE  0000002 0000003                                    %
0000002 X                            01042                %
0000003 B                     01110                       %
*Z 00011 80____                                           %
*A VE 0000001 0000002 000003                              %
*G RE  0000001 0000002                                    %
*L L1       0000001 0000002                               %
0000001 A                            01200                %
0000002 X                     01230                       %
*Z 00012 80____                                           %
*A VE 0000002 0000004 000003                              %
*G RE  0000002 0000004                                    %
*L L2       0000002 0000004                               %
0000002 X                            01233                %
0000004 C                     01300                       %
*Z 00021 80____                                           %
*A VE 0000001 0000002 000003                              %
*G RE  0000001 0000002                                    %
0000001 A                            01400                %
0000002 X                     01430                       %
*Z 00022 80____                                           %
*A VE 0000002 0000005 000003                              %
*G RE  0000002 0000005                                    %
0000002 X                            01430                %
0000005 D                     01500                       %
)";

mem_dir test_files() {
  auto const& c = hrd_5_20_26;
  auto const& b = c.core_data_;
  auto const& r = c.required_files_;
  return {{{(b / r[ATTRIBUTES][0]), ""},
           {(b / r[STATIONS][0]), stations},
           {(b / r[COORDINATES][0]), coordinates},
           {(b / r[BITFIELDS][0]), bitfields},
           {(b / r[TRACKS][0]), ""},
           {(b / r[INFOTEXT][0]), ""},
           {(b / r[BASIC_DATA][0]), basic_info},
           {(b / r[CATEGORIES][0]), categories},
           {(b / r[DIRECTIONS][0]), ""},
           {(b / r[PROVIDERS][0]), providers},
           {(b / r[THROUGH_SERVICES][0]), ""},
           {(b / r[MERGE_SPLIT_SERVICES][0]), ""},
           {(b / r[TIMEZONES][0]), timezones},
           {(b / r[FOOTPATHS][0]), ""},
           {(b / c.transfers_.station_[0]), umsteigb},
           {(b / c.transfers_.admin_[0]), umsteigv},
           {(b / c.transfers_.line_[0]), umsteigl},
           {(b / c.transfers_.trip_[0]), umsteigz},
           {(c.fplan_ / "services.101"), services}}};
}

constexpr interval<std::chrono::sys_days> period() {
  using namespace date;
  constexpr auto const from = (2020_y / March / 28).operator sys_days();
  constexpr auto const to = (2020_y / March / 31).operator sys_days();
  return {from, to};
}

timetable load(bool const transitive_footpaths) {
  auto tt = timetable{};
  tt.date_range_ = period();
  register_special_stations(tt);
  auto const d = test_files();
  load_timetable(source_idx_t{0U}, hrd_5_20_26, d, tt);
  finalize(tt, finalize_options{.adjust_footpaths_ = false,
                                .merge_dupes_intra_src_ = false,
                                .merge_dupes_inter_src_ = false,
                                .max_footpath_length_ = 20U,
                                .transitive_footpaths_ = transitive_footpaths});
  return tt;
}

unixtime_t t(char const* s) { return parse_time_tz(s, "%Y-%m-%d %H:%M %Z"); }

}  // namespace

TEST(hrd, transfer_times_umsteigb) {
  auto const tt = load(false);

  auto const x = tt.locations_.location_id_to_idx_.at(
      {std::string_view{"0000002"}, source_idx_t{0U}});
  auto const a = tt.locations_.location_id_to_idx_.at(
      {std::string_view{"0000001"}, source_idx_t{0U}});

  // X: explicit entry, A: default from first line
  EXPECT_EQ(u8_minutes{5}, tt.locations_.transfer_time_[x]);
  EXPECT_EQ(u8_minutes{3}, tt.locations_.transfer_time_[a]);

  // transfer groups at X
  auto n_transfer_children = 0U;
  for (auto const c : tt.locations_.children_[x]) {
    if (tt.locations_.types_[c] == location_type::kVirt) {
      ++n_transfer_children;
    }
  }
  EXPECT_EQ(6U, n_transfer_children);
}

TEST(hrd, transfer_times_admin_rule_longer) {
  auto const tt = load(false);

  // 80____ -> 81____ needs 10 min: the 10:35 departure (5 min gap) is not
  // reachable, the 10:42 departure is
  auto const res = raptor_search(tt, nullptr, "0000001", "0000003",
                                 "2020-03-28 09:00 Europe/Berlin");
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(t("2020-03-28 11:10 Europe/Berlin"), begin(res)->dest_time_);
}

TEST(hrd, transfer_times_admin_rule_longer_transitive) {
  // same expectation with transitive closure enabled: the rule edges
  // have to override the transitively computed footpaths
  auto const tt = load(true);

  auto const res = raptor_search(tt, nullptr, "0000001", "0000003",
                                 "2020-03-28 09:00 Europe/Berlin");
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(t("2020-03-28 11:10 Europe/Berlin"), begin(res)->dest_time_);
}

TEST(hrd, transfer_times_line_rule_shorter) {
  auto const tt = load(false);

  // (80, RE, L1) -> (80, RE, L2) needs only 2 min: the 12:33 departure is
  // reachable although the station transfer time is 5 min
  auto const res = raptor_search(tt, nullptr, "0000001", "0000004",
                                 "2020-03-28 11:00 Europe/Berlin");
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(t("2020-03-28 13:00 Europe/Berlin"), begin(res)->dest_time_);
}

TEST(hrd, transfer_times_trip_pair_bitfield) {
  auto const tt = load(false);

  // trip 21 -> trip 22: 0 min, valid only on 28.03.
  auto const res_28 = raptor_search(tt, nullptr, "0000001", "0000005",
                                    "2020-03-28 13:00 Europe/Berlin");
  ASSERT_EQ(1U, res_28.size());
  EXPECT_EQ(t("2020-03-28 15:00 Europe/Berlin"), begin(res_28)->dest_time_);

  // 29.03. (and 30.03.): rule inactive, station transfer time 5 min
  // -> the 14:30 departure is not reachable from the 14:30 arrival,
  //    earliest arrival is with the next day's trip 22
  auto const res_29 = raptor_search(tt, nullptr, "0000001", "0000005",
                                    "2020-03-29 13:00 Europe/Berlin");
  ASSERT_EQ(1U, res_29.size());
  EXPECT_EQ(t("2020-03-30 15:00 Europe/Berlin"), begin(res_29)->dest_time_);
}
