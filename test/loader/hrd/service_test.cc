#include "doctest/doctest.h"

#include "nigiri/loader/hrd/bitfield.h"
#include "nigiri/loader/hrd/service.h"
#include "nigiri/loader/hrd/station.h"
#include "nigiri/loader/hrd/timezone.h"
#include "nigiri/byte_sizes.h"
#include "nigiri/print_transport.h"
#include "nigiri/section_db.h"

using namespace nigiri;
using namespace nigiri::loader::hrd;
using std::operator""sv;

//  0     1     2     3     4     5
//  A  -  B  -  C  -  D  -  E  -  F
//    100 - 110 - 111 - 011 - 001

// 100 [0, 3] A - B - C - D
// 010 [1, 4]     B - C - D - E
// 001 [2, 5]         C - D - E - F

constexpr auto const basic_info_file_content = R"(28.03.2020
31.03.2020
Fahrplan 2020$29.03.2020 03:15:02$5.20.39$INFO+
)";

constexpr auto const stations_file_content = R"(
0000001     A
0000002     B
0000003     C
0000004     D
0000005     E
0000006     F
0000007     G
0000008     H
0000009     I
0000010     F_META
)";

constexpr auto const station_metabhf_content = R"(
0000010: 0000006
)";

constexpr auto const station_geo_file_content = R"(
0000001  32.034466  54.798343 A
0000002  34.317551  55.197393 B
0000003  36.579810  56.376671 C
0000004  38.579810  57.276672 D
0000005  40.579810  58.176673 E
0000006  41.579810  59.076673 F
0000007  42.559810  61.066673 G
0000008  43.469810  62.056673 H
0000009  44.379810  63.046673 I
0000007  41.579799  59.076849 F_META
)";

// 000001 = E6 = 11|10 0|110 = 28.03
// 000002 = F6 = 11|11 0|110 = 28.03, 29.03
// 000003 = FE = 11|11 1|110 = 28.03, 29.03, 30.03
// 000004 = DE = 11|01 1|110 =        29.03, 30.03
// 000005 = CE = 11|00 1|110 =               30.03
constexpr auto const bitfields_file_content = R"(
000001 E6
000002 F6
000003 FE
000004 DE
000005 CE
)";

constexpr auto const timezones_file_content = R"(
0000000 +0100 +0200 29032020 0200 25102020 0300 +0200 28032021 0200 31102021 0300
)";

constexpr auto const service_file_content = R"(
*Z 01337 80____       002 120                             %
*A VE 0000001 0000002 000001                              %
*A VE 0000002 0000003 000002                              %
*A VE 0000003 0000004 000003                              %
*A VE 0000004 0000005 000004                              %
*A VE 0000005 0000006 000005                              %
*G ICE 0000001 0000006                                    %
0000001 A                            02200                %
0000002 B                     02301  02302                %
0000003 C                     02704  02805                %
0000004 D                     04506  04607                %
0000005 E                     05008  05009                %
0000006 F                     05110                       %
*Z 00815 80____                                           %
*A VE 0000007 0000006 000003                              %
*G RE  0000007 0000006                                    %
0000007 G                            00230                %
0000003 C                     00320  00405                %
0000004 D                     02106  02207                %
0000005 E                     02608  02609                %
0000006 F                     02710                       %
*Z 03374 80____                                           %
*A VE 0000008 0000006 000003                              %
*G IC  0000008 0000006                                    %
0000008 H                            00030                %
0000007 G                     00043  00045                %
0000009 I                     02524  02525                %
0000005 E                     02555  02709                %
0000006 F                     02710                       %
)";

constexpr auto const categories_file_content = R"(
ICE  0 A 0  ICE       2   Intercity-Express
IC   1 B 0  IC        2   Intercity
RE   3 C 0  RE        0 N Regional-Express
)";

constexpr auto const providers_file_content = R"(
00001 K '---' L 'DB AG' V 'Deutsche Bahn AG'
00001 : 80____
)";

constexpr auto const expected_trips = R"(
TRANSPORT=0, TRAFFIC_DAYS=000010
2020-03-28 (day_idx=1)
ROUTE=0
 0: 0000001 A...............................................                               d: 28.03 23:00 [29.03 00:00]  [{name=ICE 1337, day=2020-03-28, id=80____/1337/0000001/23:00, src=0, debug=services.txt:2:14}]
 1: 0000002 B............................................... a: 29.03 00:01 [29.03 01:01]  d: 29.03 00:02 [29.03 01:02]  [{name=ICE 1337, day=2020-03-28, id=80____/1337/0000001/23:00, src=0, debug=services.txt:2:14}]
 2: 0000003 C............................................... a: 29.03 03:04 [29.03 05:04]  d: 29.03 04:05 [29.03 06:05]  [{name=ICE 1337, day=2020-03-28, id=80____/1337/0000001/23:00, src=0, debug=services.txt:2:14}]
 3: 0000004 D............................................... a: 29.03 21:06 [29.03 23:06]

---

TRANSPORT=1, TRAFFIC_DAYS=010000
2020-03-31 (day_idx=4)
ROUTE=1
 0: 0000003 C...............................................                               d: 31.03 04:05 [31.03 06:05]  [{name=ICE 1337, day=2020-03-31, id=80____/1337/0000003/04:05, src=0, debug=services.txt:2:14}]
 1: 0000004 D............................................... a: 31.03 21:06 [31.03 23:06]  d: 31.03 22:07 [01.04 00:07]  [{name=ICE 1337, day=2020-03-31, id=80____/1337/0000003/04:05, src=0, debug=services.txt:2:14}]
 2: 0000005 E............................................... a: 01.04 02:08 [01.04 04:08]  d: 01.04 02:09 [01.04 04:09]  [{name=ICE 1337, day=2020-03-31, id=80____/1337/0000003/04:05, src=0, debug=services.txt:2:14}]
 3: 0000006 F............................................... a: 01.04 03:10 [01.04 05:10]

---

TRANSPORT=2, TRAFFIC_DAYS=010000
2020-03-31 (day_idx=4)
ROUTE=2
 0: 0000003 C...............................................                               d: 31.03 06:05 [31.03 08:05]  [{name=ICE 1337, day=2020-03-31, id=80____/1337/0000003/06:05, src=0, debug=services.txt:2:14}]
 1: 0000004 D............................................... a: 31.03 23:06 [01.04 01:06]  d: 01.04 00:07 [01.04 02:07]  [{name=ICE 1337, day=2020-03-31, id=80____/1337/0000003/06:05, src=0, debug=services.txt:2:14}]
 2: 0000005 E............................................... a: 01.04 04:08 [01.04 06:08]  d: 01.04 04:09 [01.04 06:09]  [{name=ICE 1337, day=2020-03-31, id=80____/1337/0000003/06:05, src=0, debug=services.txt:2:14}]
 3: 0000006 F............................................... a: 01.04 05:10 [01.04 07:10]

---

TRANSPORT=3, TRAFFIC_DAYS=001000
2020-03-30 (day_idx=3)
ROUTE=3
 0: 0000007 G...............................................                               d: 30.03 00:30 [30.03 02:30]  [{name=RE 815, day=2020-03-30, id=80____/815/0000007/00:30, src=0, debug=services.txt:15:22}]
 1: 0000003 C............................................... a: 30.03 01:20 [30.03 03:20]  d: 30.03 02:05 [30.03 04:05]  [{name=RE 815, day=2020-03-30, id=80____/815/0000007/00:30, src=0, debug=services.txt:15:22}]
 2: 0000004 D............................................... a: 30.03 19:06 [30.03 21:06]  d: 30.03 20:07 [30.03 22:07]  [{name=RE 815, day=2020-03-30, id=80____/815/0000007/00:30, src=0, debug=services.txt:15:22}]
 3: 0000005 E............................................... a: 31.03 00:08 [31.03 02:08]  d: 31.03 00:09 [31.03 02:09]  [{name=RE 815, day=2020-03-30, id=80____/815/0000007/00:30, src=0, debug=services.txt:15:22}]
 4: 0000006 F............................................... a: 31.03 01:10 [31.03 03:10]

---

TRANSPORT=4, TRAFFIC_DAYS=000010
2020-03-28 (day_idx=1)
ROUTE=4
 0: 0000008 H...............................................                               d: 28.03 23:30 [29.03 00:30]  [{name=IC 3374, day=2020-03-28, id=80____/3374/0000008/23:30, src=0, debug=services.txt:23:30}]
 1: 0000007 G............................................... a: 28.03 23:43 [29.03 00:43]  d: 28.03 23:45 [29.03 00:45]  [{name=IC 3374, day=2020-03-28, id=80____/3374/0000008/23:30, src=0, debug=services.txt:23:30}]
 2: 0000009 I............................................... a: 29.03 23:24 [30.03 01:24]  d: 29.03 23:25 [30.03 01:25]  [{name=IC 3374, day=2020-03-28, id=80____/3374/0000008/23:30, src=0, debug=services.txt:23:30}]
 3: 0000005 E............................................... a: 29.03 23:55 [30.03 01:55]  d: 30.03 01:09 [30.03 03:09]  [{name=IC 3374, day=2020-03-28, id=80____/3374/0000008/23:30, src=0, debug=services.txt:23:30}]
 4: 0000006 F............................................... a: 30.03 01:10 [30.03 03:10]

---

TRANSPORT=5, TRAFFIC_DAYS=000001
2020-03-27 (day_idx=0)
ROUTE=5
 0: 0000008 H...............................................                               d: 27.03 23:30 [28.03 00:30]  [{name=IC 3374, day=2020-03-27, id=80____/3374/0000008/23:30, src=0, debug=services.txt:23:30}]
 1: 0000007 G............................................... a: 27.03 23:43 [28.03 00:43]  d: 27.03 23:45 [28.03 00:45]  [{name=IC 3374, day=2020-03-27, id=80____/3374/0000008/23:30, src=0, debug=services.txt:23:30}]
 2: 0000009 I............................................... a: 29.03 00:24 [29.03 01:24]  d: 29.03 00:25 [29.03 01:25]  [{name=IC 3374, day=2020-03-27, id=80____/3374/0000008/23:30, src=0, debug=services.txt:23:30}]
 3: 0000005 E............................................... a: 29.03 00:55 [29.03 01:55]  d: 29.03 01:09 [29.03 03:09]  [{name=IC 3374, day=2020-03-27, id=80____/3374/0000008/23:30, src=0, debug=services.txt:23:30}]
 4: 0000006 F............................................... a: 29.03 01:10 [29.03 03:10]

---

TRANSPORT=6, TRAFFIC_DAYS=000100
2020-03-29 (day_idx=2)
ROUTE=6
 0: 0000008 H...............................................                               d: 29.03 22:30 [30.03 00:30]  [{name=IC 3374, day=2020-03-29, id=80____/3374/0000008/22:30, src=0, debug=services.txt:23:30}]
 1: 0000007 G............................................... a: 29.03 22:43 [30.03 00:43]  d: 29.03 22:45 [30.03 00:45]  [{name=IC 3374, day=2020-03-29, id=80____/3374/0000008/22:30, src=0, debug=services.txt:23:30}]
 2: 0000009 I............................................... a: 30.03 23:24 [31.03 01:24]  d: 30.03 23:25 [31.03 01:25]  [{name=IC 3374, day=2020-03-29, id=80____/3374/0000008/22:30, src=0, debug=services.txt:23:30}]
 3: 0000005 E............................................... a: 30.03 23:55 [31.03 01:55]  d: 31.03 01:09 [31.03 03:09]  [{name=IC 3374, day=2020-03-29, id=80____/3374/0000008/22:30, src=0, debug=services.txt:23:30}]
 4: 0000006 F............................................... a: 31.03 01:10 [31.03 03:10]

---

TRANSPORT=7, TRAFFIC_DAYS=000100
2020-03-29 (day_idx=2)
ROUTE=7
 0: 0000002 B...............................................                               d: 29.03 23:02 [30.03 01:02]  [{name=ICE 1337, day=2020-03-29, id=80____/1337/0000002/23:02, src=0, debug=services.txt:2:14}]
 1: 0000003 C............................................... a: 30.03 03:04 [30.03 05:04]  d: 30.03 04:05 [30.03 06:05]  [{name=ICE 1337, day=2020-03-29, id=80____/1337/0000002/23:02, src=0, debug=services.txt:2:14}]
 2: 0000004 D............................................... a: 30.03 21:06 [30.03 23:06]  d: 30.03 22:07 [31.03 00:07]  [{name=ICE 1337, day=2020-03-29, id=80____/1337/0000002/23:02, src=0, debug=services.txt:2:14}]
 3: 0000005 E............................................... a: 31.03 02:08 [31.03 04:08]

---

TRANSPORT=8, TRAFFIC_DAYS=001000
2020-03-30 (day_idx=3)
ROUTE=8
 0: 0000002 B...............................................                               d: 30.03 01:02 [30.03 03:02]  [{name=ICE 1337, day=2020-03-30, id=80____/1337/0000002/01:02, src=0, debug=services.txt:2:14}]
 1: 0000003 C............................................... a: 30.03 05:04 [30.03 07:04]  d: 30.03 06:05 [30.03 08:05]  [{name=ICE 1337, day=2020-03-30, id=80____/1337/0000002/01:02, src=0, debug=services.txt:2:14}]
 2: 0000004 D............................................... a: 30.03 23:06 [31.03 01:06]  d: 31.03 00:07 [31.03 02:07]  [{name=ICE 1337, day=2020-03-30, id=80____/1337/0000002/01:02, src=0, debug=services.txt:2:14}]
 3: 0000005 E............................................... a: 31.03 04:08 [31.03 06:08]

---

)"sv;

TEST_CASE("loader_hrd_service, parse multiple") {
  auto const& c = configs[0];
  timetable tt;
  auto const timezones = parse_timezones(c, tt, timezones_file_content);
  auto const locations =
      parse_stations(c, source_idx_t{0U}, timezones, tt, stations_file_content,
                     station_geo_file_content, station_metabhf_content);
  auto const bitfields = parse_bitfields(c, tt, bitfields_file_content);
  auto const categories = parse_categories(c, categories_file_content);
  auto const providers = parse_providers(c, providers_file_content);
  auto const interval = parse_interval(basic_info_file_content);
  tt.begin_ = std::chrono::sys_days{interval.first};
  tt.end_ = std::chrono::sys_days{interval.second};
  write_services(c, source_idx_t{0}, "services.txt", interval, bitfields,
                 timezones, locations, categories, providers,
                 service_file_content, tt, [](std::size_t) {});

  for (auto i = 0U; i != tt.transport_stop_times_.size(); ++i) {
  }

  std::stringstream out;
  out << "\n";
  auto const num_days = (tt.end_ - tt.begin_ + 1_days) / 1_days;
  for (auto i = 0U; i != tt.transport_stop_times_.size(); ++i) {
    auto const transport_idx = transport_idx_t{i};
    auto const traffic_days =
        tt.bitfields_.at(tt.transport_traffic_days_.at(transport_idx));
    out << "TRANSPORT=" << transport_idx << ", TRAFFIC_DAYS="
        << traffic_days.to_string().substr(traffic_days.size() - num_days)
        << "\n";
    for (auto day = tt.begin_; day <= tt.end_; day += 1_days) {
      auto const day_idx = day_idx_t{
          static_cast<day_idx_t ::value_t>((day - tt.begin_) / 1_days)};
      if (traffic_days.test(to_idx(day_idx))) {
        date::to_stream(out, "%F", day);
        out << " (day_idx=" << day_idx << ")\n";
        print_transport(tt, out, transport_idx, day_idx);
        out << "\n";
      }
    }
    out << "---\n\n";
  }
  CHECK(expected_trips == out.str());
}
