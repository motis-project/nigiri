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

//  0     1     2     3     4     5
//  A  -  B  -  C  -  D  -  E  -  F
//    100 - 110 - 111 - 011 - 001

// E6 = 11|10 0|110
// F6 = 11|11 0|110
// FE = 11|11 1|110
// DE = 11|01 1|110
// CE = 11|00 1|110

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
0000007     E
0000008     F
0000009     G
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
0000008 H                            02323                %
0000009 I                     02524  02525                %
0000005 E                     02605  02609                %
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

  auto const num_days = (tt.end_ - tt.begin_ + 1_days) / 1_days;
  std::cout << "TIMETABLE_LENGTH=" << num_days << " DAYS\n\n";
  for (auto i = 0U; i != tt.transport_stop_times_.size(); ++i) {
    auto const transport_idx = transport_idx_t{i};
    auto const traffic_days =
        tt.bitfields_.at(tt.transport_traffic_days_.at(transport_idx));
    std::cout << "TRANSPORT=" << transport_idx << ", TRAFFIC_DAYS="
              << traffic_days.to_string().substr(traffic_days.size() - num_days)
              << "\n";
    for (auto day = tt.begin_; day <= tt.end_; day += 1_days) {
      auto const day_idx = day_idx_t{
          static_cast<day_idx_t ::value_t>((day - tt.begin_) / 1_days)};
      if (traffic_days.test(to_idx(day_idx))) {
        date::to_stream(std::cout, "%F", day);
        std::cout << " (day_idx=" << day_idx << ")\n";
        print_transport(tt, std::cout, transport_idx, day_idx);
        std::cout << "\n";
      }
    }
    std::cout << "---\n\n";
  }
}
