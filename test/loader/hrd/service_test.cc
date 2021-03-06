#include "doctest/doctest.h"

#include "nigiri/loader/hrd/bitfield.h"
#include "nigiri/loader/hrd/service.h"
#include "nigiri/loader/hrd/station.h"
#include "nigiri/loader/hrd/timezone.h"
#include "nigiri/loader/hrd/util.h"
#include "nigiri/byte_sizes.h"
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
0000007     F_META
)";

constexpr auto const station_metabhf_content = R"(
0000007: 0000006
)";

constexpr auto const station_geo_file_content = R"(
0000001  32.034466  54.798343 A
0000002  34.317551  55.197393 B
0000003  36.579810  56.376671 C
0000004  38.579810  57.276672 D
0000005  40.579810  58.176673 E
0000006  41.579810  59.076673 F
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
*Z 00001 admin_       002 120                             %
*A VE 0000001 0000002 000001                              %
*A VE 0000002 0000003 000002                              %
*A VE 0000003 0000004 000003                              %
*A VE 0000004 0000005 000004                              %
*A VE 0000005 0000006 000005                              %
*G abc 0000001 0000006                                    %
0000001 A                            02200                %
0000002 B                     02301  02302                %
0000003 C                     02704  02805                %
0000004 D                     04506  04607                %
0000005 E                     05008  05009                %
0000006 F                     05110                       %
*Z 00002 admin_                                           %
*A VE 0000007 0000006 000003                              %
*G abc 0000007 0000006                                    %
0000007 G                            00230                %
0000003 C                     00320  00405                %
0000004 D                     02106  02207                %
0000005 E                     02608  02609                %
0000006 F                     02710                       %
*Z 00003 admin_                                           %
*A VE 0000008 0000006 000003                              %
*G abc 0000008 0000006                                    %
0000008 H                            02323                %
0000009 I                     02524  02525                %
0000005 E                     02605  02609                %
0000006 F                     02710                       %
)";

TEST_CASE("loader_hrd_service, parse multiple") {
  for (auto const& c : configs) {
    std::vector<service> services;
    info_db db{"./test_db", 512_kB, info_db::init_type::CLEAR};
    timetable tt;
    auto const locations =
        parse_stations(c, source_idx_t{0U}, tt, stations_file_content,
                       station_geo_file_content, station_metabhf_content);
    auto const bitfields = parse_bitfields(c, db, bitfields_file_content);
    auto const timezones = parse_timezones(c, timezones_file_content);
    auto const interval = parse_interval(basic_info_file_content);
    parse_services(
        c, "services.101", interval, bitfields, timezones, service_file_content,
        [](std::size_t) {},
        [&](service&& s) { services.emplace_back(std::move(s)); });
    CHECK_EQ(services.size(), 8);
  }
}
