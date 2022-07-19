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

TEST_CASE("loader_hrd_station, parse") {
  for (auto const& c : configs) {
    std::vector<service> services;
    info_db db{"./test_db", 512_kB, info_db::init_type::CLEAR};
    timetable tt;
    auto const locations =
        parse_stations(c, source_idx_t{0U}, tt, stations_file_content,
                       station_geo_file_content, station_metabhf_content);
  }
}
