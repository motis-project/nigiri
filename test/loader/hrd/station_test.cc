#include "doctest/doctest.h"

#include "nigiri/loader/hrd/bitfield.h"
#include "nigiri/loader/hrd/service.h"
#include "nigiri/loader/hrd/station.h"
#include "nigiri/loader/hrd/timezone.h"

using namespace nigiri;
using namespace nigiri::loader::hrd;

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

constexpr auto const timezones_file_content = R"(
0000000 +0100 +0200 29032020 0200 25102020 0300 +0200 28032021 0200 31102021 0300
)";

TEST_CASE("loader_hrd_station, parse") {
  for (auto const& c : configs) {
    std::vector<service> services;
    timetable tt;
    auto const src = source_idx_t{0U};
    auto const timezones = parse_timezones(c, tt, timezones_file_content);
    auto const locations =
        parse_stations(c, src, timezones, tt, stations_file_content,
                       station_geo_file_content, station_metabhf_content);

    auto const l1 = tt.locations_.get(location_id{"0000001", src});
    CHECK_EQ(l1.id_, "0000001");
    CHECK_EQ(l1.src_, src);
  }
}
