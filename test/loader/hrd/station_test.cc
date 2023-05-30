#include "gtest/gtest.h"

#include "nigiri/loader/hrd/service/service.h"
#include "nigiri/loader/hrd/stamm/bitfield.h"
#include "nigiri/loader/hrd/stamm/station.h"
#include "nigiri/loader/hrd/stamm/timezone.h"
#include "nigiri/loader/init_finish.h"

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

TEST(hrd, parse_station) {
  for (auto const& c : configs) {
    timetable tt;
    auto const src = source_idx_t{0U};
    auto st = stamm{
        tt, timezone_map_t{
                {eva_number{0U},
                 std::pair<timezone_idx_t, tz_offsets>{
                     0U, tz_offsets{.seasons_ = {}, .offset_ = 0_minutes}}}}};
    auto const locations =
        parse_stations(c, src, tt, st, stations_file_content,
                       station_geo_file_content, station_metabhf_content);
    loader::finalize(tt);

    auto const l1 = tt.locations_.get(location_id{"0000001", src});
    EXPECT_EQ(l1.id_, "0000001");
    EXPECT_EQ(l1.src_, src);
  }
}
