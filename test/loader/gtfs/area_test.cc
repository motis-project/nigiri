#include <nigiri/loader/gtfs/booking_rule.h>

#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/area.h"
#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/parse_date.h"
#include "nigiri/loader/gtfs/parse_time.h"
#include "nigiri/timetable.h"

#include "./test_data.h"

using namespace nigiri;
using namespace nigiri::loader::gtfs;

TEST(gtfs, area) {
  timetable tt;
  source_idx_t src = source_idx_t{0};

  auto const files = example_files();

  auto const geojson = read_location_geojson(
      src, tt, files.get_file(kLocationGeojsonFile).data());

  tz_map timezones;

  auto const stops = read_stops(source_idx_t{0}, tt, timezones,
                                files.get_file(kStopFile).data(),
                                files.get_file(kTransfersFile).data(), 0U);

  auto const areas = read_areas(src, src, src, src, src, tt, stops, geojson,
                                files.get_file(kStopAreasFile).data(),
                                files.get_file(kLocationGroupsFile).data(),
                                files.get_file(kLocationGroupStopsFile).data());

  // Stop Area

  // Location Group

  // Location Group Stop
}
