#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/stop_time.h"

using namespace nigiri;
using namespace nigiri::loader::gtfs;

TEST(gtfs, stop_sequence_exceeds_uint16_max) {
  constexpr auto const kStopTimes =
      R"(trip_id,arrival_time,departure_time,stop_id,stop_sequence,stop_headsign,pickup_type,drop_off_type,shape_dist_traveled
L001I01S1FES,08:00:00,08:00:00,6,100000,,0,0,
)";

  auto trips = trip_data{};
  trips.trips_.emplace("L001I01S1FES", gtfs_trip_idx_t{0U});
  trips.data_
      .emplace_back(route_id_idx_t::invalid(), nullptr, nullptr, "L001I01S1FES",
                    kEmptyTranslation, kEmptyTranslation,
                    direction_id_t::invalid(), shape_idx_t::invalid(), false,
                    false)
      .trip_idx_ = {};
  auto tt = timetable{};
  tt.trip_debug_.emplace_back().emplace_back(trip_debug{});
  auto i18n = translator{.tt_ = tt};
  auto stops = stops_map_t{};
  stops.emplace("6", location_idx_t{0});

  EXPECT_ANY_THROW(
      read_stop_times(trips, stops, {}, {}, {}, i18n, kStopTimes, false));
}
