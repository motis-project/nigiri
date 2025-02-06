#pragma once

#include <string_view>

namespace nigiri::loader::gtfs {

// GTFS files
constexpr auto const kAgencyFile = std::string_view{"agency.txt"};
constexpr auto const kStopFile = std::string_view{"stops.txt"};
constexpr auto const kRoutesFile = std::string_view{"routes.txt"};
constexpr auto const kTripsFile = std::string_view{"trips.txt"};
constexpr auto const kShapesFile = std::string_view{"shapes.txt"};
constexpr auto const kStopTimesFile = std::string_view{"stop_times.txt"};
constexpr auto const kCalenderFile = std::string_view{"calendar.txt"};
constexpr auto const kCalendarDatesFile =
    std::string_view{"calendar_dates.txt"};
constexpr auto const kTransfersFile = std::string_view{"transfers.txt"};
constexpr auto const kFeedInfoFile = std::string_view{"feed_info.txt"};
constexpr auto const kFrequenciesFile = std::string_view{"frequencies.txt"};

// GTFS-Flex files
constexpr auto const kLocationGroupsFile =
    std::string_view{"location_groups.txt"};
constexpr auto const kLocationGroupStopsFile =
    std::string_view{"location_group_stops.txt"};
constexpr auto const kBookingRulesFile = std::string_view{"booking_rules.txt"};
constexpr auto const kLocationGeojsonFile =
    std::string_view{"location.geojson"};
constexpr auto const kAreasFile = std::string_view{"areas.txt"};
constexpr auto const kStopAreasFile = std::string_view{"stop_areas.txt"};

// Files used for Tests only
constexpr auto const kBookingRuleCalendarFile =
    std::string_view{"booking_rule_calendar.txt"};
constexpr auto const kBookingRuleCalendarDatesFile =
    std::string_view{"booking_rule_calendar_dates.txt"};
constexpr auto const kStopTimesGTFSFlexFile =
    std::string_view{"stop_times_gtfs_flex.txt"};
constexpr auto const kRtreeStopFile = std::string_view{"rtree_stops.txt"};
constexpr auto const kRtreeLocationGroupFile =
    std::string_view{"rtree_location_groups.txt"};
constexpr auto const kRtreeLocationGeojsonFile =
    std::string_view{"rtree_location.geojson"};
constexpr auto const kCalculateDurationStopTimesFile =
    std::string_view{"calculate_duration_stop_times.txt"};
constexpr auto const kLocationsWithinGeometriesStopFile =
    std::string_view{"locations_within_geometries_stops.txt"};
constexpr auto const kLocationsWithinGeometriesGeojsonFile =
    std::string_view{"locations_within_geometries.geojson"};
}  // namespace nigiri::loader::gtfs