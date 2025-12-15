#pragma once

#include <string_view>

namespace nigiri::loader::gtfs {

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
constexpr auto const kTranslationsFile = std::string_view{"translations.txt"};
constexpr auto const kFeedInfoFile = std::string_view{"feed_info.txt"};
constexpr auto const kFrequenciesFile = std::string_view{"frequencies.txt"};
constexpr auto const kFareAttributesFile =
    std::string_view{"fare_attributes.txt"};
constexpr auto const kFareRulesFile = std::string_view{"fare_rules.txt"};
constexpr auto const kLocationGroupsFile =
    std::string_view{"location_groups.txt"};
constexpr auto const kLocationGroupStopsFile =
    std::string_view{"location_group_stops.txt"};
constexpr auto const kLocationsFile = std::string_view{"locations.geojson"};
constexpr auto const kBookingRulesFile = std::string_view{"booking_rules.txt"};

// GTFS Fares v2 Files

}  // namespace nigiri::loader::gtfs
