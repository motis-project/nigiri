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
constexpr auto const kFeedInfoFile = std::string_view{"feed_info.txt"};
constexpr auto const kFrequenciesFile = std::string_view{"frequencies.txt"};
constexpr auto const kFareAttributesFile = std::string_view{"fare_attributes.txt"};
constexpr auto const kFareRulesFile = std::string_view{"fare_rules.txt"};

// GTFS Fares v2 Files
constexpr auto const kFareProductsFile = std::string_view{"fare_products.txt"};
constexpr auto const kFareMediaFile = std::string_view{"fare_media.txt"};
constexpr auto const kFareLegRulesFile = std::string_view{"fare_leg_rules.txt"};
constexpr auto const kFareLegJoinRulesFile = std::string_view{"fare_leg_join_rules.txt"};
constexpr auto const kFareTransferRulesFile = std::string_view{"fare_transfer_rules.txt"};
constexpr auto const kAreasFile = std::string_view{"areas.txt"};
constexpr auto const kStopAreasFile = std::string_view{"stop_areas.txt"};
constexpr auto const kTimeframesFile = std::string_view{"timeframes.txt"};
constexpr auto const kRiderCategoriesFile = std::string_view{"rider_categories.txt"};
constexpr auto const kNetworksFile = std::string_view{"networks.txt"};
constexpr auto const kRouteNetworksFile = std::string_view{"route_networks.txt"};

}  // namespace nigiri::loader::gtfs