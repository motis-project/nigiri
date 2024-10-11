#pragma once

#include <string_view>

namespace nigiri::loader::gtfs {

//GTFS files
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

}  // namespace nigiri::loader::gtfs

namespace nigiri::loader::gtfs_flex {
  constexpr auto const k_td_AgencyFile = std::string_view{"agency.txt"};
  constexpr auto const k_td_StopFile = std::string_view{"stops.txt"};
  constexpr auto const k_td_RoutesFile = std::string_view{"routes.txt"};
  constexpr auto const k_td_TripsFile = std::string_view{"trips.txt"};
  constexpr auto const k_td_StopTimesFile = std::string_view{"stop_times.txt"};
  constexpr auto const k_td_CalenderFile = std::string_view{"calendar.txt"};
  constexpr auto const k_td_CalendarDatesFile =
      std::string_view{"calendar_dates.txt"};
  //gtfs flex specific
  constexpr auto const k_td_LocationGroupsFile = std::string_view{"location_groups.txt"};
  constexpr auto const k_td_LocationGroupStopsFile = std::string_view{"location_group_stops.txt"};
  constexpr auto const k_td_BookingRulesFile = std::string_view{"booking_rules.txt"};
  constexpr auto const k_td_LocationGeojsonFile = std::string_view{"location.geojson"};
  constexpr auto const k_td_AreasFile = std::string_view{"areas.txt"};
  constexpr auto const k_td_StopAreasFile = std::string_view{"stop_areas.txt"};
}