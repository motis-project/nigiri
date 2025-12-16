#pragma once

#include <array>
#include <string_view>

#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::loader::gtfs {

enum t : std::uint8_t {
  kAgency,
  kStops,
  kRoutes,
  kTrips,
  kStopTimes,
  kPathways,
  kLevels,
  kFeedInfo,
  kAttributions,
  kLocationGroups,
  kBookingRules,
};

enum f : std::uint8_t {
  // agency.txt
  kAgencyName,
  kAgencyURL,
  kAgencyPhone,
  kAgencyFareUrl,
  kAgencyEmail,

  // stops.txt
  kStopCode,
  kStopName,
  kTTSStopName,
  kStopDesc,
  kStopURL,
  kPlatformCode,

  // routes.txt
  kRouteShortName,
  kRouteLongName,
  kRouteDesc,
  kRouteType,
  kRouteURL,

  // trips.txt
  kTripHeadsign,
  kTripShortName,

  // stop_times.txt
  kStopHeadsign,

  // rider_categories.txt
  kRiderCategoryName,
  kEligibilityURL,

  // fare_media.txt
  kFareMediaName,
  kFareMediaType,

  // fare_products.txt
  kFareProductName,

  // areas.txt
  kAreaName,

  // networks.txt
  kNetworkName,

  // levels.txt
  kLevelName,

  // location_groups.txt
  kLocationGroupName,

  // locations.geojson
  kLocationStopName,
  kLocationStopDesc,

  // booking_rules.txt
  kMessage,
  kPickupMessage,
  kDropOffMessage,
  kPhoneNumber,
  kInfoURL,
  kBookingURL,

  // feed_info.txt
  kFeedPublisherName,

  // attributions.txt
  kOrganizationName,
  kAttributionURL,
  kAttributionEmail,
  kAttributionPhone,
};

struct record {
  CISTA_FRIEND_COMPARABLE(record)

  generic_string record_id_;
  generic_string record_sub_id_;
};

struct translation_key {
  CISTA_FRIEND_COMPARABLE(translation_key)

  using record_t = std::variant<generic_string /* original value */, record>;

  cista::hash_t hash() const {
    auto h = cista::BASE_HASH;
    h = cista::hash_combine(h, cista::hashing<record_t>{}(record_));
    h = cista::hash_combine(h, table_);
    h = cista::hash_combine(h, field_);
    return h;
  }

  record_t record_;
  t table_;
  f field_;
};

struct translator {
  translation_idx_t get(t,
                        f,
                        std::string_view value,
                        std::string_view record_id,
                        std::string_view sub_record_id = "");

  timetable& tt_;
  hash_map<translation_key, translation_idx_t> i18n_{};
  hash_map<std::string, translation_idx_t> untranslated_{};
};

translator read_translations(timetable&,
                             std::string const& default_lang,
                             std::string_view file_content);

}  // namespace nigiri::loader::gtfs
