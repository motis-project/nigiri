#include "nigiri/loader/gtfs/translations.h"

#include "utl/get_or_create.h"
#include "utl/parser/csv_range.h"
#include "utl/progress_tracker.h"

#include "nigiri/logging.h"

namespace nigiri::loader::gtfs {

translation_idx_t translator::get(t const t,
                                  f const f,
                                  std::string_view value,
                                  std::string_view record_id,
                                  std::string_view sub_record_id) {
  {  // Lookup translation by record.
    auto const it = i18n_.find(translation_key{
        .record_ =
            record{
                generic_string{record_id, generic_string::non_owning},
                generic_string{sub_record_id, generic_string::non_owning},
            },
        .table_ = t,
        .field_ = f,
    });
    if (it != end(i18n_)) {
      return it->second;
    }
  }

  {  // Lookup translation by original field value.
    auto const it = i18n_.find(translation_key{
        .record_ = generic_string{value, generic_string::non_owning},
        .table_ = t,
        .field_ = f,
    });
    if (it != end(i18n_)) {
      return it->second;
    }
  }

  if (value.empty()) {
    return kEmptyTranslation;
  }

  return utl::get_or_create(untranslated_, value,
                            [&]() { return tt_.register_translation(value); });
}

std::optional<t> parse_table(std::string_view s) {
  using cista::hash;
  switch (hash(s)) {
    case hash("agency"): return t::kAgency;
    case hash("stops"): return t::kStops;
    case hash("routes"): return t::kRoutes;
    case hash("trips"): return t::kTrips;
    case hash("stop_times"): return t::kStopTimes;
    case hash("pathways"): return t::kPathways;
    case hash("levels"): return t::kLevels;
    case hash("feed_info"): return t::kFeedInfo;
    case hash("attributions"): return t::kAttributions;
    default: return std::nullopt;
  }
}

std::optional<f> parse_field_name(std::string_view s) {
  using cista::hash;
  switch (hash(s)) {
    case hash("agency_name"): return f::kAgencyName;
    case hash("agency_url"): return f::kAgencyURL;
    case hash("agency_phone"): return f::kAgencyPhone;
    case hash("agency_fare_url"): return f::kAgencyFareUrl;
    case hash("agency_email"): return f::kAgencyEmail;
    case hash("stop_code"): return f::kStopCode;
    case hash("stop_name"): return f::kStopName;
    case hash("tts_stop_name"): return f::kTTSStopName;
    case hash("stop_desc"): return f::kStopDesc;
    case hash("stop_url"): return f::kStopURL;
    case hash("platform_code"): return f::kPlatformCode;
    case hash("route_short_name"): return f::kRouteShortName;
    case hash("route_long_name"): return f::kRouteLongName;
    case hash("route_desc"): return f::kRouteDesc;
    case hash("route_type"): return f::kRouteType;
    case hash("route_url"): return f::kRouteURL;
    case hash("trip_headsign"): return f::kTripHeadsign;
    case hash("trip_short_name"): return f::kTripShortName;
    case hash("stop_headsign"): return f::kStopHeadsign;
    case hash("rider_category_name"): return f::kRiderCategoryName;
    case hash("eligibility_url"): return f::kEligibilityURL;
    case hash("fare_media_name"): return f::kFareMediaName;
    case hash("fare_media_type"): return f::kFareMediaType;
    case hash("fare_product_name"): return f::kFareProductName;
    case hash("area_name"): return f::kAreaName;
    case hash("network_name"): return f::kNetworkName;
    case hash("level_name"): return f::kLevelName;
    case hash("location_group_name"): return f::kLocationGroupName;
    case hash("location_stop_name"): return f::kLocationStopName;
    case hash("location_stop_desc"): return f::kLocationStopDesc;
    case hash("message"): return f::kMessage;
    case hash("pickup_message"): return f::kPickupMessage;
    case hash("drop_off_message"): return f::kDropOffMessage;
    case hash("phone_number"): return f::kPhoneNumber;
    case hash("info_url"): return f::kInfoURL;
    case hash("booking_url"): return f::kBookingURL;
    case hash("feed_publisher_name"): return f::kFeedPublisherName;
    case hash("organization_name"): return f::kOrganizationName;
    case hash("attribution_url"): return f::kAttributionURL;
    case hash("attribution_email"): return f::kAttributionEmail;
    case hash("attribution_phone"): return f::kAttributionPhone;
    default: return std::nullopt;
  }
}

translator read_translations(timetable& tt,
                             std::string const& default_lang,
                             std::string_view file_content) {
  utl::get_active_progress_tracker()->status("Parse Translations");

  struct translation_row {
    utl::csv_col<utl::cstr, UTL_NAME("table_name")> table_name_;
    utl::csv_col<utl::cstr, UTL_NAME("field_name")> field_name_;
    utl::csv_col<utl::cstr, UTL_NAME("language")> language_;
    utl::csv_col<utl::cstr, UTL_NAME("translation")> translation_;
    utl::csv_col<std::optional<utl::cstr>, UTL_NAME("record_id")> record_id_;
    utl::csv_col<std::optional<utl::cstr>, UTL_NAME("record_sub_id")>
        record_sub_id_;
    utl::csv_col<std::optional<utl::cstr>, UTL_NAME("field_value")>
        field_value_;
  };

  auto translations = hash_map<translation_key, std::vector<translation>>{};
  utl::for_each_row<translation_row>(
      file_content, [&](translation_row const& a) {
        auto key = translation_key{};

        auto const table = parse_table(a.table_name_->view());
        if (!table.has_value()) {
          return;
        }
        key.table_ = *table;

        auto const field = parse_field_name(a.field_name_->view());
        if (!field.has_value()) {
          return;
        }
        key.field_ = *field;

        if (a.record_id_->has_value()) {
          key.record_ = record{
              .record_id_ = {(*a.record_id_)->view(), generic_string::owning},
              .record_sub_id_ = {a.record_sub_id_->value_or("").view(),
                                 generic_string::owning},
          };
        } else if (a.field_value_->has_value()) {
          key.record_ =
              generic_string{(*a.field_value_)->view(), generic_string::owning};
        } else {
          log(log_lvl::error, "nigiri.loader.gtfs.translations",
              "translation w/o record_id/field_value (table={}, field={})",
              a.table_name_->view(), a.field_name_->view());
          return;
        }

        translations[key].push_back(
            translation{a.language_->view(), a.translation_->view()});
      });

  auto t = translator{.tt_ = tt};
  for (auto& [key, x] : translations) {
    auto const unsorted = x;
    utl::sort(x, [&](translation const& a, translation const& b) {
      return (a.get_language() != default_lang) <
             (b.get_language() != default_lang);
    });
    t.i18n_.emplace(key, tt.register_translation(x));
  }
  return t;
}

}  // namespace nigiri::loader::gtfs
