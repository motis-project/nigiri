#include "nigiri/loader/gtfs/agency.h"

#include "date/tz.h"

#include "utl/parser/csv_range.h"
#include "utl/progress_tracker.h"

#include "nigiri/loader/register.h"
#include "nigiri/timetable.h"

namespace nigiri::loader::gtfs {
  struct agency_row {
    utl::csv_col<utl::cstr, UTL_NAME("agency_id")> id_;
    utl::csv_col<cista::raw::generic_string, UTL_NAME("agency_name")> name_;
    utl::csv_col<utl::cstr, UTL_NAME("agency_url")> url_;
    utl::csv_col<utl::cstr, UTL_NAME("agency_timezone")> tz_name_;
    utl::csv_col<utl::cstr, UTL_NAME("agency_fare_url")> fare_url_;
    // Google Transit Ticketing extension
    utl::csv_col<utl::cstr, UTL_NAME("ticketing_deep_link_id")>
        ticketing_deep_link_id;
  };

std::pair<agency_map_t, agency_ticketing_map_t> read_agencies(
    source_idx_t const src,
    timetable& tt,
    translator& i18n,
    tz_map& timezones,
    std::string_view file_content,
    std::string_view default_tz,
    script_runner const& r) {

  utl::get_active_progress_tracker()->status("Parse Agencies");

  auto map = agency_map_t{};
  auto ticketing_map = agency_ticketing_map_t{};
  utl::for_each_row<agency_row>(file_content, [&](agency_row const& a) {
    auto const agency_id = a.id_->trim();
    auto const timezone = a.tz_name_->trim();
    auto x = agency{
        tt,
        src,
        agency_id.view(),
        i18n.get(t::kAgency, f::kAgencyName, a.name_->view(), agency_id.view()),
        i18n.get(t::kAgency, f::kAgencyURL, a.url_->view(), agency_id.view()),
        i18n.get(t::kAgency, f::kAgencyFareUrl, a.fare_url_->view(),
                 agency_id.view()),
        get_tz_idx(
            tt, timezones,
            timezone.empty() ? default_tz : timezone.view()),
        timezones};
    auto const provider_idx = process_agency(r, x) ? register_agency(tt, x)
                                                   : provider_idx_t::invalid();
    map.emplace(agency_id.view(), provider_idx);

    if (!a.ticketing_deep_link_id->empty()) {
      ticketing_map.emplace(provider_idx,
                            std::string{a.ticketing_deep_link_id->view()});
    }
  });
  return {map, ticketing_map};
}

bool has_invalid_agency_timezone(std::string_view file, std::string_view default_timezone) {
  auto invalid_tz = false;
  utl::for_each_row<agency_row>(file, [&](agency_row const& agency) {
    auto const agency_timezone = agency.tz_name_->trim();
    auto const used_timezone = agency_timezone.empty() ? default_timezone : agency_timezone.view();
    if(used_timezone.empty()) {
      std::cout<<"agency "<<std::string{agency.id_->view()}<<" has empty timezone."<<std::endl;
      invalid_tz = true;
      return;
    }
    try {
      static_cast<void>(date::locate_zone(used_timezone));
    } catch (std::exception const&) {
      std::cout<<"agency "<<std::string{agency.id_->view()}<<" has invalid timezone."<<std::endl;
      invalid_tz = true;
    }
  });
  return invalid_tz;
}

}  // namespace nigiri::loader::gtfs
