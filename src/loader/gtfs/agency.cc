#include "nigiri/loader/gtfs/agency.h"

#include "date/tz.h"

#include "utl/parser/csv_range.h"
#include "utl/progress_tracker.h"

#include "nigiri/loader/register.h"
#include "nigiri/timetable.h"

namespace nigiri::loader::gtfs {

agency_map_t read_agencies(source_idx_t const src,
                           timetable& tt,
                           translator& i18n,
                           tz_map& timezones,
                           std::string_view file_content,
                           std::string_view default_tz,
                           script_runner const& r) {
  struct agency_row {
    utl::csv_col<utl::cstr, UTL_NAME("agency_id")> id_;
    utl::csv_col<cista::raw::generic_string, UTL_NAME("agency_name")> name_;
    utl::csv_col<utl::cstr, UTL_NAME("agency_url")> url_;
    utl::csv_col<utl::cstr, UTL_NAME("agency_timezone")> tz_name_;
  };

  utl::get_active_progress_tracker()->status("Parse Agencies");

  auto map = agency_map_t{};
  utl::for_each_row<agency_row>(file_content, [&](agency_row const& a) {
    auto x = agency{
        tt,
        src,
        a.id_->view(),
        i18n.get(t::kAgency, f::kAgencyName, a.name_->view(), a.id_->view()),
        i18n.get(t::kAgency, f::kAgencyURL, a.url_->view(), a.id_->view()),
        get_tz_idx(
            tt, timezones,
            a.tz_name_->view().empty() ? default_tz : a.tz_name_->view()),
        timezones};
    map.emplace(a.id_->view(), process_agency(r, x)
                                   ? register_agency(tt, x)
                                   : provider_idx_t::invalid());
  });
  return map;
}

}  // namespace nigiri::loader::gtfs
