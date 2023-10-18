#include "nigiri/loader/netex/service_calendar.h"

#include "utl/enumerate.h"
#include "utl/parser/arg_parser.h"

#include "date/date.h"

#include "nigiri/logging.h"

namespace nigiri::loader::netex {

int resolve_day_idx(timetable const& tt, utl::cstr const& date) {
  // 0123456789
  // 2021-12-22
  auto const period_from_date = date::sys_days{date::year_month_day{
      date::year{utl::parse<int>(date.substr(0, utl::size{4}))},
      date::month{utl::parse<unsigned>(date.substr(5, utl::size{2}))},
      date::day{utl::parse<unsigned>(date.substr(8, utl::size{2}))}}};
  return (period_from_date - tt.internal_interval_days().from_).count();
}

void read_operating_periods(
    timetable const& tt,
    pugi::xml_document& doc,
    hash_map<std::string_view, bitfield>& operating_periods) {
  for (auto const n : doc.select_nodes("//UicOperatingPeriod")) {
    auto const offset =
        resolve_day_idx(tt, utl::cstr{n.node().child("FromDate").text().get()});
    auto bf = bitfield{};
    for (auto const [i, c] : utl::enumerate(
             std::string_view{n.node().child("ValidDayBits").text().get()})) {
      auto const idx = offset + static_cast<int>(i);
      if (idx < 0 || idx >= kMaxDays) {
        continue;
      }
      bf.set(static_cast<unsigned>(idx), c != '0');
    }
    operating_periods.emplace(
        std::string_view{n.node().attribute("id").value()}, bf);
  }
}

void read_service_calendar(
    timetable const& tt,
    pugi::xml_document& doc,
    hash_map<std::string_view, bitfield>& calendar,
    hash_map<std::string_view, bitfield>& operating_periods) {
  calendar.clear();
  operating_periods.clear();

  read_operating_periods(tt, doc, operating_periods);

  for (auto const n : doc.select_nodes("//DayTypeAssignment")) {
    auto const operating_period_ref = std::string_view{
        n.node().child("OperatingPeriodRef").attribute("ref").value()};

    auto const it = operating_periods.find(operating_period_ref);
    if (it == end(operating_periods)) {
      log(log_lvl::error, "loader.netex.calendar",
          "unable to resolve OperatingPeriodRef \"{}\"", operating_period_ref);
      continue;
    }

    calendar[n.node().child("DayTypeRef").attribute("ref").value()] =
        it->second;
  }
}

}  // namespace nigiri::loader::netex
