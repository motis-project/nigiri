#include <ranges>

#include "utl/parser/cstr.h"

#include "pugixml.hpp"

#include "gtest/gtest.h"

#include "nigiri/types.h"

namespace nigiri::loader::netex {

date::sys_days parse_date(utl::cstr);

hash_map<std::string_view, std::unique_ptr<bitfield>> get_operating_periods(
    pugi::xml_document const&, interval<date::sys_days> const&);

}  // namespace nigiri::loader::netex

using namespace std::string_view_literals;
using namespace nigiri;
using namespace nigiri::loader::netex;
using namespace date;

TEST(netex, operating_period) {
  constexpr auto kOperatingPeriods = R"(
<ServiceCalendarFrame>
  <ServiceCalendar>
    <operatingPeriods>
      <UicOperatingPeriod id="DE::UicOperatingPeriod:176089::" version="1763365460">
        <FromDate>2025-11-15T00:00:00</FromDate>
        <ToDate>2025-12-13T00:00:00</ToDate>
        <ValidDayBits>10000001000000000000010000001</ValidDayBits>
      </UicOperatingPeriod>
      <UicOperatingPeriod id="DE::UicOperatingPeriod:176090::" version="1763365460">
        <FromDate>2025-11-16T00:00:00</FromDate>
        <ToDate>2025-12-07T00:00:00</ToDate>
        <ValidDayBits>1000000100000000000001</ValidDayBits>
      </UicOperatingPeriod>
      <UicOperatingPeriod id="DE::UicOperatingPeriod:176092::" version="1763365460">
        <FromDate>2025-11-12T00:00:00</FromDate>
        <ToDate>2025-12-12T00:00:00</ToDate>
        <ValidDayBits>1110011111001111100111110011111</ValidDayBits>
      </UicOperatingPeriod>
    </operatingPeriods>
  </ServiceCalendar>
</ServiceCalendarFrame>
)"sv;
  auto doc = pugi::xml_document{};
  doc.load_buffer(kOperatingPeriods.data(), kOperatingPeriods.size(),
                  pugi::parse_default | pugi::parse_trim_pcdata);
  auto const operating_periods = get_operating_periods(
      doc, interval{parse_date("2025-12-02"), parse_date("2026-12-03")});
  fmt::println("operating_periods:\n\t{}",
               operating_periods | std::views::transform([](auto&& x) {
                 return std::pair{x.first, *x.second};
               }));
}

TEST(netex, parse_date) {
  EXPECT_EQ(sys_days{2025_y / December / 14}, parse_date("2025-12-14T00:00"));
  EXPECT_EQ(sys_days{2025_y / December / 14}, parse_date("2025-12-14"));
}