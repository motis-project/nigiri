#include "gtest/gtest.h"

#include "nigiri/loader/netex/service_calendar.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader::netex;

constexpr auto const netex_input =
    R"(<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<PublicationDelivery xmlns="http://www.netex.org.uk/netex" version="ntx:1.1">
    <dataObjects>
        <CompositeFrame id="DE::CompositeFrame_EU_PI_LINE_OFFER:144-SILBUS-V95" version="1641291376">
            <ValidBetween>
                <FromDate>2021-12-21T00:00:00</FromDate>
                <ToDate>2022-06-11T00:00:00</ToDate>
            </ValidBetween>
            <frames>
                <ServiceCalendarFrame id="DE::ServiceCalendarFrame_EU_PI_CALENDAR:144-SILBUS-V95" version="1641291376">
                    <TypeOfFrameRef ref="epip:EU_PI_CALENDAR" versionRef="1.0"/>
                    <ServiceCalendar id="DE::ServiceCalendar:17043::" version="1641291376">
                        <FromDate>2021-12-21</FromDate>
                        <ToDate>2022-06-11</ToDate>
                        <dayTypes>
                            <DayType id="DE::DayType:83969::" version="1641291376"/>
                            <DayType id="DE::DayType:83970::" version="1641291376"/>
                            <DayType id="DE::DayType:83971::" version="1641291376"/>
                            <DayType id="DE::DayType:83972::" version="1641291376"/>
                            <DayType id="DE::DayType:83973::" version="1641291376"/>
                            <DayType id="DE::DayType:83974::" version="1641291376"/>
                            <DayType id="DE::DayType:83975::" version="1641291376"/>
                            <DayType id="DE::DayType:83976::" version="1641291376"/>
                            <DayType id="DE::DayType:83977::" version="1641291376"/>
                        </dayTypes>
                        <operatingPeriods>
                            <UicOperatingPeriod id="DE::UicOperatingPeriod:83969::" version="1641291376">
                                <FromDate>2021-12-21T00:00:00</FromDate>
                                <ToDate>2022-06-10T00:00:00</ToDate>
                                <ValidDayBits>
                                    1100000000000000000011111001111100111110011111001111100111110011111000011100111110011111001111100111110011111000000000000000011111001111100111110011111001110000111110000111
                                </ValidDayBits>
                            </UicOperatingPeriod>
                            <UicOperatingPeriod id="DE::UicOperatingPeriod:83970::" version="1641291376">
                                <FromDate>2022-02-04T00:00:00</FromDate>
                                <ToDate>2022-04-08T00:00:00</ToDate>
                                <ValidDayBits>1000000000000000000000000000000000000000000000000000000000000001
                                </ValidDayBits>
                            </UicOperatingPeriod>
                            <UicOperatingPeriod id="DE::UicOperatingPeriod:83971::" version="1641291376">
                                <FromDate>2022-01-10T00:00:00</FromDate>
                                <ToDate>2022-06-10T00:00:00</ToDate>
                                <ValidDayBits>
                                    11111001111100111110011110001111100111110011111000011100111110011111001111100111110011110000000000000000011111001111100111110011111001110000111110000111
                                </ValidDayBits>
                            </UicOperatingPeriod>
                            <UicOperatingPeriod id="DE::UicOperatingPeriod:83972::" version="1641291376">
                                <FromDate>2021-12-22T00:00:00</FromDate>
                                <ToDate>2022-04-08T00:00:00</ToDate>
                                <ValidDayBits>
                                    100000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000001
                                </ValidDayBits>
                            </UicOperatingPeriod>
                            <UicOperatingPeriod id="DE::UicOperatingPeriod:83973::" version="1641291376">
                                <FromDate>2021-12-21T00:00:00</FromDate>
                                <ToDate>2022-06-10T00:00:00</ToDate>
                                <ValidDayBits>
                                    1000000000000000000011111001111100111110011110001111100111110011111000011100111110011111001111100111110011110000000000000000011111001111100111110011111001110000111110000111
                                </ValidDayBits>
                            </UicOperatingPeriod>
                            <UicOperatingPeriod id="DE::UicOperatingPeriod:83974::" version="1641291376">
                                <FromDate>2021-12-22T00:00:00</FromDate>
                                <ToDate>2021-12-22T00:00:00</ToDate>
                                <ValidDayBits>1</ValidDayBits>
                            </UicOperatingPeriod>
                            <UicOperatingPeriod id="DE::UicOperatingPeriod:83975::" version="1641291376">
                                <FromDate>2021-12-21T00:00:00</FromDate>
                                <ToDate>2021-12-21T00:00:00</ToDate>
                                <ValidDayBits>1</ValidDayBits>
                            </UicOperatingPeriod>
                            <UicOperatingPeriod id="DE::UicOperatingPeriod:83976::" version="1641291376">
                                <FromDate>2022-01-10T00:00:00</FromDate>
                                <ToDate>2022-06-10T00:00:00</ToDate>
                                <ValidDayBits>
                                    11111001111100111110011111001111100111110011111000011100111110011111001111100111110011111000000000000000011111001111100111110011111001110000111110000111
                                </ValidDayBits>
                            </UicOperatingPeriod>
                            <UicOperatingPeriod id="DE::UicOperatingPeriod:83977::" version="1641291376">
                                <FromDate>2021-12-21T00:00:00</FromDate>
                                <ToDate>2021-12-22T00:00:00</ToDate>
                                <ValidDayBits>11</ValidDayBits>
                            </UicOperatingPeriod>
                        </operatingPeriods>
                        <dayTypeAssignments>
                            <DayTypeAssignment id="DE::DayTypeAssignment:83969::" order="1" version="1641291376">
                                <OperatingPeriodRef ref="DE::UicOperatingPeriod:83969::" version="1641291376"/>
                                <DayTypeRef ref="DE::DayType:83969::" version="1641291376"/>
                                <isAvailable>true</isAvailable>
                            </DayTypeAssignment>
                            <DayTypeAssignment id="DE::DayTypeAssignment:83970::" order="2" version="1641291376">
                                <OperatingPeriodRef ref="DE::UicOperatingPeriod:83970::" version="1641291376"/>
                                <DayTypeRef ref="DE::DayType:83970::" version="1641291376"/>
                                <isAvailable>true</isAvailable>
                            </DayTypeAssignment>
                            <DayTypeAssignment id="DE::DayTypeAssignment:83971::" order="3" version="1641291376">
                                <OperatingPeriodRef ref="DE::UicOperatingPeriod:83971::" version="1641291376"/>
                                <DayTypeRef ref="DE::DayType:83971::" version="1641291376"/>
                                <isAvailable>true</isAvailable>
                            </DayTypeAssignment>
                            <DayTypeAssignment id="DE::DayTypeAssignment:83972::" order="4" version="1641291376">
                                <OperatingPeriodRef ref="DE::UicOperatingPeriod:83972::" version="1641291376"/>
                                <DayTypeRef ref="DE::DayType:83972::" version="1641291376"/>
                                <isAvailable>true</isAvailable>
                            </DayTypeAssignment>
                            <DayTypeAssignment id="DE::DayTypeAssignment:83973::" order="5" version="1641291376">
                                <OperatingPeriodRef ref="DE::UicOperatingPeriod:83973::" version="1641291376"/>
                                <DayTypeRef ref="DE::DayType:83973::" version="1641291376"/>
                                <isAvailable>true</isAvailable>
                            </DayTypeAssignment>
                            <DayTypeAssignment id="DE::DayTypeAssignment:83974::" order="6" version="1641291376">
                                <OperatingPeriodRef ref="DE::UicOperatingPeriod:83974::" version="1641291376"/>
                                <DayTypeRef ref="DE::DayType:83974::" version="1641291376"/>
                                <isAvailable>true</isAvailable>
                            </DayTypeAssignment>
                            <DayTypeAssignment id="DE::DayTypeAssignment:83975::" order="7" version="1641291376">
                                <OperatingPeriodRef ref="DE::UicOperatingPeriod:83975::" version="1641291376"/>
                                <DayTypeRef ref="DE::DayType:83975::" version="1641291376"/>
                                <isAvailable>true</isAvailable>
                            </DayTypeAssignment>
                            <DayTypeAssignment id="DE::DayTypeAssignment:83976::" order="8" version="1641291376">
                                <OperatingPeriodRef ref="DE::UicOperatingPeriod:83976::" version="1641291376"/>
                                <DayTypeRef ref="DE::DayType:83976::" version="1641291376"/>
                                <isAvailable>true</isAvailable>
                            </DayTypeAssignment>
                            <DayTypeAssignment id="DE::DayTypeAssignment:83977::" order="9" version="1641291376">
                                <OperatingPeriodRef ref="DE::UicOperatingPeriod:83977::" version="1641291376"/>
                                <DayTypeRef ref="DE::DayType:83977::" version="1641291376"/>
                                <isAvailable>true</isAvailable>
                            </DayTypeAssignment>
                        </dayTypeAssignments>
                    </ServiceCalendar>
                </ServiceCalendarFrame>
            </frames>
        </CompositeFrame>
    </dataObjects>
</PublicationDelivery>)";

TEST(netex, resolve_day_idx) {
  timetable tt;
  tt.date_range_ = {date::sys_days{April / 1 / 2023},
                    date::sys_days{December / 1 / 2023}};

  EXPECT_EQ(-2 + kTimetableOffset.count(),
            resolve_day_idx(tt, "2023-03-30T00:00:00"));
}

TEST(netex, read_operating_periods) {
  auto doc = pugi::xml_document{};
  auto const result = doc.load_string(netex_input);
  ASSERT_TRUE(result);

  timetable tt;
  tt.date_range_ = {date::sys_days{December / 21 / 2021},
                    date::sys_days{June / 11 / 2022}};

  auto operating_periods = hash_map<std::string_view, bitfield>{};
  read_operating_periods(tt, doc, operating_periods);

  auto const it = operating_periods.find("DE::UicOperatingPeriod:83977::");
  ASSERT_NE(end(operating_periods), it);

  auto const bf = it->second.to_string();
  EXPECT_EQ("1100000", bf.substr(bf.size() - 7, 7));
}

TEST(netex, read_service_calendar) {
  auto doc = pugi::xml_document{};
  auto const result = doc.load_string(netex_input);
  ASSERT_TRUE(result);

  timetable tt;
  tt.date_range_ = {date::sys_days{December / 21 / 2021},
                    date::sys_days{June / 11 / 2022}};

  auto calendar = hash_map<std::string_view, bitfield>{};
  auto operating_periods = hash_map<std::string_view, bitfield>{};
  read_service_calendar(tt, doc, calendar, operating_periods);

  auto const it = calendar.find("DE::DayType:83977::");
  ASSERT_NE(end(calendar), it);

  auto const bf = it->second.to_string();
  EXPECT_EQ("1100000", bf.substr(bf.size() - 7, 7));
}