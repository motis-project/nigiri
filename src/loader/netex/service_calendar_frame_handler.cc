//
// Created by mirko on 10/11/23.
//

#include "service_calendar_frame_handler.h"
#include "nigiri/loader/netex/operatingPeriod.h"

namespace nigiri::loader::netex {

/*
 * The ServiceCalendarFrame contains
 * 1. Array of DayType objects (basically just an ID),
 * 2. Array of OperatingPeriods (from when to when and on which days),
 * 3. DayTypeAssignments (an array of mapping between operatingperiods and
 * daytypes)
 */
void processServiceCalendarFrame(const pugi::xml_node& serviceCalFrame) {
  auto const serviceCalendar = serviceCalFrame.child("ServiceCalendar");

  auto const fromDate = serviceCalendar.child("FromDate").text().get();
  auto const toDate = serviceCalendar.child("ToDate").text().get();
  auto const dayTypes = serviceCalendar.child("dayTypes").children();

  std::vector<std::string>
      dayTypesVec;  // Vector of the ids of dayTypes for this xml file
  for (auto const& dayType : dayTypes) {
    dayTypesVec.emplace_back(dayType.attribute("id").value());
  }

  auto const operatingPeriods =
      serviceCalendar.child("operatingPeriods").children();
  std::vector<operatingPeriod> operatingPeriodsVec;
  for (auto const& opPeriod : operatingPeriods) {
    operatingPeriod tempOpPeriod = {opPeriod.attribute("id").value()};
    operatingPeriodsVec.push_back(tempOpPeriod);
  }

  // auto const dayTypeAssignments =
  // serviceCalendar.child("dayTypeAssignments");
  //  ServiceCalendar->{FromDate, ToDate, dayTypes, operationgPeriods,
  //  dayTypeAssignments (match dayTypes with operatingPeriods)}

  std::cout << "From Date: " << fromDate << ", to date: " << toDate
            << "DayTypes: " << dayTypesVec << ", Operating Periods: "
            << nigiri::loader::netex::toString(operatingPeriodsVec) << "\n";
}
}  // namespace nigiri::loader::netex
