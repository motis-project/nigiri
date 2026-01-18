#include "gtest/gtest.h"

#include "nigiri/loader/load.h"
#include "nigiri/resolve.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/json_to_xml.h"
#include "nigiri/rt/vdv_aus.h"

#include "pugixml.hpp"

#include "date/date.h"

#include <sstream>

using namespace date;
using namespace std::chrono_literals;
using namespace nigiri;
using namespace nigiri::rt;

constexpr auto kIn = R"(
{
  "responseTimestamp": "2025-10-25T13:34:43.96857559Z",
  "status": true,
  "validUntil": "2025-10-25T13:35:43.96857559Z",
  "estimatedJourneyVersionFrame": [
    {
      "recordedAtTime": "2025-10-25T13:34:43.96857559Z",
      "estimatedVehicleJourney": [
        {
          "lineRef": {
            "value": "7621:2025-10-25"
          },
          "datedVehicleJourneyRef": {
            "value": "7621:2025-10-25"
          },
          "cancellation": false,
          "publishedLineName": [
            {
              "value": "7621"
            }
          ],
          "vehicleJourneyName": [
            {
              "value": "7621"
            }
          ],
          "firstOrLastJourney": "FIRST_SERVICE_OF_DAY",
          "estimatedCalls": {
            "estimatedCall": []
          },
          "recordedCalls": {
            "recordedCall": [
              {
                "stopPointRef": {
                  "value": "XCY"
                },
                "order": 1,
                "stopPointName": [
                  {
                    "value": "Chambéry - Bus Station",
                    "lang": "ENG"
                  }
                ],
                "extraCall": false,
                "cancellation": false,
                "arrivalStatus": "ON_TIME",
                "aimedDepartureTime": "2025-10-25T03:00:00Z",
                "actualDepartureTime": "2025-10-25T03:01:31Z",
                "departureStatus": "DELAYED"
              },
              {
                "stopPointRef": {
                  "value": "NCY"
                },
                "order": 2,
                "stopPointName": [
                  {
                    "value": "Annecy - Bus Station",
                    "lang": "ENG"
                  }
                ],
                "extraCall": false,
                "cancellation": false,
                "aimedArrivalTime": "2025-10-25T03:45:00Z",
                "actualArrivalTime": "2025-10-25T03:39:41Z",
                "arrivalStatus": "ON_TIME",
                "aimedDepartureTime": "2025-10-25T03:50:00Z",
                "actualDepartureTime": "2025-10-25T03:52:21Z",
                "departureStatus": "DELAYED"
              },
              {
                "stopPointRef": {
                  "value": "GVA"
                },
                "order": 3,
                "stopPointName": [
                  {
                    "value": "Geneva - Airport Bus Station",
                    "lang": "ENG"
                  }
                ],
                "extraCall": false,
                "cancellation": false,
                "aimedArrivalTime": "2025-10-25T04:40:00Z",
                "actualArrivalTime": "2025-10-25T04:27:52Z",
                "arrivalStatus": "ON_TIME",
                "aimedDepartureTime": "2025-10-25T04:45:00Z",
                "actualDepartureTime": "2025-10-25T04:44:23Z",
                "departureStatus": "DEPARTED"
              },
              {
                "stopPointRef": {
                  "value": "XGC"
                },
                "order": 4,
                "stopPointName": [
                  {
                    "value": "Geneva - Bus Station",
                    "lang": "ENG"
                  }
                ],
                "extraCall": false,
                "cancellation": false,
                "aimedArrivalTime": "2025-10-25T05:00:00Z",
                "actualArrivalTime": "2025-10-25T04:57:05Z",
                "arrivalStatus": "ON_TIME",
                "departureStatus": "DEPARTED"
              }
            ]
          },
          "isCompleteStopSequence": true
        }
      ]
    }
  ],
  "version": "2.0"
}
)";

auto const kExpected =
    R"(<Siri xmlns:datex="http://datex2.eu/schema/2_0RC1/2_0" xmlns="http://www.siri.org.uk/siri" xmlns:acsb="http://www.ifopt.org.uk/acsb" xmlns:ifopt="http://www.ifopt.org.uk/ifopt" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xs="http://www.w3.org/2001/XMLSchema" version="2.0">
  <ServiceDelivery>
    <ResponseTimestamp>2025-10-25T13:34:43.96857559Z</ResponseTimestamp>
    <Status>true</Status>
    <ValidUntil>2025-10-25T13:35:43.96857559Z</ValidUntil>
    <EstimatedJourneyVersionFrame>
      <RecordedAtTime>2025-10-25T13:34:43.96857559Z</RecordedAtTime>
      <EstimatedVehicleJourney>
        <LineRef>7621:2025-10-25</LineRef>
        <DatedVehicleJourneyRef>7621:2025-10-25</DatedVehicleJourneyRef>
        <Cancellation>false</Cancellation>
        <PublishedLineName>7621</PublishedLineName>
        <VehicleJourneyName>7621</VehicleJourneyName>
        <FirstOrLastJourney>FIRST_SERVICE_OF_DAY</FirstOrLastJourney>
        <EstimatedCalls />
        <RecordedCalls>
          <RecordedCall>
            <StopPointRef>XCY</StopPointRef>
            <Order>1</Order>
            <StopPointName lang="ENG">Chambéry - Bus Station</StopPointName>
            <ExtraCall>false</ExtraCall>
            <Cancellation>false</Cancellation>
            <ArrivalStatus>ON_TIME</ArrivalStatus>
            <AimedDepartureTime>2025-10-25T03:00:00Z</AimedDepartureTime>
            <ActualDepartureTime>2025-10-25T03:01:31Z</ActualDepartureTime>
            <DepartureStatus>DELAYED</DepartureStatus>
          </RecordedCall>
          <RecordedCall>
            <StopPointRef>NCY</StopPointRef>
            <Order>2</Order>
            <StopPointName lang="ENG">Annecy - Bus Station</StopPointName>
            <ExtraCall>false</ExtraCall>
            <Cancellation>false</Cancellation>
            <AimedArrivalTime>2025-10-25T03:45:00Z</AimedArrivalTime>
            <ActualArrivalTime>2025-10-25T03:39:41Z</ActualArrivalTime>
            <ArrivalStatus>ON_TIME</ArrivalStatus>
            <AimedDepartureTime>2025-10-25T03:50:00Z</AimedDepartureTime>
            <ActualDepartureTime>2025-10-25T03:52:21Z</ActualDepartureTime>
            <DepartureStatus>DELAYED</DepartureStatus>
          </RecordedCall>
          <RecordedCall>
            <StopPointRef>GVA</StopPointRef>
            <Order>3</Order>
            <StopPointName lang="ENG">Geneva - Airport Bus Station</StopPointName>
            <ExtraCall>false</ExtraCall>
            <Cancellation>false</Cancellation>
            <AimedArrivalTime>2025-10-25T04:40:00Z</AimedArrivalTime>
            <ActualArrivalTime>2025-10-25T04:27:52Z</ActualArrivalTime>
            <ArrivalStatus>ON_TIME</ArrivalStatus>
            <AimedDepartureTime>2025-10-25T04:45:00Z</AimedDepartureTime>
            <ActualDepartureTime>2025-10-25T04:44:23Z</ActualDepartureTime>
            <DepartureStatus>DEPARTED</DepartureStatus>
          </RecordedCall>
          <RecordedCall>
            <StopPointRef>XGC</StopPointRef>
            <Order>4</Order>
            <StopPointName lang="ENG">Geneva - Bus Station</StopPointName>
            <ExtraCall>false</ExtraCall>
            <Cancellation>false</Cancellation>
            <AimedArrivalTime>2025-10-25T05:00:00Z</AimedArrivalTime>
            <ActualArrivalTime>2025-10-25T04:57:05Z</ActualArrivalTime>
            <ArrivalStatus>ON_TIME</ArrivalStatus>
            <DepartureStatus>DEPARTED</DepartureStatus>
          </RecordedCall>
        </RecordedCalls>
        <IsCompleteStopSequence>true</IsCompleteStopSequence>
      </EstimatedVehicleJourney>
    </EstimatedJourneyVersionFrame>
    <Version>2.0</Version>
  </ServiceDelivery>
</Siri>
)";

constexpr auto kGtfsTimetable = R"(
# agency.txt

# stops.txt
stop_id,stop_name,stop_lat,stop_lon
XCY,Chambéry - Bus Station,45.0,5.0
NCY,Annecy - Bus Station,46.0,6.0
GVA,Geneva - Airport Bus Station,46.2,6.1
XGC,Geneva - Bus Station,46.3,6.2

# calendar_dates.txt
service_id,date,exception_type
1,20251025,1

# routes.txt
route_id
7621

# trips.txt
trip_id,service_id,route_id
7621,1,7621

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
7621,03:00:00,03:00:00,XCY,1
7621,03:45:00,03:50:00,NCY,2
7621,04:40:00,04:45:00,GVA,3
7621,05:00:00,05:00:00,XGC,4
)";

TEST(rt, json_to_xml) {
  auto doc = to_xml(kIn);
  auto oss = std::ostringstream{};
  doc.save(oss, "  ", pugi::format_default | pugi::format_no_declaration);
  EXPECT_EQ(kExpected, oss.str());

  auto const base_day = date::sys_days{2025_y / October / 25};
  auto tt = loader::load({{.tag_ = "test",
                           .path_ = kGtfsTimetable,
                           .loader_config_ = {.default_tz_ = "UTC"}}},
                         {}, {base_day, date::sys_days{2025_y / October / 26}});

  auto rtt = rt::create_rt_timetable(tt, base_day);
  auto updater = rt::vdv_aus::updater{tt, source_idx_t{0},
                                      rt::vdv_aus::updater::xml_format::kSiri};
  updater.update(rtt, doc);

  auto const fr_rt = resolve(tt, &rtt, "7621", "20251025");
  ASSERT_EQ(4U, fr_rt.size());
  EXPECT_TRUE(fr_rt.is_rt());
  EXPECT_EQ(base_day + 3h + 1min, fr_rt[0].time(event_type::kDep));
  EXPECT_EQ(base_day + 3h + 39min, fr_rt[1].time(event_type::kArr));
  EXPECT_EQ(base_day + 3h + 52min, fr_rt[1].time(event_type::kDep));
  EXPECT_EQ(base_day + 4h + 27min, fr_rt[2].time(event_type::kArr));
  EXPECT_EQ(base_day + 4h + 44min, fr_rt[2].time(event_type::kDep));
  EXPECT_EQ(base_day + 4h + 57min, fr_rt[3].time(event_type::kArr));
}
