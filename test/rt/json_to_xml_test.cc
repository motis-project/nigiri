#include "gtest/gtest.h"

#include "nigiri/rt/json_to_xml.h"

#include <sstream>

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

TEST(rt, json_to_xml) {
  auto doc = to_xml(kIn);
  auto oss = std::ostringstream{};
  doc.save(oss, "  ", pugi::format_default | pugi::format_no_declaration);
  EXPECT_EQ(kExpected, oss.str());
}
