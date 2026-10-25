#include "gtest/gtest.h"

#include "utl/helpers/algorithm.h"

#include "nigiri/loader/load.h"
#include "nigiri/resolve.h"
#include "nigiri/timetable.h"

using namespace date;
using namespace nigiri;

// Journey pattern A -> B -> C -> D used by all service journeys.
// Local time is CET = UTC+1.
constexpr auto kNetex = R"(
# netex.xml
<PublicationDelivery>
<dataObjects>
<CompositeFrame>
  <FrameDefaults>
    <DefaultLocale><TimeZone>Europe/Vienna</TimeZone></DefaultLocale>
  </FrameDefaults>
  <frames>
    <ServiceFrame>
      <validityConditions>
        <AvailabilityCondition id="AC">
          <FromDate>2025-12-14T00:00:00</FromDate>
          <ToDate>2025-12-14T00:00:00</ToDate>
          <ValidDayBits>1</ValidDayBits>
        </AvailabilityCondition>
      </validityConditions>
      <scheduledStopPoints>
        <ScheduledStopPoint id="A"><Name>A</Name><Location><Longitude>16.00</Longitude><Latitude>48.0</Latitude></Location></ScheduledStopPoint>
        <ScheduledStopPoint id="B"><Name>B</Name><Location><Longitude>16.01</Longitude><Latitude>48.0</Latitude></Location></ScheduledStopPoint>
        <ScheduledStopPoint id="C"><Name>C</Name><Location><Longitude>16.02</Longitude><Latitude>48.0</Latitude></Location></ScheduledStopPoint>
        <ScheduledStopPoint id="D"><Name>D</Name><Location><Longitude>16.03</Longitude><Latitude>48.0</Latitude></Location></ScheduledStopPoint>
      </scheduledStopPoints>
      <journeyPatterns>
        <ServiceJourneyPattern id="JP">
          <pointsInSequence>
            <StopPointInJourneyPattern id="SP_A"><ScheduledStopPointRef ref="A"/></StopPointInJourneyPattern>
            <StopPointInJourneyPattern id="SP_B"><ScheduledStopPointRef ref="B"/></StopPointInJourneyPattern>
            <StopPointInJourneyPattern id="SP_C"><ScheduledStopPointRef ref="C"/></StopPointInJourneyPattern>
            <StopPointInJourneyPattern id="SP_D"><ScheduledStopPointRef ref="D"/></StopPointInJourneyPattern>
          </pointsInSequence>
        </ServiceJourneyPattern>
      </journeyPatterns>
    </ServiceFrame>
    <TimetableFrame>
      <vehicleJourneys>
        <!-- No time at all: trip is removed. -->
        <ServiceJourney id="NO_TIMES">
          <keyList><KeyValue><Key>TripNr</Key><Value>1</Value></KeyValue></keyList>
          <validityConditions><AvailabilityConditionRef ref="AC"/></validityConditions>
          <ServiceJourneyPatternRef ref="JP"/>
          <passingTimes>
            <TimetabledPassingTime><StopPointInJourneyPatternRef ref="SP_A"/></TimetabledPassingTime>
            <TimetabledPassingTime><StopPointInJourneyPatternRef ref="SP_B"/></TimetabledPassingTime>
          </passingTimes>
        </ServiceJourney>

        <!-- First stop A missing: gets first valid time (B arrival). -->
        <ServiceJourney id="FILL_START">
          <keyList><KeyValue><Key>TripNr</Key><Value>2</Value></KeyValue></keyList>
          <validityConditions><AvailabilityConditionRef ref="AC"/></validityConditions>
          <ServiceJourneyPatternRef ref="JP"/>
          <passingTimes>
            <TimetabledPassingTime><StopPointInJourneyPatternRef ref="SP_B"/><ArrivalTime>10:10:00</ArrivalTime><DepartureTime>10:11:00</DepartureTime></TimetabledPassingTime>
            <TimetabledPassingTime><StopPointInJourneyPatternRef ref="SP_C"/><ArrivalTime>10:20:00</ArrivalTime><DepartureTime>10:21:00</DepartureTime></TimetabledPassingTime>
            <TimetabledPassingTime><StopPointInJourneyPatternRef ref="SP_D"/><ArrivalTime>10:30:00</ArrivalTime></TimetabledPassingTime>
          </passingTimes>
        </ServiceJourney>

        <!-- Passing times out of order: assigned by journey pattern order. -->
        <ServiceJourney id="OUT_OF_ORDER">
          <keyList><KeyValue><Key>TripNr</Key><Value>3</Value></KeyValue></keyList>
          <validityConditions><AvailabilityConditionRef ref="AC"/></validityConditions>
          <ServiceJourneyPatternRef ref="JP"/>
          <passingTimes>
            <TimetabledPassingTime><StopPointInJourneyPatternRef ref="SP_D"/><ArrivalTime>10:30:00</ArrivalTime></TimetabledPassingTime>
            <TimetabledPassingTime><StopPointInJourneyPatternRef ref="SP_B"/><ArrivalTime>10:10:00</ArrivalTime><DepartureTime>10:11:00</DepartureTime></TimetabledPassingTime>
            <TimetabledPassingTime><StopPointInJourneyPatternRef ref="SP_A"/><DepartureTime>10:00:00</DepartureTime></TimetabledPassingTime>
            <TimetabledPassingTime><StopPointInJourneyPatternRef ref="SP_C"/><ArrivalTime>10:20:00</ArrivalTime><DepartureTime>10:21:00</DepartureTime></TimetabledPassingTime>
          </passingTimes>
        </ServiceJourney>

        <!-- Out of order + first stop A and middle stop C missing:
             A gets first valid time (B arrival), C gets previous time (B departure). -->
        <ServiceJourney id="COMBINED">
          <keyList><KeyValue><Key>TripNr</Key><Value>4</Value></KeyValue></keyList>
          <validityConditions><AvailabilityConditionRef ref="AC"/></validityConditions>
          <ServiceJourneyPatternRef ref="JP"/>
          <passingTimes>
            <TimetabledPassingTime><StopPointInJourneyPatternRef ref="SP_D"/><ArrivalTime>10:30:00</ArrivalTime></TimetabledPassingTime>
            <TimetabledPassingTime><StopPointInJourneyPatternRef ref="SP_B"/><ArrivalTime>10:10:00</ArrivalTime><DepartureTime>10:11:00</DepartureTime></TimetabledPassingTime>
          </passingTimes>
        </ServiceJourney>
      </vehicleJourneys>
    </TimetableFrame>
  </frames>
</CompositeFrame>
</dataObjects>
</PublicationDelivery>
)";

// UTC times in the order dep A, arr B, dep B, arr C, dep C, arr D.
std::vector<std::string> utc_times(timetable const& tt,
                                   std::string_view trip_id) {
  auto const fr = resolve(tt, nullptr, trip_id, "20251214");
  auto times = std::vector<std::string>{};
  for (auto i = stop_idx_t{0U}; i != fr.size(); ++i) {
    if (i != 0U) {
      times.push_back(date::format("%H:%M", fr[i].time(event_type::kArr)));
    }
    if (i + 1U != fr.size()) {
      times.push_back(date::format("%H:%M", fr[i].time(event_type::kDep)));
    }
  }
  return times;
}

TEST(netex, missing_times) {
  auto const tt = loader::load(
      {{.tag_ = "test", .path_ = kNetex, .loader_config_ = {}}}, {},
      {sys_days{2025_y / December / 14}, sys_days{2025_y / December / 15}});

  EXPECT_TRUE(utl::none_of(tt.trip_id_strings_,
                           [](auto&& id) { return id.view() == "NO_TIMES"; }));

  using v = std::vector<std::string>;
  EXPECT_EQ((v{"09:10", "09:10", "09:11", "09:20", "09:21", "09:30"}),
            utc_times(tt, "FILL_START"));
  EXPECT_EQ((v{"09:00", "09:10", "09:11", "09:20", "09:21", "09:30"}),
            utc_times(tt, "OUT_OF_ORDER"));
  EXPECT_EQ((v{"09:10", "09:10", "09:11", "09:11", "09:11", "09:30"}),
            utc_times(tt, "COMBINED"));
}
