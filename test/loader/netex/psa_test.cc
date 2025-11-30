#include "nigiri/loader/netex/load_timetable.h"

#include "utl/parser/cstr.h"

#include "gtest/gtest.h"

using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::netex;
using namespace date;

constexpr auto kXml = R"(
# netex.xml
<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<PublicationDelivery xmlns="http://www.netex.org.uk/netex" version="ntx:1.1">
  <dataObjects>
    <CompositeFrame id="DE::CompositeFrame_EU_PI_LINE_OFFER:123-FBBUS-FB-1" version="1763819179">
      <frames>
        <ResourceFrame id="DE::ResourceFrame_EU_PI_COMMON:DBDB-800486-S4" version="1763819179">
          <organisations>
            <Authority id="DE::Authority:151::" version="1763819179">
              <PublicCode>DBDB</PublicCode>
              <Name>DB_Gesamtnetz der deutschen Bahn</Name>
              <ShortName>DBDB</ShortName>
              <LegalName>DB_Gesamtnetz der deutschen Bahn</LegalName>
              <ContactDetails/>
              <OrganisationType>authority</OrganisationType>
            </Authority>
          </organisations>
        </ResourceFrame>

        <ServiceCalendarFrame id="DE::ServiceCalendarFrame_EU_PI_CALENDAR:DBDB-800486-S4" version="1763819179">
          <ServiceCalendar id="DE::ServiceCalendar:27522::" version="1763819179">

            <operatingPeriods>
              <UicOperatingPeriod id="DE::UicOperatingPeriod:249613::" version="1763819179">
                <FromDate>2025-12-14T00:00:00</FromDate>
                <ToDate>2026-06-13T00:00:00</ToDate>
                <ValidDayBits>11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111</ValidDayBits>
              </UicOperatingPeriod>
              <UicOperatingPeriod id="DE::UicOperatingPeriod:249614::" version="1763819179">
                <FromDate>2025-12-14T00:00:00</FromDate>
                <ToDate>2025-12-14T00:00:00</ToDate>
                <ValidDayBits>1</ValidDayBits>
              </UicOperatingPeriod>
            </operatingPeriods>

            <dayTypeAssignments>
              <DayTypeAssignment id="DE::DayTypeAssignment:249613::" order="1" version="1763819179">
                <OperatingPeriodRef ref="DE::UicOperatingPeriod:249613::" version="1763819179"/>
                <DayTypeRef ref="DE::DayType:249613::" version="1763819179"/>
                <isAvailable>true</isAvailable>
              </DayTypeAssignment>
              <DayTypeAssignment id="DE::DayTypeAssignment:249614::" order="2" version="1763819179">
                <OperatingPeriodRef ref="DE::UicOperatingPeriod:249614::" version="1763819179"/>
                <DayTypeRef ref="DE::DayType:249614::" version="1763819179"/>
                <isAvailable>true</isAvailable>
              </DayTypeAssignment>
            </dayTypeAssignments>

          </ServiceCalendar>
        </ServiceCalendarFrame>

        <SiteFrame id="DE::SiteFrame_EU_PI_STOP:123-FBBUS-FB-1" version="1763819179">
          <stopPlaces>
            <StopPlace id="DE::StopPlace:380019319_1000::" version="1763819179">
              <keyList>
                <KeyValue>
                  <Key>GlobalID</Key>
                  <Value>de:06440:19319</Value>
                </KeyValue>
              </keyList>
              <Name>Echzell-Bingenheim Raunstraße</Name>
              <Centroid>
                <Location>
                  <Longitude>8.894726</Longitude>
                  <Latitude>50.368535</Latitude>
                </Location>
              </Centroid>
              <TopographicPlaceRef ref="DE::TopographicPlace:72451::" version="1763819179"/>
              <AuthorityRef ref="DE::Authority:35::" version="1763819179"/>
              <StopPlaceType>other</StopPlaceType>
            </StopPlace>
            <StopPlace id="DE::StopPlace:22974_1000::" version="1763819179">
              <keyList>
                <KeyValue>
                  <Key>GlobalID</Key>
                  <Value>de:06440:22974</Value>
                </KeyValue>
              </keyList>
              <Name>Altenstadt Stammheimer Straße</Name>
              <Centroid>
                <Location>
                  <Longitude>8.94466</Longitude>
                  <Latitude>50.288019</Latitude>
                </Location>
              </Centroid>
              <TopographicPlaceRef ref="DE::TopographicPlace:72446::" version="1763819179"/>
              <AuthorityRef ref="DE::Authority:35::" version="1763819179"/>
              <StopPlaceType>other</StopPlaceType>
              <quays>
                <Quay id="DE::Quay:12297401_1000::" version="1763819179">
                  <keyList>
                    <KeyValue>
                      <Key>GlobalID</Key>
                      <Value>de:06440:22974</Value>
                    </KeyValue>
                  </keyList>
                  <Name>Altenstadt Stammheimer Straße | Mast 1</Name>
                  <Centroid>
                    <Location>
                      <Longitude>8.9448</Longitude>
                      <Latitude>50.287714</Latitude>
                    </Location>
                  </Centroid>
                </Quay>
                <Quay id="DE::Quay:12297402_1000::" version="1763819179">
                  <keyList>
                    <KeyValue>
                      <Key>GlobalID</Key>
                      <Value>de:06440:22974</Value>
                    </KeyValue>
                  </keyList>
                  <Name>Altenstadt Stammheimer Straße | Mast 2</Name>
                  <Centroid>
                    <Location>
                      <Longitude>8.9448</Longitude>
                      <Latitude>50.287714</Latitude>
                    </Location>
                  </Centroid>
                </Quay>
              </quays>
            </StopPlace>
            <Quay dataSourceRef="FR1:OrganisationalUnit:59:" version="1590509-23475" created="2018-04-12T07:40:58Z" changed="2024-07-05T09:25:55Z"
                  derivedFromObjectRef="FR::Quay:23475:FR1" id="FR::Quay:50114645:FR1">
                <Name>Tour Eiffel</Name>
                <PrivateCode>706</PrivateCode>
                <Centroid>
                    <Location>
                        <gml:pos srsName="EPSG:2154">648336.0 6862487.0</gml:pos>
                    </Location>
                </Centroid>
                <PostalAddress version="any" id="FR1:PostalAddress:50114645:">
                    <AddressLine1>4 AVENUE DE LA BOURDONNAIS</AddressLine1>
                    <Town>Paris 7e</Town>
                    <PostalRegion>75107</PostalRegion>
                </PostalAddress>
                <AccessibilityAssessment version="any" id="FR1:AccessibilityAssessment:50114645:">
                    <MobilityImpairedAccess>unknown</MobilityImpairedAccess>
                    <limitations>
                        <AccessibilityLimitation>
                            <WheelchairAccess>true</WheelchairAccess>
                            <AudibleSignalsAvailable>unknown</AudibleSignalsAvailable>
                            <VisualSignsAvailable>unknown</VisualSignsAvailable>
                        </AccessibilityLimitation>
                    </limitations>
                </AccessibilityAssessment>
                <TransportMode>bus</TransportMode>
                <tariffZones>
                    <TariffZoneRef ref="FR1:TariffZone:1:LOC"/>
                </tariffZones>
                <destinations>
                    <DestinationDisplayView>
                        <Name>42</Name>
                        <ShortName>42</ShortName>
                        <FrontText>Aller Cours de l'Île Seguin</FrontText>
                        <PrivateCode>C01085</PrivateCode>
                    </DestinationDisplayView>
                </destinations>
            </Quay>
          </stopPlaces>
        </SiteFrame>

        <ServiceFrame id="DE::ServiceFrame_EU_PI_NETWORK:123-FBBUS-FB-1" version="1763819179">
          <directions>
            <Direction id="DE::Direction:1::" version="1763819179"/>
            <Direction id="DE::Direction:2::" version="1763819179"/>
          </directions>

          <lines>
            <Line id="DE::Line:162620::" version="1763819179">
              <Name>S4</Name>
              <ShortName>S4</ShortName>
              <TransportMode>rail</TransportMode>
              <PublicCode>S4</PublicCode>
              <PrivateCode>S4</PrivateCode>
              <AuthorityRef ref="DE::Authority:151::" version="1763819179"/>
              <additionalOperators>
                <OperatorRef ref="DE::Operator:11662::" version="1763819179"/>
              </additionalOperators>
              <allowedDirections>
                <AllowedLineDirection id="DE::AllowedLineDirection:46777::" version="1763819179">
                  <DirectionRef ref="DE::Direction:1::" version="1763819179"/>
                </AllowedLineDirection>
                <AllowedLineDirection id="DE::AllowedLineDirection:46778::" version="1763819179">
                  <DirectionRef ref="DE::Direction:2::" version="1763819179"/>
                </AllowedLineDirection>
              </allowedDirections>
            </Line>
          </lines>

          <destinationDisplays>
            <DestinationDisplay id="DE::DestinationDisplay:35217::" version="1763819179">
              <SideText>Oschatz</SideText>
              <FrontText>Oschatz</FrontText>
            </DestinationDisplay>
          </destinationDisplays>

          <stopAssignments>
            <PassengerStopAssignment id="DE::PassengerStopAssignment:150509::" order="30" version="1763819179">
              <ScheduledStopPointRef ref="DE::ScheduledStopPoint:101376501_123_::" version="1763819179"/>
              <StopPlaceRef ref="DE::StopPlace:380019319_1000::" version="1763819179"/>
            </PassengerStopAssignment>
            <PassengerStopAssignment id="DE::PassengerStopAssignment:150510::" order="31" version="1763819179">
              <ScheduledStopPointRef ref="DE::ScheduledStopPoint:101572901_123_::" version="1763819179"/>
              <StopPlaceRef ref="DE::StopPlace:22974_1000::" version="1763819179"/>
              <QuayRef ref="DE::Quay:12297401_1000::" version="1763819179"/>
            </PassengerStopAssignment>
          </stopAssignments>

          <journeyPatterns>
            <ServiceJourneyPattern id="DE::ServiceJourneyPattern:296311804_0::" version="1763819179">
              <RouteView id="DE::RouteView:296311804_1::">
                <LineRef ref="DE::Line:162620::" version="1763819179"/>
              </RouteView>
              <DirectionRef ref="DE::Direction:1::" version="1763819179"/>
              <pointsInSequence>
                <StopPointInJourneyPattern id="DE::StopPointInJourneyPattern:296311804_1_0::" order="1" version="1763819179">
                  <ScheduledStopPointRef ref="DE::ScheduledStopPoint:101376501_123_::" version="1763819179"/>
                  <ForAlighting>true</ForAlighting>
                  <ForBoarding>true</ForBoarding>
                  <DestinationDisplayRef ref="DE::DestinationDisplay:35217::" version="1763819179"/>
                  <ChangeOfDestinationDisplay>false</ChangeOfDestinationDisplay>
                  <noticeAssignments>
                    <NoticeAssignment id="DE::NoticeAssignment:87899403::" order="1" version="1763819179">
                      <Notice id="DE::Notice:478_227594::" version="1763819179">
                        <Text lang="de-DE">Fahrradmitnahme begrenzt möglich</Text>
                        <PublicCode>F2G</PublicCode>
                      </Notice>
                    </NoticeAssignment>
                  </noticeAssignments>
                </StopPointInJourneyPattern>
              </pointsInSequence>
            </ServiceJourneyPattern>
          </journeyPatterns>

        </ServiceFrame>

        <TimetableFrame id="DE::TimetableFrame_EU_PI_TIMETABLE:DBDB-800486-S4" version="1763819179">
          <vehicleJourneys>
            <ServiceJourney id="DE::ServiceJourney:3024509099_0::" version="1763819179">
              <keyList>
                <KeyValue>
                  <Key>TripNr</Key>
                  <Value>037400</Value>
                </KeyValue>
              </keyList>
              <TransportMode>rail</TransportMode>
              <TransportSubmode>
                <RailSubmode>suburbanRailway</RailSubmode>
              </TransportSubmode>
              <DepartureTime>04:24:00</DepartureTime>
              <JourneyDuration>PT1H46M</JourneyDuration>
              <dayTypes>
                <DayTypeRef ref="DE::DayType:249613::" version="1763819179"/>
              </dayTypes>
              <ServiceJourneyPatternRef ref="DE::ServiceJourneyPattern:296311804_0::" version="1763819179"/>
              <VehicleTypeRef ref="DE::VehicleType:3834::" version="1763819179"/>
              <passingTimes>
                <TimetabledPassingTime id="DE::TimetabledPassingTime:81739562::" version="1763819179">
                  <StopPointInJourneyPatternRef ref="DE::StopPointInJourneyPattern:296311804_1_0::" version="1763819179"/>
                  <DepartureTime>04:24:00</DepartureTime>
                </TimetabledPassingTime>
                <TimetabledPassingTime id="DE::TimetabledPassingTime:81739563::" version="1763819179">
                  <StopPointInJourneyPatternRef ref="DE::StopPointInJourneyPattern:296311804_1_1::" version="1763819179"/>
                  <ArrivalTime>04:33:00</ArrivalTime>
                  <DepartureTime>04:33:00</DepartureTime>
                </TimetabledPassingTime>
                <TimetabledPassingTime id="DE::TimetabledPassingTime:81739591::" version="1763819179">
                  <StopPointInJourneyPatternRef ref="DE::StopPointInJourneyPattern:296311804_1_29::" version="1763819179"/>
                  <ArrivalTime>06:10:00</ArrivalTime>
                </TimetabledPassingTime>
              </passingTimes>
            </ServiceJourney>
          </vehicleJourneys>
        </TimetableFrame>
      </frames>
    </CompositeFrame>
  </dataObjects>
</PublicationDelivery>
)";

namespace nigiri::loader::netex {
sys_days parse_date(utl::cstr);
}

TEST(netex, parse_date) {
  EXPECT_EQ(sys_days{2025_y / December / 14}, parse_date("2025-12-14T00:00"));
  EXPECT_EQ(sys_days{2025_y / December / 14}, parse_date("2025-12-14"));
}

TEST(netex, psa) {
  auto global_bitfields = hash_map<bitfield, bitfield_idx_t>{};
  auto tt = timetable{};
  tt.date_range_ = {sys_days{2025_y / December / 14},
                    sys_days{2026_y / December / 14}};
  load_timetable({}, {}, mem_dir::read(kXml), tt, global_bitfields);
}