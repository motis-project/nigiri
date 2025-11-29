#include "nigiri/loader/netex/load_timetable.h"

#include "gtest/gtest.h"

using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::netex;

constexpr auto kXml = R"(
# netex.xml
<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<PublicationDelivery xmlns="http://www.netex.org.uk/netex" version="ntx:1.1">
  <dataObjects>
    <CompositeFrame id="DE::CompositeFrame_EU_PI_LINE_OFFER:123-FBBUS-FB-1" version="1763819179">
      <frames>
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
        </ServiceFrame>
      </frames>
    </CompositeFrame>
  </dataObjects>
</PublicationDelivery>
)";

TEST(netex, psa) {
  auto global_bitfields = hash_map<bitfield, bitfield_idx_t>{};
  auto tt = timetable{};
  load_timetable({}, {}, mem_dir::read(kXml), tt, global_bitfields);
}