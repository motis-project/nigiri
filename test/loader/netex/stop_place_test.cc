#include "gtest/gtest.h"

#include "nigiri/loader/netex/stop_place.h"

using namespace nigiri;
using namespace nigiri::loader::netex;

constexpr auto const netex_input =
    R"(<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<PublicationDelivery xmlns="http://www.netex.org.uk/netex" version="ntx:1.1">
    <dataObjects>
        <CompositeFrame id="DE::CompositeFrame_EU_PI_LINE_OFFER:144-SILBUS-V95" version="1641291376">
            <frames>
                <SiteFrame>
                    <stopPlaces>
                        <StopPlace id="DE::StopPlace:300004617_1000::" version="1641291376">
                            <keyList>
                                <KeyValue>
                                    <Key>GlobalID</Key>
                                    <Value>de:06438:4617</Value>
                                </KeyValue>
                            </keyList>
                            <Name>Rödermark-Ober-Roden Friedhof</Name>
                            <Centroid>
                                <Location>
                                    <Longitude>8.822577</Longitude>
                                    <Latitude>49.984518</Latitude>
                                </Location>
                            </Centroid>
                            <TopographicPlaceRef ref="DE::TopographicPlace:72427::" version="1641291376"/>
                            <AuthorityRef ref="DE::Authority:35::" version="1641291376"/>
                            <levels>
                                <Level id="DE::Level:48712886_0::" version="1641291376">
                                    <Name>Strassenniveau</Name>
                                    <ShortName>0</ShortName>
                                </Level>
                            </levels>
                            <StopPlaceType>other</StopPlaceType>
                            <quays>
                                <Quay id="DE::Quay:320461701_1000::" version="1641291376">
                                    <keyList>
                                        <KeyValue>
                                            <Key>GlobalID</Key>
                                            <Value>de:06438:4617:1:1</Value>
                                        </KeyValue>
                                    </keyList>
                                    <Name>NWaldacker</Name>
                                    <Centroid>
                                        <Location>
                                            <Longitude>8.822816</Longitude>
                                            <Latitude>49.984158</Latitude>
                                        </Location>
                                    </Centroid>
                                    <LevelRef ref="DE::Level:48712886_0::" version="1641291376"/>
                                </Quay>
                                <Quay id="DE::Quay:320461702_1000::" version="1641291376">
                                    <keyList>
                                        <KeyValue>
                                            <Key>GlobalID</Key>
                                            <Value>de:06438:4617:2:2</Value>
                                        </KeyValue>
                                    </keyList>
                                    <Name>VWaldacker</Name>
                                    <Centroid>
                                        <Location>
                                            <Longitude>8.822466</Longitude>
                                            <Latitude>49.984553</Latitude>
                                        </Location>
                                    </Centroid>
                                    <LevelRef ref="DE::Level:48712886_0::" version="1641291376"/>
                                </Quay>
                            </quays>
                        </StopPlace>
                    </stopPlaces>
                </SiteFrame>
            </frames>
        </CompositeFrame>
    </dataObjects>
</PublicationDelivery>)";

TEST(netex, stop_places) {
  auto doc = pugi::xml_document{};
  auto const result = doc.load_string(netex_input);
  ASSERT_TRUE(result);

  hash_map<std::string_view, stop_place> stop_map;
  read_stop_places(doc, stop_map);

  auto test_stop_place = stop_map["DE::StopPlace:300004617_1000::"];
  ASSERT_EQ(test_stop_place.name, "Rödermark-Ober-Roden Friedhof");
  ASSERT_EQ(test_stop_place.quays.size(), 2);

  ASSERT_EQ(test_stop_place.quays["DE::Quay:320461701_1000::"].name,
            "NWaldacker");
}