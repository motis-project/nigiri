#include "gtest/gtest.h"

#include "nigiri/loader/netex/route_operator.h"

using namespace nigiri;
using namespace nigiri::loader::netex;

constexpr auto const netex_input =
    R"(<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<PublicationDelivery xmlns="http://www.netex.org.uk/netex" version="ntx:1.1">
    <dataObjects>
        <CompositeFrame id="DE::CompositeFrame_EU_PI_LINE_OFFER:144-SILBUS-V95" version="1641291376">
            <frames>
              <ResourceFrame id="DE::ResourceFrame_EU_PI_COMMON:144-SILBUS-V95" version="1641291376">
<TypeOfFrameRef ref="epip:EU_PI_COMMON" versionRef="1.0"/>
<dataSources>
<DataSource id="epip_data:DataSource:General" version="1641291376">
<Name>DE</Name>
</DataSource>
</dataSources>
<responsibilitySets>
<ResponsibilitySet id="DE::ResponsibilitySet:17043::" version="1641291376">
<roles>
  <ResponsibilityRoleAssignment id="DE::ResponsibilityRoleAssignment:17043::"
                                version="1641291376">
      <DataRoleType>creates</DataRoleType>
      <ResponsibleOrganisationRef ref="DE::Authority:846::" version="1641291376"/>
  </ResponsibilityRoleAssignment>
</roles>
</ResponsibilitySet>
</responsibilitySets>
<organisations>
<Authority id="DE::Authority:846::" version="1641291376">
<PublicCode>144</PublicCode>
<Name>RMV_KVG OF - Faisy BAIM</Name>
<ShortName>144</ShortName>
<LegalName>RMV_KVG OF - Faisy BAIM</LegalName>
<ContactDetails/>
<OrganisationType>authority</OrganisationType>
<Address/>
</Authority>
<Operator id="DE::Operator:13838::" version="1641291376">
<PublicCode>SLI</PublicCode>
<Name>Schau in's Land Hain GmbH</Name>
<ShortName>SLI</ShortName>
<LegalName>Schau in's Land Hain GmbH</LegalName>
<ContactDetails/>
<OrganisationType>operator</OrganisationType>
<Address/>
</Operator>
<Operator id="DE::Operator:13839::" version="1641291376">
<PublicCode>SLI</PublicCode>
<Name>Schau in's Land Hain GmbH</Name>
<ShortName>SLI</ShortName>
<LegalName>Schau in's Land Hain GmbH</LegalName>
<ContactDetails/>
<OrganisationType>operator</OrganisationType>
<Address/>
</Operator>

<Authority id="DE::Authority:35::" version="1641291376">
<PublicCode>1000</PublicCode>
<Name>RMV_Referenz-Haltestellen RMV</Name>
<ShortName>1000</ShortName>
<LegalName>RMV_Referenz-Haltestellen RMV</LegalName>
<ContactDetails/>
<OrganisationType>authority</OrganisationType>
<Address/>
</Authority>
</organisations>
</ResourceFrame>
            </frames>
        </CompositeFrame>
    </dataObjects>
</PublicationDelivery>)";

TEST(netex, read_resource_frame) {
  hash_map<std::string_view, provider_idx_t> operatorMap;
  ASSERT_TRUE(operatorMap.empty());

  timetable tt;
  auto doc = pugi::xml_document{};
  auto const result = doc.load_string(netex_input);
  ASSERT_TRUE(result);

  read_resource_frame(tt, doc, operatorMap);

  ASSERT_EQ(operatorMap["DE::Operator:13838::"], 0);
  ASSERT_EQ(operatorMap["DE::Operator:13839::"], 1);
}
