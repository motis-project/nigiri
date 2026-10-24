#include <string>

#include "gtest/gtest.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/loader/netex/load_timetable.h"
#include "nigiri/timetable.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::netex;

namespace {

// StopPlace "s<i>" has StopPlace "s<i-1>" as parent. The last StopPlace has a
// Quay. The StopPlaces are written children first: the NeTEx loader has to
// register each stop after its parent, regardless of the order in the file.
std::string make_stop_chain_xml(unsigned const levels) {
  auto xml = std::string{R"(
# netex.xml
<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<PublicationDelivery xmlns="http://www.netex.org.uk/netex" version="ntx:1.1">
  <dataObjects>
    <CompositeFrame id="CF">
      <FrameDefaults>
        <DefaultLocale>
          <TimeZone>Europe/Berlin</TimeZone>
        </DefaultLocale>
      </FrameDefaults>
      <frames>
        <SiteFrame id="SF">
          <stopPlaces>
)"};

  for (auto i = levels; i-- != 0U;) {
    auto const id = "s" + std::to_string(i);
    xml += "<StopPlace id=\"SP:" + id + "\">";
    xml += "<keyList><KeyValue><Key>GlobalID</Key><Value>" + id +
           "</Value></KeyValue></keyList>";
    xml += "<Name>" + id + "</Name>";
    xml += "<Centroid><Location><Longitude>8.0</Longitude><Latitude>50.0"
           "</Latitude></Location></Centroid>";
    if (i != 0U) {
      xml += "<ParentSiteRef ref=\"SP:s" + std::to_string(i - 1U) + "\"/>";
    }
    if (i == levels - 1U) {
      xml += "<quays><Quay id=\"Q:q\"><keyList><KeyValue><Key>GlobalID</Key>"
             "<Value>q</Value></KeyValue></keyList><Name>q</Name></Quay>"
             "</quays>";
    }
    xml += "</StopPlace>\n";
  }

  xml += R"(
          </stopPlaces>
        </SiteFrame>
      </frames>
    </CompositeFrame>
  </dataObjects>
</PublicationDelivery>
)";
  return xml;
}

}  // namespace

TEST(netex, stop_hierarchy_with_more_than_two_levels) {
  constexpr auto kLevels = 12U;

  auto global_bitfields = hash_map<bitfield, bitfield_idx_t>{};
  auto tt = timetable{};
  tt.date_range_ = {sys_days{2025_y / December / 14},
                    sys_days{2026_y / December / 14}};
  register_special_stations(tt);
  load_timetable({}, {}, mem_dir::read(make_stop_chain_xml(kLevels)), tt,
                 global_bitfields);

  auto const find = [&](std::string_view const id) {
    auto const l = tt.find(location_id{id, {}});
    EXPECT_TRUE(l.has_value()) << id;
    return l.value_or(location_idx_t::invalid());
  };

  auto const root = find("s0");
  ASSERT_NE(location_idx_t::invalid(), root);
  EXPECT_EQ(location_idx_t::invalid(), tt.locations_.parents_[root]);

  auto parent = root;
  for (auto i = 1U; i != kLevels; ++i) {
    auto const id = "s" + std::to_string(i);
    auto const l = find(id);
    ASSERT_NE(location_idx_t::invalid(), l) << id;
    EXPECT_EQ(parent, tt.locations_.parents_[l]) << id;
    EXPECT_EQ((std::vector{l}),
              (std::vector<location_idx_t>{tt.locations_.children_[parent].begin(),
                                           tt.locations_.children_[parent].end()}))
        << id;
    parent = l;
  }

  auto const quay = find("q");
  ASSERT_NE(location_idx_t::invalid(), quay);
  EXPECT_EQ(parent, tt.locations_.parents_[quay]);
}
