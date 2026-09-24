#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/timetable.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;

namespace {

constexpr auto const kTicketingDeepLinksFile =
    std::string_view{"ticketing_deep_links.txt"};

// Agencies are registered in file order: DB -> 0, VU -> 1.
constexpr auto const kDB = provider_idx_t{0U};
constexpr auto const kVU = provider_idx_t{1U};

// One agency carries a deep link, one does not.
constexpr auto const agency = std::string_view{
    R"(agency_id,agency_name,agency_url,agency_timezone,ticketing_deep_link_id
DB,Deutsche Bahn,http://db.de,Europe/Berlin,dl:1
VU,Verkehrsverbund,http://vu.de,Europe/Berlin,
)"};

constexpr auto const stops = std::string_view{
    R"(stop_id,stop_name,stop_lat,stop_lon
S1,Stop 1,49.880,8.664
S2,Stop 2,49.878,8.661
)"};

constexpr auto const calendar = std::string_view{
    R"(service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
DAILY,1,1,1,1,1,1,1,20060701,20060731
)"};

constexpr auto const routes = std::string_view{
    R"(route_id,agency_id,route_short_name,route_long_name,route_type,ticketing_deep_link_id
R1,DB,R1,Rail Route,2,dl:2
R2,VU,R2,Bus Route,3,
)"};

constexpr auto const trips = std::string_view{
    R"(route_id,service_id,trip_id
R1,DAILY,TRAIN
R2,DAILY,BUS
)"};

constexpr auto const stop_times = std::string_view{
    R"(trip_id,arrival_time,departure_time,stop_id,stop_sequence
TRAIN,08:00:00,08:00:00,S1,1
TRAIN,08:10:00,08:10:00,S2,2
BUS,10:00:00,10:00:00,S1,1
BUS,10:10:00,10:10:00,S2,2
)"};

constexpr auto const deep_links = std::string_view{
    R"(ticketing_deep_link_id,web_url,android_intent_uri,ios_universal_link_url
dl:1,https://example.com/agency,android://agency,https://example.com/ios/agency
dl:2,https://example.com/route,android://route,https://example.com/ios/route
)"};

mem_dir test_files(std::string_view const agency_txt,
                   std::string_view const routes_txt,
                   bool const with_deep_links_file = true) {
  using std::filesystem::path;
  auto files = mem_dir::dir_t{{path{kAgencyFile}, std::string{agency_txt}},
                              {path{kStopFile}, std::string{stops}},
                              {path{kCalenderFile}, std::string{calendar}},
                              {path{kRoutesFile}, std::string{routes_txt}},
                              {path{kTripsFile}, std::string{trips}},
                              {path{kStopTimesFile}, std::string{stop_times}}};
  if (with_deep_links_file) {
    files.emplace(path{kTicketingDeepLinksFile}, std::string{deep_links});
  }
  return mem_dir{std::move(files)};
}

void load(timetable& tt, mem_dir const& files) {
  tt.date_range_ = {2006_y / 7 / 1, 2006_y / 8 / 1};
  load_timetable({}, source_idx_t{0U}, files, tt);
}

}  // namespace

// Deep link ids declared in agency.txt / routes.txt resolve to the URLs of
// ticketing_deep_links.txt -- and agencies/routes without an id keep the
// `invalid()` sentinel that consumers test against.
TEST(gtfs, ticketing_deep_links_resolve) {
  auto const files = test_files(agency, routes);
  ASSERT_TRUE(applicable(files));

  auto tt = timetable{};
  load(tt, files);

  auto const db_link = tt.providers_[kDB].ticketing_link_;
  ASSERT_NE(ticketing_link_idx_t::invalid(), db_link);
  EXPECT_EQ("https://example.com/agency",
            tt.ticketing_links_.web_[db_link].view());
  EXPECT_EQ("android://agency", tt.ticketing_links_.andoid_[db_link].view());
  EXPECT_EQ("https://example.com/ios/agency",
            tt.ticketing_links_.ios_[db_link].view());

  EXPECT_EQ(ticketing_link_idx_t::invalid(),
            tt.providers_[kVU].ticketing_link_);

  auto const& route_links =
      tt.route_ids_[source_idx_t{0U}].route_id_ticketing_link_;
  auto n_route_links = 0U;
  for (auto const link : route_links) {
    if (link != ticketing_link_idx_t::invalid()) {
      EXPECT_EQ("https://example.com/route",
                tt.ticketing_links_.web_[link].view());
      ++n_route_links;
    }
  }
  EXPECT_EQ(1U, n_route_links);
}

// An agency referencing a deep link id that ticketing_deep_links.txt does not
// define must not abort the load: nigiri::loader::load() rethrows, so one such
// feed takes down a whole multi-feed timetable build.
TEST(gtfs, ticketing_unknown_agency_deep_link_id_does_not_abort_load) {
  constexpr auto const agency_unknown_id = std::string_view{
      R"(agency_id,agency_name,agency_url,agency_timezone,ticketing_deep_link_id
DB,Deutsche Bahn,http://db.de,Europe/Berlin,dl:missing
VU,Verkehrsverbund,http://vu.de,Europe/Berlin,
)"};

  auto const files = test_files(agency_unknown_id, routes);
  ASSERT_TRUE(applicable(files));

  auto tt = timetable{};
  ASSERT_NO_THROW(load(tt, files));
  EXPECT_EQ(ticketing_link_idx_t::invalid(),
            tt.providers_[kDB].ticketing_link_);
}

// Same for routes.txt.
TEST(gtfs, ticketing_unknown_route_deep_link_id_does_not_abort_load) {
  constexpr auto const routes_unknown_id = std::string_view{
      R"(route_id,agency_id,route_short_name,route_long_name,route_type,ticketing_deep_link_id
R1,DB,R1,Rail Route,2,dl:missing
R2,VU,R2,Bus Route,3,
)"};

  auto const files = test_files(agency, routes_unknown_id);
  ASSERT_TRUE(applicable(files));

  auto tt = timetable{};
  ASSERT_NO_THROW(load(tt, files));

  auto const& route_links =
      tt.route_ids_[source_idx_t{0U}].route_id_ticketing_link_;
  for (auto const link : route_links) {
    EXPECT_EQ(ticketing_link_idx_t::invalid(), link);
  }
}

// A feed that fills ticketing_deep_link_id but ships no
// ticketing_deep_links.txt at all: the lookup table is empty, so every id is
// unknown.
TEST(gtfs, ticketing_deep_links_file_missing_does_not_abort_load) {
  auto const files = test_files(agency, routes, false);
  ASSERT_TRUE(applicable(files));

  auto tt = timetable{};
  ASSERT_NO_THROW(load(tt, files));
  EXPECT_EQ(ticketing_link_idx_t::invalid(),
            tt.providers_[kDB].ticketing_link_);
}
