#include "gtest/gtest.h"

#include <optional>
#include <string>
#include <vector>

#include "nigiri/loader/gtfs/agency.h"
#include "nigiri/loader/gtfs/translations.h"
#include "nigiri/timetable.h"

using namespace nigiri;
using namespace nigiri::loader::gtfs;

namespace {

constexpr auto const kAgencyCsv =
    R"(agency_id,agency_name,agency_url,agency_timezone
1,Hauptbahnhof AG,https://example.com,Europe/Berlin
)";

constexpr auto const kTranslationsCsv =
    R"(table_name,field_name,language,translation,record_id,record_sub_id,field_value
agency,agency_name,en,Main Station Inc,1,,
agency,agency_name,fr,Gare Centrale SA,1,,
)";

}  // namespace

TEST(gtfs, translations_default_lang_from_field_value) {
  auto tt = timetable{};
  auto timezones = tz_map{};

  auto i18n = read_translations(tt, "de", kTranslationsCsv);
  auto const agencies = read_agencies(source_idx_t{0}, tt, i18n, timezones,
                                      kAgencyCsv, "Europe/Berlin");

  auto const it = agencies.find("1");
  ASSERT_NE(it, end(agencies));
  auto const& provider = tt.providers_.at(it->second);

  EXPECT_EQ("Hauptbahnhof AG", tt.get_default_translation(provider.name_));
  EXPECT_EQ("Hauptbahnhof AG",
            tt.translate(std::optional{std::vector<std::string>{"de"}},
                         provider.name_));
  EXPECT_EQ("Main Station Inc",
            tt.translate(std::optional{std::vector<std::string>{"en"}},
                         provider.name_));
  EXPECT_EQ("Gare Centrale SA",
            tt.translate(std::optional{std::vector<std::string>{"fr"}},
                         provider.name_));
}
