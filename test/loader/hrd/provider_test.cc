#include "gtest/gtest.h"

#include "nigiri/loader/hrd/parser_config.h"
#include "nigiri/loader/hrd/stamm/provider.h"

using namespace nigiri;
using namespace nigiri::loader::hrd;

TEST(hrd, parse_providers) {
  constexpr auto const input =
      "00001 K 'DPN' L 'ABR' V 'ABELLIO Rail Mitteldeutschland GmbH'\n"
      "00001 : AM____\n"
      "00002 K 'DPN' L 'ABR' V 'ABELLIO Rail NRW GmbH'\n"
      "00002 : AR____\n"
      "00003 K 'DPN' L 'ag ' V 'agilis'\n"
      "00003 : A9____ XY____\n";
  for (auto const& c : configs) {
    auto tt = timetable{};
    auto providers = parse_providers(c, source_idx_t{0}, tt, input);

    EXPECT_EQ(4U, providers.size());

    auto const& first = tt.providers_.at(providers["AM____"]);
    EXPECT_EQ("ABR", tt.strings_.get(first.id_));
    EXPECT_EQ("ABELLIO Rail Mitteldeutschland GmbH",
              tt.strings_.get(first.name_));
    EXPECT_EQ("", tt.strings_.get(first.url_));

    auto const& second = tt.providers_.at(providers["AR____"]);
    EXPECT_EQ("ABR", tt.strings_.get(second.id_));
    EXPECT_EQ("ABELLIO Rail NRW GmbH", tt.strings_.get(second.name_));
    EXPECT_EQ("", tt.strings_.get(second.url_));

    auto const& third = tt.providers_.at(providers["A9____"]);
    EXPECT_EQ("ag ", tt.strings_.get(third.id_));
    EXPECT_EQ("agilis", tt.strings_.get(third.name_));
    EXPECT_EQ("", tt.strings_.get(third.url_));

    auto const& fourth = tt.providers_.at(providers["XY____"]);
    EXPECT_EQ("ag ", tt.strings_.get(fourth.id_));
    EXPECT_EQ("agilis", tt.strings_.get(fourth.name_));
    EXPECT_EQ("", tt.strings_.get(fourth.url_));
  }
}
