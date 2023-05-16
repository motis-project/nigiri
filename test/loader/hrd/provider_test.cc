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
    auto providers = parse_providers(c, tt, input);

    EXPECT_EQ(4U, providers.size());

    auto const& first = tt.providers_.at(providers["AM____"]);
    EXPECT_EQ("ABR", first.short_name_);
    EXPECT_EQ("ABELLIO Rail Mitteldeutschland GmbH", first.long_name_);

    auto const& second = tt.providers_.at(providers["AR____"]);
    EXPECT_EQ("ABR", second.short_name_);
    EXPECT_EQ("ABELLIO Rail NRW GmbH", second.long_name_);

    auto const& third = tt.providers_.at(providers["A9____"]);
    EXPECT_EQ("ag ", third.short_name_);
    EXPECT_EQ("agilis", third.long_name_);

    auto const& fourth = tt.providers_.at(providers["XY____"]);
    EXPECT_EQ("ag ", fourth.short_name_);
    EXPECT_EQ("agilis", fourth.long_name_);
  }
}
