
#include "doctest/doctest.h"

#include "nigiri/byte_sizes.h"
#include "nigiri/database.h"

using namespace nigiri;

TEST_CASE("db test") {
  auto const a1 = attribute{.code_ = "AB", .text_ = "Fate will decide!"};
  auto const a2 = attribute{.code_ = "CD", .text_ = "This is Sparta!!"};
  auto const p1 = provider{.short_name_ = "DB", .long_name_ = "Deutsche Bahn"};
  auto const p2 = provider{.short_name_ = "BR", .long_name_ = "Br. Railway"};

  auto db = database<nigiri::attribute, nigiri::provider>{"./db.mdb", 512_kB};
  auto const h_a1 = db.add(a1);
  auto const h_a2 = db.add(a2);
  auto const h_p1 = db.add(p1);
  auto const h_p2 = db.add(p2);
  db.flush();

  CHECK(a1 == db.get<attribute>(h_a1));
  CHECK(a2 == db.get<attribute>(h_a2));
  CHECK(p1 == db.get<provider>(h_p1));
  CHECK(p2 == db.get<provider>(h_p2));

  CHECK_THROWS(db.get<attribute>(100));
}
