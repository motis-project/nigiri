#include "gtest/gtest.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/loader/isa/load_timetable.h"
#include "nigiri/loader/register.h"
#include "nigiri/timetable.h"

using namespace nigiri::loader;

namespace {

constexpr auto const kMinimalIsa = R"(
% zeichen.asc
UTF8#5.14#0#Europe/Berlin#
% halteste.asc
2#MASTER    #      #      ##hst#11.745228# 52.508560#### Testhalt#
% bitfeld.asc
         1#F9F3#
)";

mem_dir minimal_isa_dir() {
  auto d = mem_dir::dir_t{};
  auto const src = std::string_view{kMinimalIsa};
  auto pos = std::size_t{};
  while ((pos = src.find("% ", pos)) != std::string_view::npos) {
    auto const name_begin = pos + 2U;
    auto const name_end = src.find('\n', name_begin);
    auto const content_end = std::min(src.find("% ", name_end), src.size());
    d.emplace(std::string{src.substr(name_begin, name_end - name_begin)},
              std::string{src.substr(name_end + 1U, content_end - name_end)});
    pos = content_end;
  }
  return mem_dir{std::move(d)};
}

}  // namespace

TEST(isa, applicable) { EXPECT_TRUE(isa::applicable(minimal_isa_dir())); }

TEST(isa, stops) {
  constexpr auto const kZeichen = "UTF8#5.14#0#Europe/Berlin#\n";
  constexpr auto const kHalteste =
      "%Haltestellennummer#Lieferant#...\n"
      "   8010077#DB        ####--Gl1  # 11.631800# 52.127700###Stendal "
      "Hbf########0#de:15090:800####03:00###Europe/Berlin#\n"
      "    371760#MASTER    #   8010077#DB        ### 11.631800# "
      "52.127700###Stendal, Hbf (Bus)#\n"
      "         2#MASTER    ##### 11.745228# 52.508560###L\xC3\xBC"
      "deritz, Wartehalle#\n";

  constexpr auto const kAequival =
      "%Haltestellennummer oder Zielnummer#Lieferantenkuerzel#...\n"
      "         2#MASTER    #   8010077#DB        #S #    371760#MASTER    "
      "#SB#   8010077#DB        #B #\n"
      "   8010077#DB        #    999999#XX        #S #\n";

  auto files = mem_dir::dir_t{};
  files.emplace("zeichen.asc", kZeichen);
  files.emplace("halteste.asc", kHalteste);
  files.emplace("aequival.asc", kAequival);
  auto const d = mem_dir{std::move(files)};

  auto tt = nigiri::timetable{};
  register_special_stations(tt);
  auto const c = loader_config{};
  auto const src = nigiri::source_idx_t{0U};
  auto bitfields = nigiri::hash_map<nigiri::bitfield, nigiri::bitfield_idx_t>{};
  EXPECT_ANY_THROW(isa::load_timetable(c, src, d, tt, bitfields));

  auto const db_l = tt.find(nigiri::location_id{"DB:8010077", src});
  auto const bus_l = tt.find(nigiri::location_id{"MASTER:371760", src});
  auto const warte_l = tt.find(nigiri::location_id{"MASTER:2", src});
  ASSERT_TRUE(db_l.has_value());
  ASSERT_TRUE(bus_l.has_value());
  ASSERT_TRUE(warte_l.has_value());
  auto const db = *db_l;
  auto const bus = *bus_l;
  auto const warte = *warte_l;

  EXPECT_EQ(db, tt.locations_.parents_[bus]);
  EXPECT_EQ(nigiri::location_idx_t::invalid(), tt.locations_.parents_[db]);
  EXPECT_EQ(nigiri::location_idx_t::invalid(), tt.locations_.parents_[warte]);

  EXPECT_EQ("DB:8010077", (location{tt, db}.get_id()));
  EXPECT_EQ("Stendal Hbf", (location{tt, db}.get_name()));
  EXPECT_EQ("Lüderitz, Wartehalle", (location{tt, warte}.get_name()));
  EXPECT_EQ("--Gl1", (location{tt, db}.get_stop_code()));

  EXPECT_DOUBLE_EQ(52.1277, tt.locations_.coordinates_[db].lat_);
  EXPECT_DOUBLE_EQ(11.745228, tt.locations_.coordinates_[warte].lng_);

  EXPECT_EQ(nigiri::u8_minutes{3}, tt.locations_.transfer_time_[db]);
  EXPECT_EQ(c.default_transfer_time_, tt.locations_.transfer_time_[warte]);

  EXPECT_EQ(tt.locations_.location_timezones_[db],
            tt.locations_.location_timezones_[warte]);

  EXPECT_EQ(nigiri::location_type::kStation, tt.locations_.types_[db]);
  EXPECT_EQ(nigiri::location_type::kTrack, tt.locations_.types_[bus]);

  auto eq = std::vector<nigiri::location_idx_t>{};
  for (auto const& e : tt.locations_.equivalences_[warte]) {
    eq.emplace_back(e);
  }
  ASSERT_EQ(2U, eq.size());
  EXPECT_EQ(db, eq[0]);
  EXPECT_EQ(bus, eq[1]);
  EXPECT_TRUE(tt.locations_.equivalences_[db].empty());
}

TEST(isa, not_applicable_on_gtfs) {
  auto const gtfs = mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
A,Agency,https://example.com,Europe/Berlin
# stops.txt
stop_id,stop_name,stop_lat,stop_lon
S1,Stop,50.0,8.0
)");
  EXPECT_FALSE(isa::applicable(gtfs));
}
