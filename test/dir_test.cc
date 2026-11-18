#include "gtest/gtest.h"

#include "nigiri/loader/dir.h"

#include <cstdint>
#include <fstream>
#include <iterator>
#include <vector>

#include "utl/parser/cstr.h"
#include "utl/zip.h"

#include "wyhash.h"

using namespace nigiri::loader;

constexpr auto const data = std::string_view{R"(0000001     A
0000002     B
0000003     C
0000004     D
0000005     E
0000006     F
0000007     G
0000008     H
0000009     I
0000010     J
0000011     K
0000012     L
0000013     M
0000014     N
0000015     O
0000016     P
)"};

std::vector<std::uint8_t> read_file_bytes(char const* path) {
  auto f = std::ifstream{path, std::ios::binary};
  auto buf = std::vector<std::uint8_t>{};
  buf.assign(std::istreambuf_iterator<char>{f},
             std::istreambuf_iterator<char>{});
  return buf;
}

TEST(dir, file_contents) {
  auto const fs = fs_dir{"test/test_data/mss-dayshift3"};
  auto const zip = zip_dir{"test/test_data/mss-dayshift3.zip"};
  auto const mem = mem_dir{{{"stamm/bahnhof.101", std::string{data}}}};
  auto const fs_stamm = fs.get_file("stamm/bahnhof.101");
  auto const zip_stamm = zip.get_file("stamm/bahnhof.101");
  auto const mem_stamm = mem.get_file("stamm/bahnhof.101");

  for (auto const [ref, a, b, c] : utl::czip_no_size_check(
           utl::lines(data), utl::lines(fs_stamm.data()),
           utl::lines(zip_stamm.data()), utl::lines(mem_stamm.data()))) {
    EXPECT_EQ(ref.view(), a.view());
    EXPECT_EQ(ref.view(), b.view());
    EXPECT_EQ(ref.view(), c.view());
  }
}

TEST(dir, directory_listing) {
  auto const zip = zip_dir{"test/test_data/mss-dayshift3.zip"};
  auto const fs = fs_dir{"test/test_data/mss-dayshift3"};
  auto const mem = mem_dir{mem_dir::dir_t{{"stamm/attributd_int.101", ""},
                                          {"stamm/bahnhof.101", ""},
                                          {"stamm/bitfield.101", ""},
                                          {"stamm/dbkoord_geo.101", ""},
                                          {"stamm/durchbi.101", ""},
                                          {"stamm/eckdaten.101", ""},
                                          {"stamm/gleise.101", ""},
                                          {"stamm/infotext.101", ""},
                                          {"stamm/metabhf.101", ""},
                                          {"stamm/metabhf_zusatz.101", ""},
                                          {"stamm/minct.csv", ""},
                                          {"stamm/richtung.101", ""},
                                          {"stamm/unternehmen_ris.101", ""},
                                          {"stamm/vereinig_vt.101", ""},
                                          {"stamm/zeitvs.101", ""},
                                          {"stamm/zugart_int.101", ""}}};

  EXPECT_EQ(zip.list_files("stamm/"), fs.list_files("stamm/"));
  EXPECT_EQ(mem.list_files("stamm/"), fs.list_files("stamm/"));

  EXPECT_EQ(zip.list_files("stamm/bahnhof.101"),
            fs.list_files("stamm/bahnhof.101"));
  EXPECT_EQ(mem.list_files("stamm/bahnhof.101"),
            fs.list_files("stamm/bahnhof.101"));
}

TEST(dir, in_memory_zip) {
  auto const zip =
      zip_dir{read_file_bytes("test/test_data/mss-dayshift3.zip")};

  EXPECT_TRUE(zip.exists("stamm/bahnhof.101"));
  EXPECT_FALSE(zip.exists("stamm/BAHNHOF.101"));
  EXPECT_EQ(zip.file_size("stamm/bahnhof.101"),
            std::string_view{data}.size());
  EXPECT_EQ(zip.get_file("stamm/bahnhof.101").data(), std::string_view{data});
}

// zstd fixture generation (minizip-ng 4.2.2 CLI):
//   printf '<data as above>\n' > stamm/bahnhof.101 && printf '0000001E\n0000002E\n' > stamm/zugart_int.101
//   minizip -9 -t -i <fixture> stamm
// (-t selects MZ_COMPRESS_METHOD_ZSTD, -i keeps the stamm/ prefixes, adding
// the top-level directory also creates the stamm/ directory entry required
// by zip_dir::list_files.)
TEST(dir, zstd) {
  constexpr auto const zugart = std::string_view{"0000001E\n0000002E\n"};

  auto const zip = zip_dir{"test/test_data/mss-dayshift3-zstd.zip"};
  auto const mem = mem_dir{mem_dir::dir_t{{"stamm/bahnhof.101", std::string{data}},
                                         {"stamm/zugart_int.101",
                                          std::string{zugart}}}};

  auto const fs = fs_dir{"test/test_data/mss-dayshift3"};
  EXPECT_EQ(zip.list_files("stamm/"), mem.list_files("stamm/"));
  EXPECT_TRUE(zip.exists("stamm/bahnhof.101"));
  EXPECT_FALSE(zip.exists("stamm/bahnhof.1"));
  EXPECT_EQ(zip.file_size("stamm/bahnhof.101"), std::string_view{data}.size());
  EXPECT_EQ(zip.get_file("stamm/bahnhof.101").data(), std::string_view{data});
  EXPECT_EQ(zip.get_file("stamm/bahnhof.101").data(),
            fs.get_file("stamm/bahnhof.101").data());
  EXPECT_EQ(zip.get_file("stamm/zugart_int.101").data(), zugart);

  auto const zip_mem =
      zip_dir{read_file_bytes("test/test_data/mss-dayshift3-zstd.zip")};
  EXPECT_EQ(zip_mem.get_file("stamm/bahnhof.101").data(),
            std::string_view{data});
}

TEST(dir, hash) {
  auto const raw = read_file_bytes("test/test_data/mss-dayshift3-zstd.zip");
  auto const zip = zip_dir{"test/test_data/mss-dayshift3-zstd.zip"};
  auto const mem = zip_dir{raw};
  EXPECT_EQ(zip.hash(), wyhash(raw.data(), raw.size(), 0U, _wyp));
  EXPECT_EQ(mem.hash(), zip.hash());
}

TEST(nigiri, to_dir) {
  using namespace std::string_view_literals;
  constexpr auto const gtfs = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone,agency_lang,agency_phone
"11","Schweizerische Bundesbahnen SBB","http://www.sbb.ch/","Europe/Berlin","DE","0848 44 66 88"

# calendar.txt
"TA+xce80","1","1","1","1","1","0","0","20221211","20231209"

# calendar_dates.txt
service_id,date,exception_type
"TA+xce80","20231028","1"
"TA+xce80","20231029","1"
"TA+xce80","20231030","2"
"TA+xce80","20231031","2"
"TA+xce80","20231101","2"
)"sv;

  auto const dir = mem_dir::read(gtfs);
  EXPECT_EQ(
      dir.get_file("agency.txt").data(),
      R"(agency_id,agency_name,agency_url,agency_timezone,agency_lang,agency_phone
"11","Schweizerische Bundesbahnen SBB","http://www.sbb.ch/","Europe/Berlin","DE","0848 44 66 88"
)"sv);
  EXPECT_EQ(dir.get_file("calendar_dates.txt").data(),
            R"(service_id,date,exception_type
"TA+xce80","20231028","1"
"TA+xce80","20231029","1"
"TA+xce80","20231030","2"
"TA+xce80","20231031","2"
"TA+xce80","20231101","2"
)"sv);
}
