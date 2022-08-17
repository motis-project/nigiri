#include "doctest/doctest.h"

#include "nigiri/loader/dir.h"

#include "fmt/core.h"
#include "fmt/ranges.h"

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

TEST_CASE("dir test - file contents") {
  auto const fs = fs_dir{"test/test_data/mss-dayshift3"};
  auto const zip = zip_dir{"test/test_data/mss-dayshift3.zip"};
  auto const mem = mem_dir{{{"stamm/bahnhof.101", std::string{data}}}};
  auto const fs_stamm = fs.get_file("stamm/bahnhof.101");
  auto const zip_stamm = zip.get_file("stamm/bahnhof.101");
  auto const mem_stamm = mem.get_file("stamm/bahnhof.101");
  CHECK_EQ(fs_stamm.data(), data);
  CHECK_EQ(zip_stamm.data(), data);
  CHECK_EQ(mem_stamm.data(), data);
}

TEST_CASE("dir test - directory listing") {
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

  CHECK_EQ(zip.list_files("stamm/"), fs.list_files("stamm/"));
  CHECK_EQ(mem.list_files("stamm/"), fs.list_files("stamm/"));

  CHECK_EQ(zip.list_files("stamm/bahnhof.101"),
           fs.list_files("stamm/bahnhof.101"));
  CHECK_EQ(mem.list_files("stamm/bahnhof.101"),
           fs.list_files("stamm/bahnhof.101"));
}