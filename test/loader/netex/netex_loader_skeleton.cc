//
// Created by mirko on 8/23/23.
//
#include "nigiri/loader/netex/load_timetable.h"
#include "gtest/gtest.h"

#include <filesystem>

namespace fs = std::filesystem;
using namespace nigiri::loader;

TEST(netex_loader, test_init) {

  fs::path path("test_dir");

  nigiri::source_idx_t src;
  auto const d = zip_dir{
      "test/test_data/NX-PI-01_DE_NAP_LINE_144-SILBUS-V95_20220104.zip"};
  nigiri::timetable t;

  nigiri::loader::netex::load_timetable(src, d, t);
  std::cout << t.providers_.size() << " - providers size\n";
  std::cout << t.providers_.front().long_name_ << ", "
            << t.providers_.front().short_name_ << ", "
            << t.providers_.front().tz_ << "\n";
  ASSERT_GE(t.providers_.size(), 1);

  ASSERT_TRUE(true);
}  // namespace std::filesystem