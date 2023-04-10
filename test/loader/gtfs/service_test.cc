#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/timetable.h"

#include "../service_strings.h"
#include "./test_data.h"

using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace date;

TEST(gtfs, services) {
  timetable tt;
  tt.date_range_.from_ = (2019_y / March / 15).operator sys_days();
  tt.date_range_.to_ = (2019_y / April / 15).operator sys_days();
  load_timetable(source_idx_t{0}, test_files(), tt);

  std::cout << "OUTPUT:\n";
  std::cout << "constexpr auto const expected = std::set<std::string>{";
  for (auto const& ss : service_strings(tt)) {
    std::cout << "R\"(" << ss << ")\",\n";
  }
  std::cout << "};\n";
}
