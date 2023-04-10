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
  tt.date_range_.from_ = (2006_y / July / 1).operator sys_days();
  tt.date_range_.to_ = (2006_y / July / 31).operator sys_days();
  load_timetable(source_idx_t{0}, example_files(), tt);

  std::cerr << "OUTPUT:\n";
  std::cerr << "constexpr auto const expected = std::set<std::string>{";
  for (auto const& ss : service_strings(tt)) {
    std::cerr << "R\"(" << ss << ")\",\n";
  }
  std::cerr << "};\n";
}
