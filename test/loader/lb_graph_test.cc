#include "doctest/doctest.h"

#include "nigiri/loader/hrd/load_timetable.h"

#include "hrd/hrd_timetable.h"

using namespace nigiri;
using namespace nigiri::loader::hrd;

TEST_CASE("lb_graph") {
  auto tt = timetable{};
  load_timetable(source_idx_t{0U}, hrd_5_20_26,
                 nigiri::test_data::hrd_timetable::files_simple(), tt);

  std::stringstream ss;
  for (auto i = location_idx_t{0U}; i != tt.locations_.ids_.size(); ++i) {
    if (tt.fwd_search_lb_graph_[i].empty()) {
      continue;
    }
    ss << location{tt, i} << "\n";
    for (auto const& fp : tt.fwd_search_lb_graph_[i]) {
      ss << "  " << location{tt, fp.target_} << " " << fp.duration_.count()
         << "\n";
    }
  }

  constexpr auto const raw = R"((A, 0000001)
  (B, 0000002) 60
(B, 0000002)
  (A, 0000001) 60
)";

  CHECK(std::string_view{raw} == ss.str());
}
