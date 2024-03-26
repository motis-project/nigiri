#include "gtest/gtest.h"

#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/query_generator/query_generator.h"
#include "nigiri/query_generator/transport_mode.h"

#include "../loader/hrd/hrd_timetable.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::test_data::hrd_timetable;
using namespace nigiri::query_generation;

TEST(query_generation, intermodal) {
  constexpr auto const src = source_idx_t{0U};

  timetable tt;
  tt.date_range_ = full_period();
  register_special_stations(tt);
  load_timetable(src, loader::hrd::hrd_5_20_26, files_abc(), tt);
  finalize(tt);

  query_generator qg{tt};
  qg.start_mode_ = kCar;
  qg.init_rng();

  auto q = qg.random_pretrip_query();
  for (auto const& s : q.start_) {
    std::cout << "(start_location: " << s.target_
              << ", duration: " << s.duration_ << ", start_type: " << s.type_
              << ")\n";
  }
  for (auto const& d : q.destination_) {
    std::cout << "(destination_location: " << d.target_
              << ", duration: " << d.duration_
              << ", destination_type: " << d.type_ << ")\n";
  }
}