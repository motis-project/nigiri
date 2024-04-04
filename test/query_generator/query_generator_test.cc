#include "gtest/gtest.h"

#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/query_generator/nigiri_generator.h"
#include "nigiri/query_generator/transport_mode.h"

#include "../loader/hrd/hrd_timetable.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::test_data::hrd_timetable;
using namespace nigiri::query_generation;

TEST(query_generation, pretrip_station) {
  constexpr auto const src = source_idx_t{0U};
  timetable tt;
  tt.date_range_ = full_period();
  register_special_stations(tt);
  load_timetable(src, loader::hrd::hrd_5_20_26, files_abc(), tt);
  finalize(tt);

  generator_settings gs;
  gs.start_match_mode_ = routing::location_match_mode::kEquivalent;
  gs.dest_match_mode_ = routing::location_match_mode::kEquivalent;

  nigiri_generator qg{tt, gs};

  auto q = qg.random_pretrip_query();
  ASSERT_TRUE(q.has_value());

  for (auto const& s : q.value().start_) {
    std::cout << "(start_location: " << s.target_
              << ", duration: " << s.duration_ << ", start_type: " << s.type_
              << ")\n";
  }
  for (auto const& d : q.value().destination_) {
    std::cout << "(destination_location: " << d.target_
              << ", duration: " << d.duration_
              << ", destination_type: " << d.type_ << ")\n";
  }
}

TEST(query_generation, pretrip_intermodal) {
  constexpr auto const src = source_idx_t{0U};
  timetable tt;
  tt.date_range_ = full_period();
  register_special_stations(tt);
  load_timetable(src, loader::hrd::hrd_5_20_26, files_abc(), tt);
  finalize(tt);

  generator_settings gs;
  gs.start_mode_ = kCar;

  nigiri_generator qg{tt, gs};

  auto q = qg.random_pretrip_query();
  ASSERT_TRUE(q.has_value());

  for (auto const& s : q.value().start_) {
    std::cout << "(start_location: " << s.target_
              << ", duration: " << s.duration_ << ", start_type: " << s.type_
              << ")\n";
  }
  for (auto const& d : q.value().destination_) {
    std::cout << "(destination_location: " << d.target_
              << ", duration: " << d.duration_
              << ", destination_type: " << d.type_ << ")\n";
  }
}

TEST(query_generation, ontrip_intermodal) {
  constexpr auto const src = source_idx_t{0U};
  timetable tt;
  tt.date_range_ = full_period();
  register_special_stations(tt);
  load_timetable(src, loader::hrd::hrd_5_20_26, files_abc(), tt);
  finalize(tt);

  generator_settings gs;
  nigiri_generator qg{tt, gs};

  auto q = qg.random_ontrip_query();
  ASSERT_TRUE(q.has_value());

  for (auto const& s : q.value().start_) {
    std::cout << "(start_location: " << s.target_
              << ", duration: " << s.duration_ << ", start_type: " << s.type_
              << ")\n";
  }
  for (auto const& d : q.value().destination_) {
    std::cout << "(destination_location: " << d.target_
              << ", duration: " << d.duration_
              << ", destination_type: " << d.type_ << ")\n";
  }
}