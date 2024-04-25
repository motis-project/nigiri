#include "gtest/gtest.h"

#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/query_generator/generator.h"
#include "nigiri/query_generator/generator_settings.h"
#include "nigiri/query_generator/transport_mode.h"
#include "../loader/hrd/hrd_timetable.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::test_data::hrd_timetable;
using namespace nigiri::query_generation;

bool equal_queries(routing::query const& a, routing::query const& b) {

  if (holds_alternative<unixtime_t>(a.start_time_) &&
      holds_alternative<unixtime_t>(b.start_time_)) {
    if (get<unixtime_t>(a.start_time_) != get<unixtime_t>(b.start_time_)) {
      return false;
    }
  } else if (holds_alternative<interval<unixtime_t>>(a.start_time_) &&
             holds_alternative<interval<unixtime_t>>(b.start_time_)) {
    if (get<interval<unixtime_t>>(a.start_time_).from_ !=
            get<interval<unixtime_t>>(b.start_time_).from_ ||
        get<interval<unixtime_t>>(a.start_time_).to_ !=
            get<interval<unixtime_t>>(b.start_time_).to_) {
      return false;
    }
  } else {
    return false;
  }

  auto const offset_equal = [](auto const& o0, auto const& o1) {
    return o0.target_ == o1.target_ && o0.duration_ == o1.duration_ &&
           o0.transport_mode_id_ == o1.transport_mode_id_;
  };

  if (a.start_.size() != b.start_.size()) {
    return false;
  }
  for (auto i = 0U; i < a.start_.size(); ++i) {
    if (!offset_equal(a.start_[i], b.start_[i])) {
      return false;
    }
  }

  if (a.destination_.size() != b.destination_.size()) {
    return false;
  }
  for (auto i = 0U; i < a.destination_.size(); ++i) {
    if (!offset_equal(a.destination_[i], b.destination_[i])) {
      return false;
    }
  }

  return a.start_match_mode_ == b.start_match_mode_ &&
         a.dest_match_mode_ == b.dest_match_mode_ &&
         a.use_start_footpaths_ == b.use_start_footpaths_ &&
         a.max_transfers_ == b.max_transfers_ &&
         a.min_connection_count_ == b.min_connection_count_ &&
         a.extend_interval_earlier_ == b.extend_interval_earlier_ &&
         a.extend_interval_later_ == b.extend_interval_later_ &&
         a.prf_idx_ == b.prf_idx_ && a.allowed_claszes_ == b.allowed_claszes_;
}

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

  auto qg = generator{tt, gs};

  auto const q = qg.random_pretrip_query();
  ASSERT_TRUE(q.has_value());

  for (auto const& s : q.value().start_) {
    std::cout << "(start_location: " << s.target_
              << ", duration: " << s.duration_
              << ", start_type: " << s.transport_mode_id_ << ")\n";
  }
  for (auto const& d : q.value().destination_) {
    std::cout << "(destination_location: " << d.target_
              << ", duration: " << d.duration_
              << ", destination_type: " << d.transport_mode_id_ << ")\n";
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

  auto qg = generator{tt, gs};

  auto const q = qg.random_pretrip_query();
  ASSERT_TRUE(q.has_value());

  for (auto const& s : q.value().start_) {
    std::cout << "(start_location: " << s.target_
              << ", duration: " << s.duration_
              << ", start_type: " << s.transport_mode_id_ << ")\n";
  }
  for (auto const& d : q.value().destination_) {
    std::cout << "(destination_location: " << d.target_
              << ", duration: " << d.duration_
              << ", destination_type: " << d.transport_mode_id_ << ")\n";
  }
}

TEST(query_generation, reproducibility) {
  constexpr auto const src = source_idx_t{0U};
  timetable tt;
  tt.date_range_ = full_period();
  register_special_stations(tt);
  load_timetable(src, loader::hrd::hrd_5_20_26, files_abc(), tt);
  finalize(tt);

  generator_settings const gs;
  auto const seed = 2342;
  auto const num_queries = 100U;

  auto qg0 = generator{tt, gs, seed};
  auto result_qg0 = std::vector<std::optional<routing::query>>{};
  result_qg0.reserve(num_queries);
  for (auto i = 0U; i < num_queries; ++i) {
    result_qg0.emplace_back(qg0.random_pretrip_query());
  }

  auto qg1 = generator{tt, gs, seed};
  for (auto i = 0U; i < num_queries; ++i) {
    auto const result_qg1 = qg1.random_pretrip_query();
    ASSERT_EQ(result_qg0[i].has_value(), result_qg1.has_value());
    if (result_qg0[i].has_value()) {
      EXPECT_TRUE(equal_queries(result_qg0[i].value(), result_qg1.value()));
    }
  }
}