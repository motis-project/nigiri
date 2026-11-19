#include "gtest/gtest.h"

#include "nigiri/loader/load.h"
#include "nigiri/query_generator/generator.h"
#include "nigiri/query_generator/generator_settings.h"
#include "nigiri/query_generator/transport_mode.h"
#include "nigiri/routing/query.h"

#include "../loader/hrd/hrd_timetable.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::test_data::hrd_timetable;
using namespace nigiri::query_generation;

TEST(query_generation, pretrip_station) {
  auto const tt = loader::load(files_abc(), full_period());

  generator_settings gs;
  gs.start_match_mode_ = routing::location_match_mode::kEquivalent;
  gs.dest_match_mode_ = routing::location_match_mode::kEquivalent;

  auto qg = generator{tt, gs};

  auto const sdq = qg.random_query();
  ASSERT_TRUE(sdq.has_value());
}

TEST(query_generation, pretrip_intermodal) {
  auto const tt = loader::load(files_abc(), full_period());

  generator_settings gs;
  gs.start_mode_ = kCar;

  auto qg = generator{tt, gs};

  auto const sdq = qg.random_query();
  ASSERT_TRUE(sdq.has_value());
}

TEST(query_generation, reproducibility) {
  auto const tt = loader::load(files_abc(), full_period());

  generator_settings const gs;
  auto const seed = 2342;
  auto const num_queries = 3U;

  auto qg0 = generator{tt, gs, seed};
  auto result_qg0 =
      std::vector<std::optional<query_generation::start_dest_query>>{};
  result_qg0.reserve(num_queries);
  for (auto i = 0U; i < num_queries; ++i) {
    result_qg0.emplace_back(qg0.random_query());
  }

  auto qg1 = generator{tt, gs, seed};
  for (auto i = 0U; i < num_queries; ++i) {
    auto const result_qg1 = qg1.random_query();
    ASSERT_EQ(result_qg0[i].has_value(), result_qg1.has_value());
    if (result_qg0[i].has_value()) {
      EXPECT_EQ(result_qg0[i].value().q_, result_qg1.value().q_);
    }
  }
}