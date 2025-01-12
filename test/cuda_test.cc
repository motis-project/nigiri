#include "gtest/gtest.h"

#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/routing/gpu/raptor.h"

#include "./loader/hrd/hrd_timetable.h"

namespace ngpu = nigiri::routing::gpu;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::test_data::hrd_timetable;

TEST(nigiri_cuda, test) {
  constexpr auto const src = source_idx_t{0U};

  auto tt = timetable{};
  tt.date_range_ = full_period();
  load_timetable(src, loader::hrd::hrd_5_20_26, files_abc(), tt);
  finalize(tt);

  auto const gpu_tt = ngpu::copy_timetable(tt);

  auto state = ngpu::raptor_state{.gpu_tt_ = gpu_tt};
  state.resize(tt.n_locations(), tt.n_routes(), 0U);
}