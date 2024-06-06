#pragma once

#include "nigiri/loader/dir.h"
#include "nigiri/types.h"

namespace nigiri::routing {

loader::mem_dir shortest_fp_files();

constexpr interval<std::chrono::sys_days> shortest_fp_period() {
  using namespace date;
  constexpr auto const from = (2024_y / January / 01).operator sys_days();
  constexpr auto const to = (2024_y / December / 31).operator sys_days();
  return {from, to};
}

}  // namespace nigiri::routing