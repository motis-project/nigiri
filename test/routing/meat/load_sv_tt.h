#pragma once

#include <string_view>

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"

#include "nigiri/timetable.h"

namespace nigiri::test {

inline nigiri::timetable load_tt(std::string_view sv_tt,
                         nigiri::interval<date::sys_days> date_range) {
  auto tt = nigiri::timetable{};
  auto src = nigiri::source_idx_t{0};
  tt.date_range_ = date_range;
  nigiri::loader::register_special_stations(tt);
  nigiri::loader::gtfs::load_timetable({}, src, loader::mem_dir::read(sv_tt), tt);
  nigiri::loader::finalize(tt);
  return tt;
}

}  // namespace nigiri::test