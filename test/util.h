#pragma once

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/timetable.h"

nigiri::timetable load_gtfs(auto const& files,
                            nigiri::interval<date::sys_days> const date_range) {
  using namespace date;
  nigiri::timetable tt;
  tt.date_range_ = date_range;
  nigiri::loader::register_special_stations(tt);
  nigiri::loader::gtfs::load_timetable({}, nigiri::source_idx_t{0U}, files(),
                                       tt);
  nigiri::loader::finalize(tt);
  return tt;
}