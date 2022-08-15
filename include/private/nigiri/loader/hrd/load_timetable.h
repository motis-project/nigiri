#pragma once

#include "nigiri/loader/file_list.h"
#include "nigiri/loader/hrd/parser_config.h"
#include "nigiri/timetable.h"

namespace nigiri::loader::hrd {

std::shared_ptr<timetable> load_timetable(
    config const&,
    file_list const&,
    std::vector<std::string_view> const& services);

}  // namespace nigiri::loader::hrd
