#pragma once

#include <chrono>
#include <string>
#include <utility>

#include "nigiri/common/interval.h"

namespace nigiri::loader::hrd {

std::string parse_schedule_name(std::string_view file_content);

interval<std::chrono::sys_days> parse_interval(std::string_view file_content);

}  // namespace nigiri::loader::hrd
