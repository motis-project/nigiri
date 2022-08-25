#pragma once

#include <chrono>
#include <string>
#include <utility>

namespace nigiri::loader::hrd {

using interval_t =
    std::pair<std::chrono::year_month_day, std::chrono::year_month_day>;

std::string parse_schedule_name(std::string_view file_content);

interval_t parse_interval(std::string_view file_content);

}  // namespace nigiri::loader::hrd
