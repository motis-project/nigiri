#pragma once

#include <string>

#include "date/date.h"

namespace nigiri {

date::sys_days parse_date(std::string_view);

}  // namespace nigiri