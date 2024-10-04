#include "nigiri/common/parse_date.h"

namespace nigiri {

date::sys_days parse_date(std::string const& str) {
  if (str == "TODAY") {
    return std::chrono::time_point_cast<date::days>(
        std::chrono::system_clock::now());
  }

  date::sys_days parsed;
  std::stringstream ss;
  ss << str;
  ss >> date::parse("%F", parsed);
  return parsed;
}

}  // namespace nigiri