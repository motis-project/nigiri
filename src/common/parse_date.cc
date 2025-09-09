#include "nigiri/common/parse_date.h"

namespace nigiri {

date::sys_days parse_date(std::string_view str) {
  if (str == "TODAY") {
    return std::chrono::time_point_cast<date::days>(
        std::chrono::system_clock::now());
  }
  if (str.size() > 0 && str[0] == '-') {
    return std::chrono::time_point_cast<date::days>(
        std::chrono::system_clock::now() +
        std::chrono::days{stoi(std::string{str})});
  }

  date::sys_days parsed;
  std::stringstream ss;
  ss << str;
  ss >> date::parse("%F", parsed);
  return parsed;
}

}  // namespace nigiri