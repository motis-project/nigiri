#pragma once
#include <string>
#include <vector>

namespace nigiri::loader::netex {
struct operatingPeriod {
  std::string id;
  // fromDate
  // toDate
  // validDayBits
};

// Temporary helper TODO probably delete or so
std::vector<std::string> toString(std::vector<operatingPeriod>& ops) {
  std::vector<std::string> out;
  for (auto const& op : ops) {
    out.emplace_back(op.id);
  }
  return out;
}
}  // namespace nigiri::loader::netex