#pragma once

#include <cinttypes>
#include <optional>

namespace nigiri::loader {

template <typename ServiceVector, typename Service>
std::optional<std::size_t> get_index(ServiceVector const& route_services,
                                     Service const& s) {
  auto const index = static_cast<unsigned>(std::distance(
      begin(route_services),
      std::lower_bound(begin(route_services), end(route_services), s,
                       [&](Service const& a, Service const& b) {
                         return a.utc_times_.front() % 1440 <
                                b.utc_times_.front() % 1440;
                       })));

  for (auto i = 0U; i != s.utc_times_.size(); ++i) {
    auto const is_earlier_eq =
        index > 0 && s.utc_times_[i] % 1440 <
                         route_services[index - 1].utc_times_.at(i) % 1440;
    auto const is_later_eq =
        index < route_services.size() &&
        s.utc_times_[i] % 1440 > route_services[index].utc_times_.at(i) % 1440;
    if (is_earlier_eq || is_later_eq) {
      return std::nullopt;
    }
  }

  return index;
}

}  // namespace nigiri::loader
