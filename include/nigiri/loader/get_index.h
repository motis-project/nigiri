#pragma once

#include <cinttypes>
#include <algorithm>
#include <optional>
#include <type_traits>

#include "nigiri/types.h"

namespace nigiri::loader {

// A sleeper (types.h) rides alone: night_transfer_dim judges its rides by
// their ends, which no longer orders it like the route's other trips.
template <typename Service>
bool is_sleeper(clasz const c, Service const& s) {
  using duration = std::decay_t<decltype(s.utc_times_.front())>;
  return is_sleeper_clasz(c) && s.utc_times_.back() - s.utc_times_.front() >=
                                    duration{kSleeperMinRide};
}

template <typename ServiceVector, typename Service>
std::optional<std::size_t> get_index(ServiceVector const& route_services,
                                     Service const& s,
                                     clasz const c) {
  if (is_sleeper(c, s) ||
      (!route_services.empty() && is_sleeper(c, route_services.front()))) {
    return std::nullopt;
  }
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

    // keep (traffic day, transport offset) lexicographic order == absolute
    // event time order: full times (incl. >24:00 day offsets) must be
    // row-ordered and the per-stop spread must stay below one day
    if (!route_services.empty()) {
      auto const full_before_prev =
          index > 0 &&
          s.utc_times_[i] < route_services[index - 1].utc_times_.at(i);
      auto const full_after_next =
          index < route_services.size() &&
          s.utc_times_[i] > route_services[index].utc_times_.at(i);
      auto const lo =
          std::min(s.utc_times_[i], route_services.front().utc_times_.at(i));
      auto const hi =
          std::max(s.utc_times_[i], route_services.back().utc_times_.at(i));
      if (full_before_prev || full_after_next ||
          hi - lo >= std::decay_t<decltype(hi)>{1440}) {
        return std::nullopt;
      }
    }
  }

  return index;
}

}  // namespace nigiri::loader
