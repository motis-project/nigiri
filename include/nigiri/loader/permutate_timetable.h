#pragma once

namespace nigiri {

struct timetable;

enum class route_permutation_strategy {
  kCentroid,
  kMaxEventsStop
};

constexpr auto kRoutePermutationStrategy =
    route_permutation_strategy::kCentroid;

void permutate_timetable(timetable&);

}  // namespace nigiri