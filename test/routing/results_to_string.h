#pragma once

#include <sstream>
#include <string>

#include "nigiri/routing/journey.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/timetable.h"

template <typename Journeys>
inline std::string to_string(nigiri::timetable const& tt,
                             nigiri::rt_timetable const* rtt,
                             Journeys const& results,
                             bool const debug = false) {
  std::stringstream ss;
  ss << "\n";
  for (auto const& x : results) {
    x.print(ss, tt, rtt, debug);
    ss << "\n";
  }
  return ss.str();
}

template <typename Journeys>
inline std::string to_string(nigiri::timetable const& tt,
                             Journeys const& results,
                             bool const debug = false) {
  return to_string(tt, nullptr, results, debug);
}
