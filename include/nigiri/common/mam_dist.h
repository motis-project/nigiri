#pragma once

#include <cstdint>
#include <utility>

namespace nigiri {

/*
 * Computes the minimal distance between two minutes after midnight values <
 * 1440 min, i.e., if the shorter distance between the two values spans midnight
 * that distance is returned
 * the first value of the returned pair contains the absolute distance between
 * the two values
 * the second value of the returned pair signals if and in which
 * direction midnight is passed:
 * -1: [actual -- midnight -- expected]
 *  0: [actual -- expected] / [expected -- actual]
 * +1: [expected -- midnight -- actual]
 */
std::pair<std::uint16_t, std::int16_t> mam_dist(std::uint16_t expected,
                                                std::uint16_t actual);

}  // namespace nigiri