#pragma once

#include <cstdint>
#include <utility>

#include "nigiri/types.h"

namespace nigiri {

/*
 * Computes the minimal distance between two minutes after midnight values
 * within [0, 1440[, i.e., if the shorter distance between the two values spans
 * midnight that distance is returned
 * the first value of the returned pair contains the absolute distance between
 * the two values the second value of the returned pair signals if and in which
 * direction midnight is passed:
 * -1: [actual -- midnight -- expected]
 *  0: [actual -- expected] / [expected -- actual]
 * +1: [expected -- midnight -- actual]
 */
std::pair<i32_minutes, date::days> mam_dist(i32_minutes expected,
                                            i32_minutes actual);

}  // namespace nigiri