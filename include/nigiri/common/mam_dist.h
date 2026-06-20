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
 * the two values
 * the second value of the returned pair signals if and in which direction
 * midnight is passed:
 * -1: [b -- midnight -- a]
 *  0: [b -- a] / [a -- b]
 * +1: [a -- midnight -- b]
 */
std::pair<i32_minutes, date::days> mam_dist(i32_minutes a, i32_minutes b);

}  // namespace nigiri