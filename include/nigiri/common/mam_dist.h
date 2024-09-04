#pragma once

#include <cstdint>

namespace nigiri {

/*
 * Computes the minimal distance between two minutes after midnight values <
 * 1440 min, i.e., if the shorter distance between the two values spans midnight
 * that distance is returned
 */
std::uint16_t mam_dist(std::uint16_t, std::uint16_t);

}  // namespace nigiri