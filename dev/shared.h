#pragma once

// Duplicated from libosmium

#include <cstdint>
#include <cmath>

constexpr int32_t coordinate_precision{10000000};

constexpr int32_t double_to_fix(const double c) noexcept {
    return static_cast<int32_t>(std::round(c * coordinate_precision));
}

constexpr double fix_to_double(const int32_t c) noexcept {
    return static_cast<double>(c) / coordinate_precision;
}