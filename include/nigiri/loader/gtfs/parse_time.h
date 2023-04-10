#pragma once

#include "utl/parser/cstr.h"

#include "nigiri/types.h"

namespace nigiri::loader::gtfs {

constexpr auto const kInterpolate =
    duration_t{std::numeric_limits<duration_t::rep>::min()};

duration_t hhmm_to_min(utl::cstr s);

}  // namespace nigiri::loader::gtfs
