#pragma once

#include "utl/parser/cstr.h"

namespace nigiri::loader::gtfs {

constexpr auto const kInterpolate = -1;

int hhmm_to_min(utl::cstr s);

}  // namespace nigiri::loader::gtfs
