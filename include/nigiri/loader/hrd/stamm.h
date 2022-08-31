#pragma once

#include "nigiri/loader/dir.h"
#include "nigiri/loader/hrd/attribute.h"
#include "nigiri/loader/hrd/bitfield.h"
#include "nigiri/loader/hrd/category.h"
#include "nigiri/loader/hrd/direction.h"
#include "nigiri/loader/hrd/parser_config.h"
#include "nigiri/loader/hrd/provider.h"
#include "nigiri/loader/hrd/station.h"
#include "nigiri/loader/hrd/timezone.h"
#include "nigiri/timetable.h"

namespace nigiri::loader::hrd {

struct stamm {
  stamm(config const&, timetable&, dir const&);
  location_map_t locations_;
  category_map_t categories_;
  provider_map_t providers_;
  attribute_map_t attributes_;
  direction_map_t directions_;
  bitfield_map_t bitfields_;
  timezone_map_t timezones_;
  timetable& tt_;
};

}  // namespace nigiri::loader::hrd