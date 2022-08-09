#pragma once

#include <chrono>
#include <cstdio>
#include <limits>

#include "date/tz.h"

#include "cista/reflection/comparable.h"
#include "cista/reflection/printable.h"

#include "utl/overloaded.h"

#include "geo/latlng.h"

#include "nigiri/database.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri {

using year_month_day = tuple<std::int32_t, std::uint32_t, std::uint32_t>;

struct tz_offsets {
  struct season {
    duration_t offset_{0};
    unixtime_t begin_{unixtime_t::min()}, end_{unixtime_t::max()};
    duration_t season_begin_mam_{0};
    duration_t season_end_mam_{0};
  };
  std::optional<season> season_{std::nullopt};
  duration_t offset_{0};
};

using timezone = variant<string, tz_offsets>;

struct attribute {
  CISTA_PRINTABLE(attribute)
  friend bool operator==(attribute const&, attribute const&) = default;
  string code_, text_;
};

struct provider {
  CISTA_COMPARABLE()
  string short_name_, long_name_;
};

using line_id_t = string;

using direction_t = cista::variant<location_idx_t, string>;

struct section_info {
  CISTA_COMPARABLE()
  db_index_t<attribute> attribute_idx_{0U};
  db_index_t<provider> provider_idx_{0U};
  db_index_t<direction_t> direction_idx_{0U};
  db_index_t<line_id_t> line_idx_{0U};

  std::uint32_t train_nr_{0U};
  std::uint32_t clasz_{0U};
};

using info_db = database<bitfield,
                         attribute,
                         provider,
                         line_id_t,
                         direction_t,
                         section_info>;

}  // namespace nigiri
