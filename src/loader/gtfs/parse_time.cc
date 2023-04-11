#include "nigiri/loader/gtfs/parse_time.h"

#include "utl/parser/arg_parser.h"

#include "nigiri/types.h"

namespace nigiri::loader::gtfs {

duration_t hhmm_to_min(utl::cstr s) {
  if (s.empty()) {
    return kInterpolate;
  } else {
    int hours = 0;
    parse_arg(s, hours, 0);
    if (s) {
      ++s;
    } else {
      return kInterpolate;
    }

    int minutes = 0;
    parse_arg(s, minutes, 0);

    return duration_t{hours * 60 + minutes};
  }
}

}  // namespace nigiri::loader::gtfs
