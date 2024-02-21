#include "nigiri/rust/timetable.h"

namespace nigiri::rust {

struct Journey;
struct Query;

::rust::Vec<Journey> route(Timetable const&, Query const&);

}  // namespace nigiri::rust
