#include "nigiri/rt/trip_update.h"

#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

namespace nigiri {

struct trip {
  // TODO(felix)
};

struct updater {
  updater(timetable const& tt, rt_timetable& rtt) : tt_{tt}, rtt_{rtt} {}

  void update(trip_update const& upd) {
    auto const trp = resolve_trip(upd.id_);

    if (upd.is_cancel()) {
      cancel_trip(trp);
      return;
    }

    if (upd.is_rerouting_) {
      cancel_trip(trp);
      add_trip(upd.id_, upd.info_);
      return;
    }

    if (route_stays_sorted(trp, upd.info_)) {
      update_event_times(trp, upd.info_);
    } else {
      cancel_trip(trp);
      add_trip(upd.id_, upd.info_);
    }
  }

private:
  // TODO(felix)
  trip resolve_trip(trip_id const&) {
    CISTA_UNUSED_PARAM(tt_)
    CISTA_UNUSED_PARAM(rtt_) return {};
  }
  void cancel_trip(trip const&) {
    CISTA_UNUSED_PARAM(tt_)
    CISTA_UNUSED_PARAM(rtt_)
  }
  void add_trip(trip_id const&, trip_info const&) {
    CISTA_UNUSED_PARAM(tt_)
    CISTA_UNUSED_PARAM(rtt_)
  }
  void update_event_times(trip const&, trip_info const&) {
    CISTA_UNUSED_PARAM(tt_)
    CISTA_UNUSED_PARAM(rtt_)
  }
  bool route_stays_sorted(trip const&, trip_info const&) {
    CISTA_UNUSED_PARAM(tt_)
    CISTA_UNUSED_PARAM(rtt_)
    return true;
  }

  timetable const& tt_;
  rt_timetable& rtt_;
};

void update(timetable const& tt, rt_timetable& rtt, trip_update const& upd) {
  updater{tt, rtt}.update(upd);
}

}  // namespace nigiri