#include "nigiri/rt/trip_update.h"

#include "nigiri/loader/gtfs/noon_offsets.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

namespace nigiri {

using sched_transports_t =
    cista::offset::vecvec<trip_idx_t, transport_range_t>::bucket;
using rt_transports_t =
    cista::offset::vecvec<rt_trip_idx_t, rt_transport_range_t>::bucket;

template <typename Fn>
void resolve_schedule_transport(timetable const& tt,
                                trip_id const& id,
                                date::year_month_day const day,
                                Fn&& cb) {
  auto const id_matches = [&](trip_id_idx_t const t_id_idx) {
    return tt.trip_id_src_[t_id_idx] == id.src_ &&
           tt.trip_id_strings_[t_id_idx].view() == id.id_;
  };

  auto const lb = std::lower_bound(
      begin(tt.trip_id_to_idx_), end(tt.trip_id_to_idx_), id,
      [&](pair<trip_id_idx_t, trip_idx_t> const& a, trip_id const& b) {
        return std::tuple(tt.trip_id_src_[a.first],
                          tt.trip_id_strings_[a.first].view()) <
               std::tuple(b.src_, std::string_view{b.id_});
      });

  auto const get_utc_day_idx = [&](transport_idx_t const t) {
    auto const provider =
        tt.providers_[tt.transport_section_providers_[t].front()];
    auto const tz = tt.locations_.timezones_[provider.tz_];
    auto const noon_offset = loader::gtfs::get_noon_offset(
        date::local_days{day}, reinterpret_cast<date::time_zone const*>(
                                   tz.as<pair<string, void const*>>().second));
    auto const first_dep = tt.event_mam(t, 0U, event_type::kDep);
    auto const local_first_dep = first_dep + noon_offset;
    auto const day_offset = local_first_dep.count() / 1440U;
    auto const utc_day = date::sys_days{day} - day_offset * date::days{1};
    return tt.day_idx(utc_day);
  };

  // One trip can have several transports associated to it. Reasons:
  //  - local to UTC time conversion results in different time strings, the
  //    trip_id needs to map to all of them => only one can be active!
  //  - one transport can occur in several expanded trips due to in-seat
  //    transfers (all travel combinations are built) => several can be active!
  for (auto i = lb; i != end(tt.trip_id_to_idx_) && id_matches(i->first); ++i) {
    auto const ref = tt.trip_transport_ranges_[i->second].front().first;
    auto const ref_day_idx = get_utc_day_idx(ref);
    for (auto const [t, interval] : tt.trip_transport_ranges_[i->second]) {
      auto const first_dep_day_offset =
          tt.event_mam(t, interval.from_, event_type::kDep).count() / 1440;
      auto const t_day = ref_day_idx - first_dep_day_offset;
      auto const& traffic_days = tt.bitfields_[tt.transport_traffic_days_[t]];
      if (traffic_days.test(to_idx(t_day))) {
        cb(transport{t, t_day}, interval);
      }
    }
  }
}

struct trip {
  trip(timetable const& tt,
       rt_timetable&,
       trip_id const& id,
       date::year_month_day const day) {
    resolve_schedule_transport(
        tt, id, day,
        [&](transport const& t, interval<std::uint16_t> const stop_range) {
          schedule_transports_.emplace_back(t, stop_range);
        });
  }

  std::vector<std::pair<transport, interval<std::uint16_t>>>
      schedule_transports_;
  std::vector<std::pair<rt_transport_idx_t, interval<std::uint16_t>>>
      rt_transports_;
};

struct updater {
  updater(timetable const& tt, rt_timetable& rtt) : tt_{tt}, rtt_{rtt} {}

  void update(trip_update const& upd) {
    auto const trp = trip{tt_, rtt_, upd.id_, upd.day_};

    if (upd.is_cancel()) {
      cancel_trip(trp);
      return;
    }

    if (upd.is_additional_) {
      add_trip(upd.id_, upd.info_, true);
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
  void cancel_trip(trip const&) {
    CISTA_UNUSED_PARAM(tt_)
    CISTA_UNUSED_PARAM(rtt_)
  }
  void add_trip(trip_id const&,
                trip_info const&,
                bool const is_additional = false) {
    CISTA_UNUSED_PARAM(tt_)
    CISTA_UNUSED_PARAM(rtt_)
    CISTA_UNUSED_PARAM(is_additional)
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