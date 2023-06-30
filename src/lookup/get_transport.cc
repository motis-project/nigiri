#include "nigiri/lookup/get_transport.h"

#include "nigiri/loader/gtfs/noon_offsets.h"

#include "nigiri/common/day_list.h"
#include "nigiri/timetable.h"

namespace nigiri {

template <typename Fn>
void for_each_schedule_transport(timetable const& tt,
                                 trip_id const& id,
                                 date::year_month_day const day,
                                 bool const gtfs_local_day,
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

  // One trip can have several transports associated to it. Reasons:
  //  - local to UTC time conversion results in different time strings, the
  //    trip_id needs to map to all of them => only one can be active!
  //  - one transport can occur in several expanded trips due to in-seat
  //    transfers (all travel combinations are built) => several can be active!
  for (auto i = lb; i != end(tt.trip_id_to_idx_) && id_matches(i->first); ++i) {
    for (auto const [t, interval] : tt.trip_transport_ranges_[i->second]) {
      auto tz_offset = 0_minutes;
      if (gtfs_local_day) {
        auto const provider =
            tt.providers_[tt.transport_section_providers_[t].front()];
        auto const tz = tt.locations_.timezones_[provider.tz_];
        tz_offset = loader::gtfs::get_noon_offset(
            date::local_days{day},
            reinterpret_cast<date::time_zone const*>(
                tz.template as<pair<string, void const*>>().second));
      }
      auto const first_dep = tt.event_mam(t, interval.from_, event_type::kDep);
      auto const day_offset =
          (first_dep.as_duration() + tz_offset).count() / 1440;
      auto const t_day =
          tt.day_idx(date::sys_days{day} - day_offset * date::days{1});
      auto const& traffic_days = tt.bitfields_[tt.transport_traffic_days_[t]];

      std::cout << tt.trip_id_strings_[i->first].view()
                << "first_dep=" << first_dep << ", day_offset=" << day_offset
                << ", day=" << tt.to_unixtime(t_day, 0_minutes)
                << ", active=" << traffic_days.test(to_idx(t_day))
                << ", traffic_days="
                << day_list{traffic_days, tt.internal_interval_days().from_}
                << "\n";
      if (traffic_days.test(to_idx(t_day))) {
        cb(transport{t, t_day}, interval);
      }
    }
  }
}

std::optional<std::pair<transport, interval<std::uint16_t>>> get_ref_transport(
    timetable const& tt,
    trip_id const& id,
    date::year_month_day const day,
    bool const gtfs_local_day) {
  std::optional<std::pair<transport, interval<std::uint16_t>>> t;
  for_each_schedule_transport(
      tt, id, day, gtfs_local_day,
      [&](transport const x, interval<std::uint16_t> const interval) {
        if (!t.has_value() || interval.size() > t->second.size()) {
          t = {x, interval};
        }
      });
  return t;
}

}  // namespace nigiri