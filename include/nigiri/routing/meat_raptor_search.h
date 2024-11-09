#pragma once

#include "nigiri/routing/meat/raptor/meat_raptor.h"
#include "nigiri/routing/search.h"
#include "nigiri/routing/start_times.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

namespace m = meat;
namespace mraptor = m::raptor;

struct mraptor_result {
  m::decision_graph g_;
  mraptor::meat_raptor_stats stats_;
};

mraptor_result meat_raptor_search(timetable const& tt,
                               mraptor::meat_raptor_state& m_state,
                               query q) {
  auto g = m::decision_graph{};

  auto start_time = std::visit(
      utl::overloaded{[](interval<unixtime_t> const start_interval) {
                        assert(false &&
                               "search interval (pretrip) not yet supported");
                        return *start_interval.begin();
                      },
                      [](unixtime_t const start_time) { return start_time; }},
      q.start_time_);

  auto meat_raptor = mraptor::meat_raptor{
      tt, m_state,
      day_idx_t{std::chrono::duration_cast<date::days>(
                    std::chrono::round<std::chrono::days>(start_time) -
                    tt.internal_interval().from_)
                    .count()},
      q.allowed_claszes_};

  auto add_ontrip = true;
  auto starts = std::vector<start>{};
  get_starts(direction::kForward, tt, nullptr, q.start_time_, q.start_,
             q.td_start_, q.max_start_offset_, q.start_match_mode_,
             q.use_start_footpaths_, starts, add_ontrip, q.prf_idx_,
             q.transfer_time_settings_);

  location_idx_t start_location;
  for (auto const& s : starts) {
    assert(s.time_at_start_ == start_time);
    assert(s.time_at_stop_ == start_time);
    // meat_raptor.add_start(s.stop_, s.time_at_stop_);
    start_location = s.stop_;
    break;
  }

  auto is_destination = bitvec{};
  auto dist_to_dest = std::vector<std::uint16_t>{};
  collect_destinations(tt, q.destination_, q.dest_match_mode_, is_destination,
                       dist_to_dest);

  auto end_location = location_idx_t{is_destination.next_set_bit(0).value()};
  meat_raptor.execute(start_time, start_location, end_location, q.prf_idx_, g);

  return {std::move(g), meat_raptor.get_stats()};
}

}  // namespace nigiri::routing