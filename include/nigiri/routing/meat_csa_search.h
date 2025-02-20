#pragma once

#include "nigiri/routing/meat/csa/meat_csa.h"
#include "nigiri/routing/search.h"
#include "nigiri/routing/start_times.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

namespace m = meat;
namespace mcsa = m::csa;

struct mcsa_result {
  m::decision_graph g_;
  mcsa::meat_csa_stats stats_;
};

template <typename ProfileSet>
mcsa_result meat_csa_search(timetable const& tt,
                            mcsa::meat_csa_state<ProfileSet>& m_state,
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

  auto meat_csa = mcsa::meat_csa<ProfileSet>{
      tt,
      m_state,
      day_idx_t{std::chrono::duration_cast<date::days>(
                    std::chrono::round<std::chrono::days>(start_time) -
                    tt.internal_interval().from_)
                    .count()},
      q.allowed_claszes_,
      q.max_delay_,
      q.bound_parameter_};

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
    // meat_csa.add_start(s.stop_, s.time_at_stop_);
    start_location = s.stop_;
    break;
  }

  auto is_destination = bitvec{};
  auto dist_to_dest = std::vector<std::uint16_t>{};
  collect_destinations(tt, q.destination_, q.dest_match_mode_, is_destination,
                       dist_to_dest);

  auto end_location = location_idx_t{is_destination.next_set_bit(0).value()};
  meat_csa.execute(start_time, start_location, end_location, q.prf_idx_, g);

  return {std::move(g), meat_csa.get_stats()};
}

mcsa_result meat_csa_search(
    timetable const& tt,
    mcsa::meat_csa_state<mcsa::static_profile_set>& m_state,
    query q) {
  return meat_csa_search<mcsa::static_profile_set>(tt, m_state, std::move(q));
}

mcsa_result meat_csa_search(
    timetable const& tt,
    mcsa::meat_csa_state<mcsa::dynamic_growth_profile_set>& m_state,
    query q) {
  return meat_csa_search<mcsa::dynamic_growth_profile_set>(tt, m_state,
                                                           std::move(q));
}

mcsa_result meat_csa_search(
    timetable const& tt,
    mcsa::meat_csa_state<mcsa::dynamic_profile_set>& m_state,
    query q) {
  return meat_csa_search<mcsa::dynamic_profile_set>(tt, m_state, std::move(q));
}

}  // namespace nigiri::routing