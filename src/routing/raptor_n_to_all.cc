#include "nigiri/routing/raptor_n_to_all.h"

#include <array>
#include <variant>
#include <vector>

#include "nigiri/common/delta_t.h"
#include "nigiri/routing/pareto_set.h"     // for the dummy results pareto_set
#include "nigiri/routing/journey.h"        // for dummy pareto_set type
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/routing/transfer_time_settings.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

namespace {

constexpr auto const kVias = via_offset_t{0U};


day_idx_t make_base(timetable const& tt, unixtime_t const start_time) {
  return day_idx_t{
      static_cast<day_idx_t::value_t>(
          std::chrono::duration_cast<date::days>(
              std::chrono::round<std::chrono::days>(start_time) -
              tt.internal_interval().from_)
              .count())};
}


template <bool Rt>
raptor_state run_raptor_n_to_all(timetable const& tt,
                                rt_timetable const* rtt,
                                std::vector<location_idx_t> const& origins,
                                unixtime_t const start_time,
                                std::uint8_t const max_transfers,
                                duration_t const max_travel_time) {
  auto state = raptor_state{};
  auto is_dest    = bitvec{tt.n_locations()};
  auto is_via     = std::array<bitvec, kMaxVias>{};
  auto dist_to_dest = std::vector<std::uint16_t>{};
  auto td_dest    = hash_map<location_idx_t, std::vector<td_offset>>{};
  auto via_stops  = std::vector<via_stop>{};
  auto lb = std::vector<std::uint16_t>(tt.n_locations(), 0U);

  auto const base        = make_base(tt, start_time);
  using algo_t = raptor<direction::kForward, Rt, kVias, search_mode::kOneToAll>;
  auto algo = algo_t{tt,
                     rtt,
                     state,
                     is_dest,
                     is_via,
                     dist_to_dest,
                     td_dest,
                     lb,
                     via_stops,
                     base,
                     all_clasz_allowed(),
                     false,
                     false,
                     false,
                     transfer_time_settings{}};

  algo.next_start_time();
  for (auto const& origin : origins) {
    algo.add_start(origin, start_time, origin);
  }
  auto const worst_time =
      unixtime_t{start_time + max_travel_time + duration_t{1}};


  auto dummy = pareto_set<journey>{};

  algo.execute(start_time, max_transfers, worst_time,
               profile_idx_t{0U}, dummy);

  return state;
}

std::vector<n_to_all_cell> read_results(timetable const& tt,
                                       raptor_state const& state,
                                       unixtime_t const start_time,
                                       std::uint8_t const max_transfers) {
  auto const n_locations  = tt.n_locations();
  auto const& round_times = state.get_round_times<kVias>();
  auto const& owner_tab   = state.get_owner<kVias>();
  auto const invalid      = kInvalidDelta<direction::kForward>;
  auto const base = tt.internal_interval_days().from_ +
                    static_cast<int>(
                        to_idx(day_idx_t{
                            static_cast<day_idx_t::value_t>(
                                std::chrono::duration_cast<date::days>(
                                    std::chrono::round<std::chrono::days>(start_time) -
                                    tt.internal_interval().from_)
                                    .count())})) *
                        date::days{1};

  auto const end_k =
      static_cast<unsigned>(std::min(max_transfers, kMaxTransfers)) + 2U;

  auto cells = std::vector<n_to_all_cell>(n_locations);
  for (auto l = std::uint32_t{0U}; l != n_locations; ++l) {
    for (auto k = 0U; k != end_k; ++k) {
      auto const t = round_times[k][l][kVias];
      if (t == invalid) {
        continue;
      }
      auto const arr_time   = delta_to_unix(base, t);
      auto const travel_min = static_cast<duration_t::rep>(
          (arr_time - start_time).count());
      cells[l] = n_to_all_cell{
          .owner_       = owner_tab[l][kVias],
          .travel_time_ = duration_t{travel_min},
          .transfers_   = static_cast<std::uint8_t>(k == 0U ? 0U : k - 1U),
          .reached_     = true,
      };
      break;
    }
  }
  return cells;
}

}  // namespace

n_to_all_result raptor_n_to_all(timetable const& tt,
                              rt_timetable const* rtt,
                              std::vector<location_idx_t> const& origins,
                              unixtime_t const start_time,
                              std::uint8_t const max_transfers,
                              duration_t const max_travel_time) {
  auto state = (rtt != nullptr)
      ? run_raptor_n_to_all<true>(tt, rtt, origins, start_time,
                                 max_transfers, max_travel_time)
      : run_raptor_n_to_all<false>(tt, nullptr, origins, start_time,
                                  max_transfers, max_travel_time);
  auto cells = read_results(tt, state, start_time, max_transfers);
  return n_to_all_result{
      .cells_ = std::move(cells),
      .state_ = std::move(state),
  };
}

}  // namespace nigiri::routing
