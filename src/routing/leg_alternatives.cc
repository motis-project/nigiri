#include "nigiri/routing/leg_alternatives.h"

#include <algorithm>
#include <iterator>
#include <optional>
#include <ranges>

#include "utl/helpers/algorithm.h"

#include "nigiri/for_each_meta.h"
#include "nigiri/routing/direct.h"
#include "nigiri/routing/raptor/reconstruct.h"
#include "nigiri/routing/transfer_time_settings.h"

namespace nigiri::routing {

// MSVC's debug `std::views::take_while` iterator dereferences the
// underlying iterator when comparing against the sentinel even after
// the for-loop body has run — i.e. *after* the body has done `std::move(legs)`
#if defined(_MSC_VER) && defined(_DEBUG)
#define MOVE_IF_NOT_MSVC_DBG(x) (x)
#else
#define MOVE_IF_NOT_MSVC_DBG(x) std::move(x)
#endif

query make_alternative_query(timetable const&,
                             rt_timetable const*,
                             query const& q,
                             location_idx_t const from,
                             location_idx_t const to) {
  return query{
      .start_match_mode_ = location_match_mode::kExact,
      .dest_match_mode_ = location_match_mode::kExact,
      .use_start_footpaths_ = true,
      .start_ = {{from, 0_minutes, transport_mode_id_t{0}}},
      .destination_ = {{to, 0_minutes, transport_mode_id_t{0}}},
      .prf_idx_ = q.prf_idx_,
      .allowed_claszes_ = q.allowed_claszes_,
      .require_bike_transport_ = q.require_bike_transport_,
      .require_car_transport_ = q.require_car_transport_,
      .transfer_time_settings_ = q.transfer_time_settings_,
  };
}

std::vector<journey> get_leg_alternatives(
    timetable const& tt,
    rt_timetable const* rtt,
    query const& direct_query,
    direction const search_dir,
    unixtime_t const anchor_time,
    std::optional<unixtime_t> const max_arrival,
    journey::run_enter_exit const& original,
    std::size_t const max_alternatives) {
  auto const is_transit = [](journey::leg const& l) {
    return std::holds_alternative<journey::run_enter_exit>(l.uses_);
  };

  auto alternatives = std::vector<journey>{};
  alternatives.reserve(max_alternatives);

  auto const optimize = [&](std::vector<journey::leg> const& legs) {
    auto j_alt = journey{};
    j_alt.legs_ = legs;
    optimize_footpaths(tt, rtt, direct_query, j_alt);
    j_alt.start_time_ = j_alt.legs_.front().dep_time_;
    j_alt.dest_time_ = j_alt.legs_.back().arr_time_;
    j_alt.dest_ = j_alt.legs_.back().to_;
    return j_alt;
  };
  auto const not_original = [&](journey const& j_alt) {
    return utl::none_of(j_alt.legs_, [&](journey::leg const& l) {
      return is_transit(l) &&
             std::get<journey::run_enter_exit>(l.uses_) == original;
    });
  };

  if (search_dir == direction::kBackward) {
    // EARLIER ARRIVALS: search backward from the anchor time, taking the
    // latest alternatives.
    for (auto&& j_alt : get_direct_journeys<direction::kBackward>(
                            tt, rtt, direct_query, anchor_time) |
                            std::views::transform(optimize) |
                            std::views::filter(not_original) |
                            std::views::take(max_alternatives)) {
      alternatives.push_back(MOVE_IF_NOT_MSVC_DBG(j_alt));
    }
  } else {
    // LATER DEPARTURES: search forward from the anchor time, stopping once an
    // alternative arrives later than the (optional) upper bound.
    auto const fits_arrival = [&](std::vector<journey::leg> const& legs) {
      return !max_arrival.has_value() || legs.back().arr_time_ <= *max_arrival;
    };
    for (auto&& j_alt : get_direct_journeys<direction::kForward>(
                            tt, rtt, direct_query, anchor_time) |
                            std::views::take_while(fits_arrival) |
                            std::views::transform(optimize) |
                            std::views::filter(not_original) |
                            std::views::take(max_alternatives)) {
      alternatives.push_back(MOVE_IF_NOT_MSVC_DBG(j_alt));
    }
  }

  return alternatives;
}

std::vector<journey> get_leg_alternatives(timetable const& tt,
                                          rt_timetable const* rtt,
                                          query const& q,
                                          journey const& j,
                                          std::size_t const leg_idx,
                                          std::size_t const max_alternatives) {
  // Check if it's a transit leg.
  auto const is_transit = [](journey::leg const& l) {
    return std::holds_alternative<journey::run_enter_exit>(l.uses_);
  };
  if (max_alternatives == 0U || leg_idx >= j.legs_.size() ||
      !is_transit(j.legs_[leg_idx])) {
    return {};
  }

  // Original leg.
  auto const& original_leg = j.legs_[leg_idx];
  auto const& original_ree =
      std::get<journey::run_enter_exit>(original_leg.uses_);

  // Find previous and next transit (if any).
  auto const prev_it =
      std::find_if(std::reverse_iterator{begin(j.legs_) +
                                         static_cast<std::ptrdiff_t>(leg_idx)},
                   std::rend(j.legs_), is_transit);
  auto const next_it =
      std::find_if(begin(j.legs_) + static_cast<std::ptrdiff_t>(leg_idx) + 1,
                   end(j.legs_), is_transit);

  auto const has_prev = prev_it != std::rend(j.legs_);
  auto const has_next = next_it != end(j.legs_);

  // Create direct query.
  auto direct_query = make_alternative_query(
      tt, rtt, q, has_prev ? prev_it->to_ : location_idx_t::invalid(),
      has_next ? next_it->from_ : location_idx_t::invalid());
  if (!has_prev) {
    direct_query.start_match_mode_ = q.start_match_mode_;
    direct_query.use_start_footpaths_ = q.use_start_footpaths_;
    direct_query.start_ = q.start_;
    direct_query.td_start_ = q.td_start_;
  }
  if (!has_next) {
    direct_query.dest_match_mode_ = q.dest_match_mode_;
    direct_query.destination_ = q.destination_;
    direct_query.td_dest_ = q.td_dest_;
  }

  auto const next_dep =
      has_next ? next_it->dep_time_ : j.legs_.back().arr_time_;
  if (!has_prev && has_next) {
    // First transit leg with a successor:
    // search backward from the next leg's departure.
    return get_leg_alternatives(tt, rtt, direct_query, direction::kBackward,
                                next_dep, std::nullopt, original_ree,
                                max_alternatives);
  }

  // Intermediate / single / last transit leg: search forward from the previous
  // leg's arrival (or the journey start), bounded by the next departure when a
  // successor exists.
  auto const prev_arr =
      has_prev ? prev_it->arr_time_ : j.legs_.front().dep_time_;
  return get_leg_alternatives(tt, rtt, direct_query, direction::kForward,
                              prev_arr,
                              has_next ? std::optional{next_dep} : std::nullopt,
                              original_ree, max_alternatives);
}

}  // namespace nigiri::routing
