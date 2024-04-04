#include "nigiri/query_generator/nigiri_generator.h"
#include "nigiri/routing/ontrip_train.h"

namespace nigiri::query_generation {

nigiri_generator::nigiri_generator(timetable const& tt,
                                   generator_settings const& settings)
    : generator(tt, settings), qf_{s_} {
  if (s_.start_match_mode_ == routing::location_match_mode::kIntermodal ||
      s_.dest_match_mode_ == routing::location_match_mode::kIntermodal) {
    locations_rtree_ = geo::make_point_rtree(tt_.locations_.coordinates_);
  }
}

void nigiri_generator::add_offsets_for_pos(
    std::vector<routing::offset>& o,
    geo::latlng const& pos,
    query_generation::transport_mode const& mode) {
  auto const locs_in_range = locations_rtree_.in_radius(pos, mode.range());
  for (auto const loc : locs_in_range) {
    auto const duration = duration_t{
        static_cast<std::int16_t>(
            geo::distance(pos,
                          tt_.locations_.coordinates_[location_idx_t{loc}]) /
            (mode.speed_ * 60)) +
        1};
    o.emplace_back(location_idx_t{loc}, duration, mode.mode_id_);
  }
}

std::optional<routing::query> nigiri_generator::random_pretrip_query() {

  for (auto i = 0U; i < max_gen_attempts_; ++i) {
    // start with a new query
    auto q = qf_.make_query();

    // randomize time
    auto const start_time = random_time();
    auto const start_interval =
        interval<unixtime_t>{start_time, start_time + s_.interval_size_};
    auto const dest_interval =
        interval<unixtime_t>{start_time, start_time + duration_t{1440U}};
    if (s_.interval_size_.count() == 0) {
      q.start_time_ = start_time;
    } else {
      q.start_time_ = start_interval;
    }

    // randomize start location
    auto const start_loc_idx =
        random_active_location(start_interval, event_type::kDep);
    if (!start_loc_idx.has_value()) {
      continue;
    }

    // randomize destination location
    auto const dest_loc_idx =
        random_active_location(dest_interval, event_type::kArr);
    if (!dest_loc_idx.has_value()) {
      continue;
    }

    if (start_loc_idx.value() == dest_loc_idx.value()) {
      continue;
    }

    // add start(s) to query
    if (s_.start_match_mode_ == routing::location_match_mode::kIntermodal) {
      add_offsets_for_pos(q.start_, pos_near_start(start_loc_idx.value()),
                          s_.start_mode_);
    } else {
      q.start_.emplace_back(start_loc_idx.value(), 0_minutes, 0U);
    }

    // add destination(s) to query
    if (s_.dest_match_mode_ == routing::location_match_mode::kIntermodal) {
      add_offsets_for_pos(q.destination_, pos_near_dest(dest_loc_idx.value()),
                          s_.dest_mode_);
    } else {
      q.destination_.emplace_back(dest_loc_idx.value(), 0_minutes, 0U);
    }

    return q;
  }

  std::cout << "WARNING: failed to generate a valid query after "
            << max_gen_attempts_ << " attempts\n";

  return std::nullopt;
}

std::optional<routing::query> nigiri_generator::random_ontrip_query() {

  for (auto i = 0U; i < max_gen_attempts_; ++i) {
    // start with a new query
    auto q = qf_.make_query();

    // randomize transport and stop index
    auto const [tr, stop_idx] = random_transport_active_stop(event_type::kArr);

    // generate ontrip train query for transport and stop index
    routing::generate_ontrip_train_query(tt_, tr, stop_idx, q);

    // set interval at destination
    auto const dest_interval = interval<unixtime_t>{
        std::get<unixtime_t>(q.start_time_),
        std::get<unixtime_t>(q.start_time_) + duration_t{1440U}};

    // randomize destination location
    auto const dest_loc_idx =
        random_active_location(dest_interval, event_type::kArr);
    if (!dest_loc_idx.has_value()) {
      continue;
    }

    auto const start_loc_idx =
        stop{tt_.route_location_seq_[tt_.transport_route_[tr.t_idx_]][stop_idx]}
            .location_idx();
    if (start_loc_idx == dest_loc_idx.value()) {
      continue;
    }

    // add destination(s) to query
    if (s_.dest_match_mode_ == routing::location_match_mode::kIntermodal) {
      add_offsets_for_pos(q.destination_, pos_near_dest(dest_loc_idx.value()),
                          s_.dest_mode_);
    } else {
      q.destination_.emplace_back(dest_loc_idx.value(), 0_minutes, 0U);
    }

    return q;
  }

  std::cout << "WARNING: failed to generate a valid query after "
            << max_gen_attempts_ << " attempts\n";

  return std::nullopt;
}
}  // namespace nigiri::query_generation