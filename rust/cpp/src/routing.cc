#include "nigiri/rust/routing.h"

#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/routing/search.h"

#include "nigiri-cxx/main.h"
#include "utl/to_vec.h"

namespace nr = nigiri::routing;

namespace nigiri::rust {

unixtime_t to_unixtime(std::int64_t const t) {
  return unixtime_t{
      std::chrono::duration_cast<i32_minutes>(std::chrono::seconds{t})};
}

std::int64_t to_i64(unixtime_t const t) {
  return std::chrono::duration_cast<std::chrono::seconds>(t.time_since_epoch())
      .count();
}

template <typename To, typename From>
To enum_cast(From const x) {
  return static_cast<To>(std::underlying_type_t<From>(x));
}

::rust::Vec<Leg> convert_legs(nr::journey const& j) {
  auto legs = ::rust::Vec<Leg>{};
  for (auto const& l : j.legs_) {
    auto const type =
        std::holds_alternative<nr::journey::run_enter_exit>(l.uses_)
            ? LegType::RunEnterExit
        : std::holds_alternative<footpath>(l.uses_) ? LegType::Footpath
                                                    : LegType::Offset;
    legs.emplace_back(Leg{.from = to_idx(l.from_),
                          .to = to_idx(l.to_),
                          .dep_time = to_i64(l.dep_time_),
                          .arr_time = to_i64(l.arr_time_),
                          .t = type,
                          .offset = type == LegType::Offset
                                        ? std::get<nr::offset>(l.uses_)
                                        : 0U});
  }
  return legs;
}

::rust::Vec<Journey> route(Timetable const& tt, Query const& q) {
  auto const to_offset = [](Offset const& x) {
    return nr::offset(location_idx_t{x.target}, duration_t{x.duration}, x.idx);
  };

  auto const is_interval =
      q.end_time == std::numeric_limits<decltype(q.end_time)>::min();
  auto const start_time =
      is_interval ? nr::start_time_t{interval{to_unixtime(q.start_time),
                                              to_unixtime(q.end_time)}}
                  : nr::start_time_t{to_unixtime(q.start_time)};

  auto const n_q = nr::query{
      .start_time_ = start_time,
      .start_match_mode_ =
          enum_cast<nr::location_match_mode>(q.start_match_mode),
      .dest_match_mode_ = enum_cast<nr::location_match_mode>(q.dest_match_mode),
      .use_start_footpaths_ = q.use_start_footpaths,
      .start_ = utl::to_vec(q.start, to_offset),
      .destination_ = utl::to_vec(q.dest, to_offset),
      .min_connection_count_ = q.min_connection_count,
      .extend_interval_earlier_ = q.extend_interval_earlier,
      .extend_interval_later_ = q.extend_interval_later,
      .prf_idx_ = q.profile_idx,
  };

  using algo_state_t = routing::raptor_state;
  static auto search_state = routing::search_state{};
  static auto algo_state = algo_state_t{};

  auto results = pareto_set<routing::journey>{};
  if (q.search_direction == Direction::Forward) {
    using algo_t = routing::raptor<direction::kForward, false>;
    results = *(nr::search<direction::kForward, algo_t>{
        *tt, nullptr, search_state, algo_state, std::move(n_q)}
                    .execute()
                    .journeys_);
  } else {
    using algo_t = routing::raptor<direction::kBackward, false>;
    results = *(nr::search<direction::kForward, algo_t>{
        *tt, nullptr, search_state, algo_state, std::move(n_q)}
                    .execute()
                    .journeys_);
  }

  auto journeys = ::rust::Vec<Journey>();
  for (auto const& r : results) {
    journeys.emplace_back(Journey{.legs = convert_legs(r)});
  }
  return journeys;
}

}  // namespace nigiri::rust
