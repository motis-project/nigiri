#pragma once

#include "nigiri/algo/query.h"
#include "nigiri/algo/response.h"
#include "nigiri/timetable.h"

namespace nigiri::algo {

namespace detail {

template <typename Query>
journey_response algo_journey(Query const& q) {
  return {};
}

template <typename Query>
data_response algo_data(Query const& q) {
  return {};
}

}  // namespace detail

template <typename Query>
auto algo(Query const& q, timetable const& tt) {
  if constexpr (Query::response_type == ResponseType::Journey) {
    return detail::algo_journey(q);
  } else {
    return detail::algo_data(q);
  }
}

template <typename Query, typename Data>
auto algo(Query const& q, timetable const& tt, std::unique_ptr<Data> seed) {
  if constexpr (Query::response_type == ResponseType::Journey) {
    return detail::algo_journey(q);
  } else {
    return detail::algo_data(q);
  }
}

}  // namespace nigiri::algo