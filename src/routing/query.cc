#include "nigiri/routing/query.h"

#include "utl/verify.h"

#include "nigiri/for_each_meta.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

bool query::operator==(query const& o) const {
  auto const eq = [](auto&& a, auto&& b) {
    using T = std::decay_t<decltype(a)>;
    if constexpr (std::is_floating_point_v<T>) {
      return std::abs(a - b) < 0.0001;
    } else {
      return a == b;
    }
  };
  return [&]<std::size_t... I>(std::index_sequence<I...>) {
    return (eq(std::get<I>(cista::to_tuple(o)),
               std::get<I>(cista::to_tuple(*this))) &&
            ...);
  }(std::make_index_sequence<
             std::tuple_size_v<decltype(cista::to_tuple(*this))>>());
}

void blocked_routes::verify_rtt(rt_timetable const* rtt) const {
  if (!any()) {
    return;
  }
  auto const n =
      rtt == nullptr ? 0U : static_cast<std::size_t>(rtt->n_rt_transports());
  utl::verify(rt_transports_.size() == n, "rt_transports_.size() != n");
}

blocked_routes make_blocked_routes(timetable const& tt,
                                   rt_timetable const* rtt,
                                   bitvec_map<source_idx_t> blocked_srcs) {
  auto f = blocked_routes{};
  if (!blocked_srcs.any()) {
    return f;
  }

  f.srcs_ = std::move(blocked_srcs);
  f.routes_.resize(tt.n_routes());
  for (auto r = route_idx_t{0U}; r != tt.n_routes(); ++r) {
    if (f.srcs_.test(tt.route_src_[r])) {
      f.routes_.set(r);
    }
  }

  auto const n_rt = rtt == nullptr ? 0U : rtt->n_rt_transports();
  f.rt_transports_.resize(n_rt);
  for (auto i = 0U; i != n_rt; ++i) {
    auto const rt_t = rt_transport_idx_t{i};
    if (f.srcs_.test(rtt->rt_transport_src_[rt_t])) {
      f.rt_transports_.set(rt_t);
    }
  }

  return f;
}

void sanitize_query(query& q) {
  if (q.max_travel_time_.count() < 0 || q.max_travel_time_ > kMaxTravelTime) {
    q.max_travel_time_ = kMaxTravelTime;
  }
}

void sanitize_via_stops(timetable const& tt, query& q) {
  while (q.via_stops_.size() >= 2) {
    auto updated = false;
    for (auto i = 0U; i < q.via_stops_.size() - 1; ++i) {
      auto& a = q.via_stops_[i];
      auto& b = q.via_stops_[i + 1];
      if (matches(tt, location_match_mode::kEquivalent, a.location_,
                  b.location_)) {
        a.stay_ += b.stay_;
        q.via_stops_.erase(q.via_stops_.begin() + i + 1);
        updated = true;
        break;
      }
    }
    if (!updated) {
      break;
    }
  }
}

void query::flip_dir() {
  std::swap(start_, destination_);
  std::swap(td_start_, td_dest_);
  std::swap(start_match_mode_, dest_match_mode_);
  std::reverse(begin(via_stops_), end(via_stops_));
}

void query::sanitize(timetable const& tt) {
  utl::verify(start_match_mode_ != location_match_mode::kIntermodal ||
                  !use_start_footpaths_,
              "intermodal start is incompatible with use_start_footpaths");
  sanitize_query(*this);
  sanitize_via_stops(tt, *this);
  utl::sort(start_);
  utl::sort(destination_);
}

}  // namespace nigiri::routing