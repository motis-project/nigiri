#include "nigiri/routing/query.h"

#include "utl/verify.h"

#include "nigiri/for_each_meta.h"
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

namespace {
void set_range(bitvec_map<route_idx_t>& b, interval<route_idx_t> const r) {
  using block_t = std::decay_t<decltype(b)>::block_t;
  constexpr auto const kBits = std::decay_t<decltype(b)>::bits_per_block;
  constexpr auto const kOnes = ~block_t{0U};

  auto const from = static_cast<std::size_t>(to_idx(r.from_));
  auto const to = static_cast<std::size_t>(to_idx(r.to_));
  if (from >= to) {
    return;
  }

  auto const first = from / kBits;
  auto const last = (to - 1U) / kBits;
  auto const head = kOnes << (from % kBits);
  auto const tail = (to % kBits) == 0U ? kOnes : ~(kOnes << (to % kBits));

  if (first == last) {
    b.blocks_[first] |= (head & tail);
    return;
  }
  b.blocks_[first] |= head;
  for (auto i = first + 1U; i != last; ++i) {
    b.blocks_[i] = kOnes;
  }
  b.blocks_[last] |= tail;
}

}  // namespace

blocked_feeds make_blocked_feeds(timetable const& tt,
                                 bitvec_map<source_idx_t> blocked_srcs) {
  auto f = blocked_feeds{};
  if (!blocked_srcs.any()) {
    return f;
  }

  f.srcs_ = std::move(blocked_srcs);
  f.routes_.resize(tt.n_routes());
  for (auto src = source_idx_t{0U}; src != tt.src_routes_.size(); ++src) {
    if (f.srcs_.test(src)) {
      set_range(f.routes_, tt.src_routes_[src]);
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