#pragma once

#include <cstdint>
#include <algorithm>
#include <vector>

#include "cista/reflection/comparable.h"

#include "utl/get_or_create.h"

#include "nigiri/loader/transfer_rules.h"
#include "nigiri/loader/virt/location_route_stops.h"
#include "nigiri/loader/virt/signatures.h"
#include "nigiri/loader/virt/transfer_durations.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}  // namespace nigiri

namespace nigiri::loader {

using route_stop_pair_idx_t =
    cista::strong<std::uint32_t, struct route_stop_pair_>;

constexpr auto const kBaseTransport = route_transport_idx_t{0U};

struct transfer {
  route_transport_idx_t from_;
  route_transport_idx_t to_;
  duration_t gap_;
  duration_t expected_;
};

// Types:
//
// - regular: r != invalid, sig = invalid
//   => used for (route, stop) with inhomogeneous signatures
//
// - base: r = invalid, sig = kEmptySig, stop_idx=0
//   => used to represent the base stop (unqualified / fallback transfers)
//
// - signature: r = invalid, sig != kEmptySig, sig != invalid, stop_idx=0
//   => shared between all uniform route stops at this location with this
//      signature (across routes and visits)
struct route_stop {
  CISTA_COMPARABLE()

  bool is_base() const { return sig_ == kEmptySig; }
  bool has_route() const { return route_ != route_idx_t::invalid(); }

  route_idx_t route_;
  stop_idx_t stop_idx_;
  location_idx_t stop_;
  signature_idx_t sig_{signature_idx_t::invalid()};
};

inline void narrow(window_t& w, duration_t const gap, bool const must_work) {
  if (must_work) {
    w.to_ = std::min(w.to_, gap + duration_t{1});
  } else {
    w.from_ = std::max(w.from_, gap + duration_t{1});
  }
}

inline route_stop to_base_stop(location_idx_t const l) {
  return {route_idx_t::invalid(), stop_idx_t{0U}, l, kEmptySig};
}

struct route_stop_pairs {
  using route_stop_pair_t = pair<route_stop_idx_t, route_stop_idx_t>;

  bool is_base(route_stop_idx_t const s) const {
    return route_stops_[s].is_base();
  }

  route_stop_idx_t get_or_create_route_stop(route_stop const& s) {
    return utl::get_or_create(route_stop_of_, s, [&]() {
      if (!s.is_base()) {
        get_or_create_route_stop(to_base_stop(s.stop_));
      }
      route_stops_.push_back(s);
      return route_stop_idx_t{route_stops_.size() - 1U};
    });
  }

  route_stop_idx_t get_or_create_route_stop(location_route_stop& s,
                                            location_idx_t const l) {
    if (s.route_stop_ == route_stop_idx_t::invalid()) {
      auto const rs = route_stop{s.route_, s.stop_idx_, l};
      s.route_stop_ = get_or_create_route_stop(
          s.is_uniform() ? route_stop{route_idx_t::invalid(), stop_idx_t{0U}, l,
                                      signatures::get(s.sigs_, kBaseTransport)}
                         : rs);
      route_stop_of_.emplace(rs, s.route_stop_);
    }
    return s.route_stop_;
  }

  route_stop_pair_idx_t get_or_create_pair(route_stop_idx_t const from,
                                           route_stop_idx_t const to,
                                           window_t const& hull) {
    return utl::get_or_create(pair_of_, route_stop_pair_t{from, to}, [&]() {
      auto const p = route_stop_pair_idx_t{pairs_.size()};
      pairs_.push_back(route_stop_pair_t{from, to});
      transfers_.emplace_back();
      windows_.push_back(hull);
      auto const from_route = route_stops_[from].route_;
      auto const to_route = route_stops_[to].route_;
      if (from_route != route_idx_t::invalid()) {
        of_route_[from_route].push_back(p);
      }
      if (to_route != route_idx_t::invalid() && to_route != from_route) {
        of_route_[to_route].push_back(p);
      }
      return p;
    });
  }

  void add_transfer(route_stop_pair_idx_t const p, transfer const& t) {
    auto const [from, to] = pairs_[p];
    if (route_stops_[from].has_route() || route_stops_[to].has_route()) {
      transfers_[p].push_back(t);
    }
    narrow(windows_[p], t.gap_, t.gap_ >= t.expected_);
  }

  vector_map<route_stop_idx_t, route_stop> route_stops_;
  hash_map<route_stop, route_stop_idx_t> route_stop_of_;
  vector_map<route_stop_pair_idx_t, route_stop_pair_t> pairs_;
  hash_map<route_stop_pair_t, route_stop_pair_idx_t> pair_of_;
  vector_map<route_stop_pair_idx_t, std::vector<transfer>> transfers_;
  vector_map<route_stop_pair_idx_t, window_t> windows_;
  vector_map<route_idx_t, std::vector<route_stop_pair_idx_t>> of_route_;
};

route_stop_pairs get_route_stop_pairs(
    timetable const&,
    signatures const&,
    hash_map<transfer_pair, transfer_durations> const& durations_of);

}  // namespace nigiri::loader
