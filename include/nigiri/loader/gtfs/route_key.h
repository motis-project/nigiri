#pragma once

#include "nigiri/loader/gtfs/trip.h"

namespace nigiri::loader::gtfs {

struct route_key_t {
  clasz clasz_;
  stop_seq_t stop_seq_;
  bitvec bikes_allowed_;
  bitvec cars_allowed_;
};

struct route_key_ptr_t {
  clasz clasz_;
  stop_seq_t const* stop_seq_;
  bitvec const* bikes_allowed_{nullptr};
  bitvec const* cars_allowed_{nullptr};
};

struct route_key_hash {
  using is_transparent = void;

  static cista::hash_t hash(clasz const c,
                            stop_seq_t const& seq,
                            bitvec const* bikes_allowed,
                            bitvec const* cars_allowed) {
    auto h = cista::BASE_HASH;
    h = cista::hash_combine(h, cista::hashing<stop_seq_t>{}(seq));
    h = cista::hash_combine(h, c);
    if (bikes_allowed != nullptr && !bikes_allowed->empty()) {
      h = cista::hash_combine(h, cista::hashing<bitvec>{}(*bikes_allowed));
    }
    if (cars_allowed != nullptr && !cars_allowed->empty()) {
      h = cista::hash_combine(h, cista::hashing<bitvec>{}(*cars_allowed));
    }
    return h;
  }

  cista::hash_t operator()(route_key_t const& x) const {
    return hash(x.clasz_, x.stop_seq_, &x.bikes_allowed_, &x.cars_allowed_);
  }

  cista::hash_t operator()(route_key_ptr_t const& x) const {
    return hash(x.clasz_, *x.stop_seq_, x.bikes_allowed_, x.cars_allowed_);
  }
};

struct route_key_equals {
  using is_transparent = void;

  cista::hash_t operator()(route_key_t const& a, route_key_t const& b) const {
    return std::tie(a.clasz_, a.stop_seq_, a.bikes_allowed_, a.cars_allowed_) ==
           std::tie(b.clasz_, b.stop_seq_, b.bikes_allowed_, b.cars_allowed_);
  }

  cista::hash_t operator()(route_key_ptr_t const& a,
                           route_key_t const& b) const {
    assert((a.bikes_allowed_ == nullptr) == (a.cars_allowed_ == nullptr));
    if ((a.bikes_allowed_ == nullptr || a.bikes_allowed_->empty()) !=
            b.bikes_allowed_.empty() ||
        (a.cars_allowed_ == nullptr || a.bikes_allowed_->empty()) !=
            b.cars_allowed_.empty()) {
      return false;
    }
    if (a.bikes_allowed_ == nullptr) {
      return std::tie(a.clasz_, *a.stop_seq_) ==
             std::tie(b.clasz_, b.stop_seq_);
    }
    return std::tie(a.clasz_, *a.stop_seq_, *a.bikes_allowed_,
                    *a.cars_allowed_) ==
           std::tie(b.clasz_, b.stop_seq_, b.bikes_allowed_, b.cars_allowed_);
  }
};

}  // namespace nigiri::loader::gtfs
