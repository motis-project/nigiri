#pragma once

#include <cstdint>
#include <bit>
#include <type_traits>

#include "cista/cuda_check.h"

#include "nigiri/common/delta_t.h"
#include "nigiri/routing/clasz_mask.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/mc_context.h"
#include "nigiri/types.h"

// McRAPTOR's criteria besides the arrival and its breadcrumbs, shared by the
// CPU engine (raptor/mcraptor.h) and the GPU one (gpu/mcraptor_impl.cuh).
//
// A criterion owns fields of mc_fields, the label's 16 bits. Derived from
// mc_dim (no-op updates, kRide false), it defines what applies of
//   kRide                    whether a ride changes its fields
//   at_start(f, minutes)     a start label (minutes walked to it)
//   after_ride(f, c, ctx)    after riding a trip of class c
//   after_walk(f, minutes)   after a walk (offset, footpath)
//   complete(f)              drops what only counts while the journey goes on
//   dominates(a, b)          a is no worse than b; on complete() fields, also
//                            once nothing follows
//   join(prefix, suffix, out)  out: the complete fields of both halves
//   to_journey(f, j), of_journey(j, f)  the journey's field (complete fields)
// A new criterion also needs its fields here, aliases (raptor/mcraptor.h,
// gpu/mcraptor.h) and instantiations (raptor/mcraptor.cc, gpu/mcraptor.cu).
namespace nigiri::routing {

// Every criterion's fields. A configuration's criteria set theirs, the others
// stay 0. No padding: the engines store and compare the 16 bits as an integer.
struct mc_fields {
  std::uint16_t non_transit_ : 7 {0U};  // non_transit_dim
  std::uint16_t avoided_modes_ : 1 {0U};  // avoided_modes_dim
  std::uint16_t group_ : 3 {0U};  // mode_switches_dim: group of the last trip
  std::uint16_t switches_ : 4 {0U};  // mode_switches_dim
  std::uint16_t spare_ : 1 {0U};
};
static_assert(sizeof(mc_fields) == 2U &&
                  std::has_unique_object_representations_v<mc_fields>,
              "the label's 16 bits, all of them fields");

CISTA_CUDA_COMPAT constexpr std::uint16_t saturate(std::uint32_t const v,
                                                   std::uint32_t const max) {
  return static_cast<std::uint16_t>(v < max ? v : max);
}

struct mc_dim {
  static constexpr bool kRide = false;
  CISTA_CUDA_COMPAT static void at_start(mc_fields&, std::uint32_t) {}
  CISTA_CUDA_COMPAT static void after_ride(mc_fields&,
                                           clasz,
                                           mc_context const&) {}
  CISTA_CUDA_COMPAT static void after_walk(mc_fields&, std::uint32_t) {}
  CISTA_CUDA_COMPAT static void complete(mc_fields&) {}
};

// ---- non_transit ------------------------------------------------------------
// Each walk (offset, footpath) counts its minutes floored to whole units of
// kNonTransitBin; units are summed and saturate at kNonTransitMax. Per walk,
// so the sum is order-independent (forward and backward searches agree).
inline constexpr bool kNonTransitFullResolution = false;
inline constexpr std::uint32_t kNonTransitBin =
    kNonTransitFullResolution ? 1U : 5U;
inline constexpr std::uint32_t kNonTransitMax = 0x7FU;

CISTA_CUDA_COMPAT constexpr std::uint32_t non_transit_add(
    std::uint32_t const units, std::uint32_t const minutes) {
  return saturate(units + minutes / kNonTransitBin, kNonTransitMax);
}

// Walking (offsets, footpaths; not the same-stop transfer buffer) in units of
// kNonTransitBin minutes.
struct non_transit_dim : mc_dim {
  CISTA_CUDA_COMPAT static void at_start(mc_fields& f,
                                         std::uint32_t const minutes) {
    f.non_transit_ = static_cast<std::uint16_t>(non_transit_add(0U, minutes));
  }
  CISTA_CUDA_COMPAT static void after_walk(mc_fields& f,
                                           std::uint32_t const minutes) {
    f.non_transit_ =
        static_cast<std::uint16_t>(non_transit_add(f.non_transit_, minutes));
  }
  CISTA_CUDA_COMPAT static bool dominates(mc_fields const a,
                                          mc_fields const b) {
    return a.non_transit_ <= b.non_transit_;
  }
  CISTA_CUDA_COMPAT static void join(mc_fields const a,
                                     mc_fields const b,
                                     mc_fields& out) {
    out.non_transit_ =
        saturate(a.non_transit_ + b.non_transit_, kNonTransitMax);
  }
  static void to_journey(mc_fields const f, journey& j) {
    j.criteria_cost_ =
        static_cast<std::uint16_t>(f.non_transit_ * kNonTransitBin);
  }
  static void of_journey(journey const& j, mc_fields& f) {
    f.non_transit_ =
        saturate(j.criteria_cost_ / kNonTransitBin, kNonTransitMax);
  }
};

// ---- avoided_modes
// ------------------------------------------------------------ "Uses an avoided
// vehicle class" (query::avoided_claszes_) as a Pareto dimension, not a filter.
struct avoided_modes_dim : mc_dim {
  static constexpr bool kRide = true;

  CISTA_CUDA_COMPAT static void after_ride(mc_fields& f,
                                           clasz const c,
                                           mc_context const& ctx) {
    if (is_allowed(ctx.avoided_, c)) {
      f.avoided_modes_ = 1U;
    }
  }
  CISTA_CUDA_COMPAT static bool dominates(mc_fields const a,
                                          mc_fields const b) {
    return a.avoided_modes_ <= b.avoided_modes_;
  }
  CISTA_CUDA_COMPAT static void join(mc_fields const a,
                                     mc_fields const b,
                                     mc_fields& out) {
    out.avoided_modes_ =
        static_cast<std::uint16_t>(a.avoided_modes_ | b.avoided_modes_);
  }
  static void to_journey(mc_fields const f, journey& j) {
    j.criteria_avoided_modes_ = f.avoided_modes_ != 0U;
  }
  static void of_journey(journey const& j, mc_fields& f) {
    f.avoided_modes_ = j.criteria_avoided_modes_ ? 1U : 0U;
  }
};

// ---- mode switches ----------------------------------------------------------
// Vehicle class groups; a change of group between trips is a switch.
inline constexpr std::uint32_t kNoModeGroup = 0U;  // no trip ridden yet
inline constexpr std::uint32_t kMaxModeSwitches = 0xFU;

CISTA_CUDA_COMPAT constexpr std::uint32_t mode_group(clasz const c) {
  switch (c) {
    case clasz::kHighSpeed: return 1U;
    case clasz::kLongDistance: return 2U;
    case clasz::kCoach: return 3U;
    case clasz::kNight: return 4U;
    case clasz::kRegional: return 5U;
    case clasz::kSuburban: [[fallthrough]];
    case clasz::kSubway: [[fallthrough]];
    case clasz::kTram: [[fallthrough]];
    case clasz::kBus: return 6U;
    default: return 7U;
  }
}

// Labels in different groups only dominate with a switch to spare (the next
// switch may differ); a complete journey has no group and no next switch.
struct mode_switches_dim : mc_dim {
  static constexpr bool kRide = true;

  CISTA_CUDA_COMPAT static void after_ride(mc_fields& f,
                                           clasz const c,
                                           mc_context const&) {
    auto const g = mode_group(c);
    auto const sw = f.group_ != kNoModeGroup && f.group_ != g ? 1U : 0U;
    f.group_ = static_cast<std::uint16_t>(g);
    f.switches_ = saturate(f.switches_ + sw, kMaxModeSwitches);
  }
  CISTA_CUDA_COMPAT static void complete(mc_fields& f) {
    f.group_ = kNoModeGroup;
  }
  CISTA_CUDA_COMPAT static bool dominates(mc_fields const a,
                                          mc_fields const b) {
    auto const penalty =
        (a.group_ == b.group_ || a.group_ == kNoModeGroup) ? 0U : 1U;
    return a.switches_ + penalty <= b.switches_;
  }
  // a change of group where the halves meet is one more switch
  CISTA_CUDA_COMPAT static void join(mc_fields const a,
                                     mc_fields const b,
                                     mc_fields& out) {
    auto const junction = a.group_ != kNoModeGroup &&
                          b.group_ != kNoModeGroup && a.group_ != b.group_;
    out.switches_ = saturate(a.switches_ + b.switches_ + (junction ? 1U : 0U),
                             kMaxModeSwitches);
  }
  static void to_journey(mc_fields const f, journey& j) {
    j.criteria_mode_switches_ = static_cast<std::uint8_t>(f.switches_);
  }
  static void of_journey(journey const& j, mc_fields& f) {
    f.switches_ = saturate(j.criteria_mode_switches_, kMaxModeSwitches);
  }
};

// ---- the extras -------------------------------------------------------------
// The criteria Dims on the label's 16 bits as an integer, as the engines store
// them. Field-wise (complete) dominance implies ordered(a) <= ordered(b),
// whatever the fields' layout.
template <typename... Dims>
struct mc_extras {
  static constexpr bool kAny = sizeof...(Dims) != 0U;
  static constexpr bool kRide = (... || Dims::kRide);

  CISTA_CUDA_COMPAT static mc_fields fields(std::uint32_t const x) {
    return std::bit_cast<mc_fields>(static_cast<std::uint16_t>(x));
  }
  CISTA_CUDA_COMPAT static std::uint32_t bits(mc_fields const f) {
    return std::bit_cast<std::uint16_t>(f);
  }
  CISTA_CUDA_COMPAT static mc_fields completed(std::uint32_t const x) {
    auto f = fields(x);
    (Dims::complete(f), ...);
    return f;
  }

  CISTA_CUDA_COMPAT static std::uint32_t at_start(
      [[maybe_unused]] std::uint32_t const minutes) {
    auto f = mc_fields{};
    (Dims::at_start(f, minutes), ...);
    return bits(f);
  }
  CISTA_CUDA_COMPAT static std::uint32_t after_ride(
      std::uint32_t const x,
      [[maybe_unused]] clasz const c,
      [[maybe_unused]] mc_context const& ctx) {
    auto f = fields(x);
    (Dims::after_ride(f, c, ctx), ...);
    return bits(f);
  }
  CISTA_CUDA_COMPAT static std::uint32_t after_walk(
      std::uint32_t const x, [[maybe_unused]] std::uint32_t const minutes) {
    auto f = fields(x);
    (Dims::after_walk(f, minutes), ...);
    return bits(f);
  }
  CISTA_CUDA_COMPAT static bool dominates(std::uint32_t const a,
                                          std::uint32_t const b) {
    [[maybe_unused]] auto const fa = fields(a);
    [[maybe_unused]] auto const fb = fields(b);
    return (... && Dims::dominates(fa, fb));
  }
  CISTA_CUDA_COMPAT static bool completed_dominates(std::uint32_t const a,
                                                    std::uint32_t const b) {
    [[maybe_unused]] auto const fa = completed(a);
    [[maybe_unused]] auto const fb = completed(b);
    return (... && Dims::dominates(fa, fb));
  }
  CISTA_CUDA_COMPAT static std::uint32_t ordered(std::uint32_t const x) {
    return bits(completed(x));
  }
  // a prefix and a suffix label joined at a stop fit a journey's criteria
  // (target: of_journey())
  CISTA_CUDA_COMPAT static bool join_fits(std::uint32_t const prefix,
                                          std::uint32_t const suffix,
                                          std::uint32_t const target) {
    [[maybe_unused]] auto const p = fields(prefix);
    [[maybe_unused]] auto const s = fields(suffix);
    [[maybe_unused]] auto const t = fields(target);
    [[maybe_unused]] auto j = mc_fields{};
    (Dims::join(p, s, j), ...);
    return (... && Dims::dominates(j, t));
  }
  static void to_journey(std::uint32_t const x, [[maybe_unused]] journey& j) {
    [[maybe_unused]] auto const f = completed(x);
    (Dims::to_journey(f, j), ...);
  }
  static std::uint32_t of_journey([[maybe_unused]] journey const& j) {
    auto f = mc_fields{};
    (Dims::of_journey(j, f), ...);
    return bits(f);
  }
};

// ---- breadcrumbs ------------------------------------------------------------
// The transit leg behind an arrival: the trip (transport_idx_t on traffic day
// day_, or rt_transport_idx_t if day_ is invalid), its board and alight stops,
// the arrival at the label's stop and the boarded label's breadcrumb (each
// engine has its own "start" sentinel).
struct mc_breadcrumb {
  std::uint32_t transport_;
  std::uint32_t parent_;
  day_idx_t day_;
  stop_idx_t board_;
  stop_idx_t alight_;
  delta_t arr_;
};
static_assert(sizeof(mc_breadcrumb) == 16U);

}  // namespace nigiri::routing
