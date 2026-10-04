#include "nigiri/transfer_rule_sides.h"

#include <algorithm>
#include <vector>

#include "utl/equal_ranges_linear.h"
#include "utl/erase_duplicates.h"
#include "utl/helpers/algorithm.h"
#include "utl/to_vec.h"

#include "nigiri/footpath.h"
#include "nigiri/timetable.h"

namespace nigiri {

void add_trip_rules(timetable const& tt,
                    std::span<trip_idx_t const> trips,
                    std::vector<transfer_rule_side_idx>& out) {
  auto const& tr = tt.transfer_rules_;
  for (auto const trip : trips) {
    // Collect all rules applying to the trip.
    auto const trip_rules = values_of(tr.trip_rules_, trip);
    out.insert(end(out), trip_rules.begin(), trip_rules.end());

    // Collect all rules applying to the route.
    auto const src = tt.trip_id_src_[tt.trip_ids_[trip].front()];
    for (auto const s : values_of(tr.route_rules_, tt.trip_route_id_[trip])) {
      if (tr.rules_[s.rule()].src_ == src) {
        out.push_back(s);
      }
    }
  }
}

void get_signature(timetable const& tt,
                   std::span<transfer_rule_side_idx const> arriving,
                   std::span<transfer_rule_side_idx const> departing,
                   bool const is_handover,
                   location_idx_t const base,
                   std::vector<transfer_rule_side_idx>& sig) {
  sig.clear();
  auto const add = [&](transfer_rule_side_idx const s, bool const is_arriving) {
    if (is_handover && s.is_from() != is_arriving) {
      // stay seated / block_id transfers -> only keep:
      // - the from side matching the arriving trip
      // - the to side matching the departing trip
      return;
    }

    auto const stop = tt.transfer_rules_.rules_[s.rule()].stop(s.is_from());
    if (tt.locations_.is_self_or_parent(stop, base)) {
      sig.push_back(s);
    }
  };
  for (auto const s : arriving) {
    add(s, true);
  }
  for (auto const s : departing) {
    add(s, false);
  }
  utl::erase_duplicates(sig);
}

std::vector<transfer_rule_side_idx> get_change_signature(
    timetable const& tt,
    std::span<trip_idx_t const> arriving_trips,
    std::span<trip_idx_t const> departing_trips,
    bool const is_handover,
    location_idx_t const base) {
  auto arriving = std::vector<transfer_rule_side_idx>{};
  auto departing = std::vector<transfer_rule_side_idx>{};
  add_trip_rules(tt, arriving_trips, arriving);
  add_trip_rules(tt, departing_trips, departing);
  auto sig = std::vector<transfer_rule_side_idx>{};
  get_signature(tt, arriving, departing, is_handover, base, sig);
  return sig;
}

duration_t get_transfer_time(timetable const& tt,
                             std::span<transfer_rule_side_idx const> sig,
                             location_idx_t const base) {
  auto const& rules = tt.transfer_rules_.rules_;
  auto best = transfer_rule_idx_t::invalid();
  for (auto const s : sig) {
    auto const rule = s.rule();
    if (is_applicable(tt, {rule, true}, sig, base) &&
        is_applicable(tt, {rule, false}, sig, base)) {
      best = best == transfer_rule_idx_t::invalid()
                 ? rule
                 : get_more_specific(tt, best, rule);
    }
  }
  return best != transfer_rule_idx_t::invalid()
             ? rules[best].duration_
             : to_fp_duration(tt.locations_.transfer_time_[base]);
}

std::vector<transfer_rule_side> get_transfer_rule_sides(
    timetable const& tt,
    std::span<transfer_rule_side_idx const> rules,
    location_idx_t const base) {
  auto const to_side = [&](transfer_rule_side_idx const s) {
    auto const& r = tt.transfer_rules_.rules_[s.rule()];
    auto const is_from = s.is_from();
    return transfer_rule_side{.is_from_ = is_from,
                              .rule_stop_ = r.stop(is_from),
                              .other_stop_ = r.stop(!is_from),
                              .src_ = r.src_,
                              .other_route_ = r.route(!is_from),
                              .other_trip_ = r.trip(!is_from),
                              .duration_ = r.duration_,
                              .specificity_ = r.specificity_};
  };

  // Transform to rule sides.
  auto v = utl::to_vec(rules, to_side);

  // Sides unqualified at this stop (or its parent): they apply to every trip
  // here, so they're in no signature, but they compete with its sides.
  auto unqualified = std::vector<transfer_rule_side>{};
  for (auto const stop : {base, tt.locations_.parents_[base]}) {
    if (stop != location_idx_t::invalid()) {
      for (auto const s : values_of(tt.transfer_rules_.stop_rules_, stop)) {
        unqualified.push_back(to_side(s));
      }
    }
  }

  // Lower each side's specificity to the lowest value that keeps its order
  // against competing rule sides (same partners, different duration): the same
  // rules win, but more stops get the same key and share a virtual location.
  auto const& loc = tt.locations_;
  auto const has_same_partners = [&](transfer_rule_side const& a,
                                     transfer_rule_side const& b) {
    return a.is_from_ == b.is_from_ &&
           (loc.is_self_or_parent(a.other_stop_, b.other_stop_) ||
            loc.is_self_or_parent(b.other_stop_, a.other_stop_));
  };
  utl::sort(v, [](transfer_rule_side const& a, transfer_rule_side const& b) {
    return a.specificity_ < b.specificity_;
  });
  utl::equal_ranges_linear(
      v,
      [](transfer_rule_side const& a, transfer_rule_side const& b) {
        return a.specificity_ == b.specificity_;
      },
      [&](auto const from, auto const to) {
        for (auto a = from; a != to; ++a) {
          auto lowered = transfer_rule_specificity_t{0U};

          // Iterate rules with lower specificity.
          // Higher specificity rules win anyway -> no need to check them.
          for (auto b = begin(v); b != from; ++b) {
            if (b->duration_ != a->duration_ && has_same_partners(*a, *b)) {
              // b has a lower specificity, another duration, and can apply to
              // the same transfers -> stay above it, otherwise a stop where
              // this side loses against b would get the same key and share the
              // virtual location (incorrect)
              lowered =
                  std::max(lowered, static_cast<transfer_rule_specificity_t>(
                                        b->specificity_ + 1U));
            }
          }
          for (auto const& b : unqualified) {
            if (b.specificity_ < a->specificity_ &&
                b.duration_ != a->duration_ && has_same_partners(*a, b)) {
              lowered =
                  std::max(lowered, static_cast<transfer_rule_specificity_t>(
                                        b.specificity_ + 1U));
            }
          }

          a->specificity_ = lowered;
        }
      });

  utl::erase_duplicates(v);

  return v;
}

virt_key get_virt_key(timetable const& tt,
                      std::span<transfer_rule_side_idx const> sig,
                      location_idx_t const base) {
  return {.base_ = base,
          .transfer_time_ = to_transfer_time(get_transfer_time(tt, sig, base)),
          .sides_ = get_transfer_rule_sides(tt, sig, base)};
}

}  // namespace nigiri
