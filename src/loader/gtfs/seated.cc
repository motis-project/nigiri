#include "nigiri/loader/gtfs/seated.h"

#include "nigiri/loader/netex/utc_trip.h"

#include <ranges>

// #define trace(...) fmt::println(std::clog, __VA_ARGS__)
#define trace(...)

namespace sv = std::views;

namespace nigiri::loader::gtfs {

template <typename UtcTrip, typename TripIdx>
void build_seated_trips(timetable& tt,
                        expanded_seated<UtcTrip>& seated,
                        std::function<std::string(TripIdx)> const& dbg,
                        std::function<void(UtcTrip&&)> const& consumer) {
  [[maybe_unused]] auto const base = tt.internal_interval_days().from_;

  assert(seated.expanded_.size() == seated.seated_in_.size());
  assert(seated.expanded_.size() == seated.seated_out_.size());
  assert(seated.remaining_rule_trip_.size() == seated.expanded_.data_.size());

  auto const shift = [](bitfield const& b, int const offset) {
    return offset > 0 ? b << static_cast<std::size_t>(offset)
                      : b >> static_cast<std::size_t>(-offset);
  };

  vector<UtcTrip>& remaining = seated.expanded_.data_;
  auto const get_remaining = [&](rule_trip_idx_t const i) {
    return seated.expanded_.at(i) | sv::transform([&](UtcTrip const& x) {
             return static_cast<remaining_idx_t>(remaining.index_of(&x));
           });
  };
  auto const incoming = [&](remaining_idx_t const i) {
    return seated.seated_in_[seated.remaining_rule_trip_[i]];
  };
  auto const outgoing = [&](remaining_idx_t const i) {
    return seated.seated_out_[seated.remaining_rule_trip_[i]];
  };
  [[maybe_unused]] auto const name = [&](remaining_idx_t const i) {
    return dbg(remaining[i].trips_[0]);
  };
  auto const first_dep =
      [&](remaining_idx_t const x) -> minutes_after_midnight_t {
    return remaining[x].utc_times_.front();
  };
  auto const last_arr =
      [&](remaining_idx_t const x) -> minutes_after_midnight_t {
    return remaining[x].utc_times_.back();
  };
  auto const get_day_change_offset = [&](remaining_idx_t const a,
                                         remaining_idx_t const b) {
    assert(first_dep(b) < 1440_minutes);
    auto const day_span =
        last_arr(a) / date::days{1U} - first_dep(a) / date::days{1U};
    auto const day_change = last_arr(a) % 1440 > first_dep(b) ? 1 : 0;
    return day_span + day_change;
  };
  auto const is_valid_dwell_time = [&](remaining_idx_t const a,
                                       remaining_idx_t const b) {
    // examples:
    // 1) 23:59 - 00:01 => 00:01 - 23:59 = 1 - 1339 = -1338 => -1338+1440 = 2
    // 2) 10:00 - 11:00 => 11:00 - 10:00 = 660 - 600 => 60
    assert(first_dep(b) < 1440_minutes);
    auto const diff = first_dep(b) - last_arr(a) % 1440;
    auto const dwell = diff < duration_t{0} ? diff + 1440_minutes : diff;
    return dwell < 120_minutes;
  };

  auto remaining_has_bits = bitvec_map<remaining_idx_t>{};
  remaining_has_bits.resize(remaining.size());
  remaining_has_bits.one_out();

  auto combinations = std::vector<utc_trip>{};
  auto next_remaining = remaining_has_bits.next_set_bit(0U);
  while (next_remaining.has_value()) {
    // ===============================
    // PART 1: Find maximum component.
    // -------------------------------
    auto component = hash_map<remaining_idx_t, int>{};
    auto component_traffic_days = remaining[*next_remaining].utc_traffic_days_;
    {
      // Collect all trips reachable from this trip connected by stay-seated
      // transfers from here (forward+backward, direct + transitive) while
      // building the traffic day intersection of all visited trips. Stop early
      // if the intersection would be empty.
      auto q = hash_map<remaining_idx_t,
                        int /* offset relative to its traffic days */>{};
      q.emplace(static_cast<remaining_idx_t>(
                    remaining.index_of(&remaining[*next_remaining])),
                0U);
      while (!q.empty()) {
        // Extract next queue element.
        auto const curr_it = q.begin();
        auto const [curr, offset] = *curr_it;
        auto& current = remaining[curr];
        q.erase(curr_it);

        trace("\nEXTRACT {}, offset={}", name(curr), offset);

        // Intersect traffic days.
        auto const next_traffic_days =
            shift(current.utc_traffic_days_, -offset) & component_traffic_days;
        trace(
            "      current: {}\n"
            "      shifted: {}\n"
            "    component: {}\n"
            "         next: {}",
            day_list{current.utc_traffic_days_, base},
            day_list{shift(current.utc_traffic_days_, -offset), base},
            day_list{component_traffic_days, base},
            day_list{next_traffic_days, base});
        if (next_traffic_days.none()) {
          trace("-> EMPTY INTERSECTION");
          continue;  // Nothing left, skip.
        }

        // Non-empty intersection!
        // Add trip to component + update component traffic days.
        trace("UPDATE: {}", day_list{next_traffic_days, base});
        component_traffic_days = next_traffic_days;
        component.emplace(curr, offset);

        // Expand search to neighbors.
        for (auto const& out_trp : outgoing(curr)) {
          for (auto const out : get_remaining(out_trp)) {
            if (component.contains(out) || !is_valid_dwell_time(curr, out)) {
              continue;
            }

            auto const o = offset + get_day_change_offset(curr, out);
            trace(
                "    EXPAND OUT: {}, day_change_offset={} "
                "(last_arr(curr)={}, first_dep(out)={}) => offset={}",
                name(out), get_day_change_offset(curr, out), last_arr(curr),
                first_dep(out), o);
            q.emplace(out, o);
          }
        }

        for (auto const& in_trp : incoming(curr)) {
          for (auto const in : get_remaining(in_trp)) {
            if (component.contains(in) || !is_valid_dwell_time(in, curr)) {
              continue;
            }

            auto const o = offset - get_day_change_offset(in, curr);
            trace(
                "    EXPAND IN: {}, day_change_offset={} (last_arr(in)={}, "
                "first_dep(curr)={}) => offset={}",
                name(in), get_day_change_offset(in, curr), last_arr(in),
                first_dep(curr), o);
            q.emplace(in, o);
          }
        }
      }  // END while (!q.empty())
    }

    // =========================
    // PART 2: Queue all starts.
    // -------------------------
    trace("\n=> COMPONENT: {}", day_list{component_traffic_days, base});

    // Finds the remaining_idx
    // that represents a given gtfs_trip_idx in this specific component.
    auto const get_representative = [&](rule_trip_idx_t const t)
        -> std::optional<std::pair<remaining_idx_t, int>> {
      auto const it = utl::find_if(
          component, [&](std::pair<remaining_idx_t, int> const& x) {
            return seated.remaining_rule_trip_[x.first] == t;
          });
      return it == end(component) ? std::nullopt : std::optional{*it};
    };

    // Initialize queue with all remaining_idx that do not have any incoming
    // seated transfers *in this component*.
    auto q =
        std::vector<std::tuple<UtcTrip,  // concatenated transport
                               rule_trip_idx_t,  // last rule trip in chain
                               int  // transport offset relative to component
                               >>{};
    for (auto const& [remaining_idx, offset] : component) {
      trace("  -> {}: {} on {}", name(remaining_idx), offset,
            day_list{shift(component_traffic_days, offset), base});
      auto const is_entry = std::ranges::empty(
          incoming(remaining_idx)  //
          | sv::transform(get_representative)  //
          | sv::filter([](auto&& r) { return r.has_value(); }));
      if (is_entry) {
        auto const transport_traffic_days =
            shift(component_traffic_days, offset);
        assert((remaining.at(remaining_idx).utc_traffic_days_ &
                transport_traffic_days) == transport_traffic_days);
        trace("    -> PUSH {}, on={} (transport_offset={})",
              name(remaining_idx), day_list{transport_traffic_days, base},
              -offset);
        auto start = remaining.at(remaining_idx);
        start.utc_traffic_days_ = transport_traffic_days;
        q.emplace_back(std::move(start),
                       seated.remaining_rule_trip_[remaining_idx], -offset);
      }
    }

    // ============
    // PART 3: DFS.
    // ------------
    while (!q.empty()) {
      auto [curr, prev_rule_trip, transport_offset] = std::move(q.back());
      q.resize(q.size() - 1U);

      // Expand search.
      auto has_next = false;
      for (auto const& next : seated.seated_out_[prev_rule_trip]) {
        auto const r = get_representative(next);
        if (r.has_value()) {
          auto copy = curr;
          auto const [remaining_idx, offset] = *r;
          auto const& next_r = remaining.at(remaining_idx);
          auto next_times = next_r.utc_times_;
          for (auto& t : next_times) {
            t += (transport_offset + offset) * date::days{1};
            assert(t >= duration_t{0});
          }
          copy.trips_.push_back(remaining[remaining_idx].trips_[0]);
          copy.utc_times_.insert(end(copy.utc_times_), begin(next_times),
                                 end(next_times));
          copy.stop_seq_.insert(end(copy.stop_seq_),
                                std::next(begin(next_r.stop_seq_)),
                                end(next_r.stop_seq_));
          trace("append {} (offset={}): {}", name(remaining_idx), offset,
                copy.utc_times_ | sv::transform(std::identity{}));
          assert(std::ranges::is_sorted(copy.utc_times_));
          if (!std::ranges::is_sorted(copy.utc_times_)) {
            fmt::println(
                std::clog,
                "ERROR! NOT SORTED: transport_offset={}, offset={}"
                "\ntrips:\n\t{}\nstops:\n\t{}\ntimes:\n\t{}\nnext_times:\n\t{}"
                "\n",
                transport_offset, offset,
                fmt::join(copy.trips_ | sv::transform(dbg), "\n\t"),
                fmt::join(copy.stop_seq_ |
                              sv::transform([&](stop::value_type const& s) {
                                return loc{tt, stop{s}.location_idx()};
                              }),
                          "\n\t"),
                fmt::join(copy.utc_times_ | sv::transform(std::identity{}),
                          "\n\t"),
                fmt::join(next_r.utc_times_ | sv::transform(std::identity{}),
                          "\n\t"));
          }
          q.emplace_back(std::move(copy),
                         seated.remaining_rule_trip_[remaining_idx],
                         transport_offset);
          has_next = true;
        }
      }

      if (!has_next) {  // Terminal?
        // No outgoing seated-transfer *in this component*.
        // Pass finished transport to consumer.
        trace("adding trips={}, stops={}, times={}",
              curr.trips_ | sv::transform(dbg),
              curr.stop_seq_ | sv::transform([&](stop::value_type const& s) {
                return nigiri::loc{tt, stop{s}.location_idx()};
              }),
              curr.utc_times_ | sv::transform(std::identity{}));
        consumer(std::move(curr));
      }
    }

    // ===========================
    // PART 4: Update Traffic Days
    // ---------------------------
    for (auto const& [remaining_idx, offset] : component) {
      remaining.at(remaining_idx).utc_traffic_days_ &=
          ~shift(component_traffic_days, offset);

      if (remaining.at(remaining_idx).utc_traffic_days_.none()) {
        remaining_has_bits.set(remaining_idx, false);
      }
    }

    trace("------------\n");

    next_remaining = remaining_has_bits.next_set_bit(*next_remaining);
  }  // END while (!utl::all_of(remaining, is_empty))
}

template void build_seated_trips<gtfs::utc_trip, gtfs_trip_idx_t>(
    timetable&,
    expanded_seated<gtfs::utc_trip>&,
    std::function<std::string(gtfs_trip_idx_t)> const&,
    std::function<void(gtfs::utc_trip&&)> const&);

template void build_seated_trips<netex::utc_trip, netex::service_journey_idx_t>(
    timetable&,
    expanded_seated<netex::utc_trip>&,
    std::function<std::string(netex::service_journey_idx_t)> const&,
    std::function<void(netex::utc_trip&&)> const&);

}  // namespace nigiri::loader::gtfs
