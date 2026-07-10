#pragma once

#include <cassert>


#include "nigiri/common/delta_t.h"
#include "nigiri/common/linear_lower_bound.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/routing/raptor/debug.h"
#include "nigiri/routing/raptor/raptor_state.h"
#include "nigiri/routing/raptor/reconstruct.h"
#include "nigiri/routing/transfer_time_settings.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/special_stations.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

#include "cista/containers/vecvec.h"
#include "utl/erase_if.h"

namespace nigiri::routing {

    template <direction SearchDir,
        bool Rt,
        via_offset_t Vias,
        search_mode SearchMode>
    struct mcraptor {

        struct bag_entry {
            delta_t time_;
            bool flag = true;


            bag_entry(delta_t time) : time_(time) {
                //#ifdef kFalseFlag
                flag = (static_cast<unsigned long long>(time_) % 2); // prefer even (only for testing)
                //#endif
            }

            bool is_invalid() const {
                return time_ == kInvalid;
            }

            bool operator<=(const delta_t time) const {
                return kFwd ? (time_ <= time) : (time_ >= time);
            }

            bool operator<=(const bag_entry& cmp) const {
                return kFwd ? (time_ <= cmp.time_) : (time_ >= cmp.time_);
            }

            bool operator==(const bag_entry& cmp) const {
                return time_ == cmp.time_;
            }

            bool operator<(const delta_t time) const {
                return kFwd ? (time_ < time) : (time_ > time);
            }

            bool operator<(const bag_entry& cmp) const {
                return kFwd ? (time_ < cmp.time_) : (time_ > cmp.time_);
            }

        };

        struct bag {
            std::vector<bag_entry> pareto_set_;

            bag(delta_t t) {
                pareto_set_.clear();
                if (t != kInvalid) {
                    pareto_set_.push_back(bag_entry(t));
                }
            }

            bag(bag_entry be) {
                pareto_set_.push_back(be);
            }

            bag() {
                pareto_set_.clear();
            }

            bool is_better(bag b) const {
                if (pareto_set_.empty()) {
                    return false
                }
                for (auto this_ele : pareto_set_) {
                    for (auto b_ele : b.pareto_set_) {
                        if (b_ele < this_ele)
                            return false;
                    }
                }
                return true;
            }

            bool is_invalid() const {
                return pareto_set_.empty();
            }

            bool is_better(delta_t time) const {
                if (pareto_set_.empty()) {
                    return false;
                }
                for (auto e : pareto_set_)
                    if (!(e < time)) return false;
                return true;
            }

            bool is_better_with_offset(delta_t offset, bag b) const {
                for (auto this_ele : pareto_set_) {
                    for (auto b_ele : b.pareto_set_) {
                        bag_entry offset_ele = bag_entry(this_ele.time_ + offset);
                        if (!(offset_ele < b_ele))
                            return false;
                    }
                }
                return true;
            }

            void add(bag_entry be) {

                if (be.is_invalid()) {
                    return;
                }

                std::vector<bag_entry> bad_entries;
                bool should_add = false;
                for (auto elem : pareto_set_)
                {
                    if (be <= elem) {
                        should_add = true;
                        bad_entries.push_back(elem);
                    }
                }
                
                utl::erase_if(pareto_set_
                    , [&](auto const& toDelete) { return std::find(bad_entries.begin(), bad_entries.end(), toDelete) != bad_entries.end();});
                if (should_add) { pareto_set_.push_back(be); }
            }

            void add(delta_t t) {
                if (t == kInvalid) {
                    return;
                }

                std::vector<bag_entry> bad_entries;
                bool should_add = false;
                for (auto elem : pareto_set_)
                {
                    if (!(elem <= t)) {
                        should_add = true;
                        bad_entries.push_back(elem);
                    }
                }

                utl::erase_if(pareto_set_
                    , [&](auto const& toDelete) { return std::find(bad_entries.begin(), bad_entries.end(), toDelete) != bad_entries.end();});
                if (should_add) { pareto_set_.push_back(bag_entry(t)); }
            }

            void add(bag bg) {
                for (auto elem : bg.pareto_set_) {
                    add(elem);
                }
            }

            template <typename... Args>
            bag copy(Args... t) const{
                bag ret = bag();
                ret.add(*this);
                for (auto e : ret.pareto_set_) {
                    e.time_ = clamp((e.time_ + ... + t));
                }
                return ret;
            }

            // Depricated
            delta_t get_any_time() const {
                if (pareto_set_.empty()) {
                    return delta_t{ kInvalid };
                }

                return pareto_set_.at(0).time_;
            }

            // Depricated 
            void replace_any_time(delta_t t) {
                if (pareto_set_.empty()) {
                    pareto_set_.push_back(bag_entry(t));
                    return;
                }

                pareto_set_.at(0).time_ = t;
            }

            // Depricated
            void replace_time(delta_t t) {
                if (t == kInvalid) {
                    return;
                }

                if (t < get_any_time()) {
                    replace_any_time(t);
                }
            }


        };

        using algo_state_t = raptor_state;
        using algo_stats_t = raptor_stats;

        #pragma region constexpr_func
        static constexpr bool kUseLowerBounds = true;
        static constexpr auto const kFwd = (SearchDir == direction::kForward);
        static constexpr auto const kBwd = (SearchDir == direction::kBackward);
        static constexpr auto const kInvalid = kInvalidDelta<SearchDir>;
        static constexpr auto const kUnreachable =
            std::numeric_limits<std::uint16_t>::max();
        static constexpr auto const kIntermodalTarget =
            to_idx(get_special_station(special_station::kEnd));
        static constexpr auto const kInvalidArray = []() {
            auto a = std::array<delta_t, Vias + 1>{};
            a.fill(kInvalid);
            return a;
            }();
        #pragma endregion

        #pragma region static_func
        static bool is_better(auto a, auto b) { return kFwd ? a < b : a > b; }
        static bool is_better_or_eq(auto a, auto b) { return kFwd ? a <= b : a >= b; }
        static auto get_best(auto a, auto b) { return is_better(a, b) ? a : b; }
        static auto get_best(auto x, auto... y) {
            ((x = get_best(x, y)), ...);
            return x;
        }
        static auto dir(auto a) { return (kFwd ? 1 : -1) * a; }
        #pragma endregion

        #pragma region constructor
        mcraptor(
            timetable const& tt,
            rt_timetable const* rtt,
            raptor_state& state,
            bitvec& is_dest,
            std::array<bitvec, kMaxVias>& is_via,
            std::vector<std::uint16_t>& dist_to_dest,
            hash_map<location_idx_t, std::vector<td_offset>> const& td_dist_to_dest,
            std::vector<std::uint16_t>& lb,
            std::vector<via_stop> const& via_stops,
            day_idx_t const base,
            clasz_mask_t const allowed_claszes,
            bool const require_bike_transport,
            bool const require_car_transport,
            bool const is_wheelchair,
            transfer_time_settings const& tts)
            : tt_{ tt },
            rtt_{ rtt },
            n_days_{ tt_.internal_interval_days().size().count() },
            n_locations_{ tt_.n_locations() },
            n_routes_{ tt.n_routes() },
            n_rt_transports_{ Rt ? rtt->n_rt_transports() : 0U },
            state_{ state.resize(n_locations_, n_routes_, n_rt_transports_) },
            is_dest_{ is_dest },
            is_via_{ is_via },
            dist_to_end_{ dist_to_dest },
            td_dist_to_end_{ td_dist_to_dest },
            lb_{ lb },
            via_stops_{ via_stops },
            base_{ base },
            allowed_claszes_{ allowed_claszes },
            require_bike_transport_{ require_bike_transport },
            require_car_transport_{ require_car_transport },
            is_wheelchair_{ is_wheelchair },
            transfer_time_settings_{ tts } {
            #pragma region inside constructor
            //TODO: avoid copy
            auto all_tmp_times = state.get_tmp<Vias>();
            auto all_best_times = state.get_best<Vias>();

            new_tmp_.resize(all_tmp_times.size());
            for (unsigned long i = 0; i < all_tmp_times.size(); ++i) {
                for (unsigned long v = 0; v < Vias + 1; ++v) {
                    new_tmp_[i][v] = bag(all_tmp_times[i][v]);
                }             
            }
            new_best_.resize(all_best_times.size());
            for (unsigned long i = 0; i < all_best_times.size(); ++i) {
                for (unsigned long v = 0; v < Vias + 1; ++v) {
                    new_best_[i][v] = bag(all_best_times[i][v]);
                }
            }

            auto src_span = state.get_round_times<Vias>();
            round_times_.clear();
            for (unsigned long k = 0; k < src_span.n_rows_; ++k) {
                std::vector<std::array<bag, Vias + 1>> row;
                for (unsigned long i = 0; i < src_span.n_columns_; ++i) {
                    std::array<bag, Vias + 1> target_array;
                    auto src_array = src_span[k][i];
                    for (unsigned long v = 0; v < Vias + 1; ++v) {
                        target_array[v] = bag(src_array[v]);
                    }
                    row.push_back(target_array);
                }
                round_times_.emplace_back(row);
            }

            assert(Vias == via_stops_.size());
            reset_arrivals();
            if (!dist_to_end_.empty()) {
                // only used for intermodal queries (dist_to_dest != empty)
                end_reachable_.resize(n_locations_);
                for (auto i = 0U; i != dist_to_end_.size(); ++i) {
                    if (dist_to_end_[i] != kUnreachable) {
                        end_reachable_.set(i, true);
                    }
                }
                for (auto const& [l, _] : td_dist_to_end_) {
                    end_reachable_.set(to_idx(l), true);
                }
            }
            #pragma endregion
        }
        #pragma endregion

        #pragma region public_functions
        algo_stats_t get_stats() const { return stats_; }

        void reset_arrivals() {
            utl::fill(time_at_dest_, bag());
            //round_times_.reset(kInvalidArray);
            for (auto ke : round_times_) {
                for (auto& ie : ke) {
                    for (auto& ve : ie) {
                        ve = bag();
                    }
                }
            }
        }

        void next_start_time() {
            utl::fill(new_best_, []() {
                auto a = std::array<bag, Vias + 1>{};
                a.fill(bag());
                return a;
                }());
            utl::fill(new_tmp_, []() {
                auto a = std::array<bag, Vias + 1>{};
                a.fill(bag());
                return a;
                }());
            utl::fill(state_.prev_station_mark_.blocks_, 0U);
            utl::fill(state_.station_mark_.blocks_, 0U);
            utl::fill(state_.route_mark_.blocks_, 0U);
            if constexpr (Rt) {
                utl::fill(state_.rt_transport_mark_.blocks_, 0U);
            }
        }

        void add_start(location_idx_t const l, unixtime_t const t) {
            auto const v = (Vias != 0 && is_via_[0][to_idx(l)]) ? 1U : 0U;
            trace_upd(
                "adding start [fwd={}] {}: {}, v={} [current: best={}, round={} => "
                "best={}]\n",
                kFwd, loc{ tt_, l }, t, v, to_unix(new_best_[to_idx(l)][v]),
                to_unix(round_times_[0U][to_idx(l)][v].get_any_time()),
                get_best(t, to_unix(new_best_[to_idx(l)][v].get_any_time())));
            new_best_[to_idx(l)][v].add(unix_to_delta(base(), t));
            round_times_[0U][to_idx(l)][v].add( unix_to_delta(base(), t));
            state_.station_mark_.set(to_idx(l), true);
        }

        void execute(unixtime_t const start_time,
            std::uint8_t const max_transfers,
            unixtime_t const worst_time_at_dest,
            profile_idx_t const prf_idx,
            pareto_set<journey>& results) {
            auto const end_k = std::min(max_transfers, kMaxTransfers) + 2U;

            auto const d_worst_at_dest = unix_to_delta(base(), worst_time_at_dest);
            for (auto& time_at_dest : time_at_dest_) {
                time_at_dest.add(d_worst_at_dest);
            }

            trace_print_init_state();

            for (auto k = 1U; k != end_k; ++k) {
                for (auto i = 0U; i != n_locations_; ++i) {
                    for (auto v = 0U; v != Vias + 1; ++v) {
                        // new_best_[i][v].add(round_time_[k][i][v]);
                        new_best_[i][v].add(round_times_[k][i][v]);
                    }
                }
                //COMMENT: bestimmt bestmöglichen Ankunftzeit für i
                is_dest_.for_each_set_bit([&](std::uint64_t const i) {
                    update_time_at_dest(k, new_best_[i][Vias]);
                    });

                auto any_marked = false;
                state_.station_mark_.for_each_set_bit([&](std::uint64_t const i) {
                    //COMMENT: markiert die Station (falls es Umstiegsmöglichkeiten gibt) und bestimmt alle Routen, die von der markierten Stationen ausgehn
                    for (auto const& r : tt_.location_routes_[location_idx_t{ i }]) {
                        any_marked = true;
                        state_.route_mark_.set(to_idx(r), true);
                    }
                    if constexpr (Rt) {
                        //COMMENT: Echtzeit-Fall
                        for (auto const& rt_t :
                            rtt_->location_rt_transports_[location_idx_t{ i }]) {
                            any_marked = true;
                            state_.rt_transport_mark_.set(to_idx(rt_t), true);
                        }
                    }
                    });

                if (!any_marked) {
                    trace_print_state_after_round();
                    break;
                }

                std::swap(state_.prev_station_mark_, state_.station_mark_);
                utl::fill(state_.station_mark_.blocks_, 0U);

                bool const clasz_filter = allowed_claszes_ != all_clasz_allowed();
                uint8_t const filters =
                    static_cast<uint8_t>(clasz_filter << 3) |
                    static_cast<uint8_t>(require_bike_transport_ << 2) |
                    static_cast<uint8_t>(require_car_transport_ << 1) |
                    static_cast<uint8_t>(is_wheelchair_ << 0);

                any_marked |= [&]() {
                    switch (filters) {
                    case 0b0000: return loop_routes<false, false, false, false>(k);
                    case 0b0001: return loop_routes<false, false, false, true>(k);
                    case 0b0010: return loop_routes<false, false, true, false>(k);
                    case 0b0011: return loop_routes<false, false, true, true>(k);
                    case 0b0100: return loop_routes<false, true, false, false>(k);
                    case 0b0101: return loop_routes<false, true, false, true>(k);
                    case 0b0110: return loop_routes<false, true, true, false>(k);
                    case 0b0111: return loop_routes<false, true, true, true>(k);
                    case 0b1000: return loop_routes<true, false, false, false>(k);
                    case 0b1001: return loop_routes<true, false, false, true>(k);
                    case 0b1010: return loop_routes<true, false, true, false>(k);
                    case 0b1011: return loop_routes<true, false, true, true>(k);
                    case 0b1100: return loop_routes<true, true, false, false>(k);
                    case 0b1101: return loop_routes<true, true, false, true>(k);
                    case 0b1110: return loop_routes<true, true, true, false>(k);
                    case 0b1111: return loop_routes<true, true, true, true>(k);
                    default: std::unreachable();
                    }
                    }();

                if constexpr (Rt) {
                    any_marked |= [&]() {
                        switch (filters) {
                        case 0b0000: return loop_rt_routes<false, false, false, false>(k);
                        case 0b0001: return loop_rt_routes<false, false, false, true>(k);
                        case 0b0010: return loop_rt_routes<false, false, true, false>(k);
                        case 0b0011: return loop_rt_routes<false, false, true, true>(k);
                        case 0b0100: return loop_rt_routes<false, true, false, false>(k);
                        case 0b0101: return loop_rt_routes<false, true, false, true>(k);
                        case 0b0110: return loop_rt_routes<false, true, true, false>(k);
                        case 0b0111: return loop_rt_routes<false, true, true, true>(k);
                        case 0b1000: return loop_rt_routes<true, false, false, false>(k);
                        case 0b1001: return loop_rt_routes<true, false, false, true>(k);
                        case 0b1010: return loop_rt_routes<true, false, true, false>(k);
                        case 0b1011: return loop_rt_routes<true, false, true, true>(k);
                        case 0b1100: return loop_rt_routes<true, true, false, false>(k);
                        case 0b1101: return loop_rt_routes<true, true, false, true>(k);
                        case 0b1110: return loop_rt_routes<true, true, true, false>(k);
                        case 0b1111: return loop_rt_routes<true, true, true, true>(k);
                        default: std::unreachable();
                        }
                        }();
                }

                if (!any_marked) {
                    trace_print_state_after_round();
                    break;
                }

                utl::fill(state_.route_mark_.blocks_, 0U);
                utl::fill(state_.rt_transport_mark_.blocks_, 0U);

                std::swap(state_.prev_station_mark_, state_.station_mark_);
                utl::fill(state_.station_mark_.blocks_, 0U);

                update_transfers(k);
                update_intermodal_footpaths(k);
                update_footpaths(k, prf_idx);
                update_td_offsets(k, prf_idx);

                trace_print_state_after_round();
            }

            if constexpr (SearchMode == search_mode::kOneToAll) {
                return;
            }

            //COMMENT: hier werden die journeys erstellt
            is_dest_.for_each_set_bit([&](auto const i) {
                for (auto k = 1U; k != end_k; ++k) {
                    auto const dest_time = round_times_[k][i][Vias];
                    if (!dest_time.is_invalid()) {
                        trace("ADDING JOURNEY: start={}, dest={} @ {}, transfers={}\n",
                            start_time, delta_to_unix(base(), round_times_[k][i][Vias].get_any_time()),
                            loc{ tt_, location_idx_t{i} }, k - 1);
                        for (auto label : dest_time.pareto_set_) {
                            // TODO: added criteria should also be added in journey
                            auto const [optimal, it, dominated_by] = results.add(
                                journey{ .legs_ = {},
                                        .start_time_ = start_time,
                                        .dest_time_ = delta_to_unix(base(), label.time_),
                                        .dest_ = location_idx_t{i},
                                        .transfers_ = static_cast<std::uint8_t>(k - 1) });
                            if (!optimal) {
                                trace("  DOMINATED BY: start={}, dest={} @ {}, transfers={}\n",
                                    dominated_by->start_time_, dominated_by->dest_time_,
                                    loc{ tt_, dominated_by->dest_ }, dominated_by->transfers_);
                            }
                        }
                    }
                }
                });
        }

        void reconstruct(query const& q, journey& j) {
            if constexpr (SearchMode == search_mode::kOneToAll) {
                return;
            }
            trace("reconstruct({} - {}, {} transfers", j.departure_time(),
                j.arrival_time(), j.transfers_);
            reconstruct_journey<SearchDir>(tt_, rtt_, q, state_, j, base(), base_);
        }
        #pragma endregion
    private:
        date::sys_days base() const {
            return tt_.internal_interval_days().from_ + as_int(base_) * date::days{ 1 };
        }

        #pragma region private_func
        template <bool WithClaszFilter,
            bool WithBikeFilter,
            bool WithCarFilter,
            bool WithWheelchairFilter>
        bool loop_routes(unsigned const k) {
            auto any_marked = false;
            state_.route_mark_.for_each_set_bit([&](auto const r_idx) {
                auto const r = route_idx_t{ r_idx };

                if constexpr (WithClaszFilter) {
                    if (!is_allowed(allowed_claszes_, tt_.route_clasz_[r])) {
                        return;
                    }
                }

                auto section_bike_filter = false;
                if constexpr (WithBikeFilter) {
                    auto const bikes_allowed_on_all_sections =
                        tt_.route_bikes_allowed_.test(r_idx * 2);
                    if (!bikes_allowed_on_all_sections) {
                        auto const bikes_allowed_on_some_sections =
                            tt_.route_bikes_allowed_.test(r_idx * 2 + 1);
                        if (!bikes_allowed_on_some_sections) {
                            return;
                        }
                        section_bike_filter = true;
                    }
                }

                auto section_car_filter = false;
                if constexpr (WithCarFilter) {
                    auto const cars_allowed_on_all_sections =
                        tt_.route_cars_allowed_.test(r_idx * 2);
                    if (!cars_allowed_on_all_sections) {
                        auto const cars_allowed_on_some_sections =
                            tt_.route_cars_allowed_.test(r_idx * 2 + 1);
                        if (!cars_allowed_on_some_sections) {
                            return;
                        }
                        section_car_filter = true;
                    }
                }

                auto section_wheelchair_filter = false;
                if constexpr (WithWheelchairFilter) {
                    auto const wheelchair_accessibility_on_all_sections =
                        tt_.route_wheelchair_accessible_.test(r_idx * 2);
                    if (!wheelchair_accessibility_on_all_sections) {
                        auto const wheelchair_accessibility_on_some_sections =
                            tt_.route_wheelchair_accessible_.test(r_idx * 2 + 1);
                        if (!wheelchair_accessibility_on_some_sections) {
                            return;
                        }
                        section_wheelchair_filter = true;
                    }
                }

                ++stats_.n_routes_visited_;
                trace("┊ ├k={} updating route {}\n", k, r);

                uint8_t const filters =
                    static_cast<uint8_t>(section_bike_filter << 2) |
                    static_cast<uint8_t>(section_car_filter << 1) |
                    static_cast<uint8_t>(section_wheelchair_filter << 0);

                any_marked |= [&]() {
                    switch (filters) {
                    case 0b000: return update_route<false, false, false>(k, r);
                    case 0b001: return update_route<false, false, true>(k, r);
                    case 0b010: return update_route<false, true, false>(k, r);
                    case 0b011: return update_route<false, true, true>(k, r);
                    case 0b100: return update_route<true, false, false>(k, r);
                    case 0b101: return update_route<true, false, true>(k, r);
                    case 0b110: return update_route<true, true, false>(k, r);
                    case 0b111: return update_route<true, true, true>(k, r);
                    default: std::unreachable();
                    }
                    }();
                });
            return any_marked;
        }

        template <bool WithClaszFilter,
            bool WithBikeFilter,
            bool WithCarFilter,
            bool WithWheelchairFilter>
        bool loop_rt_routes(unsigned const k) {
            auto any_marked = false;
            state_.rt_transport_mark_.for_each_set_bit([&](auto const rt_t_idx) {
                auto const rt_t = rt_transport_idx_t{ rt_t_idx };

                if constexpr (WithClaszFilter) {
                    if (!is_allowed(allowed_claszes_,
                        rtt_->rt_transport_section_clasz_[rt_t][0])) {
                        return;
                    }
                }

                auto section_bike_filter = false;
                if constexpr (WithBikeFilter) {
                    auto const bikes_allowed_on_all_sections =
                        rtt_->rt_transport_bikes_allowed_.test(rt_t_idx * 2);
                    if (!bikes_allowed_on_all_sections) {
                        auto const bikes_allowed_on_some_sections =
                            rtt_->rt_transport_bikes_allowed_.test(rt_t_idx * 2 + 1);
                        if (!bikes_allowed_on_some_sections) {
                            return;
                        }
                        section_bike_filter = true;
                    }
                }

                auto section_car_filter = false;
                if constexpr (WithCarFilter) {
                    auto const cars_allowed_on_all_sections =
                        rtt_->rt_transport_cars_allowed_.test(rt_t_idx * 2);
                    if (!cars_allowed_on_all_sections) {
                        auto const cars_allowed_on_some_sections =
                            rtt_->rt_transport_cars_allowed_.test(rt_t_idx * 2 + 1);
                        if (!cars_allowed_on_some_sections) {
                            return;
                        }
                        section_car_filter = true;
                    }
                }

                auto section_wheelchair_filter = false;
                if constexpr (WithWheelchairFilter) {
                    auto const wheelchair_accessible_on_all_sections =
                        rtt_->rt_transport_wheelchair_accessibility_.test(rt_t_idx * 2);
                    if (!wheelchair_accessible_on_all_sections) {
                        auto const wheelchair_accessible_on_some_sections =
                            rtt_->rt_transport_wheelchair_accessibility_.test(rt_t_idx * 2 +
                                1);
                        if (!wheelchair_accessible_on_some_sections) {
                            return;
                        }
                        section_wheelchair_filter = true;
                    }
                }

                ++stats_.n_routes_visited_;
                trace("┊ ├k={} updating rt transport {}\n", k, rt_t);

                uint8_t const filters =
                    static_cast<uint8_t>(section_bike_filter << 2) |
                    static_cast<uint8_t>(section_car_filter << 1) |
                    static_cast<uint8_t>(section_wheelchair_filter << 0);

                any_marked |= [&]() {
                    switch (filters) {
                    case 0b000: return update_rt_transport<false, false, false>(k, rt_t);
                    case 0b001: return update_rt_transport<false, false, true>(k, rt_t);
                    case 0b010: return update_rt_transport<false, true, false>(k, rt_t);
                    case 0b011: return update_rt_transport<false, true, true>(k, rt_t);
                    case 0b100: return update_rt_transport<true, false, false>(k, rt_t);
                    case 0b101: return update_rt_transport<true, false, true>(k, rt_t);
                    case 0b110: return update_rt_transport<true, true, false>(k, rt_t);
                    case 0b111: return update_rt_transport<true, true, true>(k, rt_t);
                    default: std::unreachable();
                    }
                    }();
                });
            return any_marked;
        }

        void update_transfers(unsigned const k) {
            state_.prev_station_mark_.for_each_set_bit([&](auto&& i) {
                for (auto v = 0U; v != Vias + 1; ++v) {
                    auto const tmp_bag = new_tmp_[i][v];
                    if (tmp_bag.is_invalid()) {
                        continue;
                    }

                    auto const is_via = v != Vias && is_via_[v][i];
                    auto const target_v = is_via ? v + 1 : v;
                    auto const is_dest = target_v == Vias && is_dest_[i];
                    auto const stay = is_via ? via_stops_[v].stay_ : 0_minutes;

                    trace(
                        "  loc={}, v={}, tmp={}, is_dest={}, is_via={}, target_v={}, "
                        "stay={}\n",
                        loc{ tt_, location_idx_t{i} }, v, to_unix(tmp_bag.get_any_time()), is_dest, is_via,
                        target_v, stay);

                    auto const transfer_time =
                        (!is_intermodal_dest() && is_dest)
                        ? 0
                        : dir(adjusted_transfer_time(
                            transfer_time_settings_,
                            tt_.locations_.transfer_time_[location_idx_t{ i }]
                            .count()));
                    auto const fp_target_time = tmp_bag.copy(transfer_time + dir(stay.count()));

                    trace(
                        "    transfer_time={}, fp_target_time={}, best@target={}, "
                        "dest={}\n",
                        transfer_time, to_unix(fp_target_time.get_any_time()), to_unix(new_best_[i][target_v].get_any_time()),
                        to_unix(time_at_dest_[k].get_any_time()));

                    if (!new_best_[i][target_v].is_better(fp_target_time) &&
                        fp_target_time.is_better( time_at_dest_[k])) {
                        if (lb_[i] == kUnreachable ||
                            !fp_target_time.is_better_with_offset(static_cast<delta_t>(dir(lb_[i])), time_at_dest_[k])) {
                            ++stats_.fp_update_prevented_by_lower_bound_;
                            return;
                        }

                        //COMMENT: gegeben aus if-Bedingung: fp_target_time < round_times_[k] und best_
                        ++stats_.n_earliest_arrival_updated_by_footpath_;
                        round_times_[k][i][target_v].add(fp_target_time);
                        new_best_[i][target_v].add(fp_target_time);
                        state_.station_mark_.set(i, true);
                        if (is_dest) {
                            update_time_at_dest(k, fp_target_time);
                        }
                    }
                }
                });
        }

        void update_footpaths(unsigned const k, profile_idx_t const prf_idx) {
            state_.prev_station_mark_.for_each_set_bit([&](std::uint64_t const i) {
                auto const l_idx = location_idx_t{ i };
                if constexpr (Rt) {
                    if (prf_idx != 0U && (kFwd ? rtt_->has_td_footpaths_out_
                        : rtt_->has_td_footpaths_in_)[prf_idx]
                        .test(l_idx)) {
                        return;
                    }
                }

                auto const& fps = kFwd ? tt_.locations_.footpaths_out_[prf_idx][l_idx]
                    : tt_.locations_.footpaths_in_[prf_idx][l_idx];

                for (auto const& fp : fps) {
                    ++stats_.n_footpaths_visited_;

                    auto const target = to_idx(fp.target());

                    for (auto v = 0U; v != Vias + 1; ++v) {
                        auto const tmp_bag = new_tmp_[i][v];
                        if (tmp_bag.is_invalid()) {
                            continue;
                        }

                        auto const start_is_via =
                            v != Vias && is_via_[v][static_cast<bitvec::size_type>(i)];
                        auto const start_v = start_is_via ? v + 1 : v;

                        auto const target_is_via =
                            start_v != Vias && is_via_[start_v][target];
                        auto const target_v = target_is_via ? start_v + 1 : start_v;
                        auto stay = 0_minutes;
                        if (start_is_via) {
                            stay += via_stops_[v].stay_;
                        }
                        if (target_is_via) {
                            stay += via_stops_[start_v].stay_;
                        }

                        auto const fp_target_time = tmp_bag.copy(dir(adjusted_transfer_time(transfer_time_settings_,
                            fp.duration().count()) +
                            stay.count()));

                        if (!new_best_[target][target_v].is_better(fp_target_time) &&
                            fp_target_time.is_better( time_at_dest_[k])) {
                            auto const lower_bound = lb_[target];
                            if (lower_bound == kUnreachable ||
                                !fp_target_time.is_better_with_offset(static_cast<delta_t>(dir(lower_bound)), time_at_dest_[k])) {
                                ++stats_.fp_update_prevented_by_lower_bound_;
                                trace_upd(
                                    "┊ ├k={} *** LB NO UPD: (from={}, tmp={}) --{}--> (to={}, "
                                    "best={}) --> update => {}, LB={}, LB_AT_DEST={}, DEST={}\n",
                                    k, loc{ tt_, l_idx }, to_unix(new_tmp_[to_idx(l_idx)][v].get_any_time()),
                                    adjusted_transfer_time(transfer_time_settings_,
                                        fp.duration()),
                                    loc{ tt_, fp.target() }, new_best_[target][target_v].get_any_time(),
                                    to_unix(fp_target_time.get_any_time()), lower_bound,
                                    to_unix(clamp(fp_target_time.get_any_time() + dir(lower_bound))),
                                    to_unix(time_at_dest_[k].get_any_time()));
                                continue;
                            }

                            trace_upd(
                                "┊ ├k={}   footpath: ({}, tmp={}) --{}--> ({}, best={}) --> "
                                "update => {}, v={}->{}, stay={}\n",
                                k, loc{ tt_, l_idx }, to_unix(new_tmp_[to_idx(l_idx)][v].get_any_time()),
                                adjusted_transfer_time(transfer_time_settings_, fp.duration()),
                                loc{ tt_, fp.target() }, to_unix(new_best_[target][target_v].get_any_time()),
                                to_unix(fp_target_time.get_any_time()), v, target_v, stay);

                            //COMMENT: gegeben aus if-Bedingung: fp_target_time < round_times_[k] und best_
                            ++stats_.n_earliest_arrival_updated_by_footpath_;
                            round_times_[k][target][target_v].add(fp_target_time);
                            new_best_[target][target_v].add(fp_target_time);
                            state_.station_mark_.set(target, true);
                            if (target_v == Vias && is_dest_[target]) {
                                update_time_at_dest(k, fp_target_time);
                            }
                        }
                        else {
                            trace(
                                "┊ ├k={}   NO FP UPDATE: {} [best={}] --{}--> {} "
                                "[best={}, time_at_dest={}]\n",
                                k, loc{ tt_, l_idx }, to_unix(new_best_[to_idx(l_idx)][target_v].get_any_time()),
                                adjusted_transfer_time(transfer_time_settings_, fp.duration()),
                                loc{ tt_, fp.target() }, to_unix(new_best_[target][target_v].get_any_time()),
                                to_unix(time_at_dest_[k].get_any_time()));
                        }
                    }
                }
                });
        }

        void update_td_offsets(unsigned const k, profile_idx_t const prf_idx) {
            if constexpr (!Rt) {
                return;
            }

            if (prf_idx == 0U) {
                return;
            }

            state_.prev_station_mark_.for_each_set_bit([&](std::uint64_t const i) {
                auto const l_idx = location_idx_t{ i };
                if (!(kFwd ? rtt_->has_td_footpaths_out_
                    : rtt_->has_td_footpaths_in_)[prf_idx]
                    .test(l_idx)) {
                    return;
                }

                auto const& fps = kFwd ? rtt_->td_footpaths_out_[prf_idx][l_idx]
                    : rtt_->td_footpaths_in_[prf_idx][l_idx];

                for (auto v = 0U; v != Vias + 1; ++v) {
                    auto const tmp_bag = new_tmp_[i][v];
                    if (new_tmp_[i][v].is_invalid()) {
                        continue;
                    }
                    for_each_footpath<
                        SearchDir>(fps, to_unix(tmp_bag.get_any_time()), [&](footpath const fp) {
                        ++stats_.n_footpaths_visited_;

                        auto const target = to_idx(fp.target());

                        auto const start_is_via =
                            v != Vias && is_via_[v][static_cast<bitvec::size_type>(i)];
                        auto const start_v = start_is_via ? v + 1 : v;

                        auto const target_is_via =
                            start_v != Vias && is_via_[start_v][target];
                        auto const target_v = target_is_via ? start_v + 1 : start_v;
                        auto stay = 0_minutes;
                        if (start_is_via) {
                            stay += via_stops_[v].stay_;
                        }
                        if (target_is_via) {
                            stay += via_stops_[start_v].stay_;
                        }

                        auto const fp_target_time = tmp_bag.copy(dir(fp.duration().count() + stay.count()));

                        if (!new_best_[target][target_v].is_better(fp_target_time) &&
                            fp_target_time.is_better(time_at_dest_[k])) {
                            auto const lower_bound = lb_[target];
                            if (lower_bound == kUnreachable ||
                                !fp_target_time.is_better_with_offset(static_cast<delta_t>(dir(lower_bound)), time_at_dest_[k])) {
                                ++stats_.fp_update_prevented_by_lower_bound_;
                                trace_upd(
                                    "┊ ├k={} *** LB NO TD FP UPD: (from={}, tmp={}) --{}--> "
                                    "(to={}, best={}) --> update => {}, LB={}, LB_AT_DEST={}, "
                                    "DEST={}\n",
                                    k, loc{ tt_, l_idx }, to_unix(new_tmp_[to_idx(l_idx)][v].get_any_time()),
                                    fp.duration(), loc{ tt_, fp.target() }, new_best_[target][target_v].get_any_time(),
                                    fp_target_time.get_any_time(), lower_bound,
                                    to_unix(clamp(fp_target_time.get_any_time() + dir(lower_bound))),
                                    to_unix(time_at_dest_[k].get_any_time()));
                                return utl::cflow::kContinue;
                            }

                            trace_upd(
                                "┊ ├k={}   td footpath: ({}, tmp={}) --{}--> ({}, best={}) --> "
                                "update => {}, v={}->{}, stay={}\n",
                                k, loc{ tt_, l_idx }, to_unix(new_tmp_[to_idx(l_idx)][v].get_any_time()),
                                fp.duration(), loc{ tt_, fp.target() },
                                to_unix(new_best_[target][target_v].get_any_time()), to_unix(fp_target_time.get_any_time()), v,
                                target_v, stay);

                            //COMMENT: gegeben aus if-Bedingung: fp_target_time < round_times_[k] und best_
                            ++stats_.n_earliest_arrival_updated_by_footpath_;
                            round_times_[k][target][target_v].add(fp_target_time);
                            new_best_[target][target_v].add(fp_target_time);
                            state_.station_mark_.set(target, true);
                            if (is_dest_[target]) {
                                update_time_at_dest(k, fp_target_time);
                            }
                        }
                        else {
                            trace(
                                "┊ ├k={}   NO TD FP UPDATE: {} [best={}] --{}--> {} "
                                "[best={}, time_at_dest={}]\n",
                                k, loc{ tt_, l_idx }, new_best_[to_idx(l_idx)][v].get_any_time(),
                                adjusted_transfer_time(transfer_time_settings_, fp.duration()),
                                loc{ tt_, fp.target() }, new_best_[target][v].get_any_time(),
                                to_unix(time_at_dest_[k].get_any_time()));
                        }

                        return utl::cflow::kContinue;
                            });
                }
                });
        }

        void update_intermodal_footpaths(unsigned const k) {
            if (dist_to_end_.empty()) {
                return;
            }

            state_.prev_station_mark_.for_each_set_bit([&](auto const i) {
                if (!end_reachable_.test(i)) {
                    trace_upd("┊ ├k={}   no end_reachable: {}\n", k,
                        loc{ tt_, location_idx_t{i} });
                    [[likely]];
                    return;
                }

                trace_upd("┊ ├k={}   end_reachable: {}\n", k,
                    loc{ tt_, location_idx_t{i} });

                auto const l = location_idx_t{ i };
                if (dist_to_end_[i] != std::numeric_limits<std::uint16_t>::max()) {
                    [[likely]];

                    // Case 1: l is last via -> add stay
                    if constexpr (Vias != 0U) {
                        constexpr auto v = Vias - 1U;
                        if (!new_tmp_[i][v].is_invalid() && is_via_[v][i]) {
                            auto const end_time = new_tmp_[i][v].copy(dir(via_stops_[v].stay_.count()), dir(dist_to_end_[i]));

                            trace_upd(
                                "┊ ├k={}, INTERMODAL FOOTPATH FROM LAST VIA: ({}, tmp={}) "
                                "--({} +stay={})--> "
                                "({}, best={})",
                                k, loc{ tt_, l }, to_unix(new_tmp_[to_idx(l)][v].get_any_time()), dist_to_end_[i],
                                via_stops_[v].stay_,
                                loc{ tt_, location_idx_t{kIntermodalTarget} },
                                to_unix(new_best_[kIntermodalTarget][Vias].get_any_time()), to_unix(end_time.get_any_time()));

                            if (!new_best_[kIntermodalTarget][Vias].is_better(end_time)) {
                                round_times_[k][kIntermodalTarget][Vias].add(end_time);
                                new_best_[kIntermodalTarget][Vias].add(end_time);
                                update_time_at_dest(k, end_time);
                                trace_upd(" -> update\n");
                            }
                            else {
                                trace_upd(" -> no update\n");
                            }
                        }
                    }

                    // Case 2: l is no via -> don't add stay
                    auto const tmp_bag = new_tmp_[i][Vias];
                    if (tmp_bag.is_invalid()) {
                        trace_upd("┊ ├k={}, loc={} NOT REACHED\n", k, loc{ tt_, l });
                        return;
                    }

                    auto const end_time = tmp_bag.copy(dir(dist_to_end_[i]));

                    trace_upd(
                        "┊ ├k={}, INTERMODAL FOOTPATH: ({}, tmp={}) --{}--> "
                        "({}, best={})",
                        k, loc{ tt_, l }, to_unix(new_tmp_[to_idx(l)][Vias].get_any_time()), dist_to_end_[i],
                        loc{ tt_, location_idx_t{kIntermodalTarget} },
                        to_unix(new_best_[kIntermodalTarget][Vias].get_any_time()), to_unix(end_time.get_any_time()));

                    if (!new_best_[kIntermodalTarget][Vias].is_better(end_time)) {
                        round_times_[k][kIntermodalTarget][Vias].add(end_time);
                        new_best_[kIntermodalTarget][Vias].add(end_time);
                        update_time_at_dest(k, end_time);
                        trace_upd(" -> update\n");
                    }
                    else {
                        trace_upd(" -> no update\n");
                    }
                }

                if (auto const it = td_dist_to_end_.find(l); it != end(td_dist_to_end_)) {
                    [[unlikely]];

                    auto const fp_start_time = new_tmp_[i][Vias];
                    if (fp_start_time.is_invalid()) {
                        return;
                    }
                    //TODO: für mich zu kompliziert. evtl muss für jeden label ein instanz erstellt werden (siehe: td_footpath.h)
                    //COMMENT: fp bzw. duration wird nur für trace verwendet -> wird erstmal ignoriert
                    auto const fp =
                        get_td_duration<SearchDir>(it->second, to_unix(fp_start_time.get_any_time()));
                    if (fp.has_value()) {
                        auto const& [duration, _] = *fp;
                        auto const end_time = fp_start_time.copy(dir(duration.count()));

                        if (!new_best_[kIntermodalTarget][Vias].is_better(end_time)) {
                            round_times_[k][kIntermodalTarget][Vias].add(end_time);
                            new_best_[kIntermodalTarget][Vias].add(end_time);
                            update_time_at_dest(k, end_time);

                            trace(
                                "┊ │k={}  TD INTERMODAL FOOTPATH: location={}, "
                                "start_time={}, dist_to_end={} --> update to {}\n",
                                k, loc{ tt_, l }, to_unix(fp_start_time.get_any_time()), duration,
                                to_unix(end_time.get_any_time()));
                        }
                        else {
                            trace(
                                "┊ │k={}  TD INTERMODAL FOOTPATH: location={}, "
                                "start_time={}, dist_to_end={} --> NO update to {} best={}\n",
                                k, loc{ tt_, l }, to_unix(fp_start_time.get_any_time()), duration,
                                to_unix(end_time.get_any_time()), new_best_[kIntermodalTarget][Vias].get_any_time());
                        }
                    }
                }
                });
        }

        template <bool WithSectionBikeFilter,
            bool WithSectionCarFilter,
            bool WithSectionWheelchairFilter>
        bool update_rt_transport(unsigned const k, rt_transport_idx_t const rt_t) {
            auto const stop_seq = rtt_->rt_transport_location_seq_[rt_t];
            auto et = std::array<bool, Vias + 1>{};
            auto v_offset = std::array<std::size_t, Vias + 1>{};
            auto any_marked = false;

            for (auto i = 0U; i != stop_seq.size(); ++i) {
                auto const stop_idx =
                    static_cast<stop_idx_t>(kFwd ? i : stop_seq.size() - i - 1U);
                auto const stp = stop{ stop_seq[stop_idx] };
                auto const l_idx = cista::to_idx(stp.location_idx());
                auto const is_first = i == 0U;
                auto const is_last = i == stop_seq.size() - 1U;

                if constexpr (WithSectionBikeFilter) {
                    if (!is_first &&
                        !rtt_->rt_bikes_allowed_per_section_[rt_t][kFwd ? stop_idx - 1
                        : stop_idx]) {
                        et.fill(false);
                        v_offset.fill(0);
                    }
                }

                if constexpr (WithSectionCarFilter) {
                    if (!is_first &&
                        !rtt_->rt_cars_allowed_per_section_[rt_t][kFwd ? stop_idx - 1
                        : stop_idx]) {
                        et.fill(false);
                        v_offset.fill(0);
                    }
                }

                if constexpr (WithSectionWheelchairFilter) {
                    if (!is_first && !rtt_->rt_wheelchair_accessible_per_section_
                        [rt_t][kFwd ? stop_idx - 1 : stop_idx]) {
                        et.fill(false);
                        v_offset.fill(0);
                    }
                }

                if ((kFwd && stop_idx != 0U) ||
                    (kBwd && stop_idx != stop_seq.size() - 1U)) {
                    auto const by_transport = rt_time_at_stop(
                        rt_t, stop_idx, kFwd ? event_type::kArr : event_type::kDep);
                    for (auto j = 0U; j != Vias + 1; ++j) {
                        auto const v = Vias - j;
                        auto target_v = v + v_offset[v];
                        if (et[v] && stp.can_finish<SearchDir>(is_wheelchair_)) {
                            auto const is_via = target_v != Vias && is_via_[target_v][l_idx];
                            auto const is_no_stay_via =
                                is_via && via_stops_[target_v].stay_ == 0_minutes;

                            if (is_no_stay_via) {
                                ++v_offset[v];
                                ++target_v;
                            }

                            auto current_best =
                                get_best(round_times_[k - 1][l_idx][target_v].get_any_time(),
                                    new_tmp_[l_idx][target_v].get_any_time(), new_best_[l_idx][target_v].get_any_time());

                            if (is_better(by_transport, time_at_dest_[k].get_any_time()) &&
                                lb_[l_idx] != kUnreachable &&
                                is_better(by_transport + dir(lb_[l_idx]), time_at_dest_[k].get_any_time())) {
                                trace_upd(
                                    "┊ │k={}    RT | name={}, dbg={}, time_by_transport={}, "
                                    "BETTER THAN current_best={} => update, {} marking station "
                                    "{}!\n",
                                    k, rtt_->default_trip_short_name(tt_, rt_t),
                                    rtt_->dbg(tt_, rt_t), to_unix(by_transport),
                                    to_unix(current_best),
                                    !is_better(by_transport, current_best) ? "NOT" : "",
                                    loc{ tt_, stp.location_idx() });

                                ++stats_.n_earliest_arrival_updated_by_route_;
                                new_tmp_[l_idx][target_v].add(get_best(by_transport));
                                state_.station_mark_.set(l_idx, true);
                                if (is_better(by_transport, current_best)) {
                                    current_best = by_transport;
                                }
                                any_marked = true;
                            }
                        }
                    }
                }

                if (lb_[l_idx] == kUnreachable) {
                    break;
                }

                if (is_last || !(stp.can_start<SearchDir>(is_wheelchair_)) ||
                    !state_.prev_station_mark_[l_idx]) {
                    continue;
                }

                auto const by_transport = rt_time_at_stop(
                    rt_t, stop_idx, kFwd ? event_type::kDep : event_type::kArr);
                for (auto v = 0U; v != Vias + 1; ++v) {
                    auto const target_v = v + v_offset[v];
                    auto const prev_round_time = round_times_[k - 1][l_idx][target_v].get_any_time();
                    if (is_better_or_eq(prev_round_time, by_transport)) {
                        et[v] = true;
                        v_offset[v] = 0;
                    }
                }
            }
            return any_marked;
        }

        template <bool WithSectionBikeFilter,
            bool WithSectionCarFilter,
            bool WithSectionWheelchairFilter>
        bool update_route(unsigned const k, route_idx_t const r) {
            auto const stop_seq = tt_.route_location_seq_[r];
            bool any_marked = false;

            auto et = std::array<transport, Vias + 1>{};
            auto v_offset = std::array<std::size_t, Vias + 1>{};

            for (auto i = 0U; i != stop_seq.size(); ++i) {
                auto const stop_idx =
                    static_cast<stop_idx_t>(kFwd ? i : stop_seq.size() - i - 1U);
                auto const stp = stop{ stop_seq[stop_idx] };
                auto const l_idx = cista::to_idx(stp.location_idx());
                auto const is_first = i == 0U;
                auto const is_last = i == stop_seq.size() - 1U;

                auto current_best = std::array<delta_t, Vias + 1>{};
                current_best.fill(kInvalid);

                // v = via state when entering the transport
                // v + v_offset = via state at the current stop after entering the
                // transport (v_offset > 0 if the transport passes via stops)
                for (auto j = 0U; j != Vias + 1; ++j) {
                    auto const v = Vias - j;
                    if (!et[v].is_valid() && !state_.prev_station_mark_[l_idx]) {
                        trace(
                            "┊ │k={} v={}  stop_idx={} {}: not marked, no et - "
                            "skip\n",
                            k, v, stop_idx, loc{ tt_, location_idx_t{l_idx} });
                        continue;
                    }

                    trace(
                        "┊ │k={} v={}(+{})  stop_idx={}, location={}, round_times={}, "
                        "best={}, "
                        "tmp={}\n",
                        k, v, v_offset[v], stop_idx, loc{ tt_, stp.location_idx() },
                        to_unix(round_times_[k - 1][l_idx][v].get_any_time()), to_unix(new_best_[l_idx][v].get_any_time()),
                        to_unix(new_tmp_[l_idx][v].get_any_time()));

                    if constexpr (WithSectionBikeFilter) {
                        if (!is_first &&
                            !tt_.route_bikes_allowed_per_section_[r][kFwd ? stop_idx - 1
                            : stop_idx]) {
                            et[v] = {};
                            v_offset[v] = 0;
                        }
                    }

                    if constexpr (WithSectionCarFilter) {
                        if (!is_first &&
                            !tt_.route_cars_allowed_per_section_[r][kFwd ? stop_idx - 1
                            : stop_idx]) {
                            et[v] = {};
                            v_offset[v] = 0;
                        }
                    }

                    if constexpr (WithSectionWheelchairFilter) {
                        if (!is_first && !tt_.route_wheelchair_accessibility_per_section_
                            [r][kFwd ? stop_idx - 1 : stop_idx]) {
                            et[v] = {};
                            v_offset[v] = 0;
                        }
                    }

                    auto target_v = v + v_offset[v];

                    if (et[v].is_valid() && stp.can_finish<SearchDir>(is_wheelchair_)) {
                        auto const by_transport = time_at_stop(
                            r, et[v], stop_idx, kFwd ? event_type::kArr : event_type::kDep);

                        auto const is_via = target_v != Vias && is_via_[target_v][l_idx];
                        auto const is_no_stay_via =
                            is_via && via_stops_[target_v].stay_ == 0_minutes;

                        if (Vias != 0) {
                            trace_upd(
                                "┊ │k={} v={}(+{})={} via_count={} is_via_dest={} stay={} "
                                "is_via={} is_dest={}\n",
                                k, v, v_offset[v], target_v, Vias,
                                target_v != Vias ? is_via_[target_v][l_idx] : is_dest_[l_idx],
                                via_stops_[target_v].stay_, is_no_stay_via, is_dest_[l_idx]);
                        }

                        if (is_no_stay_via) {
                            ++v_offset[v];
                            ++target_v;
                        }

                        current_best[v] =
                            get_best(round_times_[k - 1][l_idx][target_v].get_any_time(),
                                new_tmp_[l_idx][target_v].get_any_time(), new_best_[l_idx][target_v].get_any_time());

                        assert(by_transport != std::numeric_limits<delta_t>::min() &&
                            by_transport != std::numeric_limits<delta_t>::max());
                        if (is_better(by_transport, time_at_dest_[k].get_any_time()) &&
                            lb_[l_idx] != kUnreachable &&
                            is_better(by_transport + dir(lb_[l_idx]), time_at_dest_[k].get_any_time())) {
                            trace_upd(
                                "┊ │k={} v={}->{}    name={}, dbg={}, time_by_transport={}, "
                                "BETTER THAN current_best={} => update, {} marking station "
                                "{}!\n",
                                k, v, target_v, tt_.transport_name(et[v].t_idx_),
                                tt_.dbg(et[v].t_idx_), to_unix(by_transport),
                                to_unix(current_best[v]),
                                !is_better(by_transport, current_best[v]) ? "NOT" : "",
                                loc{ tt_, stp.location_idx() });

                            ++stats_.n_earliest_arrival_updated_by_route_;
                            new_tmp_[l_idx][target_v].add(get_best(by_transport));
                            state_.station_mark_.set(l_idx, true);
                            if (is_better(by_transport, current_best[v])) {
                                current_best[v] = by_transport;
                            }
                            any_marked = true;
                        }
                        else {
                            trace(
                                "┊ │k={} v={}->{}    *** NO UPD: at={}, name={}, dbg={}, "
                                "time_by_transport={}, current_best=min({}, {}, {})={} => {} "
                                "- "
                                "LB={}, LB_AT_DEST={}, TIME_AT_DEST={}, "
                                "(is_better(by_transport={}={}, current_best={}={})={}, "
                                "is_better(by_transport={}={}, time_at_dest_={}={})={}, "
                                "reachable={}, "
                                "is_better(lb={}={}, time_at_dest_={}={})={})!\n",
                                k, v, target_v, loc{ tt_, location_idx_t{l_idx} },
                                tt_.transport_name(et[v].t_idx_), tt_.dbg(et[v].t_idx_),
                                to_unix(by_transport),
                                to_unix(round_times_[k - 1][l_idx][target_v].get_any_time()),
                                to_unix(new_best_[l_idx][target_v].get_any_time()), to_unix(new_tmp_[l_idx][target_v].get_any_time()),
                                to_unix(current_best[v]), loc{ tt_, location_idx_t{l_idx} },
                                lb_[l_idx], to_unix(time_at_dest_[k].get_any_time()),
                                to_unix(clamp(by_transport + dir(lb_[l_idx]))), by_transport,
                                to_unix(by_transport), current_best[v],
                                to_unix(current_best[v]),
                                is_better(by_transport, current_best[v]), by_transport,
                                to_unix(by_transport), time_at_dest_[k].get_any_time(),
                                to_unix(time_at_dest_[k].get_any_time()),
                                is_better(by_transport, time_at_dest_[k].get_any_time()),
                                lb_[l_idx] != kUnreachable, by_transport + dir(lb_[l_idx]),
                                to_unix(clamp(by_transport + dir(lb_[l_idx]))),
                                time_at_dest_[k].get_any_time(), to_unix(time_at_dest_[k].get_any_time()),
                                to_unix(time_at_dest_[k].get_any_time()),
                                is_better(clamp(by_transport + dir(lb_[l_idx])),
                                    time_at_dest_[k].get_any_time()));
                        }
                    }
                    else {
                        trace(
                            "┊ │k={} v={}->{}    *** NO UPD: no_trip={}, in_allowed={}, "
                            "out_allowed={}, label_allowed={}\n",
                            k, v, target_v, !et[v].is_valid(), stp.in_allowed(),
                            stp.out_allowed(), (kFwd ? stp.out_allowed() : stp.in_allowed()));
                    }
                }

                if (is_last || !stp.can_start<SearchDir>(is_wheelchair_) ||
                    !state_.prev_station_mark_[l_idx]) {
                    continue;
                }

                if (lb_[l_idx] == kUnreachable) {
                    break;
                }

                for (auto v = 0U; v != Vias + 1; ++v) {
                    if (!et[v].is_valid() && !state_.prev_station_mark_[l_idx]) {
                        continue;
                    }

                    auto const target_v = v + v_offset[v];
                    auto const et_time_at_stop =
                        et[v].is_valid()
                        ? time_at_stop(r, et[v], stop_idx,
                            kFwd ? event_type::kDep : event_type::kArr)
                        : kInvalid;
                    auto const prev_round_time = round_times_[k - 1][l_idx][target_v].get_any_time();
                    if (prev_round_time != kInvalid &&
                        is_better_or_eq(prev_round_time, et_time_at_stop)) {
                        auto const [day, mam] = split(prev_round_time);
                        auto const new_et = get_earliest_transport(k, r, stop_idx, day, mam,
                            stp.location_idx());
                        current_best[v] = get_best(current_best[v], new_best_[l_idx][target_v].get_any_time(),
                            new_tmp_[l_idx][target_v].get_any_time());
                        if (new_et.is_valid() &&
                            (current_best[v] == kInvalid ||
                                is_better_or_eq(
                                    time_at_stop(r, new_et, stop_idx,
                                        kFwd ? event_type::kDep : event_type::kArr),
                                    et_time_at_stop))) {
                            et[v] = new_et;
                            v_offset[v] = 0;
                            trace("┊ │k={} v={}    update et: time_at_stop={}\n", k, v,
                                to_unix(et_time_at_stop));
                        }
                        else if (new_et.is_valid()) {
                            trace("┊ │k={} v={}    update et: no update time_at_stop={}\n", k,
                                v, to_unix(et_time_at_stop));
                        }
                    }
                }
            }
            return any_marked;
        }

        transport get_earliest_transport(unsigned const k,
            route_idx_t const r,
            stop_idx_t const stop_idx,
            day_idx_t const day_at_stop,
            minutes_after_midnight_t const mam_at_stop,
            location_idx_t const l) {
            ++stats_.n_earliest_trip_calls_;

            auto const event_times = tt_.event_times_at_stop(
                r, stop_idx, kFwd ? event_type::kDep : event_type::kArr);

            auto const seek_first_day = [&]() {
                return linear_lb(get_begin_it(event_times), get_end_it(event_times),
                    mam_at_stop,
                    [&](delta const a, minutes_after_midnight_t const b) {
                        return is_better(a.mam(), b.count());
                    });
                };

            trace("┊ │k={}    et: current_best_at_stop={}, stop_idx={}, location={}\n",
                k, tt_.to_unixtime(day_at_stop, mam_at_stop), stop_idx,
                loc{ tt_, stop{tt_.route_location_seq_[r][stop_idx]}.location_idx() });

            auto const n_days_to_iterate = kMaxTravelTime / std::chrono::days{ 1 } + 1U;
            for (auto i = day_idx_t::value_t{ 0U }; i != n_days_to_iterate; ++i) {
                auto const day = kFwd ? day_at_stop + i : day_at_stop - i;

                if (!tt_.is_route_active(r, day)) {
                    continue;
                }

                auto const ev_time_range =
                    it_range{ i == 0U ? seek_first_day() : get_begin_it(event_times),
                             get_end_it(event_times) };
                if (ev_time_range.empty()) {
                    continue;
                }
                for (auto it = begin(ev_time_range); it != end(ev_time_range); ++it) {
                    auto const t_offset =
                        static_cast<std::size_t>(&*it - event_times.data());
                    auto const ev = *it;
                    auto const ev_mam = ev.mam();

                    if (is_better_or_eq(time_at_dest_[k].get_any_time(),
                        to_delta(day, ev_mam) + dir(lb_[to_idx(l)]))) {
                        trace(
                            "┊ │k={}      => name={}, dbg={}, day={}={}, best_mam={}, "
                            "transport_mam={}, transport_time={} => TIME AT DEST {} IS "
                            "BETTER!\n",
                            k, tt_.transport_name(tt_.route_transport_ranges_[r][t_offset]),
                            tt_.dbg(tt_.route_transport_ranges_[r][t_offset]), day,
                            tt_.to_unixtime(day, 0_minutes), mam_at_stop, ev_mam,
                            tt_.to_unixtime(day, duration_t{ ev_mam }),
                            to_unix(time_at_dest_[k].get_any_time()));
                        return { transport_idx_t::invalid(), day_idx_t::invalid() };
                    }

                    auto const t = tt_.route_transport_ranges_[r][t_offset];
                    if (i == 0U && !is_better_or_eq(mam_at_stop.count(), ev_mam)) {
                        trace(
                            "┊ │k={}      => transport={}, name={}, dbg={}, day={}/{}, "
                            "best_mam={}, "
                            "transport_mam={}, transport_time={} => NO REACH!\n",
                            k, t, tt_.transport_name(t), tt_.dbg(t), i, day, mam_at_stop,
                            ev_mam, ev);
                        continue;
                    }

                    auto const ev_day_offset = ev.days();
                    auto const start_day =
                        static_cast<day_idx_t>(as_int(day) - ev_day_offset);
                    if (!is_transport_active(t, start_day)) {
                        trace(
                            "┊ │k={}      => transport={}, name={}, dbg={}, day={}/{}, "
                            "ev_day_offset={}, "
                            "best_mam={}, "
                            "transport_mam={}, transport_time={} => NO TRAFFIC!\n",
                            k, t, tt_.transport_name(t), tt_.dbg(t), i, day, ev_day_offset,
                            mam_at_stop, ev_mam, ev);
                        continue;
                    }

                    trace(
                        "┊ │k={}      => ET FOUND: name={}, dbg={}, at day {} "
                        "(day_offset={}) - ev_mam={}, ev_time={}, ev={}\n",
                        k, tt_.transport_name(t), tt_.dbg(t), day, ev_day_offset, ev_mam,
                        ev, tt_.to_unixtime(day, duration_t{ ev_mam }));
                    return { t, static_cast<day_idx_t>(as_int(day) - ev_day_offset) };
                }
            }
            return {};
        }

        bool is_transport_active(transport_idx_t const t, day_idx_t const day) const {
            if constexpr (Rt) {
                return rtt_->is_transport_active(t, day);
            }
            else {
                return tt_.is_transport_active(t, day);
            }
        }

        delta_t time_at_stop(route_idx_t const r,
            transport const t,
            stop_idx_t const stop_idx,
            event_type const ev_type) {
            return to_delta(t.day_,
                tt_.event_mam(r, t.t_idx_, stop_idx, ev_type).count());
        }

        delta_t rt_time_at_stop(rt_transport_idx_t const rt_t,
            stop_idx_t const stop_idx,
            event_type const ev_type) {
            return to_delta(rtt_->base_day_idx_,
                rtt_->event_time(rt_t, stop_idx, ev_type));
        }

        delta_t to_delta(day_idx_t const day, std::int16_t const mam) {
            return clamp((as_int(day) - as_int(base_)) * 1440 + mam);
        }

        unixtime_t to_unix(delta_t const t) { return delta_to_unix(base(), t); }

        std::pair<day_idx_t, minutes_after_midnight_t> split(delta_t const x) {
            return split_day_mam(base_, x);
        }

        bool is_intermodal_dest() const { return !dist_to_end_.empty(); }

        void update_time_at_dest(unsigned const k, bag const b) {
            if constexpr (SearchMode == search_mode::kOneToAll) {
                return;
            }
            for (auto i = k; i != time_at_dest_.size(); ++i) {
                time_at_dest_[i].add(b);
            }
        }

        int as_int(day_idx_t const d) const { return static_cast<int>(d.v_); }

        template <typename T>
        auto get_begin_it(T const& t) {
            if constexpr (kFwd) {
                return t.begin();
            }
            else {
                return t.rbegin();
            }
        }

        template <typename T>
        auto get_end_it(T const& t) {
            if constexpr (kFwd) {
                return t.end();
            }
            else {
                return t.rend();
            }
        }
        #pragma endregion

        #pragma region members
        timetable const& tt_;
        rt_timetable const* rtt_{ nullptr };
        int n_days_;
        std::uint32_t n_locations_, n_routes_, n_rt_transports_;
        raptor_state& state_;
        bitvec end_reachable_;
        //TODO: replace vector with more memory efficient type; and rename to best_/tmp_
        std::vector<std::array<bag, Vias + 1>> new_tmp_;
        std::vector<std::array<bag, Vias + 1>> new_best_;

        //COMMENT: [n_rows_ -1] -> last matrix_row; [n_columns - 1] -> last Span_entry; [span.size()] -> last array element 
        //COMMENT: kein resize zum befüllen über loops; größe des matrix dest aber über span von state abhängig
        //TODO: workaround:ersetze flat_matrix_view mit vec<vec<arr>>
        //flat_matrix_view<std::array<bag, Vias + 1>> round_times_;
        vecvec< unsigned, std::array<bag, Vias + 1>> round_times_;
        //std::vector<std::vecctor<std::array<bag, Vias + 1>>> round_times_;

        bitvec const& is_dest_;
        std::array<bitvec, kMaxVias> const& is_via_;
        std::vector<std::uint16_t> const& dist_to_end_;
        hash_map<location_idx_t, std::vector<td_offset>> const& td_dist_to_end_;
        std::vector<std::uint16_t> const& lb_;
        std::vector<via_stop> const& via_stops_;

        std::array<bag, kMaxTransfers + 2> time_at_dest_;
        day_idx_t base_;
        raptor_stats stats_;
        clasz_mask_t allowed_claszes_;
        bool require_bike_transport_;
        bool require_car_transport_;
        bool is_wheelchair_;
        transfer_time_settings transfer_time_settings_;
        #pragma endregion
    };

}  // namespace nigiri::routing
