#include "nigiri/routing/meat/csa/decision_graph_extractor.h"

#include "utl/overloaded.h"

#include "nigiri/routing/journey.h"
#include "nigiri/routing/meat/compact_representation.h"
#include "nigiri/routing/meat/csa/binary_search.h"

namespace nigiri::routing::meat::csa {

namespace {
template <class F>
void forall_optimal_outgoing_connections(profile const& p,
                                         delta_t when,
                                         delta_t dur,
                                         const F& f) {
  auto i = binary_find_first_true(
      std::begin(p), std::end(p),
      [&](profile_entry e) { return when <= e.dep_time_; });

  if (i == std::end(p)) {
    return;
  }
  while (i->dep_time_ < when + dur) {
    f(i);
    ++i;
  }
  f(i);
}
}  // namespace

std::vector<profile_entry const*>
decision_graph_extractor::extract_relevant_entries(
    profile_set const& profile_set,
    location_idx_t source_stop,
    delta_t source_time,
    location_idx_t target_stop,
    delta_t max_delay) const {
  std::vector<profile_entry const*> relevant;
  is_enter_conn_relevant_.resize(profile_set.n_entry_idxs());

  auto on_new_relevant_entry =
      [&](std::reverse_iterator<
          std::vector<profile_entry>::const_iterator> const& e_it) {
        auto const& e = *e_it;
        auto e_bit_idx = profile_set.global_index_of(e_it);
        assert(e.dep_time_ == profile_set.entry_[e_bit_idx].dep_time_ &&
               e.meat_ == profile_set.entry_[e_bit_idx].meat_);
        assert(&(*e_it) == &profile_set.entry_[e_bit_idx]);
        if (!is_enter_conn_relevant_[e_bit_idx]) {
          relevant.push_back(&(*e_it));
          is_enter_conn_relevant_.set(e_bit_idx);
          std::visit(
              utl::overloaded{
                  [&](walk const& w) {
                    if (w.fp_.target() != target_stop) {
                      stack_.push(&(*e_it));
                    }
                  },
                  [&](ride const& r) {
                    if (stop{tt_.fwd_connections_[r.exit_conn_].arr_stop_}
                            .location_idx() != target_stop) {
                      stack_.push(&(*e_it));
                    }
                  }},
              e.uses_);
          // if (stop{tt_.fwd_connections_[e.ride_.exit_conn_].arr_stop_}
          //         .location_idx() != target_stop) {
          //   // push(e.ride.exit_conn);
          //   stack_.push(*e_it);

          // // TODO footpath
          // if(fp_dis_to_target_[stop{tt_.fwd_connections_[e.ride_.exit_conn_].arr_stop_}.location_idx()]
          // != std::numeric_limits<meat_t>::infinity() &&
          // fp_dis_to_target_[stop{tt_.fwd_connections_[e.ride_.exit_conn_].arr_stop_}.location_idx()]
          // + e.arr_time_ == e.meat_){
          //   relevant.push_back(/*add final footpath*/);
          // }
          // //
          // }
        }
      };

  forall_optimal_outgoing_connections(profile_set.for_stop(source_stop),
                                      source_time, 0, on_new_relevant_entry);
  while (!stack_.empty()) {
    auto const& pe = stack_.top();

    std::visit(
        utl::overloaded{
            [&](walk const& w) {
              delta_t arr_time = pe->dep_time_ + w.fp_.duration().count();
              stack_.pop();
              forall_optimal_outgoing_connections(
                  profile_set.for_stop(w.fp_.target()), arr_time,
                  tt_.locations_.transfer_time_[w.fp_.target()].count() +
                      max_delay,
                  on_new_relevant_entry);
            },
            [&](ride const& r) {
              auto p_exit_conn = tt_.fwd_connections_[r.exit_conn_];
              auto p_exit_stop_idx = stop{p_exit_conn.arr_stop_}.location_idx();
              auto p_enter_conn = tt_.fwd_connections_[r.enter_conn_];
              delta_t arr_time =
                  pe->dep_time_ +
                  (p_exit_conn.arr_time_ - p_enter_conn.dep_time_).count();
              stack_.pop();
              forall_optimal_outgoing_connections(
                  profile_set.for_stop(p_exit_stop_idx), arr_time,
                  tt_.locations_.transfer_time_[p_exit_stop_idx].count() +
                      max_delay,
                  on_new_relevant_entry);
            }},
        pe->uses_);

    // if (profile_set.fp_dis_to_target_[p_exit_conn_idx] !=
    // std::numeric_limits<meat_t>::infinity() &&
    // profile_set.fp_dis_to_target_[p_exit_conn_idx] + arr_time == p.meat_){
    //   relevant.push_back(/*add final footpath*/); // TODO relevant hat nur
    //   ref auf profile_entry, ändern oder doch erst später anfügen
    // }
  }

  is_enter_conn_relevant_.reset();

  return relevant;
}

decision_graph decision_graph_extractor::operator()(
    profile_set const& profile_set,
    location_idx_t source_stop,
    delta_t source_time,
    location_idx_t target_stop,
    delta_t max_delay) const {
  if (profile_set.is_stop_empty(source_stop)) {
    return decision_graph{
        {{source_stop, {}, {}}, {target_stop, {}, {}}}, {}, 0, 1, -1};
  }

  auto entry_list = extract_relevant_entries(
      profile_set, source_stop, source_time, target_stop, max_delay);

  decision_graph g;
  int node_count = 0;

  auto discover_stop = [&](location_idx_t s) {
    if (to_node_id_[s] == -1) {
      g.nodes_.push_back({s, {}, {}});
      to_node_id_[s] = node_count++;
    }
    return to_node_id_[s];
  };

  for (auto const& e : entry_list) {
    std::visit(
        utl::overloaded{
            [&](walk const& w) {
              int dep_node = discover_stop(w.from_);
              int arr_node = discover_stop(w.fp_.target());
              int arc_id = g.arcs_.size();
              delta_t arr_time = e->dep_time_ + w.fp_.duration().count();
              g.arcs_.push_back({dep_node, arr_node, to_unix(e->dep_time_),
                                 to_unix(arr_time), to_unix(e->meat_), w.fp_});
              g.nodes_[dep_node].out_.push_back(arc_id);
              g.nodes_[arr_node].in_.push_back(arc_id);
            },
            [&](ride const& r) {
              auto const& enter_conn = tt_.fwd_connections_[r.enter_conn_];
              auto const& exit_conn = tt_.fwd_connections_[r.exit_conn_];

              int dep_node =
                  discover_stop(stop{enter_conn.dep_stop_}.location_idx());
              int arr_node =
                  discover_stop(stop{exit_conn.arr_stop_}.location_idx());
              int arc_id = g.arcs_.size();
              delta_t arr_time =
                  e->dep_time_ +
                  (exit_conn.arr_time_ - enter_conn.dep_time_).count();
              auto [day, mam] = split(e->dep_time_);
              day = day - enter_conn.dep_time_.days();
              g.arcs_.push_back(
                  {dep_node, arr_node, to_unix(e->dep_time_), to_unix(arr_time),
                   to_unix(e->meat_),
                   journey::run_enter_exit{
                       {.t_ = transport{enter_conn.transport_idx_, day},
                        .stop_range_ = interval<stop_idx_t>{0, 0}},
                       enter_conn.trip_con_idx_,
                       static_cast<stop_idx_t>(exit_conn.trip_con_idx_ + 1)}});
              g.nodes_[dep_node].out_.push_back(arc_id);
              g.nodes_[arr_node].in_.push_back(arc_id);

              auto fp_dis_to_target =
                  profile_set.fp_dis_to_target_[stop{exit_conn.arr_stop_}
                                                    .location_idx()];
              bool add_final_footpath =
                  stop{exit_conn.arr_stop_}.location_idx() != target_stop &&
                  fp_dis_to_target != std::numeric_limits<meat_t>::infinity() &&
                  fp_dis_to_target + arr_time == e->meat_;
              if (add_final_footpath) {
                dep_node = arr_node;
                arr_node = discover_stop(target_stop);
                auto dep_time = arr_time;
                auto arr_time = e->meat_;
                arc_id = g.arcs_.size();
                g.arcs_.push_back(
                    {dep_node, arr_node, to_unix(dep_time), to_unix(arr_time),
                     to_unix(e->meat_),
                     footpath(target_stop, duration_t{static_cast<delta_t>(
                                               fp_dis_to_target)})});
                g.nodes_[dep_node].out_.push_back(arc_id);
                g.nodes_[arr_node].in_.push_back(arc_id);
              }
            }},
        e->uses_);
  }

  g.source_node_ = to_node_id_[source_stop];
  g.target_node_ = to_node_id_[target_stop];
  g.first_arc_ = 0;
  {
    auto min_dep_time = g.arcs_[0].dep_time_;
    for (auto i = 1; i < g.arc_count(); ++i)
      if (g.arcs_[i].dep_time_ < min_dep_time) {
        min_dep_time = g.arcs_[i].dep_time_;
        g.first_arc_ = i;
      }
  }

  for (auto x : g.nodes_) to_node_id_[x.stop_id_] = -1;

  return g;
}

std::pair<decision_graph, delta_t> extract_small_sub_decision_graph(
    decision_graph_extractor const& e,
    profile_set const& profile_set,
    location_idx_t source_stop,
    delta_t source_time,
    location_idx_t target_stop,
    delta_t max_delay,
    int max_ride_count,
    int max_arrow_count) {

  if (max_ride_count == std::numeric_limits<int>::max() &&
      max_arrow_count == std::numeric_limits<int>::max())
    return {e(profile_set, source_stop, source_time, target_stop, max_delay),
            max_delay};
  else if (max_arrow_count == std::numeric_limits<int>::max()) {
    delta_t min_delay = 0;
    while (min_delay != max_delay) {
      delta_t mid_delay = (min_delay + max_delay + 1) / 2;
      auto g = e(profile_set, source_stop, source_time, target_stop, mid_delay);
      if (g.arc_count() <= max_ride_count) {
        min_delay = mid_delay;
      } else {
        max_delay = mid_delay - 1;
      }
    }
    return {e(profile_set, source_stop, source_time, target_stop, max_delay),
            max_delay};
  } else {
    delta_t min_delay = 0;
    while (min_delay != max_delay) {
      delta_t mid_delay = (min_delay + max_delay + 1) / 2;
      auto g = e(profile_set, source_stop, source_time, target_stop, mid_delay);
      if (g.arc_count() <= max_ride_count) {
        if (compact_representation(g).arrow_count() <= max_arrow_count)
          min_delay = mid_delay;
        else
          max_delay = mid_delay - 1;
      } else {
        max_delay = mid_delay - 1;
      }
    }
    return {e(profile_set, source_stop, source_time, target_stop, max_delay),
            max_delay};
  }
}
}  // namespace nigiri::routing::meat::csa