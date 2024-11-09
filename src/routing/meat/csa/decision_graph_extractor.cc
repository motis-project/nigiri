#include "nigiri/routing/meat/csa/decision_graph_extractor.h"

#include <cmath>

#include "utl/overloaded.h"

#include "nigiri/routing/journey.h"
#include "nigiri/routing/meat/binary_search.h"

namespace nigiri::routing::meat::csa {

namespace {
template <class F>
void for_first_optimal_outgoing_connection(profile const& p,
                                           delta_t when,
                                           const F& f) {
  auto i = binary_find_first_true(
      std::begin(p), std::end(p),
      [&](profile_entry const& e) { return when <= e.dep_time_; });
  f(i);
}

template <class F>
void forall_optimal_outgoing_connections(profile const& p,
                                         delta_t when,
                                         delta_t dur,
                                         const F& f) {
  auto i = binary_find_first_true(
      std::begin(p), std::end(p),
      [&](profile_entry const& e) { return when <= e.dep_time_; });

  while (i->dep_time_ <= when + dur) {
    f(i);
    ++i;
  }
  if (i == (std::end(p) - 1)) {
    return;
  }
  f(i);
}
}  // namespace

template <typename ProfileSet>
std::vector<profile_entry const*>
decision_graph_extractor<ProfileSet>::extract_relevant_entries(
    location_idx_t source_stop,
    delta_t source_time,
    location_idx_t target_stop,
    delta_t max_delay) const {
  std::vector<profile_entry const*> relevant;
  is_enter_conn_relevant_.resize(
      static_cast<bitvec::size_type>(profile_set_.n_entry_idxs()));

  auto on_new_relevant_entry =
      [&](std::reverse_iterator<
          std::vector<profile_entry>::const_iterator> const& e_it) {
        auto const& e = *e_it;
        auto e_bit_idx = profile_set_.global_index_of(e_it);
        if (!is_enter_conn_relevant_[static_cast<bitvec::size_type>(
                e_bit_idx)]) {
          relevant.push_back(&(*e_it));
          is_enter_conn_relevant_.set(
              static_cast<bitvec::size_type>(e_bit_idx));
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
        }
      };

  assert(!profile_set_.is_stop_empty(source_stop));
  for_first_optimal_outgoing_connection(profile_set_.for_stop(source_stop),
                                        source_time, on_new_relevant_entry);
  while (!stack_.empty()) {
    auto const& pe = stack_.top();

    std::visit(
        utl::overloaded{
            [&](walk const& w) {
              delta_t arr_time = pe->dep_time_ + w.fp_.duration().count();
              stack_.pop();
              forall_optimal_outgoing_connections(
                  profile_set_.for_stop(w.fp_.target()), arr_time, 0,
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
                  profile_set_.for_stop(p_exit_stop_idx), arr_time,
                  tt_.locations_.transfer_time_[p_exit_stop_idx].count() +
                      max_delay,
                  on_new_relevant_entry);
            }},
        pe->uses_);
  }

  return relevant;
}

template <typename ProfileSet>
decision_graph decision_graph_extractor<ProfileSet>::operator()(
    location_idx_t source_stop,
    delta_t source_time,
    location_idx_t target_stop,
    delta_t max_delay) const {
  if (profile_set_.is_stop_empty(source_stop)) {
    // TODO remove
    assert(false && "Should not be empty");
    return decision_graph{{{source_stop, {}, {}}, {target_stop, {}, {}}},
                          {},
                          dg_node_idx_t{0},
                          dg_node_idx_t{1},
                          dg_arc_idx_t::invalid()};
  }

  auto entry_list = extract_relevant_entries(source_stop, source_time,
                                             target_stop, max_delay);

  // TODO sort entry_list by e.dep_time_? then g.arcs_ and g.nodes_[x].out_
  // are sorted by dep_time_, except the ones added by add_final_fps()

  auto g = decision_graph{};
  auto node_count = dg_node_idx_t{0};

  auto discover_stop = [&](location_idx_t s) {
    if (to_node_id_[s] == dg_node_idx_t::invalid()) {
      g.nodes_.emplace_back(s, vector_map<dg_arc_2idx_t, dg_arc_idx_t>{},
                            vector_map<dg_arc_2idx_t, dg_arc_idx_t>{});
      to_node_id_[s] = node_count++;
    }
    return to_node_id_[s];
  };

  for (auto const& e : entry_list) {
    std::visit(
        utl::overloaded{
            [&](walk const& w) {
              auto dep_node = discover_stop(w.from_);
              auto arr_node = discover_stop(w.fp_.target());
              auto arc_id = dg_arc_idx_t{g.arcs_.size()};
              delta_t arr_time = e->dep_time_ + w.fp_.duration().count();
              g.arcs_.emplace_back(dep_node, arr_node, to_unix(e->dep_time_),
                                   to_unix(arr_time), to_unix(e->meat_), w.fp_,
                                   0.0);
              g.nodes_[dep_node].out_.push_back(arc_id);
              g.nodes_[arr_node].in_.push_back(arc_id);
            },
            [&](ride const& r) {
              auto const& enter_conn = tt_.fwd_connections_[r.enter_conn_];
              auto const& exit_conn = tt_.fwd_connections_[r.exit_conn_];

              auto dep_node =
                  discover_stop(stop{enter_conn.dep_stop_}.location_idx());
              auto arr_node =
                  discover_stop(stop{exit_conn.arr_stop_}.location_idx());
              auto arc_id = dg_arc_idx_t{g.arcs_.size()};
              delta_t arr_time =
                  e->dep_time_ +
                  (exit_conn.arr_time_ - enter_conn.dep_time_).count();
              auto [day, mam] = split(e->dep_time_);
              assert(enter_conn.dep_time_.days() >= 0);
              day = static_cast<day_idx_t>(as_int(day) -
                                           enter_conn.dep_time_.days());
              g.arcs_.emplace_back(
                  dep_node, arr_node, to_unix(e->dep_time_), to_unix(arr_time),
                  to_unix(e->meat_),
                  journey::run_enter_exit{
                      {.t_ = transport{enter_conn.transport_idx_, day},
                       .stop_range_ = interval<stop_idx_t>{0, 0}},
                      enter_conn.trip_con_idx_,
                      static_cast<stop_idx_t>(exit_conn.trip_con_idx_ + 1)},
                  0.0);
              g.nodes_[dep_node].out_.push_back(arc_id);
              g.nodes_[arr_node].in_.push_back(arc_id);
            }},
        e->uses_);
  }

  g.source_node_ = to_node_id_[source_stop];
  g.target_node_ = discover_stop(target_stop);
  add_final_fps(g, target_stop, g.target_node_);

  g.first_arc_ = dg_arc_idx_t{0};
  {
    auto min_dep_time = g.arcs_[dg_arc_idx_t{0}].dep_time_;
    for (auto i = dg_arc_idx_t{1}; i < g.arc_count(); ++i)
      if (g.arcs_[i].dep_time_ < min_dep_time) {
        min_dep_time = g.arcs_[i].dep_time_;
        g.first_arc_ = i;
      }
  }

  for (auto x : g.nodes_) to_node_id_[x.stop_id_] = dg_node_idx_t::invalid();

  return g;
}

template <typename ProfileSet>
void decision_graph_extractor<ProfileSet>::add_final_fps(
    decision_graph& g,
    location_idx_t target_stop,
    dg_node_idx_t target_node_id) const {
  for (auto const& n : g.nodes_) {
    auto const fp_dis_to_target = profile_set_.fp_dis_to_target_[n.stop_id_];
    if (n.stop_id_ == target_stop ||
        (!std::isfinite(fp_dis_to_target) && !std::signbit(fp_dis_to_target))) {
      continue;
    }
    auto fp_dep_s = std::set<unixtime_t>{};
    auto const in_size = n.in_.size();
    for (auto in_idx = dg_arc_2idx_t{0}; in_idx < in_size; ++in_idx) {
      auto const a_idx = n.in_[in_idx];
      auto const& a = g.arcs_[a_idx];
      bool const add_final_footpath =
          !std::holds_alternative<footpath>(a.uses_) &&
          to_duration(fp_dis_to_target) + a.arr_time_ == a.meat_;
      if (!add_final_footpath) {
        continue;
      }
      auto const dep_node = a.arr_node_;
      auto const arr_node = target_node_id;
      auto const dep_time = a.arr_time_;
      auto const arr_time = a.meat_;
      if (!fp_dep_s.insert(dep_time).second) {
        continue;
      }

      auto const arc_id = dg_arc_idx_t{g.arcs_.size()};
      g.arcs_.emplace_back(dep_node, arr_node, dep_time, arr_time, a.meat_,
                           footpath(target_stop, to_duration(fp_dis_to_target)),
                           0.0);
      g.nodes_[dep_node].out_.push_back(arc_id);
      g.nodes_[arr_node].in_.push_back(arc_id);
    }
  }
}

template struct decision_graph_extractor<static_profile_set>;
template struct decision_graph_extractor<dynamic_profile_set>;
template struct decision_graph_extractor<dynamic_growth_profile_set>;

}  // namespace nigiri::routing::meat::csa