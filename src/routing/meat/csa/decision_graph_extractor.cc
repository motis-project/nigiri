#include "nigiri/routing/meat/csa/decision_graph_extractor.h"

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

  while (i->dep_time_ < when + dur) {
    f(*i);
    ++i;
  }
  f(*i);
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

  // auto heap_begin = queue_.begin();
  // auto heap_end = queue_.begin();
  //
  //// TODO
  // auto heap_order = [&](profile_entry* l,
  //                       profile_entry* r) {  // ??? min-heap
  //   return l->dep_time_ > r->dep_time_;
  // };
  //
  // auto empty = [&]() { return heap_begin == heap_end; };
  //// TODO
  // auto push = [&](profile_entry* exit_conn) {
  //   *heap_end = exit_conn;
  //   ++heap_end;
  //   std::push_heap(heap_begin, heap_end, heap_order);
  // };
  //// TODO
  // auto pop = [&]() {
  //   std::pop_heap(heap_begin, heap_end, heap_order);
  //   --heap_end;
  //   return *heap_end;
  // };
  //  TODO
  auto on_new_relevant_entry = [&](profile_entry const& e) {
    auto conn_rel_day = as_int(split(e.dep_time_).first) % kMaxSearchDays;
    auto conn_bit_idx = conn_rel_day * tt_.fwd_connections_.size() +
                        to_idx(e.ride_.enter_conn_);
    if (!is_enter_conn_relevant_[conn_bit_idx]) {
      relevant.push_back(&e);
      is_enter_conn_relevant_.set(conn_bit_idx);
      if (stop{tt_.fwd_connections_[e.ride_.exit_conn_].arr_stop_}.location_idx() !=
          target_stop) {
        // push(e.ride.exit_conn);
        stack_.push(&e);
      }
    }
  };

  forall_optimal_outgoing_connections(profile_set.for_stop(source_stop),
                                      source_time, 0, on_new_relevant_entry);
  while (!stack_.empty()) {
    // auto& c = n.conn[pop()];
    auto const& p = stack_.top();
    stack_.pop();
    auto p_exit_conn = tt_.fwd_connections_[p->ride_.exit_conn_];
    auto p_enter_conn = tt_.fwd_connections_[p->ride_.enter_conn_];
    auto arr_time =
        p->dep_time_ + (p_exit_conn.arr_time_ - p_enter_conn.dep_time_).count();
    forall_optimal_outgoing_connections(
        profile_set.for_stop(stop{p_exit_conn.arr_stop_}.location_idx()), arr_time,
        tt_.locations_.transfer_time_[stop{p_exit_conn.arr_stop_}.location_idx()]
                .count() +
            max_delay,
        on_new_relevant_entry);
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
  if (profile_set.is_stop_empty(source_stop))
    return decision_graph{
        {{source_stop, {}, {}}, {target_stop, {}, {}}}, {}, 0, 1, -1};

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

  for (auto e : entry_list) {
    auto const& enter_conn = tt_.fwd_connections_[e->ride_.enter_conn_];
    auto const& exit_conn = tt_.fwd_connections_[e->ride_.exit_conn_];

    int dep_node = discover_stop(stop{enter_conn.dep_stop_}.location_idx());
    int arr_node = discover_stop(stop{exit_conn.arr_stop_}.location_idx());
    int arc_id = g.arcs_.size();
    auto trav_time = (exit_conn.arr_time_ - enter_conn.dep_time_).as_duration();
    g.arcs_.push_back({dep_node, arr_node, to_unix(e->dep_time_),
                       to_unix(e->dep_time_) + trav_time, to_unix(e->meat_),
                       e->ride_});
    g.nodes_[dep_node].out_.push_back(arc_id);
    g.nodes_[arr_node].in_.push_back(arc_id);
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