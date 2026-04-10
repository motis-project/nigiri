//! LRU ngram match cache with subset-based incremental update.
//!
//! Mirrors C++ `adr/cache.h`. Stores previous bigram match count vectors
//! keyed by ngram sets, enabling incremental updates when the new query
//! is a superset of a cached query.

use std::collections::{BTreeSet, HashMap, VecDeque};

use crate::types::Ngram;

/// Ordered set of bigram keys. C++ `ngram_set_t = std::set<ngram_t>`.
pub type NgramSet = BTreeSet<Ngram>;

/// Match count vector indexed by `StringIdx`. C++ `string_match_count_vector_t`.
pub type StringMatchCounts = Vec<u8>;

/// Check if `subset` is a subset of `superset`.
fn is_subset(subset: &NgramSet, superset: &NgramSet) -> bool {
    subset.len() <= superset.len() && subset.iter().all(|x| superset.contains(x))
}

/// Return elements in `superset` that are NOT in `subset`.
fn missing_elements(subset: &NgramSet, superset: &NgramSet) -> NgramSet {
    superset.difference(subset).copied().collect()
}

/// LRU cache for ngram match count vectors.
///
/// C++ `adr::cache`. Thread safety: the C++ uses a mutex; callers in Rust
/// should wrap in `Mutex<Cache>` if needed for concurrent access.
pub struct Cache {
    n_strings: usize,
    max_size: usize,
    insert_order: VecDeque<NgramSet>,
    entries: HashMap<NgramSet, StringMatchCounts>,
}

impl Cache {
    pub fn new(n_strings: usize, max_size: usize) -> Self {
        Self {
            n_strings,
            max_size,
            insert_order: VecDeque::new(),
            entries: HashMap::new(),
        }
    }

    /// Insert a match count vector for the given ngram set.
    pub fn add(&mut self, key: NgramSet, v: StringMatchCounts) {
        self.entries.insert(key.clone(), v);
        self.insert_order.push_back(key);

        while self.entries.len() > self.max_size && !self.insert_order.is_empty() {
            let front = self.insert_order.pop_front().unwrap();
            if self.entries.remove(&front).is_some() {
                break;
            }
        }
    }

    /// Find the closest cached superset of `ref_set` and return a copy of its
    /// match counts along with the missing ngrams that still need processing.
    ///
    /// If an exact match exists, returns it with an empty `missing` set.
    pub fn get_closest(&self, ref_set: &NgramSet) -> (StringMatchCounts, NgramSet) {
        // Exact match?
        if let Some(existing) = self.entries.get(ref_set) {
            return (existing.clone(), NgramSet::new());
        }

        // Find largest cached subset of ref_set.
        let mut max_size = 0;
        let mut best: Option<&NgramSet> = None;
        for key in self.entries.keys() {
            if is_subset(key, ref_set) && key.len() > max_size {
                max_size = key.len();
                best = Some(key);
            }
        }

        match best {
            Some(best_key) => {
                let missing = missing_elements(best_key, ref_set);
                let counts = self.entries[best_key].clone();
                (counts, missing)
            }
            None => {
                let missing = ref_set.clone();
                (vec![0u8; self.n_strings], missing)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_exact_hit() {
        let mut cache = Cache::new(10, 5);
        let key: NgramSet = [1, 2, 3].into_iter().collect();
        let counts = vec![0u8; 10];
        cache.add(key.clone(), counts.clone());

        let (result, missing) = cache.get_closest(&key);
        assert!(missing.is_empty());
        assert_eq!(result, counts);
    }

    #[test]
    fn cache_subset_incremental() {
        let mut cache = Cache::new(10, 5);
        let subset: NgramSet = [1, 2].into_iter().collect();
        let mut counts = vec![0u8; 10];
        counts[0] = 5;
        cache.add(subset.clone(), counts.clone());

        let superset: NgramSet = [1, 2, 3].into_iter().collect();
        let (result, missing) = cache.get_closest(&superset);
        assert_eq!(missing.len(), 1);
        assert!(missing.contains(&3));
        assert_eq!(result[0], 5);
    }

    #[test]
    fn cache_eviction() {
        let mut cache = Cache::new(10, 2);
        let k1: NgramSet = [1].into_iter().collect();
        let k2: NgramSet = [2].into_iter().collect();
        let k3: NgramSet = [3].into_iter().collect();

        cache.add(k1.clone(), vec![1u8; 10]);
        cache.add(k2.clone(), vec![2u8; 10]);
        cache.add(k3.clone(), vec![3u8; 10]);

        // k1 should have been evicted
        assert!(cache.entries.len() <= 2);
    }
}
