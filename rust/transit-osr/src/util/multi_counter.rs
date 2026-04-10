//! Translation of osr/include/osr/util/multi_counter.h
//!
//! Multi-counter tracks elements that appear multiple times.
//! Uses two bitvecs: once_ (seen at least once) and multi_ (seen more than once).

use bitvec::prelude::*;

/// Counter that tracks which indices have been seen once vs multiple times
#[derive(Debug, Clone)]
pub struct MultiCounter {
    /// Bits set for indices seen at least once
    once: BitVec,
    /// Bits set for indices seen more than once
    multi: BitVec,
}

impl MultiCounter {
    /// Create a new multi-counter
    pub fn new() -> Self {
        Self {
            once: BitVec::new(),
            multi: BitVec::new(),
        }
    }

    /// Create with reserved capacity
    pub fn with_capacity(size: usize) -> Self {
        Self {
            once: BitVec::with_capacity(size),
            multi: BitVec::with_capacity(size),
        }
    }

    /// Check if index has been seen multiple times
    pub fn is_multi(&self, i: usize) -> bool {
        self.multi.get(i).map(|b| *b).unwrap_or(false)
    }

    /// Increment count for index i
    pub fn increment(&mut self, i: usize) {
        // Resize if needed
        let new_size = i + 1;
        if self.once.len() < new_size {
            self.once.resize(new_size, false);
        }
        if self.multi.len() < new_size {
            self.multi.resize(new_size, false);
        }

        // First time: set once
        // Second time: set multi
        if self.once[i] {
            if !self.multi[i] {
                self.multi.set(i, true);
            }
        } else {
            self.once.set(i, true);
        }
    }

    /// Get the size (highest index + 1)
    pub fn size(&self) -> usize {
        self.once.len()
    }

    /// Reserve capacity
    pub fn reserve(&mut self, size: usize) {
        if self.once.len() < size {
            self.once.resize(size, false);
        }
        if self.multi.len() < size {
            self.multi.resize(size, false);
        }
    }
}

impl Default for MultiCounter {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_counter_single() {
        let mut counter = MultiCounter::new();
        counter.increment(5);

        assert!(!counter.is_multi(5));
        assert_eq!(counter.size(), 6);
    }

    #[test]
    fn test_multi_counter_double() {
        let mut counter = MultiCounter::new();
        counter.increment(5);
        counter.increment(5);

        assert!(counter.is_multi(5));
    }

    #[test]
    fn test_multi_counter_triple() {
        let mut counter = MultiCounter::new();
        counter.increment(3);
        counter.increment(3);
        counter.increment(3);

        assert!(counter.is_multi(3));
    }

    #[test]
    fn test_multi_counter_different_indices() {
        let mut counter = MultiCounter::new();
        counter.increment(1);
        counter.increment(2);
        counter.increment(2);
        counter.increment(3);
        counter.increment(3);
        counter.increment(3);

        assert!(!counter.is_multi(1));
        assert!(counter.is_multi(2));
        assert!(counter.is_multi(3));
    }

    #[test]
    fn test_multi_counter_with_capacity() {
        let mut counter = MultiCounter::with_capacity(100);
        counter.increment(50);
        counter.increment(50);

        assert!(counter.is_multi(50));
    }
}
