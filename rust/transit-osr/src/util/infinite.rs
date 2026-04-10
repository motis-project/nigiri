//! Translation of osr/include/osr/util/infinite.h
//!
//! Infinite iterator that wraps around when reaching the end.
//! Used for routing algorithms that need to iterate indefinitely.

use std::iter::Iterator;

/// Iterator that can wrap around to the beginning when is_infinite is true
pub struct InfiniteIterator<I: Iterator> {
    items: Vec<I::Item>,
    index: usize,
    is_infinite: bool,
}

impl<I> InfiniteIterator<I>
where
    I: Iterator,
    I::Item: Clone,
{
    /// Create a new infinite iterator
    pub fn new(iter: I, is_infinite: bool) -> Self {
        let items: Vec<I::Item> = iter.collect();
        Self {
            items,
            index: 0,
            is_infinite,
        }
    }
}

impl<I> Iterator for InfiniteIterator<I>
where
    I: Iterator,
    I::Item: Clone,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if self.items.is_empty() {
            return None;
        }

        if self.index >= self.items.len() {
            if self.is_infinite {
                self.index = 0; // Wrap around
            } else {
                return None; // Stop at end
            }
        }

        let result = self.items.get(self.index).cloned();
        self.index += 1;

        result
    }
}

/// Create an infinite iterator from a collection
pub fn infinite<I>(iter: I, is_infinite: bool) -> InfiniteIterator<I>
where
    I: Iterator,
    I::Item: Clone,
{
    InfiniteIterator::new(iter, is_infinite)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infinite_wraps_around() {
        let data = vec![1, 2, 3];
        let mut iter = infinite(data.into_iter(), true);

        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), Some(3));
        // Should wrap around
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(2));
    }

    #[test]
    fn test_non_infinite_stops() {
        let data = vec![1, 2, 3];
        let mut iter = infinite(data.into_iter(), false);

        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), Some(3));
        // Should stop
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_empty_iterator() {
        let data: Vec<i32> = vec![];
        let mut iter = infinite(data.into_iter(), true);

        assert_eq!(iter.next(), None);
    }
}
