//! Translation of osr/include/osr/routing/dial.h
//!
//! Dial's algorithm priority queue - specialized for integer costs.
//! Uses bucketing to achieve O(1) push and pop for bounded integer priorities.

/// Dial priority queue for items with integer costs
///
/// This is a bucket-based priority queue optimized for routing with integer costs.
/// Instead of heap operations, items are placed in buckets based on their cost.
/// Popping extracts from the lowest non-empty bucket.
///
/// Complexity:
/// - push: O(1)
/// - pop: O(1) amortized
/// - Space: O(max_cost)
pub struct Dial<T, F>
where
    F: Fn(&T) -> usize,
{
    get_bucket: F,
    current_bucket: usize,
    size: usize,
    /// Contiguous allocation of bucket vectors, matching C++ std::array<std::vector, N>.
    /// Using Box<[Vec<T>]> instead of Vec<Vec<T>> eliminates one level of heap indirection.
    buckets: Box<[Vec<T>]>,
}

impl<T, F> Dial<T, F>
where
    F: Fn(&T) -> usize,
{
    /// Create a new Dial priority queue with the given bucket function
    pub fn new(get_bucket: F) -> Self {
        Self {
            get_bucket,
            current_bucket: 0,
            size: 0,
            buckets: Box::new([]),
        }
    }

    /// Push an element into the queue
    ///
    /// The element is placed in a bucket determined by get_bucket(el).
    /// Panics if the bucket index >= n_buckets.
    pub fn push(&mut self, el: T) {
        let dist = (self.get_bucket)(&el);
        assert!(
            dist < self.buckets.len(),
            "Bucket {} out of range (max: {})",
            dist,
            self.buckets.len()
        );

        self.buckets[dist].push(el);
        self.current_bucket = self.current_bucket.min(dist);
        self.size += 1;
    }

    /// Pop the element with the lowest cost
    ///
    /// Panics if the queue is empty.
    pub fn pop(&mut self) -> T {
        assert!(!self.empty(), "Cannot pop from empty Dial queue");

        self.current_bucket = self.get_next_bucket();
        assert!(!self.buckets[self.current_bucket].is_empty());

        let item = self.buckets[self.current_bucket].pop().unwrap();
        self.size -= 1;
        item
    }

    /// Returns the number of elements in the queue
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns true if the queue is empty
    pub fn empty(&self) -> bool {
        self.size == 0
    }

    /// Clear all elements from the queue
    pub fn clear(&mut self) {
        self.current_bucket = 0;
        self.size = 0;
        for bucket in &mut self.buckets {
            bucket.clear();
        }
    }

    /// Set the number of buckets (reallocates as a contiguous Box<[Vec<T>]>)
    pub fn set_n_buckets(&mut self, n: usize) {
        if self.buckets.len() == n {
            // Same size — just clear existing buckets
            self.clear();
            return;
        }
        self.current_bucket = 0;
        self.size = 0;
        self.buckets = (0..n).map(|_| Vec::new()).collect::<Vec<_>>().into_boxed_slice();
    }

    /// Get the current number of buckets
    pub fn n_buckets(&self) -> usize {
        self.buckets.len()
    }

    /// Find the next non-empty bucket starting from current_bucket
    ///
    /// Panics if the queue is empty (size == 0).
    fn get_next_bucket(&self) -> usize {
        assert!(self.size != 0, "get_next_bucket called on empty queue");

        let mut bucket = self.current_bucket;
        while bucket < self.buckets.len() && self.buckets[bucket].is_empty() {
            bucket += 1;
        }
        bucket
    }

    /// Peek the index of the next non-empty bucket without asserting.
    /// Returns `None` if the queue is empty or no non-empty bucket found.
    pub fn peek_min_bucket(&self) -> Option<usize> {
        if self.size == 0 {
            return None;
        }
        let mut bucket = self.current_bucket;
        while bucket < self.buckets.len() && self.buckets[bucket].is_empty() {
            bucket += 1;
        }
        if bucket < self.buckets.len() {
            Some(bucket)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq)]
    struct Item {
        cost: usize,
        value: i32,
    }

    fn get_cost(item: &Item) -> usize {
        item.cost
    }

    #[test]
    fn test_dial_basic() {
        let mut dial: Dial<Item, _> = Dial::new(get_cost);
        dial.set_n_buckets(100);

        assert!(dial.empty());
        assert_eq!(dial.size(), 0);

        dial.push(Item { cost: 5, value: 1 });
        dial.push(Item { cost: 3, value: 2 });
        dial.push(Item { cost: 7, value: 3 });

        assert_eq!(dial.size(), 3);
        assert!(!dial.empty());

        // Should pop in cost order
        let item1 = dial.pop();
        assert_eq!(item1.cost, 3);

        let item2 = dial.pop();
        assert_eq!(item2.cost, 5);

        let item3 = dial.pop();
        assert_eq!(item3.cost, 7);

        assert!(dial.empty());
    }

    #[test]
    fn test_dial_same_bucket() {
        let mut dial: Dial<Item, _> = Dial::new(get_cost);
        dial.set_n_buckets(100);

        // Multiple items in same bucket
        dial.push(Item { cost: 5, value: 1 });
        dial.push(Item { cost: 5, value: 2 });
        dial.push(Item { cost: 5, value: 3 });

        assert_eq!(dial.size(), 3);

        // All have same cost, order within bucket is LIFO
        let item1 = dial.pop();
        assert_eq!(item1.value, 3);

        let item2 = dial.pop();
        assert_eq!(item2.value, 2);

        let item3 = dial.pop();
        assert_eq!(item3.value, 1);
    }

    #[test]
    fn test_dial_clear() {
        let mut dial: Dial<Item, _> = Dial::new(get_cost);
        dial.set_n_buckets(100);

        dial.push(Item { cost: 5, value: 1 });
        dial.push(Item { cost: 10, value: 2 });

        assert_eq!(dial.size(), 2);

        dial.clear();
        assert!(dial.empty());
        assert_eq!(dial.size(), 0);
    }

    #[test]
    #[should_panic(expected = "Cannot pop from empty")]
    fn test_dial_pop_empty() {
        let mut dial: Dial<Item, _> = Dial::new(get_cost);
        dial.set_n_buckets(100);
        dial.pop();
    }

    #[test]
    #[should_panic(expected = "Bucket")]
    fn test_dial_push_out_of_range() {
        let mut dial: Dial<Item, _> = Dial::new(get_cost);
        dial.set_n_buckets(10);
        dial.push(Item {
            cost: 100,
            value: 1,
        });
    }
}
