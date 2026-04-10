//! Translation of osr/include/osr/util/reverse.h
//!
//! Conditional reverse iterator - iterates forward or backward based on flag.
//! Used for bidirectional routing algorithms.

/// Iterator that conditionally reverses based on is_reverse flag
pub struct ReverseIterator<I: Iterator> {
    forward_items: Vec<I::Item>,
    index: usize,
    is_reverse: bool,
}

impl<I> ReverseIterator<I>
where
    I: Iterator,
    I::Item: Clone,
{
    /// Create a new reverse iterator
    pub fn new(iter: I, is_reverse: bool) -> Self {
        let forward_items: Vec<I::Item> = iter.collect();
        let index = if is_reverse { forward_items.len() } else { 0 };

        Self {
            forward_items,
            index,
            is_reverse,
        }
    }
}

impl<I> Iterator for ReverseIterator<I>
where
    I: Iterator,
    I::Item: Clone,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if self.is_reverse {
            if self.index == 0 {
                None
            } else {
                self.index -= 1;
                self.forward_items.get(self.index).cloned()
            }
        } else {
            if self.index >= self.forward_items.len() {
                None
            } else {
                let result = self.forward_items.get(self.index).cloned();
                self.index += 1;
                result
            }
        }
    }
}

/// Create a conditional reverse iterator
pub fn reverse<I>(iter: I, is_reverse: bool) -> ReverseIterator<I>
where
    I: Iterator,
    I::Item: Clone,
{
    ReverseIterator::new(iter, is_reverse)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reverse_forward() {
        let data = vec![1, 2, 3, 4, 5];
        let iter = reverse(data.into_iter(), false);
        let result: Vec<i32> = iter.collect();

        assert_eq!(result, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_reverse_backward() {
        let data = vec![1, 2, 3, 4, 5];
        let iter = reverse(data.into_iter(), true);
        let result: Vec<i32> = iter.collect();

        assert_eq!(result, vec![5, 4, 3, 2, 1]);
    }

    #[test]
    fn test_reverse_empty() {
        let data: Vec<i32> = vec![];
        let iter = reverse(data.into_iter(), false);
        let result: Vec<i32> = iter.collect();

        assert_eq!(result, Vec::<i32>::new());
    }

    #[test]
    fn test_reverse_single_element() {
        let data = vec![42];
        let iter_fwd = reverse(data.clone().into_iter(), false);
        let iter_bwd = reverse(data.into_iter(), true);

        assert_eq!(iter_fwd.collect::<Vec<i32>>(), vec![42]);
        assert_eq!(iter_bwd.collect::<Vec<i32>>(), vec![42]);
    }
}
