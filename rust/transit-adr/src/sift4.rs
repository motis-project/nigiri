//! SIFT4 approximate string distance algorithm.
//!
//! A fast edit distance approximation that handles character transpositions.
//! Mirrors the C++ `adr::sift4()` implementation.

use crate::types::EditDist;

/// Tracks character offset positions for transposition detection.
#[derive(Debug, Clone)]
pub struct SiftOffset {
    pub c1: u8,
    pub c2: u8,
    pub trans: bool,
}

/// Compute the SIFT4 approximate string distance between `s1` and `s2`.
///
/// # Parameters
/// - `s1`, `s2`: The two strings to compare (should be ASCII/normalized).
/// - `max_offset`: Search window for transpositions (typically 3).
/// - `max_distance`: Early-exit threshold (0 = no limit).
/// - `offset_arr`: Reusable buffer for offset tracking.
///
/// # Returns
/// Approximate edit distance (lower = more similar).
pub fn sift4(
    s1: &str,
    s2: &str,
    max_offset: EditDist,
    max_distance: EditDist,
    offset_arr: &mut Vec<SiftOffset>,
) -> EditDist {
    debug_assert!(s1.len() < EditDist::MAX as usize);
    debug_assert!(s2.len() < EditDist::MAX as usize);

    offset_arr.clear();

    let b1 = s1.as_bytes();
    let b2 = s2.as_bytes();

    if b1.is_empty() {
        return if b2.is_empty() {
            0
        } else {
            b2.len() as EditDist
        };
    }
    if b2.is_empty() {
        return b1.len() as EditDist;
    }

    let l1 = b1.len();
    let l2 = b2.len();

    let mut c1: usize = 0; // cursor for s1
    let mut c2: usize = 0; // cursor for s2
    let mut lcss: usize = 0; // largest common subsequence
    let mut local_cs: usize = 0; // local common substring
    let mut trans: usize = 0; // transpositions

    while c1 < l1 && c2 < l2 {
        if b1[c1] == b2[c2] {
            local_cs += 1;
            let mut is_trans = false;

            let mut i = 0;
            while i < offset_arr.len() {
                let ofs = &offset_arr[i];
                if c1 <= ofs.c1 as usize || c2 <= ofs.c2 as usize {
                    is_trans = abs_diff(c2, c1) >= abs_diff(ofs.c2 as usize, ofs.c1 as usize);
                    if is_trans {
                        trans += 1;
                    } else if !ofs.trans {
                        offset_arr[i].trans = true;
                        trans += 1;
                    }
                    break;
                } else if c1 > ofs.c2 as usize && c2 > ofs.c1 as usize {
                    offset_arr.remove(i);
                } else {
                    i += 1;
                }
            }
            offset_arr.push(SiftOffset {
                c1: c1 as u8,
                c2: c2 as u8,
                trans: is_trans,
            });
        } else {
            lcss += local_cs;
            local_cs = 0;
            if c1 != c2 {
                let m = c1.min(c2);
                c1 = m;
                c2 = m;
            }
            if max_distance > 0 {
                let temp_dist = c1.max(c2) - lcss + trans;
                if temp_dist > max_distance as usize {
                    return temp_dist as EditDist;
                }
            }
            let max_off = max_offset as usize;
            for i in 0..max_off {
                if c1 + i >= l1 && c2 + i >= l2 {
                    break;
                }
                if c1 + i < l1 && b1[c1 + i] == b2[c2] {
                    c1 = c1.wrapping_add(i).wrapping_sub(1);
                    c2 = c2.wrapping_sub(1);
                    break;
                }
                if c2 + i < l2 && b1[c1] == b2[c2 + i] {
                    c1 = c1.wrapping_sub(1);
                    c2 = c2.wrapping_add(i).wrapping_sub(1);
                    break;
                }
            }
        }
        c1 = c1.wrapping_add(1);
        c2 = c2.wrapping_add(1);

        if c1 >= l1 || c2 >= l2 {
            lcss += local_cs;
            local_cs = 0;
            let m = c1.min(c2);
            c1 = m;
            c2 = m;
        }
    }
    lcss += local_cs;

    (l1.max(l2) - lcss + trans) as EditDist
}

#[inline]
fn abs_diff(a: usize, b: usize) -> usize {
    a.abs_diff(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// C++ test: `TEST(adr, sift4)`
    ///
    /// Computes distance between "Landkreis Aschaffenburg" and
    /// "mainaschaff aschaffenburg".
    #[test]
    fn sift4_aschaffenburg() {
        let mut offsets = Vec::new();
        let dist = sift4(
            "Landkreis Aschaffenburg",
            "mainaschaff aschaffenburg",
            4,
            10,
            &mut offsets,
        );
        // The C++ test only prints the result; we verify it computes without
        // panic and returns a reasonable distance.
        assert!(dist <= 25, "distance {dist} should be within string length");
    }

    #[test]
    fn sift4_identical() {
        let mut offsets = Vec::new();
        let dist = sift4("hello", "hello", 3, 10, &mut offsets);
        assert_eq!(dist, 0);
    }

    #[test]
    fn sift4_empty() {
        let mut offsets = Vec::new();
        assert_eq!(sift4("", "", 3, 10, &mut offsets), 0);
        assert_eq!(sift4("abc", "", 3, 10, &mut offsets), 3);
        assert_eq!(sift4("", "abc", 3, 10, &mut offsets), 3);
    }

    #[test]
    fn sift4_single_edit() {
        let mut offsets = Vec::new();
        let dist = sift4("kitten", "sitten", 3, 10, &mut offsets);
        assert!(dist <= 2, "single substitution should yield small distance");
    }

    #[test]
    fn sift4_transposition() {
        let mut offsets = Vec::new();
        let dist = sift4("ab", "ba", 3, 10, &mut offsets);
        assert!(dist <= 2);
    }
}
