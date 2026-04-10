//! Translation of osr/include/osr/preprocessing/elevation/resolution.h
//!
//! Resolution tracking for elevation data with X/Y components.

/// Resolution of elevation data (in degrees per pixel)
///
/// Tracks the finest resolution available, updating when better data is found.
/// Uses NaN as initial state to indicate "no data yet".
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Resolution {
    /// X-axis resolution (longitude, in degrees)
    pub x: f64,
    /// Y-axis resolution (latitude, in degrees)
    pub y: f64,
}

impl Resolution {
    /// Create a new Resolution with NaN values (no data)
    pub fn new() -> Self {
        Self {
            x: f64::NAN,
            y: f64::NAN,
        }
    }

    /// Update resolution to track the finest available
    ///
    /// If `other` has better (smaller) resolution in either dimension,
    /// update to that value.
    pub fn update(&mut self, other: &Resolution) {
        if self.x.is_nan() || other.x < self.x {
            self.x = other.x;
        }
        if self.y.is_nan() || other.y < self.y {
            self.y = other.y;
        }
    }

    /// Check if resolution has been set (not NaN)
    pub fn is_valid(&self) -> bool {
        !self.x.is_nan() && !self.y.is_nan()
    }
}

impl Default for Resolution {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolution_creation() {
        let res = Resolution::new();
        assert!(res.x.is_nan());
        assert!(res.y.is_nan());
        assert!(!res.is_valid());
    }

    #[test]
    fn test_resolution_update() {
        let mut res = Resolution::new();

        res.update(&Resolution { x: 0.001, y: 0.002 });
        assert_eq!(res.x, 0.001);
        assert_eq!(res.y, 0.002);
        assert!(res.is_valid());

        // Better resolution in X
        res.update(&Resolution {
            x: 0.0005,
            y: 0.003,
        });
        assert_eq!(res.x, 0.0005); // Updated
        assert_eq!(res.y, 0.002); // Kept finer value
    }

    #[test]
    fn test_resolution_update_nan() {
        let mut res = Resolution { x: 0.001, y: 0.002 };

        res.update(&Resolution {
            x: f64::NAN,
            y: 0.001,
        });
        assert_eq!(res.x, 0.001); // Kept existing
        assert_eq!(res.y, 0.001); // Updated to finer
    }

    #[test]
    fn test_resolution_is_valid() {
        let res1 = Resolution { x: 0.001, y: 0.002 };
        assert!(res1.is_valid());

        let res2 = Resolution {
            x: f64::NAN,
            y: 0.002,
        };
        assert!(!res2.is_valid());

        let res3 = Resolution {
            x: 0.001,
            y: f64::NAN,
        };
        assert!(!res3.is_valid());
    }
}
