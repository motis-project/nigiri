//! Translation of osr/include/osr/routing/mode.h + src/routing/mode.cc
//!
//! Routing mode enumeration.

use std::fmt;

use rkyv::{Archive, Deserialize, Serialize};

/// Routing mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Archive, Serialize, Deserialize)]
#[repr(u8)]
pub enum Mode {
    Foot = 0,
    Wheelchair = 1,
    Bike = 2,
    Car = 3,
}

impl Mode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Mode::Foot => "foot",
            Mode::Wheelchair => "wheelchair",
            Mode::Bike => "bike",
            Mode::Car => "car",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "foot" => Some(Mode::Foot),
            "wheelchair" => Some(Mode::Wheelchair),
            "bike" => Some(Mode::Bike),
            "car" => Some(Mode::Car),
            _ => None,
        }
    }
}

impl fmt::Display for Mode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mode_to_str() {
        assert_eq!(Mode::Foot.as_str(), "foot");
        assert_eq!(Mode::Bike.as_str(), "bike");
    }

    #[test]
    fn test_mode_from_str() {
        assert_eq!(Mode::from_str("car"), Some(Mode::Car));
        assert_eq!(Mode::from_str("invalid"), None);
    }
}
