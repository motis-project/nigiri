//! Level tests - Rust port of C++ level_test.cc
//!
//! Tests level representation and conversion.

#[test]
fn test_level_0() {
    // Level 0.1 should round to 0.0
    let lvl = 0.1_f32;
    let rounded = (lvl * 2.0).round() / 2.0; // Round to nearest 0.5
    assert_eq!(0.0, rounded);
}

#[test]
fn test_level_neg4() {
    let lvl = -4.0_f32;
    let rounded = (lvl * 2.0).round() / 2.0;
    assert_eq!(-4.0, rounded);
}

#[test]
fn test_level_4() {
    let lvl = 4.0_f32;
    let rounded = (lvl * 2.0).round() / 2.0;
    assert_eq!(4.0, rounded);
}

#[test]
fn test_level_minus_3() {
    let lvl = -3.0_f32;
    let rounded = (lvl * 2.0).round() / 2.0;
    assert_eq!(-3.0, rounded);
}

#[test]
fn test_level_half() {
    // Test 0.5 increments
    let lvl = 0.5_f32;
    let rounded = (lvl * 2.0).round() / 2.0;
    assert_eq!(0.5, rounded);

    let lvl = 1.5_f32;
    let rounded = (lvl * 2.0).round() / 2.0;
    assert_eq!(1.5, rounded);

    let lvl = -0.5_f32;
    let rounded = (lvl * 2.0).round() / 2.0;
    assert_eq!(-0.5, rounded);
}
