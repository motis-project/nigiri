//! Google polyline encoding/decoding.
//!
//! Implements the algorithm described at:
//! <https://developers.google.com/maps/documentation/utilities/polylinealgorithm>

/// Encode a single coordinate value (lat or lng) as delta from `prev`,
/// appending the result to `out`.
fn encode_value(value: f64, prev: f64, precision: u32, out: &mut String) {
    let factor = 10_f64.powi(precision as i32);
    let current = (value * factor).round() as i64;
    let previous = (prev * factor).round() as i64;
    let mut v = current - previous;

    // Left-shift and invert if negative.
    v = if v < 0 { (v << 1) ^ (!0) } else { v << 1 };

    // Break into 5-bit chunks from LSB.
    loop {
        let mut chunk = (v & 0x1F) as u8;
        v >>= 5;
        if v > 0 {
            chunk |= 0x20;
        }
        out.push((chunk + 63) as char);
        if v == 0 {
            break;
        }
    }
}

/// Encode a sequence of (latitude, longitude) pairs into a Google polyline string.
pub fn encode_polyline(coordinates: &[(f64, f64)], precision: u32) -> String {
    let mut result = String::new();
    let mut prev_lat = 0.0;
    let mut prev_lng = 0.0;

    for &(lat, lng) in coordinates {
        encode_value(lat, prev_lat, precision, &mut result);
        encode_value(lng, prev_lng, precision, &mut result);
        prev_lat = lat;
        prev_lng = lng;
    }

    result
}

/// Decode a single value from the encoded byte slice starting at `idx`,
/// returning the decoded integer and updating `idx` past the consumed bytes.
fn decode_value(bytes: &[u8], idx: &mut usize) -> i64 {
    let mut result: i64 = 0;
    let mut shift = 0;

    loop {
        let b = bytes[*idx] as i64 - 63;
        *idx += 1;
        result |= (b & 0x1F) << shift;
        shift += 5;
        if b < 0x20 {
            break;
        }
    }

    if result & 1 != 0 {
        result = !(result >> 1);
    } else {
        result >>= 1;
    }

    result
}

/// Decode a Google polyline string into (latitude, longitude) pairs.
pub fn decode_polyline(encoded: &str, precision: u32) -> Vec<(f64, f64)> {
    let bytes = encoded.as_bytes();
    let mut idx = 0;
    let factor = 10_f64.powi(precision as i32);
    let mut lat: i64 = 0;
    let mut lng: i64 = 0;
    let mut result = Vec::new();

    while idx < bytes.len() {
        lat += decode_value(bytes, &mut idx);
        lng += decode_value(bytes, &mut idx);
        result.push((lat as f64 / factor, lng as f64 / factor));
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_known_polyline() {
        // Known Google example: (38.5, -120.2), (40.7, -120.95), (43.252, -126.453)
        let coords = vec![(38.5, -120.2), (40.7, -120.95), (43.252, -126.453)];
        let encoded = encode_polyline(&coords, 5);
        assert_eq!(encoded, "_p~iF~ps|U_ulLnnqC_mqNvxq`@");
    }

    #[test]
    fn roundtrip() {
        let coords = vec![(38.5, -120.2), (40.7, -120.95), (43.252, -126.453)];
        let encoded = encode_polyline(&coords, 5);
        let decoded = decode_polyline(&encoded, 5);
        assert_eq!(coords.len(), decoded.len());
        for (a, b) in coords.iter().zip(decoded.iter()) {
            assert!((a.0 - b.0).abs() < 1e-5);
            assert!((a.1 - b.1).abs() < 1e-5);
        }
    }

    #[test]
    fn empty_input() {
        assert_eq!(encode_polyline(&[], 5), "");
        assert_eq!(decode_polyline("", 5), vec![]);
    }

    #[test]
    fn single_point() {
        let coords = vec![(0.0, 0.0)];
        let encoded = encode_polyline(&coords, 5);
        let decoded = decode_polyline(&encoded, 5);
        assert_eq!(decoded.len(), 1);
        assert!((decoded[0].0).abs() < 1e-5);
    }

    #[test]
    fn negative_coordinates() {
        let coords = vec![(-33.8688, 151.2093)]; // Sydney
        let encoded = encode_polyline(&coords, 5);
        let decoded = decode_polyline(&encoded, 5);
        assert!((decoded[0].0 - (-33.8688)).abs() < 1e-5);
        assert!((decoded[0].1 - 151.2093).abs() < 1e-5);
    }

    #[test]
    fn precision_6() {
        let coords = vec![(38.5, -120.2), (40.7, -120.95)];
        let encoded = encode_polyline(&coords, 6);
        let decoded = decode_polyline(&encoded, 6);
        for (a, b) in coords.iter().zip(decoded.iter()) {
            assert!((a.0 - b.0).abs() < 1e-6);
            assert!((a.1 - b.1).abs() < 1e-6);
        }
    }
}
