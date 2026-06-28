//! Thin boundary around the JSON parser.
//!
//! Kept intentionally small so the parser can be swapped without touching
//! decode logic.

use simd_json::BorrowedValue;

/// Parse one row of JSON using the `simd-json` borrowed tape.
///
/// On parse failure the caller decides whether to null the row or raise (per `ErrorMode`).
pub fn parse_borrowed<'a>(
    scratch: &'a mut Vec<u8>,
    raw: &str,
) -> Result<BorrowedValue<'a>, simd_json::Error> {
    // The borrowed tape mutates its input buffer in place, so the raw string is
    // copied into a reused `scratch` buffer (avoiding per-row allocation) before
    // parsing. The returned value borrows from `scratch`.
    scratch.clear();
    scratch.extend_from_slice(raw.as_bytes());
    simd_json::to_borrowed_value(scratch.as_mut_slice())
}

/// Parse a buffer that already holds the raw JSON bytes, in place.
///
/// The returned value borrows from `buf` - `buf` is mutated by the parser and must not be modified again while the value is alive.
/// Used by the column-wise decode path, which keeps one distinct buffer per row so all parsed values can
/// coexist.
pub fn parse_borrowed_buf(buf: &mut [u8]) -> Result<BorrowedValue<'_>, simd_json::Error> {
    simd_json::to_borrowed_value(buf)
}
