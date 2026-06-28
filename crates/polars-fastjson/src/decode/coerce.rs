//! Coercion table.
//!
//! Conservative defaults, applied only when `coerce = true`.
//! Maps the JSON value kinds onto the target leaf dtype.
//! When `coerce = false`, only the directly matching JSON kind is accepted, anything else becomes a null field.
//!
//! Temporal values are parsed from ISO-8601 strings (see [`parse_date`], [`parse_time`], [`parse_datetime`]).

use simd_json::{BorrowedValue, StaticNode};

use crate::ir::TimeUnit;

/// Coerce a JSON value to an `i128`, the wide intermediate that callers then
/// range-check into a concrete integer width.
///
/// Rules (coerce = true):
/// - JSON integer -> accepted.
/// - JSON float -> only if it has no fractional part and is finite.
/// - JSON string -> parsed as an integer (rejecting `"1.0"`-style floats).
/// - JSON bool -> rejected.
///
/// When `coerce = false` only a JSON integer is accepted.
fn coerce_to_i128(value: &BorrowedValue, coerce: bool) -> Option<i128> {
    match value {
        BorrowedValue::Static(StaticNode::I64(i)) => Some(*i as i128),
        BorrowedValue::Static(StaticNode::U64(u)) => Some(*u as i128),
        BorrowedValue::Static(StaticNode::F64(f)) if coerce => {
            if f.fract() == 0.0 && f.is_finite() {
                Some(*f as i128)
            } else {
                None
            }
        }
        BorrowedValue::String(s) if coerce => s.trim().parse::<i128>().ok(),
        _ => None,
    }
}

/// Coerce to a signed integer of width `T`, range-checking the result.
pub fn coerce_int<T>(value: &BorrowedValue, coerce: bool) -> Option<T>
where
    T: TryFrom<i128>,
{
    coerce_to_i128(value, coerce).and_then(|v| T::try_from(v).ok())
}

/// Coerce to an unsigned integer of width `T`, range-checking the result.
///
/// Identical rules to [`coerce_int`] but the i128 intermediate must be
/// non-negative and fit the unsigned width.
pub fn coerce_uint<T>(value: &BorrowedValue, coerce: bool) -> Option<T>
where
    T: TryFrom<i128>,
{
    coerce_to_i128(value, coerce).and_then(|v| T::try_from(v).ok())
}

/// Coerce a JSON value to `f64` (int/float; parseable string; bool rejected).
/// When `coerce = false`, only a JSON float is accepted.
pub fn coerce_f64(value: &BorrowedValue, coerce: bool) -> Option<f64> {
    match value {
        BorrowedValue::Static(StaticNode::F64(f)) => Some(*f),
        BorrowedValue::Static(StaticNode::I64(i)) if coerce => Some(*i as f64),
        BorrowedValue::Static(StaticNode::U64(u)) if coerce => Some(*u as f64),
        BorrowedValue::String(s) if coerce => s.trim().parse::<f64>().ok(),
        _ => None,
    }
}

/// Coerce a JSON value to `f32` (via [`coerce_f64`] then narrowing).
pub fn coerce_f32(value: &BorrowedValue, coerce: bool) -> Option<f32> {
    coerce_f64(value, coerce).map(|f| f as f32)
}

/// Coerce a JSON value to a `String`.
///
/// - JSON string -> verbatim.
/// - JSON number/bool -> stringified (`42 -> "42"`, `true -> "true"`).
/// - JSON null -> null (None).
/// - JSON object/array -> null (None) in v1.
///
/// When `coerce = false`, only a JSON string is accepted.
pub fn coerce_str(value: &BorrowedValue, coerce: bool) -> Option<String> {
    match value {
        BorrowedValue::String(s) => Some(s.to_string()),
        _ if !coerce => None,
        BorrowedValue::Static(StaticNode::Bool(b)) => Some(b.to_string()),
        BorrowedValue::Static(StaticNode::I64(i)) => Some(i.to_string()),
        BorrowedValue::Static(StaticNode::U64(u)) => Some(u.to_string()),
        BorrowedValue::Static(StaticNode::F64(f)) => Some(f.to_string()),
        BorrowedValue::Static(StaticNode::Null) => None,
        _ => None, // array / object -> null in v1
    }
}

/// Coerce a JSON value to a `bool`.
///
/// - JSON bool -> accepted.
/// - JSON string `"true"`/`"false"` (case-insensitive) -> bool; other strings -> null.
/// - JSON number -> null (no 0/1 magic in v1).
///
/// When `coerce = false`, only a JSON bool is accepted.
pub fn coerce_bool(value: &BorrowedValue, coerce: bool) -> Option<bool> {
    match value {
        BorrowedValue::Static(StaticNode::Bool(b)) => Some(*b),
        BorrowedValue::String(s) if coerce => match s.trim().to_ascii_lowercase().as_str() {
            "true" => Some(true),
            "false" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

/// Coerce a JSON value to bytes.
///
/// v1: a JSON string maps to its UTF-8 bytes; everything else -> null. No base64
/// decoding is attempted.
pub fn coerce_binary(value: &BorrowedValue, _coerce: bool) -> Option<Vec<u8>> {
    match value {
        BorrowedValue::String(s) => Some(s.as_bytes().to_vec()),
        _ => None,
    }
}

/// Number of whole days from the Unix epoch (1970-01-01) to `y-m-d`.
///
/// Uses a standard civil-date -> day-number algorithm (Howard Hinnant's
/// `days_from_civil`). Valid for the proleptic Gregorian calendar.
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400; // [0, 399]
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) as i64 + 2) / 5 + d as i64 - 1; // [0,365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146097 + doe - 719468
}

/// Parse a `YYYY-MM-DD` date into days since the Unix epoch (i32).
pub fn parse_date(s: &str) -> Option<i32> {
    let s = s.trim();
    let mut parts = s.splitn(3, '-');
    // Handle a leading sign for negative years is out of scope; reject.
    let y: i64 = parts.next()?.parse().ok()?;
    let m: u32 = parts.next()?.parse().ok()?;
    let d: u32 = parts.next()?.parse().ok()?;
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    i32::try_from(days_from_civil(y, m, d)).ok()
}

/// Parse a `HH:MM:SS[.fffffffff]` time into nanoseconds since midnight (i64).
pub fn parse_time(s: &str) -> Option<i64> {
    let s = s.trim();
    let (hms, frac) = match s.split_once('.') {
        Some((a, b)) => (a, Some(b)),
        None => (s, None),
    };
    let mut parts = hms.splitn(3, ':');
    let h: i64 = parts.next()?.parse().ok()?;
    let mi: i64 = parts.next()?.parse().ok()?;
    let sec: i64 = parts.next()?.parse().ok()?;
    if !(0..=23).contains(&h) || !(0..=59).contains(&mi) || !(0..=59).contains(&sec) {
        return None;
    }
    let nanos = parse_fractional_nanos(frac)?;
    Some(((h * 3600 + mi * 60 + sec) * 1_000_000_000) + nanos)
}

/// Parse an optional fractional-second string (the part after `.`) into nanos.
///
/// Accepts up to 9 digits; pads/truncates to nanosecond resolution. `None` input
/// -> 0. A non-digit fraction -> parse failure.
fn parse_fractional_nanos(frac: Option<&str>) -> Option<i64> {
    match frac {
        None => Some(0),
        Some(f) => {
            // Strip any trailing timezone designator that a caller left attached.
            let f = f.trim();
            if f.is_empty() || !f.bytes().all(|b| b.is_ascii_digit()) {
                return None;
            }
            // Pad/truncate the fraction to nanosecond resolution (9 digits).
            // `{:0<9}` right-pads with zeros; slicing handles the >9 case.
            let digits = if f.len() > 9 { &f[..9] } else { f };
            format!("{digits:0<9}").parse::<i64>().ok()
        }
    }
}

/// Convert a count of nanoseconds to the given [`TimeUnit`].
fn nanos_to_unit(nanos: i128, tu: TimeUnit) -> Option<i64> {
    let v = match tu {
        TimeUnit::Ns => nanos,
        TimeUnit::Us => nanos / 1_000,
        TimeUnit::Ms => nanos / 1_000_000,
    };
    i64::try_from(v).ok()
}

/// Parse an ISO-8601 / RFC-3339 datetime into the physical i64 the given
/// [`TimeUnit`] expects (elapsed since the Unix epoch).
///
/// Accepts `YYYY-MM-DD` optionally followed by `T`/space and `HH:MM:SS[.fff]`,
/// optionally followed by a `Z` or `±HH:MM` offset. Time zone naming on the
/// schema side does not affect the stored physical value (which is epoch-based);
/// an explicit offset in the string shifts the instant to UTC.
pub fn parse_datetime(s: &str, tu: TimeUnit) -> Option<i64> {
    let s = s.trim();
    // Separate the date part from the time part on 'T' or ' '.
    let (date_part, rest) = if let Some(idx) = s.find(['T', 't', ' ']) {
        (&s[..idx], Some(&s[idx + 1..]))
    } else {
        (s, None)
    };

    let days = parse_date(date_part)? as i128;
    let mut nanos: i128 = days * 86_400 * 1_000_000_000;

    if let Some(time_str) = rest {
        // Extract a trailing timezone offset if present.
        let (time_only, offset_secs) = split_offset(time_str)?;
        nanos += parse_time(time_only)? as i128;
        // Subtract the offset to normalize to UTC.
        nanos -= (offset_secs as i128) * 1_000_000_000;
    }

    nanos_to_unit(nanos, tu)
}

/// Split a trailing timezone designator (`Z` or `±HH:MM` / `±HHMM`) from a time
/// string, returning `(time_without_offset, offset_in_seconds)`.
fn split_offset(s: &str) -> Option<(&str, i64)> {
    if let Some(stripped) = s.strip_suffix(['Z', 'z']) {
        return Some((stripped, 0));
    }
    // Look for a '+' or a '-' that introduces an offset (after the seconds).
    // We scan from the right so we don't confuse it with date separators (none
    // here since the date was already split off).
    for (i, c) in s.char_indices().rev() {
        if c == '+' || c == '-' {
            let (time, off) = (&s[..i], &s[i..]);
            let sign = if c == '-' { -1 } else { 1 };
            let off = &off[1..];
            let (oh, om) = if let Some((h, m)) = off.split_once(':') {
                (h.parse::<i64>().ok()?, m.parse::<i64>().ok()?)
            } else if off.len() == 4 {
                (off[..2].parse::<i64>().ok()?, off[2..].parse::<i64>().ok()?)
            } else if off.len() == 2 {
                (off.parse::<i64>().ok()?, 0)
            } else {
                return None;
            };
            return Some((time, sign * (oh * 3600 + om * 60)));
        }
    }
    Some((s, 0))
}

/// Parse a duration string into the physical i64 for the given [`TimeUnit`].
///
/// v1 keeps duration minimal: a JSON integer-or-string-of-integer is interpreted
/// as an already-scaled count in the target unit. (ISO-8601 duration grammar,
/// e.g. `PT1H`, is out of scope for v1.)
pub fn coerce_duration(value: &BorrowedValue, _tu: TimeUnit, coerce: bool) -> Option<i64> {
    coerce_int::<i64>(value, coerce)
}

/// Parse a date from a JSON value (string only) -> days since epoch (i32).
pub fn coerce_date(value: &BorrowedValue, _coerce: bool) -> Option<i32> {
    match value {
        BorrowedValue::String(s) => parse_date(s),
        _ => None,
    }
}

/// Parse a time from a JSON value (string only) -> ns since midnight (i64).
pub fn coerce_time(value: &BorrowedValue, _coerce: bool) -> Option<i64> {
    match value {
        BorrowedValue::String(s) => parse_time(s),
        _ => None,
    }
}

/// Parse a datetime from a JSON value (string only) -> physical i64 in `tu`.
pub fn coerce_datetime(value: &BorrowedValue, tu: TimeUnit, _coerce: bool) -> Option<i64> {
    match value {
        BorrowedValue::String(s) => parse_datetime(s, tu),
        _ => None,
    }
}
