use serde::{Deserialize, Serialize};

/// How to handle parse failures and top-level structural mismatches.
///
/// Field-level leniency (missing/extra/wrong-type at leaves) is governed by the
/// `coerce` option regardless of `ErrorMode`.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ErrorMode {
    /// A bad/mismatched row becomes a null struct.
    #[default]
    Null,
    /// Raise (parity escape hatch with `str.json_decode`).
    Error,
}
