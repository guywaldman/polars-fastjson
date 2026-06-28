//! Opt-in decode diagnostics.
//!
//! The hot path does not use this module. Callers opt into the diagnostic decode
//! entry point, which threads row identity and path context through the builders
//! and records clustered summaries for values that decode to null.

use std::collections::HashMap;

use simd_json::{BorrowedValue, StaticNode};

use crate::ir::SchemaType;

const MAX_SAMPLES_PER_CLUSTER: usize = 3;
const MAX_IDS_PER_CLUSTER: usize = 20;
const MAX_SAMPLE_CHARS: usize = 120;

/// Runtime diagnostics mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticsMode {
    Off,
    Summary,
}

/// Options for a diagnostic decode pass.
#[derive(Debug, Clone, Default)]
pub struct DiagnosticsOptions {
    /// Optional ID value per input row, already rendered as user-facing text.
    pub ids: Option<Vec<Option<String>>>,
}

/// A row value plus its originating input row index.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DiagRowValue<'a, 'v> {
    pub row_idx: usize,
    pub value: Option<&'a BorrowedValue<'v>>,
}

pub(crate) type DiagRowValues<'a, 'v> = &'a [DiagRowValue<'a, 'v>];

/// A completed diagnostic decode result.
pub struct DecodeDiagnostics {
    pub series: polars::prelude::Series,
    pub summary: DiagnosticsSummary,
}

/// Batch-level summary of clustered decode issues.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagnosticsSummary {
    pub column: String,
    pub issues: usize,
    pub clusters: Vec<DiagnosticsCluster>,
}

impl DiagnosticsSummary {
    pub fn is_empty(&self) -> bool {
        self.issues == 0
    }

    pub fn render(&self) -> String {
        let issue_word = if self.issues == 1 { "issue" } else { "issues" };
        let mut lines = vec![format!(
            "fastjson decode produced {} {issue_word} in column {:?}",
            self.issues, self.column
        )];

        for cluster in &self.clusters {
            lines.push(String::new());
            if cluster.path == "$" {
                lines.push(format!("{}  count={}", cluster.kind, cluster.count));
            } else {
                lines.push(format!(
                    "{} {}  count={}",
                    cluster.path, cluster.kind, cluster.count
                ));
            }
            if let Some(reason) = &cluster.reason {
                lines.push(format!("  reason: {reason}"));
            }
            if let Some(expected) = &cluster.expected {
                lines.push(format!("  expected: {expected}"));
            }
            if let Some(found) = &cluster.found {
                lines.push(format!("  found: {found}"));
            }
            if !cluster.ids.is_empty() {
                lines.push(format!(
                    "  ids: {}",
                    render_limited_list(&cluster.ids, cluster.omitted_ids)
                ));
            }
            if !cluster.samples.is_empty() {
                lines.push(format!(
                    "  samples: {}",
                    render_limited_list(&cluster.samples, cluster.omitted_samples)
                ));
            }
        }

        lines.join("\n")
    }
}

/// One clustered class of decode issue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagnosticsCluster {
    pub kind: String,
    pub path: String,
    pub expected: Option<String>,
    pub found: Option<String>,
    pub reason: Option<String>,
    pub count: usize,
    pub samples: Vec<String>,
    pub omitted_samples: usize,
    pub ids: Vec<String>,
    pub omitted_ids: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ClusterKey {
    kind: String,
    path: String,
    expected: Option<String>,
    found: Option<String>,
    reason: Option<String>,
}

/// Incremental collector used by the diagnostic decode path.
pub(crate) struct DiagnosticsCollector {
    column: String,
    ids: Option<Vec<Option<String>>>,
    clusters: Vec<DiagnosticsCluster>,
    index: HashMap<ClusterKey, usize>,
    issues: usize,
}

impl DiagnosticsCollector {
    pub fn new(column: String, options: DiagnosticsOptions) -> Self {
        Self {
            column,
            ids: options.ids,
            clusters: Vec::new(),
            index: HashMap::new(),
            issues: 0,
        }
    }

    pub fn finish(self) -> DiagnosticsSummary {
        DiagnosticsSummary {
            column: self.column,
            issues: self.issues,
            clusters: self.clusters,
        }
    }

    pub fn record_parse_error(&mut self, row_idx: usize, raw: &str, error: &simd_json::Error) {
        let reason = normalize_parse_error(error);
        self.record(
            ClusterKey {
                kind: "invalid_json".to_string(),
                path: "$".to_string(),
                expected: Some("json".to_string()),
                found: None,
                reason: Some(reason),
            },
            row_idx,
            Some(format_raw_sample(raw)),
        );
    }

    pub fn record_value_mismatch(
        &mut self,
        row_idx: usize,
        path: &str,
        expected: &str,
        found: &BorrowedValue,
    ) {
        if is_json_null(found) {
            return;
        }
        self.record(
            ClusterKey {
                kind: "type_mismatch".to_string(),
                path: path.to_string(),
                expected: Some(expected.to_string()),
                found: Some(json_kind(found).to_string()),
                reason: None,
            },
            row_idx,
            Some(format_json_sample(found)),
        );
    }

    pub fn record_required_field_failure(
        &mut self,
        row_idx: usize,
        path: &str,
        expected: &str,
        found: Option<&str>,
        reason: &str,
    ) {
        self.record(
            ClusterKey {
                kind: "required_field".to_string(),
                path: path.to_string(),
                expected: Some(expected.to_string()),
                found: found.map(str::to_string),
                reason: Some(reason.to_string()),
            },
            row_idx,
            None,
        );
    }

    fn record(&mut self, key: ClusterKey, row_idx: usize, sample: Option<String>) {
        self.issues += 1;

        let cluster_idx = if let Some(idx) = self.index.get(&key) {
            *idx
        } else {
            let idx = self.clusters.len();
            self.clusters.push(DiagnosticsCluster {
                kind: key.kind.clone(),
                path: key.path.clone(),
                expected: key.expected.clone(),
                found: key.found.clone(),
                reason: key.reason.clone(),
                count: 0,
                samples: Vec::new(),
                omitted_samples: 0,
                ids: Vec::new(),
                omitted_ids: 0,
            });
            self.index.insert(key, idx);
            idx
        };

        let cluster = &mut self.clusters[cluster_idx];
        cluster.count += 1;

        if let Some(id) = self
            .ids
            .as_ref()
            .and_then(|ids| ids.get(row_idx))
            .and_then(|id| id.as_ref())
        {
            if !cluster.ids.contains(id) {
                if cluster.ids.len() < MAX_IDS_PER_CLUSTER {
                    cluster.ids.push(id.clone());
                } else {
                    cluster.omitted_ids += 1;
                }
            }
        }

        if let Some(sample) = sample {
            if !cluster.samples.contains(&sample) {
                if cluster.samples.len() < MAX_SAMPLES_PER_CLUSTER {
                    cluster.samples.push(sample);
                } else {
                    cluster.omitted_samples += 1;
                }
            }
        }
    }
}

pub(crate) fn expected_type(ir: &SchemaType) -> &'static str {
    match ir {
        SchemaType::Null => "null",
        SchemaType::Bool => "bool",
        SchemaType::I8 => "i8",
        SchemaType::I16 => "i16",
        SchemaType::I32 => "i32",
        SchemaType::I64 => "i64",
        SchemaType::U8 => "u8",
        SchemaType::U16 => "u16",
        SchemaType::U32 => "u32",
        SchemaType::U64 => "u64",
        SchemaType::F32 => "f32",
        SchemaType::F64 => "f64",
        SchemaType::Str => "string",
        SchemaType::Binary => "binary",
        SchemaType::Date => "date",
        SchemaType::Time => "time",
        SchemaType::Datetime { .. } => "datetime",
        SchemaType::Duration { .. } => "duration",
        SchemaType::Decimal { .. } => "decimal",
        SchemaType::List { .. } => "array",
        SchemaType::Struct { .. } => "object",
    }
}

pub(crate) fn json_kind(v: &BorrowedValue) -> &'static str {
    match v {
        BorrowedValue::Static(StaticNode::Null) => "null",
        BorrowedValue::Static(StaticNode::Bool(_)) => "bool",
        BorrowedValue::Static(StaticNode::I64(_)) | BorrowedValue::Static(StaticNode::U64(_)) => {
            "integer"
        }
        BorrowedValue::Static(StaticNode::F64(_)) => "float",
        BorrowedValue::String(_) => "string",
        BorrowedValue::Array(_) => "array",
        BorrowedValue::Object(_) => "object",
    }
}

pub(crate) fn is_json_null(v: &BorrowedValue) -> bool {
    matches!(v, BorrowedValue::Static(StaticNode::Null))
}

fn normalize_parse_error(error: &simd_json::Error) -> String {
    let message = error.to_string();
    let mut normalized = String::with_capacity(message.len());
    let mut chars = message.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch.is_ascii_digit() {
            normalized.push('#');
            while chars.peek().is_some_and(|c| c.is_ascii_digit()) {
                chars.next();
            }
        } else {
            normalized.push(ch);
        }
    }
    normalized
}

fn format_raw_sample(raw: &str) -> String {
    quote_string(&truncate(raw.trim()))
}

fn format_json_sample(value: &BorrowedValue) -> String {
    match value {
        BorrowedValue::Static(StaticNode::Null) => "null".to_string(),
        BorrowedValue::Static(StaticNode::Bool(b)) => b.to_string(),
        BorrowedValue::Static(StaticNode::I64(i)) => i.to_string(),
        BorrowedValue::Static(StaticNode::U64(u)) => u.to_string(),
        BorrowedValue::Static(StaticNode::F64(f)) => f.to_string(),
        BorrowedValue::String(s) => quote_string(&truncate(s)),
        BorrowedValue::Array(arr) => {
            let mut parts: Vec<String> = arr.iter().take(3).map(format_json_sample).collect();
            if arr.len() > 3 {
                parts.push("...".to_string());
            }
            format!("[{}]", parts.join(", "))
        }
        BorrowedValue::Object(map) => {
            let mut parts: Vec<String> = map
                .iter()
                .take(3)
                .map(|(k, v)| format!("{}: {}", quote_string(k), format_json_sample(v)))
                .collect();
            if map.len() > 3 {
                parts.push("...".to_string());
            }
            format!("{{{}}}", parts.join(", "))
        }
    }
}

fn truncate(value: &str) -> String {
    let mut out = String::new();
    for (idx, ch) in value.chars().enumerate() {
        if idx == MAX_SAMPLE_CHARS {
            out.push_str("...");
            return out;
        }
        out.push(ch);
    }
    out
}

fn quote_string(value: &str) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| format!("{value:?}"))
}

fn render_limited_list(values: &[String], omitted: usize) -> String {
    let mut rendered = values.join(", ");
    if omitted > 0 {
        if !rendered.is_empty() {
            rendered.push_str(", ");
        }
        rendered.push_str(&format!("... (+{omitted} omitted)"));
    }
    rendered
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_borrowed(raw: &str, f: impl FnOnce(&BorrowedValue)) {
        let mut bytes = raw.as_bytes().to_vec();
        let value = simd_json::to_borrowed_value(&mut bytes).unwrap();
        f(&value);
    }

    #[test]
    fn collector_clusters_equivalent_errors() {
        let mut collector = DiagnosticsCollector::new(
            "payload".to_string(),
            DiagnosticsOptions {
                ids: Some(vec![Some("a".to_string()), Some("b".to_string())]),
            },
        );
        with_borrowed(r#""bad""#, |value| {
            collector.record_value_mismatch(0, "$.score", "f64", value);
        });
        with_borrowed(r#""worse""#, |value| {
            collector.record_value_mismatch(1, "$.score", "f64", value);
        });

        let summary = collector.finish();
        assert_eq!(summary.issues, 2);
        assert_eq!(summary.clusters.len(), 1);
        assert_eq!(summary.clusters[0].path, "$.score");
        assert_eq!(summary.clusters[0].count, 2);
        assert_eq!(summary.clusters[0].ids, ["a", "b"]);
        assert_eq!(summary.clusters[0].samples, [r#""bad""#, r#""worse""#]);
    }

    #[test]
    fn collector_caps_samples_and_ids() {
        let ids = (0..25).map(|i| Some(format!("id-{i}"))).collect();
        let mut collector =
            DiagnosticsCollector::new("payload".to_string(), DiagnosticsOptions { ids: Some(ids) });

        for i in 0..25 {
            let raw = format!(r#""bad-{i}""#);
            with_borrowed(&raw, |value| {
                collector.record_value_mismatch(i, "$.score", "f64", value);
            });
        }

        let summary = collector.finish();
        let cluster = &summary.clusters[0];
        assert_eq!(cluster.count, 25);
        assert_eq!(cluster.ids.len(), MAX_IDS_PER_CLUSTER);
        assert_eq!(cluster.omitted_ids, 5);
        assert_eq!(cluster.samples.len(), MAX_SAMPLES_PER_CLUSTER);
        assert_eq!(cluster.omitted_samples, 22);
    }

    #[test]
    fn render_uses_normalized_paths() {
        let mut collector = DiagnosticsCollector::new("payload".to_string(), Default::default());
        with_borrowed(r#""oops""#, |value| {
            collector.record_value_mismatch(0, "$.tags[]", "string", value);
        });

        let rendered = collector.finish().render();
        assert!(rendered.contains("$.tags[] type_mismatch"));
        assert!(rendered.contains("expected: string"));
        assert!(rendered.contains("found: string"));
    }
}
