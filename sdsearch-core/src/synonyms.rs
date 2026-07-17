//! Query-time synonym/translation dictionary (cross-lingual ES↔EN via OMW).
//! Loaded lazily and used only when a query sets `synonyms: true`. std-only.

use std::collections::HashMap;

use crate::analysis::fold_accents;

/// Upper bound on expansion terms per query token. Applied both when the bundle
/// is built and again at lookup time as a safety net against a malformed blob,
/// so the postings union a single token can trigger stays bounded.
pub const MAX_SYNONYMS_PER_TOKEN: usize = 8;

/// A flat `normalized-lemma → canonical expansion terms` map. Keys are lowercased
/// and accent-folded so lookups are accent-insensitive by construction; values stay
/// in canonical (accented) form because they become query terms against the index.
pub struct SynonymDict {
    map: HashMap<String, Vec<String>>,
}

/// The lookup/storage key form: lowercase, then fold Spanish accents.
fn norm_key(s: &str) -> String {
    fold_accents(&s.to_lowercase())
}

/// Build a dictionary from `(key, values)` pairs. Keys are normalized; values are
/// deduplicated (order-preserving) and truncated to `MAX_SYNONYMS_PER_TOKEN`.
/// Entries sharing a normalized key merge their value lists.
pub fn from_pairs(pairs: &[(&str, &[&str])]) -> SynonymDict {
    let mut map: HashMap<String, Vec<String>> = HashMap::new();
    for (key, values) in pairs {
        let entry = map.entry(norm_key(key)).or_default();
        for v in *values {
            let v = (*v).to_string();
            if !entry.contains(&v) {
                entry.push(v);
            }
        }
        entry.truncate(MAX_SYNONYMS_PER_TOKEN);
    }
    SynonymDict { map }
}

impl SynonymDict {
    /// Expansion terms for `token`, capped at `MAX_SYNONYMS_PER_TOKEN`. The probe is
    /// lowercased and accent-folded before lookup. Empty slice if the token is absent.
    #[must_use]
    pub fn expand(&self, token: &str) -> &[String] {
        match self.map.get(&norm_key(token)) {
            Some(v) => &v[..v.len().min(MAX_SYNONYMS_PER_TOKEN)],
            None => &[],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_is_accent_and_case_insensitive() {
        let d = from_pairs(&[("avión", &["aeroplane", "plane"])]);
        // folded + lowercased probe finds the accented, mixed-case key
        assert_eq!(
            d.expand("Avion"),
            &["aeroplane".to_string(), "plane".to_string()]
        );
        assert_eq!(
            d.expand("avión"),
            &["aeroplane".to_string(), "plane".to_string()]
        );
    }

    #[test]
    fn missing_token_returns_empty() {
        let d = from_pairs(&[("laptop", &["notebook"])]);
        assert!(d.expand("nonexistent").is_empty());
    }

    #[test]
    fn values_are_deduped() {
        let d = from_pairs(&[("laptop", &["notebook", "notebook", "portátil"])]);
        assert_eq!(
            d.expand("laptop"),
            &["notebook".to_string(), "portátil".to_string()]
        );
    }

    #[test]
    fn lookup_caps_at_max() {
        let many: Vec<&str> = vec!["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"];
        let d = from_pairs(&[("x", &many)]);
        assert_eq!(d.expand("x").len(), MAX_SYNONYMS_PER_TOKEN);
    }
}
