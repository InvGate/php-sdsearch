//! sdsearch-core: embedded search engine (pure Rust, no PHP dependency).

pub mod analysis;
pub mod distance;
pub mod doc;
pub mod index;
pub mod query;
pub mod score;
pub mod search;
pub mod segment;
pub mod serialize;
pub mod zsl;

/// returns the crate version, used as an end-to-end smoke test
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_not_empty() {
        assert!(!version().is_empty());
    }
}
