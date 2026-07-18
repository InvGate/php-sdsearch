//! sdsearch-core: embedded search engine (pure Rust, no PHP dependency).

pub mod analysis;
pub mod distance;
pub mod doc;
pub mod hybrid;
pub mod index;
pub mod mlt;
pub mod prf;
pub mod query;
pub mod score;
pub mod search;
pub mod segment;
pub mod serialize;
pub mod synonyms;
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

    /// The FFI boundary in sdsearch-php relies on `catch_unwind`, which only works
    /// with `panic = "unwind"`. If a profile ever sets `panic = "abort"`, reader
    /// panics become uncatchable process aborts. This documents/enforces the invariant.
    #[test]
    #[allow(clippy::assertions_on_constants)] // intentional: a compile-config invariant guard
    fn panic_is_unwind() {
        assert!(
            !cfg!(panic = "abort"),
            "sdsearch requires panic = \"unwind\" so the FFI catch_unwind net works"
        );
    }
}
