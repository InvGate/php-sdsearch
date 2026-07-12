//! analyzer: exact replica of Zend_Search_Lucene Utf8Num_CaseInsensitive.
//! tokenization regex deviates from stock Zend Lucene: tokens keep - _ . : # / @
//! (emails, URLs, and ticket refs stay as a single term).

use regex::Regex;

static TOKEN_RE: std::sync::LazyLock<Regex> =
    std::sync::LazyLock::new(|| Regex::new(r"[\p{L}\p{N}\-_.:#/@]+[\p{L}\p{N}]?").unwrap());

/// tokenizes and lowercases, replicating the legacy analyzer
pub fn analyze(text: &str) -> Vec<String> {
    TOKEN_RE
        .find_iter(text)
        .map(|m| m.as_str().to_lowercase())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_on_whitespace_and_lowercases() {
        assert_eq!(analyze("Hello World"), vec!["hello", "world"]);
    }

    #[test]
    fn keeps_ticket_ref_as_one_token() {
        // the hyphen stays inside the token (deviation from stock Zend Lucene)
        assert_eq!(analyze("TICKET-12345"), vec!["ticket-12345"]);
    }

    #[test]
    fn keeps_email_and_url_as_one_token() {
        assert_eq!(
            analyze("Mail user@example.com"),
            vec!["mail", "user@example.com"]
        );
        assert_eq!(analyze("see https://a.b/c"), vec!["see", "https://a.b/c"]);
    }

    #[test]
    fn emits_punctuation_only_tokens() {
        // legacy quirk: a run of only punctuation from the set matches
        assert_eq!(analyze("a --- b"), vec!["a", "---", "b"]);
    }

    #[test]
    fn unicode_letters_and_numbers() {
        assert_eq!(analyze("Über 2 Ítems"), vec!["über", "2", "ítems"]);
    }
}
