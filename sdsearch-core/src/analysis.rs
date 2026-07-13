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

/// folds Spanish acute accents and the diéresis to the base vowel; preserves ñ.
/// á é í ó ú ü → a e i o u; ñ stays ñ. Used only for accent-insensitive query
/// expansion — the analyzer itself does NOT fold (that would diverge from the
/// terms stored in the existing ZendLucene indexes).
pub fn fold_accents(s: &str) -> String {
    s.chars().map(fold_char).collect()
}

fn fold_char(c: char) -> char {
    match c {
        'á' => 'a',
        'é' => 'e',
        'í' => 'i',
        'ó' => 'o',
        'ú' | 'ü' => 'u',
        'Á' => 'A',
        'É' => 'E',
        'Í' => 'I',
        'Ó' => 'O',
        'Ú' | 'Ü' => 'U',
        other => other, // ñ/Ñ and everything else unchanged
    }
}

/// accent variants of a token for accent-insensitive matching. Spanish allows at
/// most one written tilde per word, so the candidate set is LINEAR: the folded
/// base plus one variant per vowel position carrying a single accent. `u` yields
/// both `ú` and `ü` (diéresis). `ñ` is preserved. Folding the input first means
/// this works whether the user typed the accented or the plain form.
pub fn accent_variants(token: &str) -> Vec<String> {
    let base: Vec<char> = fold_accents(token).chars().collect();
    let mut out = vec![base.iter().collect::<String>()];
    for (i, c) in base.iter().enumerate() {
        for &accented in accented_forms(*c) {
            let mut variant = base.clone();
            variant[i] = accented;
            out.push(variant.into_iter().collect());
        }
    }
    out
}

/// the single-accent forms a base vowel can take (empty for non-vowels).
fn accented_forms(c: char) -> &'static [char] {
    match c {
        'a' => &['á'],
        'e' => &['é'],
        'i' => &['í'],
        'o' => &['ó'],
        'u' => &['ú', 'ü'],
        _ => &[],
    }
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

    #[test]
    fn fold_accents_strips_acute_and_dieresis() {
        assert_eq!(fold_accents("avión"), "avion");
        assert_eq!(fold_accents("gestión"), "gestion");
        assert_eq!(fold_accents("pingüino"), "pinguino");
    }

    #[test]
    fn fold_accents_preserves_enye() {
        // folding año -> ano would conflate distinct Spanish words: ñ must survive.
        assert_eq!(fold_accents("año"), "año");
        assert_eq!(fold_accents("niño"), "niño");
    }

    #[test]
    fn accent_variants_includes_base_and_real_word_from_plain_input() {
        // user typed "avion": we must produce the plain base and the real aguda "avión".
        let v = accent_variants("avion");
        assert!(v.contains(&"avion".to_string()), "base missing: {v:?}");
        assert!(v.contains(&"avión".to_string()), "aguda missing: {v:?}");
    }

    #[test]
    fn accent_variants_from_accented_input_yields_plain() {
        // user typed "avión": folding first must also produce the plain "avion".
        let v = accent_variants("avión");
        assert!(v.contains(&"avion".to_string()), "plain missing: {v:?}");
        assert!(v.contains(&"avión".to_string()), "original missing: {v:?}");
    }

    #[test]
    fn accent_variants_covers_front_accented_words() {
        // esdrújula/llana accent the front: must generate every position, not just the last.
        let v = accent_variants("publico");
        assert!(
            v.contains(&"público".to_string()),
            "esdrújula missing: {v:?}"
        );
        assert!(v.contains(&"publicó".to_string()), "aguda missing: {v:?}");
        // llana with the tilde on the first syllable
        assert!(
            accent_variants("arbol").contains(&"árbol".to_string()),
            "front-accented llana missing"
        );
    }

    #[test]
    fn accent_variants_generates_both_u_forms() {
        let v = accent_variants("pinguino");
        assert!(
            v.contains(&"pingüino".to_string()),
            "diéresis missing: {v:?}"
        );
        assert!(
            v.contains(&"pingúino".to_string()),
            "acute u missing: {v:?}"
        );
    }

    #[test]
    fn accent_variants_never_touches_enye() {
        // every variant of "año" must keep the ñ (no variant should read "ano...").
        for variant in accent_variants("año") {
            assert!(variant.contains('ñ'), "ñ lost in variant {variant:?}");
        }
    }
}
