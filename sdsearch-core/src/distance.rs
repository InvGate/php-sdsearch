//! classic Levenshtein edit distance over bytes,
//! equivalent to PHP's levenshtein() function that ZSL uses (byte-based, no transposition).

/// classic Levenshtein (insert/delete/substitute = 1, no transposition) over bytes
pub fn levenshtein_bytes(a: &[u8], b: &[u8]) -> usize {
    let n = a.len();
    let m = b.len();
    if n == 0 {
        return m;
    }
    if m == 0 {
        return n;
    }
    let mut prev: Vec<usize> = (0..=m).collect();
    let mut cur: Vec<usize> = vec![0; m + 1];
    for i in 1..=n {
        cur[0] = i;
        for j in 1..=m {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[m]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_known_distances() {
        assert_eq!(levenshtein_bytes(b"", b""), 0);
        assert_eq!(levenshtein_bytes(b"abc", b"abc"), 0);
        assert_eq!(levenshtein_bytes(b"", b"abc"), 3);
        assert_eq!(levenshtein_bytes(b"abc", b""), 3);
        assert_eq!(levenshtein_bytes(b"kitten", b"sitting"), 3);
        assert_eq!(levenshtein_bytes(b"ting", b"ted"), 3);
        // no transposition: "ab"->"ba" is 2 edits, not 1
        assert_eq!(levenshtein_bytes(b"ab", b"ba"), 2);
    }
}
