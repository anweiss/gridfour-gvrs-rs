pub enum GvrsCacheSize {
    Small = 2 * 1024 * 1024,
    Medium = 12 * 1024 * 1024,
    Large = 256 * 1024 * 1024,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gvrs_cache_sizes() {
        assert_eq!(
            (
                GvrsCacheSize::Small as i64,
                GvrsCacheSize::Medium as i64,
                GvrsCacheSize::Large as i64
            ),
            (2097152, 12582912, 268435456)
        );
    }
}
