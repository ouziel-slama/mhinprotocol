use crate::types::Amount;

/// Configuration used to drive the MHIN protocol behaviour.
#[derive(Debug, Clone)]
pub struct MhinConfig {
    /// Minimum leading zeros required for a transaction to earn MHIN (default: 6).
    pub min_zero_count: u8,
    /// Base reward for the best transaction in a block (default: 4096).
    pub base_reward: Amount,
    /// Prefix bytes for custom distribution OP_RETURN data (default: "MHIN").
    pub mhin_prefix: Vec<u8>,
}

impl Default for MhinConfig {
    fn default() -> Self {
        Self {
            min_zero_count: 6,
            base_reward: 4096,
            mhin_prefix: b"MHIN".to_vec(),
        }
    }
}
