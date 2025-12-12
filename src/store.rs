use crate::types::{Amount, UtxoKey};

/// Abstraction over the persistence layer used by `ZeldProtocol`.
///
/// Any backend that can read and write ZELD balances using the provided key can
/// be used. Higher-level lifecycle management (transactions, staging, etc.) is
/// left to concrete implementations.
pub trait ZeldStore {
    /// Fetches the ZELD balance attached to a given UTXO key.
    fn get(&mut self, key: &UtxoKey) -> Amount;

    /// Removes the entry for the UTXO key, returning its current ZELD balance.
    /// Implementations should return `0` if the key is not present.
    fn pop(&mut self, key: &UtxoKey) -> Amount;

    /// Sets the ZELD balance assigned to a UTXO key.
    fn set(&mut self, key: UtxoKey, value: Amount);
}
