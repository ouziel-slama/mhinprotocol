use bitcoin::Txid;

/// 8-byte key identifying a UTXO, derived from txid and vout via xxHash.
pub type UtxoKey = [u8; 8];

/// MHIN balance type (unsigned 64-bit integer).
pub type Amount = u64;

/// Represents a transaction output with MHIN-relevant data.
#[derive(Debug, Clone)]
pub struct MhinOutput {
    /// Unique key identifying this UTXO.
    pub utxo_key: UtxoKey,
    /// Satoshi value of this output.
    pub value: Amount,
    /// MHIN reward assigned to this output.
    pub reward: Amount,
    /// Custom distribution amount from OP_RETURN (if any).
    pub distribution: Amount,
    /// Output index within the transaction.
    pub vout: u32,
}

/// Represents a transaction input with MHIN-relevant data.
#[derive(Debug, Clone)]
pub struct MhinInput {
    /// Key of the UTXO being spent.
    pub utxo_key: UtxoKey,
}

/// Pre-processed transaction with all MHIN-relevant fields.
#[derive(Debug, Clone)]
pub struct MhinTransaction {
    /// Transaction ID.
    pub txid: Txid,
    /// Transaction inputs.
    pub inputs: Vec<MhinInput>,
    /// Transaction outputs (excluding OP_RETURN).
    pub outputs: Vec<MhinOutput>,
    /// Number of leading zeros in the txid.
    pub zero_count: u8,
    /// Total MHIN reward for this transaction.
    pub reward: Amount,
    /// Whether custom distribution was specified via OP_RETURN.
    pub has_op_return_distribution: bool,
}

/// Pre-processed block with all MHIN-relevant transactions.
#[derive(Debug, Clone)]
pub struct MhinBlock {
    /// Transactions in the block (excluding coinbase).
    pub transactions: Vec<MhinTransaction>,
    /// Highest leading zero count among all transactions.
    pub max_zero_count: u8,
}
