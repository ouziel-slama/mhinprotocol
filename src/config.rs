use crate::types::Amount;

/// Static parameters that describe how the MHIN protocol behaves on a network.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MhinConfig {
    /// Minimum leading zeros required for a transaction to earn MHIN.
    pub min_zero_count: u8,
    /// Base reward for the best transaction in a block.
    pub base_reward: Amount,
    /// Prefix bytes for custom distribution OP_RETURN data.
    pub mhin_prefix: &'static [u8],
}

/// Bitcoin networks supported by the MHIN protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MhinNetwork {
    Mainnet,
    Testnet4,
    Signet,
    Regtest,
}

impl MhinConfig {
    /// MHIN parameters for Bitcoin mainnet.
    pub const MAINNET: Self = Self {
        min_zero_count: 6,
        base_reward: 4_096 * 10u64.pow(8),
        mhin_prefix: b"MHIN",
    };

    /// MHIN parameters for Bitcoin testnet4.
    pub const TESTNET4: Self = Self {
        min_zero_count: 2,
        base_reward: 4_096 * 10u64.pow(8),
        mhin_prefix: b"MHIN",
    };

    /// MHIN parameters for Bitcoin signet.
    pub const SIGNET: Self = Self {
        min_zero_count: 2,
        base_reward: 4_096 * 10u64.pow(8),
        mhin_prefix: b"MHIN",
    };

    /// MHIN parameters for Bitcoin regtest.
    pub const REGTEST: Self = Self {
        min_zero_count: 2,
        base_reward: 4_096 * 10u64.pow(8),
        mhin_prefix: b"MHIN",
    };

    /// Returns the configuration associated with the provided Bitcoin network.
    pub const fn for_network(network: MhinNetwork) -> Self {
        match network {
            MhinNetwork::Mainnet => Self::MAINNET,
            MhinNetwork::Testnet4 => Self::TESTNET4,
            MhinNetwork::Signet => Self::SIGNET,
            MhinNetwork::Regtest => Self::REGTEST,
        }
    }
}

impl Default for MhinConfig {
    fn default() -> Self {
        Self::MAINNET
    }
}

impl From<MhinNetwork> for MhinConfig {
    fn from(network: MhinNetwork) -> Self {
        Self::for_network(network)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_config(
        config: MhinConfig,
        min_zero_count: u8,
        base_reward: Amount,
        prefix: &'static [u8],
    ) {
        assert_eq!(config.min_zero_count, min_zero_count);
        assert_eq!(config.base_reward, base_reward);
        assert_eq!(config.mhin_prefix, prefix);
    }

    #[test]
    fn constants_expose_expected_parameters() {
        let base_reward = 4_096 * 10u64.pow(8);
        assert_config(MhinConfig::MAINNET, 6, base_reward, b"MHIN");
        assert_config(MhinConfig::TESTNET4, 2, base_reward, b"MHIN");
        assert_config(MhinConfig::SIGNET, 2, base_reward, b"MHIN");
        assert_config(MhinConfig::REGTEST, 2, base_reward, b"MHIN");
    }

    #[test]
    fn for_network_routes_to_correct_constants() {
        assert_eq!(
            MhinConfig::for_network(MhinNetwork::Mainnet),
            MhinConfig::MAINNET
        );
        assert_eq!(
            MhinConfig::for_network(MhinNetwork::Testnet4),
            MhinConfig::TESTNET4
        );
        assert_eq!(
            MhinConfig::for_network(MhinNetwork::Signet),
            MhinConfig::SIGNET
        );
        assert_eq!(
            MhinConfig::for_network(MhinNetwork::Regtest),
            MhinConfig::REGTEST
        );
    }

    #[test]
    fn from_network_matches_for_network() {
        let networks = [
            MhinNetwork::Mainnet,
            MhinNetwork::Testnet4,
            MhinNetwork::Signet,
            MhinNetwork::Regtest,
        ];

        for network in networks {
            assert_eq!(MhinConfig::from(network), MhinConfig::for_network(network));
        }
    }

    #[test]
    fn default_matches_mainnet() {
        assert_eq!(MhinConfig::default(), MhinConfig::MAINNET);
    }
}
