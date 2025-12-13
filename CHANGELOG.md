# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.1] - 2025-12-13

### Changed

- `helpers` module is now public, exposing `compute_utxo_key`, `leading_zero_count`, `parse_op_return`, `calculate_reward`, and `calculate_proportional_distribution` for external use.

## [0.3.0] - 2025-12-12

### Changed

- `UtxoKey` expanded to 12 bytes using xxh3-128 (truncated) for faster hashing and lower collision risk.

## [0.2.3] - 2025-12-04

### Fixed

- Corrected the configured ZELD base reward to `4_096 * 10u64.pow(8)` across all networks.

## [0.2.2] - 2025-12-04

### Fixed

- Corrected the configured ZELD base reward to `4_096 * (10 ^ 8)` across all networks.

## [0.2.1] - 2025-12-02

### Fixed

- `leading_zero_count` now iterates bytes in the displayed order so it counts the true leading zeros of a txid hash despite the internal little-endian storage
- `ZeldProtocol::process_block` now skips persisting zero-value outputs so the ZELD store only tracks spendable UTXOs

## [0.2.0] - 2025-11-30

### Added

- `ZeldNetwork` enum plus per-network `ZeldConfig` constants for mainnet, testnet4, signet, and regtest
- Documentation note added in `docs/protocol.typ` clarifying OP_RETURN behavior
- `ZeldStore::pop` method for retrieving and removing ZELD balances in one call

### Changed

- `ZeldConfig::zeld_prefix` now stores a static byte slice to make the configuration compile-time constant
- `ZeldProtocol::process_block` now uses `ZeldStore::pop` to burn spent inputs atomically

## [0.1.1] - 2025-11-30

### Added

- `ProcessedZeldBlock` return type that surfaces reward entries, totals, and processing stats
- `Reward` type describing each rewarded output when processing a block

### Changed

- `ZeldBlock` has been renamed to `PreProcessedZeldBlock` throughout the public API
- `ZeldProtocol::process_block` no longer requires a block index and now returns `ProcessedZeldBlock`

## [0.1.0] - 2025-11-29

### Added

- Initial release of the ZELD protocol implementation
- `ZeldProtocol` struct for processing Bitcoin blocks
- `ZeldStore` trait for storage backend abstraction
- `ZeldConfig` for protocol configuration (min_zero_count, base_reward, zeld_prefix)
- `ZeldBlock`, `ZeldTransaction`, `ZeldInput`, `ZeldOutput` types
- Two-phase block processing:
  - `pre_process_block`: parallelizable extraction of ZELD-relevant data
  - `process_block`: sequential balance updates
- Proportional ZELD distribution based on output values
- Custom distribution via OP_RETURN with CBOR-encoded data
- Comprehensive test suite with 28 unit tests

[Unreleased]: https://github.com/ouziel-slama/zeldhash-protocol/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/ouziel-slama/zeldhash-protocol/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/ouziel-slama/zeldhash-protocol/compare/v0.2.3...v0.3.0
[0.1.1]: https://github.com/ouziel-slama/zeldhash-protocol/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ouziel-slama/zeldhash-protocol/releases/tag/v0.1.0

