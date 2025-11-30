# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2025-11-30

### Added

- `MhinNetwork` enum plus per-network `MhinConfig` constants for mainnet, testnet4, signet, and regtest
- Documentation note added in `docs/protocol.typ` clarifying OP_RETURN behavior
- `MhinStore::pop` method for retrieving and removing MHIN balances in one call

### Changed

- `MhinConfig::mhin_prefix` now stores a static byte slice to make the configuration compile-time constant
- `MhinProtocol::process_block` now uses `MhinStore::pop` to burn spent inputs atomically

## [0.1.1] - 2025-11-30

### Added

- `ProcessedMhinBlock` return type that surfaces reward entries, totals, and processing stats
- `Reward` type describing each rewarded output when processing a block

### Changed

- `MhinBlock` has been renamed to `PreProcessedMhinBlock` throughout the public API
- `MhinProtocol::process_block` no longer requires a block index and now returns `ProcessedMhinBlock`

## [0.1.0] - 2025-11-29

### Added

- Initial release of the MHIN protocol implementation
- `MhinProtocol` struct for processing Bitcoin blocks
- `MhinStore` trait for storage backend abstraction
- `MhinConfig` for protocol configuration (min_zero_count, base_reward, mhin_prefix)
- `MhinBlock`, `MhinTransaction`, `MhinInput`, `MhinOutput` types
- Two-phase block processing:
  - `pre_process_block`: parallelizable extraction of MHIN-relevant data
  - `process_block`: sequential balance updates
- Proportional MHIN distribution based on output values
- Custom distribution via OP_RETURN with CBOR-encoded data
- Comprehensive test suite with 28 unit tests

[Unreleased]: https://github.com/ouziel-slama/mhinprotocol/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/ouziel-slama/mhinprotocol/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ouziel-slama/mhinprotocol/releases/tag/v0.1.0

