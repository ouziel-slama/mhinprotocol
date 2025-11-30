# mhinprotocol

[![Tests](https://github.com/ouziel-slama/mhinprotocol/actions/workflows/tests.yml/badge.svg)](https://github.com/ouziel-slama/mhinprotocol/actions/workflows/tests.yml)
[![Coverage](https://codecov.io/github/ouziel-slama/mhinprotocol/graph/badge.svg?token=KB51O71CZR)](https://codecov.io/github/ouziel-slama/mhinprotocol)
[![Format](https://github.com/ouziel-slama/mhinprotocol/actions/workflows/fmt.yml/badge.svg)](https://github.com/ouziel-slama/mhinprotocol/actions/workflows/fmt.yml)
[![Clippy](https://github.com/ouziel-slama/mhinprotocol/actions/workflows/clippy.yml/badge.svg)](https://github.com/ouziel-slama/mhinprotocol/actions/workflows/clippy.yml)
[![Crates.io](https://img.shields.io/crates/v/mhinprotocol.svg)](https://crates.io/crates/mhinprotocol)
[![Docs.rs](https://docs.rs/mhinprotocol/badge.svg)](https://docs.rs/mhinprotocol)

A Rust implementation of the **MY HASH IS NICE (MHIN)** protocol — a system that rewards Bitcoin transactions with aesthetically pleasing transaction IDs (those starting with leading zeros).

## Overview

This library provides a database-agnostic and block parser-agnostic implementation of the MHIN protocol. It focuses purely on the protocol logic, allowing you to integrate it with any Bitcoin block source and any storage backend.

For the complete protocol specification, see [docs/protocol.typ](docs/protocol.typ).

## Key Features

- **Storage Agnostic**: Bring your own database by implementing the `MhinStore` trait
- **Block Parser Agnostic**: Works with any source of `bitcoin::Block` data
- **Parallel Pre-processing**: `pre_process_block` can be executed in parallel across multiple blocks
- **Sequential Processing**: `process_block` must be called sequentially, block after block

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
mhinprotocol = "0.1"
bitcoin = "0.32"
```

## Usage

### Two-Phase Block Processing

The protocol processes blocks in two phases:

1. **Pre-processing** (`pre_process_block`): Extracts all MHIN-relevant data from a Bitcoin block. This phase is **parallelizable** — you can pre-process multiple blocks concurrently.

2. **Processing** (`process_block`): Updates MHIN balances in the store. This phase is **sequential** — blocks must be processed in order, one after another.

```rust
use mhinprotocol::{MhinProtocol, MhinConfig, MhinStore, Amount, UtxoKey};

// Implement your own store
struct MyStore { /* ... */ }

impl MhinStore for MyStore {
    fn get(&mut self, key: &UtxoKey) -> Amount {
        // Fetch MHIN balance for the given UTXO key
    }
    
    fn set(&mut self, key: UtxoKey, value: Amount) {
        // Store MHIN balance for the given UTXO key
    }
}

fn main() {
    // Create protocol instance with default configuration
    let protocol = MhinProtocol::new(MhinConfig::default());
    let mut store = MyStore::new();
    
    // Fetch blocks from your Bitcoin source
    let blocks: Vec<bitcoin::Block> = fetch_blocks();
    
    // Phase 1: Pre-process blocks (can be parallelized)
    let mhin_blocks: Vec<_> = blocks
        .par_iter()  // Using rayon for parallelism
        .map(|block| protocol.pre_process_block(block))
        .collect();
    
    // Phase 2: Process blocks sequentially
    for mhin_block in &mhin_blocks {
        protocol.process_block(mhin_block, &mut store);
    }
}
```

### Configuration

```rust
use mhinprotocol::MhinConfig;

let config = MhinConfig {
    // Minimum leading zeros required to earn MHIN (default: 6)
    min_zero_count: 6,
    
    // Base reward for the best transaction in a block (default: 4096)
    base_reward: 4096,
    
    // Prefix for custom distribution OP_RETURN data (default: "MHIN")
    mhin_prefix: b"MHIN".to_vec(),
};
```

## Protocol Summary

### Mining MHIN

- Broadcast a Bitcoin transaction whose txid starts with at least 6 zeros
- The best transaction in a block (most leading zeros) earns 4096 MHIN
- Each fewer zero reduces the reward by 16x (256, 16, 1, ...)
- Coinbase transactions are not eligible

### Distribution

When MHIN is earned or transferred:

- Single output: receives the entire amount
- Multiple outputs: distributed proportionally by satoshi value (excluding last output)
- Remainder goes to the first output

### Custom Distribution

Include an OP_RETURN output with:
- 4-byte prefix: `MHIN`
- CBOR-encoded `Vec<u64>` specifying exact amounts per output

## Types

| Type | Description |
|------|-------------|
| `MhinProtocol` | Main entry point for processing blocks |
| `MhinConfig` | Protocol configuration parameters |
| `MhinStore` | Trait for storage backend implementation |
| `PreProcessedMhinBlock` | Pre-processed block data |
| `MhinTransaction` | Transaction with MHIN-relevant fields |
| `UtxoKey` | 8-byte key identifying a UTXO (`[u8; 8]`) |
| `Amount` | MHIN balance type (`u64`) |

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

