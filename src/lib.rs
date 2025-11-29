//! # mhinprotocol
//!
//! A Rust implementation of the **MY HASH IS NICE (MHIN)** protocol — a system
//! that rewards Bitcoin transactions with aesthetically pleasing transaction IDs
//! (those starting with leading zeros).
//!
//! ## Overview
//!
//! This library provides a database-agnostic and block parser-agnostic
//! implementation of the MHIN protocol. It focuses purely on the protocol logic,
//! allowing you to integrate it with any Bitcoin block source and any storage backend.
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use mhinprotocol::{MhinProtocol, MhinConfig, MhinStore};
//!
//! let protocol = MhinProtocol::new(MhinConfig::default());
//!
//! // Pre-process blocks (parallelizable)
//! let mhin_block = protocol.pre_process_block(&bitcoin_block);
//!
//! // Process blocks sequentially
//! protocol.process_block(block_index, &mhin_block, &mut store);
//! ```

/// Protocol configuration.
pub mod config;
mod helpers;
/// Core protocol implementation.
pub mod protocol;
/// Storage trait abstraction.
pub mod store;
/// Core types used by the protocol.
pub mod types;

pub use config::MhinConfig;
pub use protocol::MhinProtocol;
pub use store::MhinStore;
pub use types::{Amount, MhinBlock, MhinInput, MhinOutput, MhinTransaction, UtxoKey};
