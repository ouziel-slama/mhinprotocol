use std::io::Cursor;

use bitcoin::{opcodes, script::Instruction, ScriptBuf, Txid};
use ciborium::de::from_reader;
use xxhash_rust::xxh3::xxh3_128;

use crate::types::{Amount, UtxoKey};

pub fn compute_utxo_key(txid: &Txid, vout: u32) -> UtxoKey {
    let mut payload = [0u8; 36];
    payload[..32].copy_from_slice(txid.as_ref());
    payload[32..].copy_from_slice(&vout.to_le_bytes());

    // xxh3_128 is extremely fast and provides enough entropy; truncate to 96 bits.
    let hash = xxh3_128(&payload).to_le_bytes();
    let mut key = [0u8; 12];
    key.copy_from_slice(&hash[..12]);
    key
}

pub fn leading_zero_count(txid: &Txid) -> u8 {
    let mut count: u8 = 0;
    let bytes: &[u8] = txid.as_ref();

    // bitcoin::Txid stores the hash in little-endian order; scan it in
    // reverse so we count the human-visible leading zeros.
    for &byte in bytes.iter().rev() {
        if byte == 0 {
            count += 2;
            continue;
        }

        if byte >> 4 == 0 {
            count += 1;
        }
        break;
    }

    count
}

pub fn parse_op_return(script: &ScriptBuf, prefix: &[u8]) -> Option<Vec<u64>> {
    let mut instructions = script.instructions();

    let op_return = instructions.next()?;
    match op_return.ok()? {
        Instruction::Op(opcodes::all::OP_RETURN) => {}
        _ => return None,
    }

    let push = instructions.next()?;
    let data = match push.ok()? {
        Instruction::PushBytes(bytes) => bytes.as_bytes(),
        _ => return None,
    };

    if data.len() < prefix.len() || &data[..prefix.len()] != prefix {
        return None;
    }

    let mut reader = Cursor::new(&data[prefix.len()..]);
    from_reader::<Vec<u64>, _>(&mut reader).ok()
}

pub fn calculate_reward(
    zero_count: u8,
    max_zero_count: u8,
    min_zero_count: u8,
    base_reward: Amount,
) -> Amount {
    if zero_count < min_zero_count {
        return 0;
    }

    let diff = max_zero_count.saturating_sub(zero_count);
    let mut reward = base_reward;

    for _ in 0..diff {
        reward /= 16;
        if reward == 0 {
            break;
        }
    }

    reward
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitcoin::{hashes::Hash, opcodes, script::PushBytesBuf, ScriptBuf, Txid};
    use ciborium::ser::into_writer;
    use std::str::FromStr;

    fn txid_from_hex(hex: &str) -> Txid {
        Txid::from_str(hex).expect("invalid txid hex")
    }

    fn txid_from_bytes(bytes: [u8; 32]) -> Txid {
        Txid::from_slice(&bytes).expect("invalid txid bytes")
    }

    fn build_op_return_script(data: &[u8]) -> ScriptBuf {
        let push = PushBytesBuf::try_from(data.to_vec()).expect("invalid push data length");
        ScriptBuf::builder()
            .push_opcode(opcodes::all::OP_RETURN)
            .push_slice(push)
            .into_script()
    }

    fn encode_values(values: &[u64]) -> Vec<u8> {
        let mut encoded = Vec::new();
        into_writer(values, &mut encoded).expect("failed to encode cbor");
        encoded
    }

    #[test]
    fn compute_utxo_key_varies_with_inputs() {
        let txid_a =
            txid_from_hex("0101010101010101010101010101010101010101010101010101010101010101");
        let txid_b =
            txid_from_hex("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");

        let key_a0 = compute_utxo_key(&txid_a, 0);
        let key_a1 = compute_utxo_key(&txid_a, 1);
        let key_b0 = compute_utxo_key(&txid_b, 0);

        assert_eq!(key_a0.len(), 12);
        assert_ne!(key_a0, key_a1);
        assert_ne!(key_a0, key_b0);
    }

    #[test]
    fn leading_zero_count_handles_full_and_partial_bytes() {
        let all_zero = txid_from_bytes([0u8; 32]);
        assert_eq!(leading_zero_count(&all_zero), 64);

        let mut half = [0xffu8; 32];
        half[31] = 0x0f;
        let half_byte = txid_from_bytes(half);
        assert_eq!(leading_zero_count(&half_byte), 1);

        let mut bytes = [0xffu8; 32];
        bytes[31] = 0xf0;
        let non_zero = txid_from_bytes(bytes);
        assert_eq!(leading_zero_count(&non_zero), 0);
    }

    #[test]
    fn leading_zero_count_ignores_trailing_zero_bytes() {
        let mut bytes = [0xffu8; 32];
        bytes[0] = 0x00;
        let txid = txid_from_bytes(bytes);

        assert_eq!(leading_zero_count(&txid), 0);
    }

    #[test]
    fn parse_op_return_succeeds_with_valid_prefix_and_cbor() {
        const PREFIX: &[u8] = b"ZELD";
        let mut payload = PREFIX.to_vec();
        payload.extend(encode_values(&[1, 2, 3]));
        let script = build_op_return_script(&payload);

        let parsed = parse_op_return(&script, PREFIX).expect("expected values");
        assert_eq!(parsed, vec![1, 2, 3]);
    }

    #[test]
    fn parse_op_return_rejects_missing_op_return() {
        let script = ScriptBuf::builder().push_slice(b"data").into_script();
        assert!(parse_op_return(&script, b"ZELD").is_none());
    }

    #[test]
    fn parse_op_return_rejects_non_push_instruction() {
        let script = ScriptBuf::builder()
            .push_opcode(opcodes::all::OP_RETURN)
            .push_opcode(opcodes::all::OP_ADD)
            .into_script();
        assert!(parse_op_return(&script, b"ZELD").is_none());
    }

    #[test]
    fn parse_op_return_enforces_prefix_length() {
        let script = build_op_return_script(b"\x01");
        assert!(parse_op_return(&script, b"ZELD").is_none());
    }

    #[test]
    fn parse_op_return_enforces_prefix_match() {
        const PREFIX: &[u8] = b"ZELD";
        let mut payload = b"BADP".to_vec();
        payload.extend(encode_values(&[9]));
        let script = build_op_return_script(&payload);

        assert!(parse_op_return(&script, PREFIX).is_none());
    }

    #[test]
    fn parse_op_return_rejects_invalid_cbor() {
        const PREFIX: &[u8] = b"ZELD";
        let mut payload = PREFIX.to_vec();
        payload.extend([0xff, 0x00]);
        let script = build_op_return_script(&payload);

        assert!(parse_op_return(&script, PREFIX).is_none());
    }

    #[test]
    fn parse_op_return_rejects_empty_script() {
        let script = ScriptBuf::new();
        assert!(parse_op_return(&script, b"ZELD").is_none());
    }

    #[test]
    fn parse_op_return_bubbles_error_before_op_return() {
        let bytes = vec![opcodes::all::OP_PUSHDATA1.to_u8(), 0x02];
        let script = ScriptBuf::from_bytes(bytes);
        assert!(parse_op_return(&script, b"ZELD").is_none());
    }

    #[test]
    fn parse_op_return_requires_push_after_op_return() {
        let script = ScriptBuf::builder()
            .push_opcode(opcodes::all::OP_RETURN)
            .into_script();
        assert!(parse_op_return(&script, b"ZELD").is_none());
    }

    #[test]
    fn parse_op_return_bubbles_error_after_op_return() {
        let bytes = vec![
            opcodes::all::OP_RETURN.to_u8(),
            opcodes::all::OP_PUSHDATA1.to_u8(),
            0x01,
        ];
        let script = ScriptBuf::from_bytes(bytes);
        assert!(parse_op_return(&script, b"ZELD").is_none());
    }

    #[test]
    fn calculate_reward_returns_zero_when_below_min() {
        assert_eq!(calculate_reward(1, 5, 2, 1_000), 0);
    }

    #[test]
    fn calculate_reward_scales_by_zero_difference() {
        let reward = calculate_reward(5, 5, 0, 1_000);
        assert_eq!(reward, 1_000);

        let scaled = calculate_reward(4, 5, 0, 1_000);
        assert_eq!(scaled, 62);
    }

    #[test]
    fn calculate_reward_exhausts_to_zero() {
        let reward = calculate_reward(0, 10, 0, 1);
        assert_eq!(reward, 0);
    }
}
