use std::io::Cursor;

use bitcoin::{opcodes, script::Instruction, ScriptBuf, Txid};
use ciborium::de::from_reader;
use xxhash_rust::xxh64::xxh64;

use crate::types::{Amount, MhinOutput, UtxoKey};

pub(crate) fn compute_utxo_key(txid: &Txid, vout: u32) -> UtxoKey {
    let mut payload = [0u8; 36];
    payload[..32].copy_from_slice(txid.as_ref());
    payload[32..].copy_from_slice(&vout.to_le_bytes());
    xxh64(&payload, 0).to_le_bytes()
}

pub(crate) fn leading_zero_count(txid: &Txid) -> u8 {
    let mut count: u8 = 0;
    let bytes: &[u8] = txid.as_ref();

    for &byte in bytes {
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

pub(crate) fn parse_op_return(script: &ScriptBuf, prefix: &[u8]) -> Option<Vec<u64>> {
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

pub(crate) fn calculate_reward(
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

pub(crate) fn calculate_proportional_distribution(
    total_reward: Amount,
    outputs: &[MhinOutput],
) -> Vec<Amount> {
    if outputs.is_empty() {
        return Vec::new();
    }

    if total_reward == 0 {
        return vec![0; outputs.len()];
    }

    if outputs.len() == 1 {
        return vec![total_reward];
    }

    let eligible_outputs_len = outputs.len() - 1;
    let mut shares = vec![0; outputs.len()];

    let mut total_value: Amount = 0;
    for output in &outputs[..eligible_outputs_len] {
        total_value = total_value.saturating_add(output.value);
    }

    if total_value == 0 {
        shares[0] = total_reward;
        return shares;
    }

    let mut distributed: Amount = 0;
    for (idx, output) in outputs[..eligible_outputs_len].iter().enumerate() {
        let share = ((total_reward as u128 * output.value as u128) / total_value as u128) as Amount;
        shares[idx] = share;
        distributed = distributed.saturating_add(share);
    }

    if distributed < total_reward {
        shares[0] = shares[0].saturating_add(total_reward - distributed);
    }

    shares
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

    fn make_output(value: Amount) -> MhinOutput {
        MhinOutput {
            utxo_key: [0; 8],
            value,
            reward: 0,
            distribution: 0,
            vout: 0,
        }
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

        assert_eq!(key_a0.len(), 8);
        assert_ne!(key_a0, key_a1);
        assert_ne!(key_a0, key_b0);
    }

    #[test]
    fn leading_zero_count_handles_full_and_partial_bytes() {
        let all_zero = txid_from_bytes([0u8; 32]);
        assert_eq!(leading_zero_count(&all_zero), 64);

        let mut half = [0xffu8; 32];
        half[0] = 0x0f;
        let half_byte = txid_from_bytes(half);
        assert_eq!(leading_zero_count(&half_byte), 1);

        let mut bytes = [0xffu8; 32];
        bytes[0] = 0xf0;
        let non_zero = txid_from_bytes(bytes);
        assert_eq!(leading_zero_count(&non_zero), 0);
    }

    #[test]
    fn parse_op_return_succeeds_with_valid_prefix_and_cbor() {
        const PREFIX: &[u8] = b"MHIN";
        let mut payload = PREFIX.to_vec();
        payload.extend(encode_values(&[1, 2, 3]));
        let script = build_op_return_script(&payload);

        let parsed = parse_op_return(&script, PREFIX).expect("expected values");
        assert_eq!(parsed, vec![1, 2, 3]);
    }

    #[test]
    fn parse_op_return_rejects_missing_op_return() {
        let script = ScriptBuf::builder().push_slice(b"data").into_script();
        assert!(parse_op_return(&script, b"MHIN").is_none());
    }

    #[test]
    fn parse_op_return_rejects_non_push_instruction() {
        let script = ScriptBuf::builder()
            .push_opcode(opcodes::all::OP_RETURN)
            .push_opcode(opcodes::all::OP_ADD)
            .into_script();
        assert!(parse_op_return(&script, b"MHIN").is_none());
    }

    #[test]
    fn parse_op_return_enforces_prefix_length() {
        let script = build_op_return_script(b"\x01");
        assert!(parse_op_return(&script, b"MHIN").is_none());
    }

    #[test]
    fn parse_op_return_enforces_prefix_match() {
        const PREFIX: &[u8] = b"MHIN";
        let mut payload = b"BADP".to_vec();
        payload.extend(encode_values(&[9]));
        let script = build_op_return_script(&payload);

        assert!(parse_op_return(&script, PREFIX).is_none());
    }

    #[test]
    fn parse_op_return_rejects_invalid_cbor() {
        const PREFIX: &[u8] = b"MHIN";
        let mut payload = PREFIX.to_vec();
        payload.extend([0xff, 0x00]);
        let script = build_op_return_script(&payload);

        assert!(parse_op_return(&script, PREFIX).is_none());
    }

    #[test]
    fn parse_op_return_rejects_empty_script() {
        let script = ScriptBuf::new();
        assert!(parse_op_return(&script, b"MHIN").is_none());
    }

    #[test]
    fn parse_op_return_bubbles_error_before_op_return() {
        let bytes = vec![opcodes::all::OP_PUSHDATA1.to_u8(), 0x02];
        let script = ScriptBuf::from_bytes(bytes);
        assert!(parse_op_return(&script, b"MHIN").is_none());
    }

    #[test]
    fn parse_op_return_requires_push_after_op_return() {
        let script = ScriptBuf::builder()
            .push_opcode(opcodes::all::OP_RETURN)
            .into_script();
        assert!(parse_op_return(&script, b"MHIN").is_none());
    }

    #[test]
    fn parse_op_return_bubbles_error_after_op_return() {
        let bytes = vec![
            opcodes::all::OP_RETURN.to_u8(),
            opcodes::all::OP_PUSHDATA1.to_u8(),
            0x01,
        ];
        let script = ScriptBuf::from_bytes(bytes);
        assert!(parse_op_return(&script, b"MHIN").is_none());
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

    #[test]
    fn calculate_proportional_distribution_handles_empty_cases() {
        assert!(calculate_proportional_distribution(100, &[]).is_empty());

        let single = vec![make_output(10)];
        assert_eq!(calculate_proportional_distribution(100, &single), vec![100]);

        let outputs = vec![make_output(0), make_output(5)];
        assert_eq!(calculate_proportional_distribution(0, &outputs), vec![0, 0]);
    }

    #[test]
    fn calculate_proportional_distribution_assigns_total_when_no_value() {
        let outputs = vec![make_output(0), make_output(0)];
        let shares = calculate_proportional_distribution(250, &outputs);
        assert_eq!(shares, vec![250, 0]);
    }

    #[test]
    fn calculate_proportional_distribution_distributes_and_handles_remainder() {
        let outputs = vec![make_output(1), make_output(2), make_output(0)];
        let shares = calculate_proportional_distribution(5, &outputs);

        assert_eq!(shares[1], 3);
        assert_eq!(shares[2], 0);
        assert_eq!(shares[0], 2);
        assert_eq!(shares.iter().sum::<Amount>(), 5);
    }
}
