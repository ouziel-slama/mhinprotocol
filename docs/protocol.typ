#set document(
  title: "ZELDHASH Protocol",
  author: "Ouziel Slama"
)
#set page(
  paper: "a4",
  margin: (x: 2.5cm, y: 2.5cm),
  numbering: "1",
)
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.1")
#show heading: it => [
  #it
  #v(0.8em)
]

#align(center)[
  #text(size: 24pt, weight: "bold")[ZELDHASH PROTOCOL]
]
#align(center)[
  #text(size: 14pt)[*Hunt for Bitcoin's Rarest Transactions and Earn ZELD.*]
]
#align(center)[
  #text(size: 12pt)[By Ouziel Slama]
]

#v(2em)

= Motivations

- For the thrill of the hunt. Every transaction becomes an opportunity to discover something rare — a digital treasure hidden in plain sight on the blockchain.

- These patterns of leading zeros aren't just rare — they could also enhance compression, potentially streamlining blockchain storage and processing efficiency.

- Anyone can earn ZELD by hunting rare transactions — no single-winner-per-block like in Bitcoin block mining. The hunt is open to all.

- If successful, ZELD tokens could eventually reimburse transaction fees — rewarding hunters who uncover the rarest finds!

= ZELD mining

To mine ZELD you must broadcast a Bitcoin transaction whose txid starts with at least 6 zeros. The reward is calculated based on how your transaction compares to the transaction(s) with the most leading zeros in that block:

- In a given block, transactions starting with the most zeros earn 4096 ZELD

- Transactions with one fewer zero than the best earn 4096 / 16 = 256 ZELD

- Transactions with two fewer zeros earn 4096 / 16 / 16 = 16 ZELD

- etc.

The formula used is therefore as follows:

#align(center)[
  ```
  reward = 4096 / 16 ^ (max_zero_count - zero_count)
  ```
]

Where `max_zero_count` is the leading zero count of the best transaction in the block, and `zero_count` is the leading zero count of the transaction being evaluated.

*Note:* Coinbase transactions are not eligible for ZELD rewards.

= ZELD distribution

Throughout this document, *first non-OP_RETURN output* refers to the lowest-index output in the transaction that is not an OP_RETURN.

ZELD rewards earned by mining always attach to the first non-OP_RETURN output.

= Moving ZELD

== Method 1: Automatic Distribution

When spending UTXOs that hold ZELD, all ZELD is transferred to the first non-OP_RETURN output by default.

== Method 2: Custom Distribution via OP_RETURN

You can specify exactly how ZELDs should be distributed by including an OP_RETURN output in your transaction with custom distribution data. This allows for precise control over ZELD transfers.

=== OP_RETURN Format:

- The OP_RETURN script must contain data that starts with the 4-byte prefix "ZELD"

- Following the prefix, the data must be encoded in CBOR format

- The CBOR data must be an array of unsigned 64-bit integers

- Each integer specifies how many ZELDs to send to the corresponding output

=== Distribution Rules:

- The number of values in the distribution array is automatically adjusted to match the number of non-OP_RETURN outputs

- If the array is too long, extra values are removed

- If the array is too short, zeros are appended

- The total sum of the distribution values cannot exceed the total ZELDs being spent

- If the sum is less than the total, the remainder is added to the first non-OP_RETURN output

- If the sum exceeds the total, the custom distribution is ignored and all spent ZELD attaches to the first non-OP_RETURN output

- Newly mined ZELD rewards always attach to the first non-OP_RETURN output, regardless of the custom distribution

=== Example:

If you have 1000 ZELDs to distribute across 3 outputs and want to send 600 to the first, 300 to the second, and 100 to the third, your OP_RETURN would contain "ZELD" followed by the CBOR encoding of [600, 300, 100].

*Notes:*
- If no valid OP_RETURN distribution is found, the automatic distribution (Method 1) applies.
- If all outputs are OP_RETURN, any ZELD attached to the transaction's inputs #strong[and any newly earned reward] is permanently burned because there are no spendable outputs to receive them.
- When multiple OP_RETURN outputs are present, only the last one carrying a valid `ZELD`+CBOR payload is considered.
