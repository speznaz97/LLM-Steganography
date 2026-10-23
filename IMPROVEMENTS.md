# Improvements: shorter, more natural output text

Deferred design work. Ordered by payoff.

## Where output length comes from

Real run (Qwen3.5-4B output model, 0.8B compressor, `stego_temp=1.22`, `syncpool_topk=47`), message `"Yes, 5pm works."`:

| Part                      | Bits |
|---------------------------|------|
| Compressed message        | 55   |
| Checksum                  | 16   |
| Variation value (attempt counter) | 3 (was 32) |
| **Total**                 | 74 (was 103) |

Before item 1 that was ~65 encoded tokens plus a 0–17 token trailing section, with **47% of the encoded data fixed overhead** and the 32-bit variation value alone costing ~35 tokens. Fixed overhead is now 19 bits (~26%).

Observed readability issue: a WhatsApp reply answered naturally, then kept going past the point where the reply would naturally end, and filled the rest with emoji. Its trailing section was 0 tokens, so all of the unwanted text was generated *during the encoding*.

## 1. ~~Derive the per-message variation value from the conversation instead of sending it~~ (done)

Nonce = SHA-256 over the canonical JSON of `messages` plus a 3-bit attempt counter (`cfg.attempt_bits`); only the counter is sent, masked with a keyed pad. Saves 29 bits per message.

- Caveat: sending twice in an identical context with the same counter reuses the keystream.

## 2. ~~Let the message end where the model wants it to end~~ (done)

Both steps implemented. End-of-turn (`<|im_end|>` / `eos`) is now a real candidate.

**Step A — tail can end on the turn-end token.** `SyncPool.step` takes `allow_end`; when set, all end-of-turn ids are returned as a single extra pool with empty surface bytes (kept out of the prefix grouping, since `b""` is a prefix of everything). The tail now passes `allow_end=True` and stops as soon as that pool is chosen; `tail_max` + the sentence-ender fallback are unchanged. Empty bytes mean the decoder never sees it, and it decodes no tail bits, so nothing else changed receiver-side.

**Step B — end-of-turn during the encoding → multi-message output.** In the payload phase end-of-turn is an ordinary Discop symbol (`_end_allowed`): picking it closes the current message and the rest of the payload continues in a follow-up assistant turn. The follow-up's prefix is taken from the chat template itself (`_followup_ids`: render history + a marker assistant message, keep the tokens after the marker) and appended **token-wise** on both sides — re-rendering the history would re-tokenise the cover canonically, which SyncPool's tokens are not. The keystream nonce/ctx is still derived from the **original** `messages`, so one keystream spans all parts. The decoder, at end-of-text with bits still due, reads the bits that end-of-turn choice carried, then moves to the next message across the same extended context. The model distribution (endings included) is untouched.

- API change: `generate_stego` returns `list[str]`; `extract_stego` takes `list[str]` (a bare `str` is still accepted). Callers updated: `cli_poc.py`, `gui_poc.py`, `optimize.py`, `optimize2.0.py`, `param_tester.py`, README. `compression_optimize.py` doesn't use these functions.
- Degenerate cases: end-of-turn is disallowed as a message's first token (no empty messages) and in the last allowed message (`cfg.max_messages`, default 3) so the payload always has somewhere to finish; if the cap is hit the final message just keeps generating like before. The sender-side round-trip check runs per message (and rejects an empty one).
- Wire format bumped: `eot=split-v1|max_messages=N` added to `session_fingerprint`.
- Skipped the optional length-hint idea, as noted.
- Verified with the 0.8B model (cover+codec): 8/8 round trips OK across 4 conversations × 2 secrets; the short secret split into 2 messages on 2/4 convs, the longer one into 2–3 messages on all 4 (`end_in_payload` == `n_msgs − 1` every time). Truncated cover lists are rejected by the CRC. Step A tails observed ending on `im_end`.

## 3. Use min-p instead of top-k

The tuned settings combine T≈1.22–1.3 with top-k 35–47. A fixed k with a high temperature keeps unlikely candidates when the model is confident (the source of the emoji). min-p cuts candidates relative to the most likely token: it keeps many options where the model is unsure (which carries bits) and few where it's sure (which keeps the text natural).

- Deterministic on both sides, so it fits into `SyncPool.step` in place of `top_k` / `prob_floor`.
- Add it to the version identifier.
- Re-run the tuning script afterwards.

## 4. Shrink the compressed message

- The compressor runs at `T=1.12`. Temperature 1.0 usually compresses best. Tune it separately from the output settings, on bits per message only.
- Try a larger compressor model, or a domain-specific primer instead of `"Message:\n"`.
- The 16-bit checksum could drop to 8 bits if a 1/256 false-accept rate is acceptable.

## 5. Measure before tuning further

- Log per step: token uncertainty, grouping uncertainty (after SyncPool grouping), and bits actually encoded by the selection process. This shows how much encoding space is lost to grouping (Qwen has many prefix tokens) and to forced choices in the selection process, versus uncertainty that simply isn't there.
- Fix the tuning script and `param_tester.py` to score the tokens actually sent (`cur[plen:]`), not `model.tokenize(output_text)`. SyncPool's tokens aren't the canonical split, so bits/token and perplexity are currently measured on a different sequence.
- Note: the variation value is now derived from the conversation, so generation is deterministic per (context, secret, config) again; tuning scores no longer need averaging for that reason.