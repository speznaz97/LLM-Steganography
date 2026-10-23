# Multi-message covers: turn-spanning payloads

## The problem we're fixing

Today a secret that doesn't fit in one assistant turn is split across several
*consecutive* assistant messages (`generate_stego` returns a list, the model is
allowed to emit end-of-turn mid-payload, and the rest continues in a follow-up
turn with no user message in between).

That split is unnatural at its root. The model emits end-of-turn exactly when it
considers the reply finished, so the follow-up message is generated from a state
where the model has nothing left to say. It then restates, answers an imagined
user reply, or drifts. We tested three ways of joining the two messages (fresh
assistant turn, paragraph break, a hidden "keep going" instruction between them):
none produced a consistent conversation — each just changed the *flavour* of
filler. The bad coherence is a property of the split, not of the join, so a judge
model would only measure it, not fix it.

Two sample failures from the current scheme (2B model):

- `"I did, and I've already saved the relevant data..."` → `"Yes, I reviewed the
  file earlier today..."` (re-answers the same question)
- `"...no changes need to be made. You have a great morning."` → `"You're
  welcome, and thank you for sharing..."` (thanks a user who never spoke)

## The two modes we actually need

The channel is multi-purpose. A design has to serve both:

1. **One-shot** — the sender gets a single turn (post a reply, send one message)
   and the whole secret must ride in it. No reply from the other side to lean on.
2. **Conversation** — both parties are chatting back and forth over time, and the
   secret can be spread across several real exchanges.

The current code collapses these into one path (multi-message-per-turn), which is
wrong for *both*: mode 1 shouldn't exist as multi-message at all, and mode 2
shouldn't fake the follow-up turns.

## The design

Drop "multiple consecutive assistant messages per turn." One assistant turn
produces exactly **one** cover message, always ending where the model chooses.
The payload is carried across however many *real* messages the context provides.

### Mode A — one-shot (single turn, single message)

- Encode secret → payload bits (CRC + body, as today).
- Generate one cover message. The model may end its turn whenever it likes; the
  message is whatever it wrote up to that point.
- **The entire payload must fit in that one message**, or the attempt fails.
- This is where capacity (`stego_temp`, `syncpool_minp`) matters: higher
  bits/token → a short secret fits in one natural message. A secret too long for
  one message in this mode simply can't be sent one-shot; the caller must use
  mode B. We surface that clearly instead of forcing an ugly split.
- No message cap, no forced run-on last message, no cross-message keystream. The
  whole "end-of-turn mid-payload" machinery is removed from this path.

### Mode B — conversation (payload spread across real turns)

- Secret → payload bits, split into **chunks**, one chunk per assistant turn.
- Each turn:
  - takes the real conversation so far (including the other party's latest
    message) as context,
  - generates one natural, single-turn reply that embeds the next chunk,
  - ends its turn where the model wants to.
- The receiver, holding the same conversation, extracts each turn's chunk and
  concatenates them. A length header (or the CRC + a continuation flag) tells it
  when the secret is complete.
- Because every message is a genuine reply to a real preceding message,
  coherence is automatic — we never generate "a second message with nothing to
  reply to."

## Shared machinery

Both modes reuse most of what exists:

- **Codec / payload framing** — `codec.encode_bits`, `crc_bits`, the attempt
  counter and derived nonce are unchanged.
- **Per-step embedding** — SyncPool + Discop, the banned-byte set, the tail that
  completes a sentence: all identical. A "cover for one turn" is exactly what the
  inner embed loop already produces *before* the message-split logic was added.
- **Integrity / handshake** — the session fingerprint keeps pinning wire params;
  it just drops `eot=split-v1` / `max_messages` and gains a chunk-framing tag.

What's removed: `_followup_ids` / the seam, `_end_allowed`'s message-cap arm,
`max_messages`, the `capped` stat and its benchmark constraint, and the
list-of-covers return for a single turn.

## Chunk framing (mode B)

The receiver needs to know, per turn, "is this the last chunk, and where does the
payload stop inside this message?" Options, cheapest first:

1. **Fixed bits-per-turn budget** — agree a chunk size; the last chunk is
   zero-padded and the length header (sent in chunk 0) bounds the real end. The
   tail absorbs the rest of each message. Simplest; wastes a little on the last
   chunk.
2. **Per-turn continuation flag** — one bit at the front of each turn's payload:
   1 = more chunks follow, 0 = this is the last. Length header optional.
3. **Length header only** — chunk 0 carries the total bit length; the receiver
   keeps pulling bits across turns until it has that many, then checks the CRC.
   No per-turn flag.

Leaning toward **(3) length header + CRC**: the receiver reads a fixed-width
length from the first turn, then accumulates bits from each subsequent turn until
it reaches that length, then verifies the CRC. One framing cost, paid once.

## API shape

```
# Mode A: one-shot. Raises if the secret doesn't fit one message.
cover: str = generate_oneshot(messages, secret, model, codec, cfg)
secret: str = extract_oneshot(messages, cover, model, codec, cfg)

# Mode B: conversation. Stateful across turns.
tx = StegoSender(secret, codec, cfg)          # splits secret into chunks
cover: str = tx.next_cover(messages, model)   # one reply per real turn; may
                                              # raise StopIteration when done
rx = StegoReceiver(codec, cfg)
done: bool = rx.absorb(messages, cover, model)  # True when CRC verifies
secret: str = rx.secret
```

`messages` is always the *real* conversation up to that point. Mode B holds only
the pending-bit buffer (sender) and the accumulated-bit buffer (receiver); the
nonce/CRC/length header already carry the framing.

## Benchmark changes

- Mode A: measure, per config, *what fraction of a secret set fits in one
  message* and the bits/token of those that do. Drop the `capped` constraint.
- Mode B: simulate a back-and-forth (a fixed script of other-party replies) and
  measure bits per turn and turns-to-complete, plus per-turn coherence if we add
  a judge later.
- The cap-hit constraint and `max_messages` come out of `optimize2.0.py`.

## Migration / open questions

- `gui_poc.py` and `cli_poc.py` currently consume a list of covers. Mode A
  returns a single string; mode B yields one per turn. Both demos move to the new
  API.
- The wire fingerprint changes, so this is a breaking format change (expected —
  endpoints fail the handshake rather than silently desync).
- Decide whether mode A and mode B share one `generate` with a flag, or stay two
  entry points. Two entry points reads clearer and keeps mode A free of
  conversation state.
- Open: in mode B, what if the other party stops replying mid-secret? The secret
  stalls with no further covers. Acceptable for a real chat; document it.
