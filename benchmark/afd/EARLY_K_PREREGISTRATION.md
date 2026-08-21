# Pre-registration: which key may be early

Written before the measurement, so the decision rule cannot be chosen after seeing the numbers.

## The conjecture

A key that feeds a state REUSED ACROSS TIME must be the current one. Every other use of the key
may take the early one, projected from the shifted residual `h_(l-1)`.

The arrangement has two such states in a linear layer, and the conjecture says both are protected:

| what the key feeds | reused across time? | which key |
|---|---|---|
| the recurrent state `S <- alpha S + beta (v - alpha S k) k^T` | yes, forever | current |
| the short convolution's ring, which holds the last 3 steps | yes, 3 steps | current |
| the query coefficient `q~ = q - beta (k.q) k` | no, this step only | **early** |

The reason the conjecture gives: an error in something time-reused compounds, and an error in a
per-step quantity does not. So the cut is not "output against state" -- which is where this was
headed -- but "does this value survive the step".

## What it implies that was not noticed before

The convolution runs BEFORE the split into query, key and value, so an early key computed the
ordinary way would pass through `causal_conv1d_update` and be written into the ring, where it
would persist for three steps. Under the conjecture that is forbidden. The early key must be
convolved against the ring WITHOUT writing to it -- a read-only convolution -- and the current key
does the write.

An implementation that missed this would be testing something other than the conjecture, and it
would look like the conjecture failing.

## The arms

    exact        every key current                                          the model
    mixed        early key in the coefficient only; read-only convolution   the conjecture
    all_early    early key everywhere, including both state updates         the control

## The decision rule, fixed now

Measured in bits per byte over fineweb-edu, against `exact`:

* The conjecture SURVIVES if `mixed` is within **0.5%** of `exact` in bits per byte AND `all_early`
  is at least **four times** further from `exact` than `mixed` is.
* The conjecture FAILS if `mixed` is worse than 0.5%. Then the coefficient's key is not free either,
  and Early-K is not available at any granularity.
* The conjecture is UNINFORMATIVE if `all_early` is also within 0.5%. That would mean the state
  updates tolerate an early key too, which is a different and larger claim than this one, and it
  would need its own measurement over a long generation rather than a fixed corpus -- a corpus of
  4000-character documents cannot show a state error that takes 1/(1-alpha) steps to saturate.

The third branch is the one worth naming in advance. `state_precision.py` measured that the
recurrent state's error saturates at about 1/(1-alpha) steps; a document short enough not to reach
that point would report `all_early` as harmless when it is not.

## What is NOT being claimed

That `mixed` is free. It buys the round trip hiding inside the previous feed-forward, and it costs
a second key projection on the pool -- about 12 us a layer, 0.56 ms a decode step. Whether that
trade is worth taking is a separate question from whether the conjecture about WHICH key is true.
