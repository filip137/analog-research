# Proposal after exp-001: next_experiment

exp-001 came back contradicted. Verify-and-retry cut the first-read error from 0.0294 to 0.0147, but by 3600 s at most 4.8% of the gain was left. So write noise is not what limits the 1 h error. The only other knob in the envelope is drift pre-compensation, so it is the next thing to test, and the README question can't be answered without it.

The Direction adds a constraint: the first-read error must not be worse than baseline. In this toy no drift has happened by the 1 s read. Any pre-compensation used alone therefore adds a bias at first read, so precomp with no retry can never qualify. The only route that can qualify combines two things. A verify scheme within the write budget lowers write noise at first read, which leaves room for extra error there. Mild pre-compensation then uses that room.

I picked the arm values from the toy's fixed parameters in runner.py. The verdict will come from the data, not from these predictions:
- Drift mean is −(0.01 + 0.02g) per decade, which gives about 0.021 RMS bias per decade across targets.
- max_retries 3 with tol 0.03 should cost about 0.45 extra writes per device and leave about 0.017 write noise. That is better than exp-001's retry1 (0.0222 at 0.49 writes).
- With that verify, first-read error should cross baseline somewhere between t_ref 10 s and 20 s. At t_ref 10 s the predicted 1 h gain is about 25%.

The 8 arms are:
- baseline
- r0 + precomp at 3600 s, to check how much of the 1 h error is drift
- verify alone with the new tol
- verify plus t_ref 3, 10, 20 and 3600 s, to bracket the first-read limit and show the floor when that limit is ignored

I used 5 seeds and 50k devices to tighten the comparisons near the first-read limit. The runner always uses seeds 0..n-1, so the paired arms share targets. The run should take only a few seconds, well under 120 s. That leaves 2 experiments to narrow t_ref near the limit and confirm.

Proposed experiment: [exp-002](../experiments/exp-002.md)

Proposer cost $0.375, 2026-10-31T21:24:24+00:00
