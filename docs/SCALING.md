# How large a network can the simulator handle?

Measured 2026-09-14 on the DGX Spark (GB10) with the GPU otherwise idle:
`benchmarks/throughput_large_n.py`, one period at rate 3 (≈ 3n ministeps),
four structural effects, float32, batch sized to saturate the device.

| n | batch | panels/s | s per panel | 10⁵ panels | 10⁶ panels |
|---|---|---|---|---|---|
| 50 | 4096 | 24,400 | 0.04 ms | 4 s | 41 s |
| 100 | 2048 | 3,800 | 0.26 ms | 26 s | 4.4 min |
| 200 | 1024 | 570 | 1.7 ms | 3 min | 29 min |
| 400 | 256 | 83 | 12 ms | 20 min | 3.3 h |
| 800 | 64 | 10.5 | 95 ms | 2.7 h | 26 h |
| 1,600 | 16 | 1.3 | 0.75 s | 21 h | 9 days |
| 3,200 | 4 | 0.16 | 6.2 s | 7 days | — |

Cost per panel scales as **n³**: the number of ministeps grows with n (each
actor gets ~rate opportunities per period) and each ministep is an n² batched
matrix–vector product over the dense adjacency.

What this means in practice:

- **n ≤ 200** — routine. A population estimator over 20–200 actors is a
  half-hour of generation per 10⁶ panels (the M4 Glasgow estimator lives here).
- **n ≈ 400–800** — a fixed-start estimator for one large dataset is a few
  hours of generation at 10⁵ panels; a population estimator across this range
  is a multi-day job.
- **n ≥ 1,500** — needs a sparse simulator. Real networks at that size have
  tie fractions around 1 %, so dense n² arrays waste ~99 % of the arithmetic;
  the ministep only needs the focal actor's row and column and the two-path
  counts to its candidates, which sparse adjacency structures give in O(degree)
  rather than O(n). That is an engineering item, not a modelling one.

Two framing points for the paper. First, amortization pays off with the
**number** of networks, not the size of one: for a single very large network
RSiena fits once while we would simulate many networks of that size to train.
Second, three-wave and co-evolution panels cost roughly 2–3× the numbers above
(two periods; behaviour ministeps), and generation of a population across
sizes is dominated by its largest members — the Glasgow population
(n ≤ 200, three waves, rates to 20) ran at ~60 panels/s while sharing the GPU.

RSiena's own cost for reference: 11 s for one two-wave fit at n = 50, 24 s
for three waves at n = 50, 219 s (network) and 287 s (co-evolution) at n = 129.

## A second Spark, and where training time goes

Added 2026-11-11. Two DGX Sparks joined by their 200 Gb/s port, driven by
`tools/spark.sh`. What the second node buys:

- **Generation halves.** Shards are independent by seed, so `spark.sh shards`
  deals the seeds across the nodes and `collect` brings them to the training
  node over the direct link (1.7 GB of M6 shards in 7 s).
- **Paired runs take one run's time**, one arm per node, and SBC runs on the
  node that is not training instead of queueing behind it.
- **One training run is not faster.** The measurements below say why.

Every shard and estimator now records the node and commit that produced it.

### The step, not the loader

`benchmarks/profile_train.py`, one M6 shard, the M6 flow (8 transforms, 128
hidden, 745k parameters):

| feed | batch | fetch ms | step ms | rows/s |
|---|---|---|---|---|
| sbi `DataLoader` (CPU) | 1,024 | 3.6 | 30.9 | 29,600 |
| tensors on the GPU | 1,024 | 0.1 | 31.4 | 32,500 |
| sbi `DataLoader` (CPU) | 4,096 | 17.8 | 41.7 | 68,900 |
| tensors on the GPU | 4,096 | 0.1 | 39.3 | 104,000 |
| sbi `DataLoader` (CPU) | 16,384 | 75.5 | 97.5 | 94,700 |
| tensors on the GPU | 16,384 | 0.1 | 95.7 | 171,000 |

At the batch the estimators are trained with, the loader is a tenth of a step.
The step costs about 31 ms whatever feeds it, most of it fixed per-step
overhead in the spline flow: sixteen times the rows cost three times the time.
The GPU is doing the work (the same step on the CPU takes 187 ms). The table
predicts 5.6 min per 10⁷-row epoch; M8's 202 epochs in 18 h 12 m is 5.4. So
splitting one training run across two nodes would synchronise steps that are
already overhead-bound, and the only large lever is the batch size.

### Does a larger batch train the same estimator?

The M6 run repeated on the same ten shards, seed, flow and summaries, one arm
per node, with the learning rate scaled by the square root of the batch ratio
(so each arm changes two things) and the epoch cap raised. Fresh-population
SBC as for M6: 4,000 panels, 1,000 draws.

| | `npe_m6` | batch 4,096 | batch 16,384 |
|---|---|---|---|
| learning rate | 5e-4 | 1e-3 | 2e-3 |
| training time | 10 h 55 m | 11 h 20 m | **4 h 50 m** |
| epochs (best at) | 126 (106) | 251 (231) | 131 (111) |
| min per epoch | 5.2 | 2.7 | 2.2 |
| best validation loss | −0.036 | **−0.061** | +0.037 |
| mean 90 % coverage | 0.898 | 0.897 | 0.896 |
| mean 50 % coverage | 0.498 | 0.496 | 0.494 |
| rank KS p < 0.05 | 3 of 8 | 1 of 8 | 5 of 8 |

Neither arm is a free speed-up. Batch 16,384 finishes in under half the time
and is a worse estimator: 0.07 nats behind on a million validation rows, and
five rank distributions rejected against the baseline's three. Mean coverage
does not show it, which is a reminder that mean coverage is a blunt instrument.
Batch 4,096 took as long as the baseline, because early stopping let it run
twice the epochs, and ended 0.025 nats ahead with one rejection. Whether that
is the batch, the learning rate or simply the extra epochs this run cannot say.

One seed per arm. The three estimators also disagree on the s50 rate by about
half a posterior sd (means 5.2, 5.8, 5.0), which is the size of difference a
second seed would be needed to interpret.

Use: batch 16,384 for screening runs and ablations where a five-hour answer
is wanted and the SBC is read anyway; not for an estimator that ships.
