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
