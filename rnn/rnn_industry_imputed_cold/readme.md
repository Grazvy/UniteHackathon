1. NACE-sector priors — after loading data, builds a dict mapping each nace_code → top eclasses ranked by (order_count + revenue/1e6) across all training customers in that sector.
  Also builds a global_prior fallback for unknown NACE codes.
  2. Inference split — at prediction time:
    - Cold-start (cid not in sequences): skip the model entirely, use the NACE-sector ranked list directly with the economic filter. This gives industry-specific, data-driven
  predictions instead of the model getting confused by zero-padded input.
    - Warm-start (cid in sequences): unchanged — runs through the RNN as before.


€246,072.17
New best score!

Savings: €255,412.17
Fees: €9,340.00
Hits: 826
Spend Captured: 10.19%
