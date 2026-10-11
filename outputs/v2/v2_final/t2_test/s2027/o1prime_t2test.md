# v2 pilot — O1' on T2-test (Amendment 8, §17 Q4-Q5)

Evaluation windows of `/content/final/stage/corpus` (T2-test, unseen by every arm). A0 = `A0`. One seed: every Δ is printed, **not a result**.

## T2-test

| arm | micro | macro | clip oracle | window AUC | O1 (registered) |
|---|---|---|---|---|---|
| A0 | 0.6226 | 0.6348 | 0.7037 | 0.6581 | PASS |
| A1 | 0.6645 | 0.6413 | 0.7037 | 0.7260 | **FAIL** |
| A2 | 0.7193 | 0.7094 | 0.7037 | 0.7808 | **FAIL** |
| A3 | 0.7116 | 0.6875 | 0.7037 | 0.7754 | **FAIL** |

## T2-val (batch diag.json)

| arm | micro | macro | clip oracle | window AUC | O1 (registered) |
|---|---|---|---|---|---|
| A0 | 0.6454 | 0.6638 | 0.6986 | 0.6877 | PASS |
| A1 | 0.6675 | 0.6637 | 0.6986 | 0.7194 | **FAIL** |
| A2 | 0.7185 | 0.6995 | 0.6986 | 0.7768 | **FAIL** |
| A3 | 0.7104 | 0.7104 | 0.6986 | 0.7474 | **FAIL** |

## Q5 verdict

* **A0** (reference): O1 on T2-val PASS
* **A1**: O1 T2-val FAIL (registered, kept) · O1 T2-test FAIL (printed)
  * T2-val  O1' (Amendment 8): Δ window AUC +0.0317, Δ macro -0.0001, Δ micro +0.0222; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * T2-test O1' (Amendment 8): Δ window AUC +0.0678, Δ macro +0.0065, Δ micro +0.0419; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * → **survivor, by O1' (Amendment 8)**
* **A2**: O1 T2-val FAIL (registered, kept) · O1 T2-test FAIL (printed)
  * T2-val  O1' (Amendment 8): Δ window AUC +0.0891, Δ macro +0.0357, Δ micro +0.0731; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * T2-test O1' (Amendment 8): Δ window AUC +0.1227, Δ macro +0.0746, Δ micro +0.0967; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * → **survivor, by O1' (Amendment 8)**
* **A3**: O1 T2-val FAIL (registered, kept) · O1 T2-test FAIL (printed)
  * T2-val  O1' (Amendment 8): Δ window AUC +0.0597, Δ macro +0.0466, Δ micro +0.0651; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * T2-test O1' (Amendment 8): Δ window AUC +0.1173, Δ macro +0.0526, Δ micro +0.0890; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * → **survivor, by O1' (Amendment 8)**

**Motion Stream: kept**
