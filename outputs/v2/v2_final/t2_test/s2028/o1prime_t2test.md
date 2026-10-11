# v2 pilot — O1' on T2-test (Amendment 8, §17 Q4-Q5)

Evaluation windows of `/content/final/stage/corpus` (T2-test, unseen by every arm). A0 = `A0`. One seed: every Δ is printed, **not a result**.

## T2-test

| arm | micro | macro | clip oracle | window AUC | O1 (registered) |
|---|---|---|---|---|---|
| A0 | 0.6356 | 0.6299 | 0.7037 | 0.6778 | **FAIL** |
| A1 | 0.6619 | 0.6449 | 0.7037 | 0.7284 | **FAIL** |
| A2 | 0.7106 | 0.7006 | 0.7037 | 0.7723 | **FAIL** |
| A3 | 0.7092 | 0.6974 | 0.7037 | 0.7749 | **FAIL** |

## T2-val (batch diag.json)

| arm | micro | macro | clip oracle | window AUC | O1 (registered) |
|---|---|---|---|---|---|
| A0 | 0.6458 | 0.6587 | 0.6986 | 0.6835 | PASS |
| A1 | 0.6680 | 0.6717 | 0.6986 | 0.6952 | PASS |
| A2 | 0.7057 | 0.6798 | 0.6986 | 0.7710 | **FAIL** |
| A3 | 0.7103 | 0.7013 | 0.6986 | 0.7622 | **FAIL** |

## Q5 verdict

* **A0** (reference): O1 on T2-val PASS
* **A1**: O1 T2-val PASS (registered, kept) · O1 T2-test FAIL (printed)
  * T2-val  O1' (Amendment 8): Δ window AUC +0.0118, Δ macro +0.0130, Δ micro +0.0221; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * T2-test O1' (Amendment 8): Δ window AUC +0.0506, Δ macro +0.0150, Δ micro +0.0263; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * → **survivor, by O1**
* **A2**: O1 T2-val FAIL (registered, kept) · O1 T2-test FAIL (printed)
  * T2-val  O1' (Amendment 8): Δ window AUC +0.0875, Δ macro +0.0211, Δ micro +0.0599; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * T2-test O1' (Amendment 8): Δ window AUC +0.0945, Δ macro +0.0707, Δ micro +0.0750; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * → **survivor, by O1' (Amendment 8)**
* **A3**: O1 T2-val FAIL (registered, kept) · O1 T2-test FAIL (printed)
  * T2-val  O1' (Amendment 8): Δ window AUC +0.0788, Δ macro +0.0426, Δ micro +0.0645; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * T2-test O1' (Amendment 8): Δ window AUC +0.0971, Δ macro +0.0675, Δ micro +0.0736; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * → **survivor, by O1' (Amendment 8)**

**Motion Stream: kept**
