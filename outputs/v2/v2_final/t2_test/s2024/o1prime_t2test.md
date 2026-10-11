# v2 pilot — O1' on T2-test (Amendment 8, §17 Q4-Q5)

Evaluation windows of `/content/final/stage/corpus` (T2-test, unseen by every arm). A0 = `A0`. One seed: every Δ is printed, **not a result**.

## T2-test

| arm | micro | macro | clip oracle | window AUC | O1 (registered) |
|---|---|---|---|---|---|
| A0 | 0.6189 | 0.6277 | 0.7037 | 0.6654 | PASS |
| A1 | 0.6563 | 0.6370 | 0.7037 | 0.7039 | **FAIL** |
| A2 | 0.7155 | 0.7126 | 0.7037 | 0.7633 | **FAIL** |
| A3 | 0.7144 | 0.7001 | 0.7037 | 0.7676 | **FAIL** |

## T2-val (batch diag.json)

| arm | micro | macro | clip oracle | window AUC | O1 (registered) |
|---|---|---|---|---|---|
| A0 | 0.6446 | 0.6473 | 0.6986 | 0.6732 | PASS |
| A1 | 0.6738 | 0.6500 | 0.6986 | 0.7029 | **FAIL** |
| A2 | 0.7109 | 0.6828 | 0.6986 | 0.7791 | **FAIL** |
| A3 | 0.7126 | 0.7092 | 0.6986 | 0.7574 | **FAIL** |

## Q5 verdict

* **A0** (reference): O1 on T2-val PASS
* **A1**: O1 T2-val FAIL (registered, kept) · O1 T2-test FAIL (printed)
  * T2-val  O1' (Amendment 8): Δ window AUC +0.0297, Δ macro +0.0027, Δ micro +0.0292; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * T2-test O1' (Amendment 8): Δ window AUC +0.0385, Δ macro +0.0093, Δ micro +0.0374; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * → **survivor, by O1' (Amendment 8)**
* **A2**: O1 T2-val FAIL (registered, kept) · O1 T2-test FAIL (printed)
  * T2-val  O1' (Amendment 8): Δ window AUC +0.1059, Δ macro +0.0355, Δ micro +0.0663; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * T2-test O1' (Amendment 8): Δ window AUC +0.0979, Δ macro +0.0849, Δ micro +0.0966; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * → **survivor, by O1' (Amendment 8)**
* **A3**: O1 T2-val FAIL (registered, kept) · O1 T2-test FAIL (printed)
  * T2-val  O1' (Amendment 8): Δ window AUC +0.0842, Δ macro +0.0619, Δ micro +0.0680; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * T2-test O1' (Amendment 8): Δ window AUC +0.1022, Δ macro +0.0724, Δ micro +0.0955; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * → **survivor, by O1' (Amendment 8)**

**Motion Stream: kept**
