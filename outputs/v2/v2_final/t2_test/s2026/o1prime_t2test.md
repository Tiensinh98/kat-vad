# v2 pilot — O1' on T2-test (Amendment 8, §17 Q4-Q5)

Evaluation windows of `/content/final/stage/corpus` (T2-test, unseen by every arm). A0 = `A0`. One seed: every Δ is printed, **not a result**.

## T2-test

| arm | micro | macro | clip oracle | window AUC | O1 (registered) |
|---|---|---|---|---|---|
| A0 | 0.6169 | 0.6232 | 0.7037 | 0.6641 | PASS |
| A1 | 0.6619 | 0.6263 | 0.7037 | 0.7166 | **FAIL** |
| A2 | 0.7230 | 0.7156 | 0.7037 | 0.7767 | **FAIL** |
| A3 | 0.7084 | 0.6989 | 0.7037 | 0.7663 | **FAIL** |

## T2-val (batch diag.json)

| arm | micro | macro | clip oracle | window AUC | O1 (registered) |
|---|---|---|---|---|---|
| A0 | 0.6417 | 0.6326 | 0.6986 | 0.6717 | **FAIL** |
| A1 | 0.6615 | 0.6469 | 0.6986 | 0.6996 | **FAIL** |
| A2 | 0.7115 | 0.6990 | 0.6986 | 0.7718 | **FAIL** |
| A3 | 0.7147 | 0.7071 | 0.6986 | 0.7493 | **FAIL** |

## Q5 verdict

* **A0** (reference): O1 on T2-val **FAIL**
* **A1**: O1 T2-val FAIL (registered, kept) · O1 T2-test FAIL (printed)
  * T2-val  O1' (Amendment 8): Δ window AUC +0.0279, Δ macro +0.0142, Δ micro +0.0198; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * T2-test O1' (Amendment 8): Δ window AUC +0.0525, Δ macro +0.0031, Δ micro +0.0450; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * → **survivor, by O1' (Amendment 8)**
* **A2**: O1 T2-val FAIL (registered, kept) · O1 T2-test FAIL (printed)
  * T2-val  O1' (Amendment 8): Δ window AUC +0.1001, Δ macro +0.0664, Δ micro +0.0697; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * T2-test O1' (Amendment 8): Δ window AUC +0.1127, Δ macro +0.0925, Δ micro +0.1061; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * → **survivor, by O1' (Amendment 8)**
* **A3**: O1 T2-val FAIL (registered, kept) · O1 T2-test FAIL (printed)
  * T2-val  O1' (Amendment 8): Δ window AUC +0.0776, Δ macro +0.0745, Δ micro +0.0730; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * T2-test O1' (Amendment 8): Δ window AUC +0.1022, Δ macro +0.0757, Δ micro +0.0916; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * → **survivor, by O1' (Amendment 8)**

**Motion Stream: kept**
