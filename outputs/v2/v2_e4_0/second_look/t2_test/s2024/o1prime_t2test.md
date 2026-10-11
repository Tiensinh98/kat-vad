# v2 pilot — O1' on T2-test (Amendment 8, §17 Q4-Q5)

Evaluation windows of `/content/second_look/stage/corpus` (T2-test, unseen by every arm). A0 = `A0`. One seed: every Δ is printed, **not a result**.

## T2-test

| arm | micro | macro | clip oracle | window AUC | O1 (registered) |
|---|---|---|---|---|---|
| A0 | 0.6189 | 0.6277 | 0.7037 | 0.6654 | PASS |
| A3ign | 0.7389 | 0.7259 | 0.7037 | 0.7692 | **FAIL** |

## T2-val (batch diag.json)

| arm | micro | macro | clip oracle | window AUC | O1 (registered) |
|---|---|---|---|---|---|
| A0 | 0.6446 | 0.6473 | 0.6986 | 0.6732 | PASS |
| A3ign | 0.7444 | 0.7366 | 0.6986 | 0.7824 | **FAIL** |

## Q5 verdict

* **A0** (reference): O1 on T2-val PASS
* **A3ign**: O1 T2-val FAIL (registered, kept) · O1 T2-test FAIL (printed)
  * T2-val  O1' (Amendment 8): Δ window AUC +0.1093, Δ macro +0.0893, Δ micro +0.0998; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * T2-test O1' (Amendment 8): Δ window AUC +0.1039, Δ macro +0.0982, Δ micro +0.1200; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * → **survivor, by O1' (Amendment 8)**

**Motion Stream: kept**
