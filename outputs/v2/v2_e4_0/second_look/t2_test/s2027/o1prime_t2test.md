# v2 pilot — O1' on T2-test (Amendment 8, §17 Q4-Q5)

Evaluation windows of `/content/second_look/stage/corpus` (T2-test, unseen by every arm). A0 = `A0`. One seed: every Δ is printed, **not a result**.

## T2-test

| arm | micro | macro | clip oracle | window AUC | O1 (registered) |
|---|---|---|---|---|---|
| A0 | 0.6226 | 0.6348 | 0.7037 | 0.6581 | PASS |
| A3ign | 0.7315 | 0.7061 | 0.7037 | 0.7810 | **FAIL** |

## T2-val (batch diag.json)

| arm | micro | macro | clip oracle | window AUC | O1 (registered) |
|---|---|---|---|---|---|
| A0 | 0.6454 | 0.6638 | 0.6986 | 0.6877 | PASS |
| A3ign | 0.7375 | 0.7334 | 0.6986 | 0.7496 | **FAIL** |

## Q5 verdict

* **A0** (reference): O1 on T2-val PASS
* **A3ign**: O1 T2-val FAIL (registered, kept) · O1 T2-test FAIL (printed)
  * T2-val  O1' (Amendment 8): Δ window AUC +0.0618, Δ macro +0.0696, Δ micro +0.0921; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * T2-test O1' (Amendment 8): Δ window AUC +0.1229, Δ macro +0.0713, Δ micro +0.1089; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * → **survivor, by O1' (Amendment 8)**

**Motion Stream: kept**
