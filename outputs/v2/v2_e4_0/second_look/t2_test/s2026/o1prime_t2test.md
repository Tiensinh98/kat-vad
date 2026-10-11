# v2 pilot — O1' on T2-test (Amendment 8, §17 Q4-Q5)

Evaluation windows of `/content/second_look/stage/corpus` (T2-test, unseen by every arm). A0 = `A0`. One seed: every Δ is printed, **not a result**.

## T2-test

| arm | micro | macro | clip oracle | window AUC | O1 (registered) |
|---|---|---|---|---|---|
| A0 | 0.6169 | 0.6232 | 0.7037 | 0.6641 | PASS |
| A3ign | 0.7346 | 0.7279 | 0.7037 | 0.7728 | **FAIL** |

## T2-val (batch diag.json)

| arm | micro | macro | clip oracle | window AUC | O1 (registered) |
|---|---|---|---|---|---|
| A0 | 0.6417 | 0.6326 | 0.6986 | 0.6717 | **FAIL** |
| A3ign | 0.7441 | 0.7350 | 0.6986 | 0.7542 | **FAIL** |

## Q5 verdict

* **A0** (reference): O1 on T2-val **FAIL**
* **A3ign**: O1 T2-val FAIL (registered, kept) · O1 T2-test FAIL (printed)
  * T2-val  O1' (Amendment 8): Δ window AUC +0.0824, Δ macro +0.1024, Δ micro +0.1024; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * T2-test O1' (Amendment 8): Δ window AUC +0.1087, Δ macro +0.1047, Δ micro +0.1178; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * → **survivor, by O1' (Amendment 8)**

**Motion Stream: kept**
