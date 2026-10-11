# v2 pilot — O1' on T2-test (Amendment 8, §17 Q4-Q5)

Evaluation windows of `/content/second_look/stage/corpus` (T2-test, unseen by every arm). A0 = `A0`. One seed: every Δ is printed, **not a result**.

## T2-test

| arm | micro | macro | clip oracle | window AUC | O1 (registered) |
|---|---|---|---|---|---|
| A0 | 0.6356 | 0.6299 | 0.7037 | 0.6778 | **FAIL** |
| A3ign | 0.7308 | 0.7160 | 0.7037 | 0.7710 | **FAIL** |

## T2-val (batch diag.json)

| arm | micro | macro | clip oracle | window AUC | O1 (registered) |
|---|---|---|---|---|---|
| A0 | 0.6458 | 0.6587 | 0.6986 | 0.6835 | PASS |
| A3ign | 0.7410 | 0.7345 | 0.6986 | 0.7616 | **FAIL** |

## Q5 verdict

* **A0** (reference): O1 on T2-val PASS
* **A3ign**: O1 T2-val FAIL (registered, kept) · O1 T2-test FAIL (printed)
  * T2-val  O1' (Amendment 8): Δ window AUC +0.0781, Δ macro +0.0758, Δ micro +0.0952; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * T2-test O1' (Amendment 8): Δ window AUC +0.0932, Δ macro +0.0861, Δ micro +0.0952; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * → **survivor, by O1' (Amendment 8)**

**Motion Stream: kept**
