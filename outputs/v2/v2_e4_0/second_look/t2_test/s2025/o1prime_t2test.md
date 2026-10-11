# v2 pilot — O1' on T2-test (Amendment 8, §17 Q4-Q5)

Evaluation windows of `/content/second_look/stage/corpus` (T2-test, unseen by every arm). A0 = `A0`. One seed: every Δ is printed, **not a result**.

## T2-test

| arm | micro | macro | clip oracle | window AUC | O1 (registered) |
|---|---|---|---|---|---|
| A0 | 0.6200 | 0.6378 | 0.7037 | 0.6680 | PASS |
| A3ign | 0.7310 | 0.7249 | 0.7037 | 0.7628 | **FAIL** |

## T2-val (batch diag.json)

| arm | micro | macro | clip oracle | window AUC | O1 (registered) |
|---|---|---|---|---|---|
| A0 | 0.6591 | 0.6616 | 0.6986 | 0.6856 | PASS |
| A3ign | 0.7467 | 0.7296 | 0.6986 | 0.7807 | **FAIL** |

## Q5 verdict

* **A0** (reference): O1 on T2-val PASS
* **A3ign**: O1 T2-val FAIL (registered, kept) · O1 T2-test FAIL (printed)
  * T2-val  O1' (Amendment 8): Δ window AUC +0.0951, Δ macro +0.0679, Δ micro +0.0876; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * T2-test O1' (Amendment 8): Δ window AUC +0.0947, Δ macro +0.0871, Δ micro +0.1110; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * → **survivor, by O1' (Amendment 8)**

**Motion Stream: kept**
