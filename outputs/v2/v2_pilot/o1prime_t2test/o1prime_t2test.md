# v2 pilot — O1' on T2-test (Amendment 8, §17 Q4-Q5)

Evaluation windows of `/content/o1p/stage/corpus` (T2-test, unseen by every arm). A0 = `A0`. One seed: every Δ is printed, **not a result**.

## T2-test

| arm | micro | macro | clip oracle | window AUC | O1 (registered) |
|---|---|---|---|---|---|
| A0 | 0.6320 | 0.6389 | 0.7037 | 0.6637 | PASS |
| A1 | 0.6558 | 0.6350 | 0.7037 | 0.7117 | **FAIL** |
| A2 | 0.7192 | 0.7136 | 0.7037 | 0.7650 | **FAIL** |
| A3 | 0.7090 | 0.6897 | 0.7037 | 0.7804 | **FAIL** |

## T2-val (batch diag.json)

| arm | micro | macro | clip oracle | window AUC | O1 (registered) |
|---|---|---|---|---|---|
| A0 | 0.6586 | 0.6737 | 0.6986 | 0.6863 | PASS |
| A1 | 0.6683 | 0.6705 | 0.6986 | 0.6828 | PASS |
| A2 | 0.7120 | 0.6882 | 0.6986 | 0.7767 | **FAIL** |
| A3 | 0.7161 | 0.7052 | 0.6986 | 0.7637 | **FAIL** |

## Q5 verdict

* **A0** (reference): O1 on T2-val PASS
* **A1**: O1 T2-val PASS (registered, kept) · O1 T2-test FAIL (printed)
  * T2-val  O1' (Amendment 8): Δ window AUC -0.0035, Δ macro -0.0031, Δ micro +0.0097; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * T2-test O1' (Amendment 8): Δ window AUC +0.0480, Δ macro -0.0039, Δ micro +0.0238; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * → **survivor, by O1**
* **A2**: O1 T2-val FAIL (registered, kept) · O1 T2-test FAIL (printed)
  * T2-val  O1' (Amendment 8): Δ window AUC +0.0905, Δ macro +0.0145, Δ micro +0.0534; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * T2-test O1' (Amendment 8): Δ window AUC +0.1014, Δ macro +0.0747, Δ micro +0.0872; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * → **survivor, by O1' (Amendment 8)**
* **A3**: O1 T2-val FAIL (registered, kept) · O1 T2-test FAIL (printed)
  * T2-val  O1' (Amendment 8): Δ window AUC +0.0775, Δ macro +0.0315, Δ micro +0.0575; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * T2-test O1' (Amendment 8): Δ window AUC +0.1167, Δ macro +0.0508, Δ micro +0.0770; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * → **survivor, by O1' (Amendment 8)**

**Motion Stream: kept**
