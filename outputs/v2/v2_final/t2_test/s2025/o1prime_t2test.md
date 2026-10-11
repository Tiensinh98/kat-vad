# v2 pilot — O1' on T2-test (Amendment 8, §17 Q4-Q5)

Evaluation windows of `/content/final/stage/corpus` (T2-test, unseen by every arm). A0 = `A0`. One seed: every Δ is printed, **not a result**.

## T2-test

| arm | micro | macro | clip oracle | window AUC | O1 (registered) |
|---|---|---|---|---|---|
| A0 | 0.6200 | 0.6378 | 0.7037 | 0.6680 | PASS |
| A1 | 0.6523 | 0.6390 | 0.7037 | 0.7073 | **FAIL** |
| A2 | 0.7081 | 0.7018 | 0.7037 | 0.7642 | **FAIL** |
| A3 | 0.7036 | 0.6911 | 0.7037 | 0.7690 | **FAIL** |

## T2-val (batch diag.json)

| arm | micro | macro | clip oracle | window AUC | O1 (registered) |
|---|---|---|---|---|---|
| A0 | 0.6591 | 0.6616 | 0.6986 | 0.6856 | PASS |
| A1 | 0.6698 | 0.6643 | 0.6986 | 0.6868 | **FAIL** |
| A2 | 0.6939 | 0.6847 | 0.6986 | 0.7685 | **FAIL** |
| A3 | 0.7139 | 0.6889 | 0.6986 | 0.7664 | **FAIL** |

## Q5 verdict

* **A0** (reference): O1 on T2-val PASS
* **A1**: O1 T2-val FAIL (registered, kept) · O1 T2-test FAIL (printed)
  * T2-val  O1' (Amendment 8): Δ window AUC +0.0012, Δ macro +0.0027, Δ micro +0.0107; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * T2-test O1' (Amendment 8): Δ window AUC +0.0393, Δ macro +0.0013, Δ micro +0.0323; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * → **survivor, by O1' (Amendment 8)**
* **A2**: O1 T2-val FAIL (registered, kept) · O1 T2-test FAIL (printed)
  * T2-val  O1' (Amendment 8): Δ window AUC +0.0829, Δ macro +0.0230, Δ micro +0.0348; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * T2-test O1' (Amendment 8): Δ window AUC +0.0962, Δ macro +0.0641, Δ micro +0.0880; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * → **survivor, by O1' (Amendment 8)**
* **A3**: O1 T2-val FAIL (registered, kept) · O1 T2-test FAIL (printed)
  * T2-val  O1' (Amendment 8): Δ window AUC +0.0808, Δ macro +0.0272, Δ micro +0.0548; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * T2-test O1' (Amendment 8): Δ window AUC +0.1010, Δ macro +0.0533, Δ micro +0.0836; not_collapsed PASS, micro_within_a0_margin PASS → **PASS**
  * → **survivor, by O1' (Amendment 8)**

**Motion Stream: kept**
