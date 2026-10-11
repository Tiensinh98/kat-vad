# KAT-VAD v2 — E3 read-out (2 x 2 x 5 seeds)

Rules: PREREG\_ADDENDUM.md §18 (Amendment 9, M1-M10); proposal §10.1-§10.3. Seeds s2024, s2025, s2026, s2027, s2028; decision interval \= paired t95 (t \= 2.776). **MDE\_E3 \= 0.0282** (E3-internal, not E0b; M7).

## Decision: **adopt A3** (F \= A0)

* v2 as a whole (CRN \+ V2-S) improves DoTA-CAP (n/1397) over A0.  
* The video stream (V2-S) improves DoTA-CAP (n/1397) on top of CRN (A3 \- A1).  
* Never write "motion" of the stream: D6 found no temporal-order effect (§15 G3).

## Contrasts (decision interval, per-seed macro Δ)

| contrast | set | mean Δ | t95 | per seed | clip-level Δ (printed) | reading |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| A1 \- A0 | DoTA-dev | \-0.0151 | \[-0.0372, \+0.0070\] | \+0.0050 / \-0.0228 / \+0.0029 / \-0.0276 / \-0.0330 | \-0.0191 \[-0.0358, \-0.0028\] | not detectable at MDE\_E3 \= 0.0282 |
| A2 \- A0 | DoTA-CAP-dev (n/1397) | \+0.0924 | \[+0.0505, \+0.1343\] | \+0.1183 / \+0.0522 / \+0.1349 / \+0.0842 / \+0.0725 | \+0.0768 \[+0.0546, \+0.0999\] | excludes 0 |
| A3 \- A0 | DoTA-CAP-dev (n/1397) | \+0.0855 | \[+0.0573, \+0.1137\] | \+0.1161 / \+0.0678 / \+0.1022 / \+0.0779 / \+0.0636 | \+0.0808 \[+0.0613, \+0.1000\] | excludes 0 |
| A3 \- A1 | DoTA-CAP-dev (n/1397) | \+0.0978 | \[+0.0863, \+0.1093\] | \+0.1094 / \+0.0869 / \+0.0926 / \+0.1053 / \+0.0949 | \+0.0999 \[+0.0833, \+0.1172\] | excludes 0 |
| A2 \- A1 | DoTA-CAP-dev (n/1397) | \+0.1047 | \[+0.0796, \+0.1298\] | \+0.1116 / \+0.0713 / \+0.1253 / \+0.1116 / \+0.1038 | \+0.0959 \[+0.0713, \+0.1204\] | excludes 0 |

&nbsp;

Interaction (A3 \- A2) \- (A1 \- A0) on dota\_cap\_dev; descriptive only (§10.3): \+0.0054.

## Guardrails (T2-val, per seed)

| arm | rule | per seed | hold |
| :---- | :---- | :---- | :---- |
| A0 | O1 (printed) | ✓ ✓ ✗ ✓ ✓ | **no** |
| A1 | O1' every seed (M2) | ✓ ✓ ✓ ✓ ✓ | yes |
| A2 | O1' every seed (M2) | ✓ ✓ ✓ ✓ ✓ | yes |
| A3 | O1' every seed (M2) | ✓ ✓ ✓ ✓ ✓ | yes |

&nbsp;

## Free rule for A1 (M5)

* dota\_dev\_mean\_ge\_0: FAIL  
* t2\_val\_macro\_mean\_ge\_0: PASS  
* no\_reversed\_share\_bin: FAIL  
* T2-val macro A1 \- A0 (seed mean): \+0.0065  
* DoTA-dev clip-level Δ(A1 \- A0) per share bin: \<30 \-0.0114 \[-0.0366, \+0.0133\], 30-50 \-0.0171 \[-0.0498, \+0.0157\], 50-70 \-0.0480 \[-0.0918, \-0.0052\], \>70 \-0.0389 \[-0.1784, \+0.1156\]

## Printed, never decided on (M6)

| arm | DoTA-dev macro | DoTA-CAP-dev macro | T2-val micro | T2-val macro | window AUC | shortcut V^t | shortcut y^bin | pos R² V^t | rho\_u last | Spearman(y, p\_T2) |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| A0 | 0.6793 \[0.6592, 0.6996\] | 0.6768 \[0.6538, 0.6994\] | 0.6473 | 0.6528 | 0.6803 | 0.9998 | 0.5558 | \-0.2322 | — | 0.3312 |
| A1 | 0.6602 \[0.6426, 0.6779\] | 0.6577 \[0.6379, 0.6766\] | 0.6681 | 0.6593 | 0.7008 | 0.5690 | 0.5301 | \-0.0310 | — | 0.2759 |
| A2 | — | 0.7536 \[0.7351, 0.7725\] | 0.7081 | 0.6892 | 0.7734 | 0.9997 | 0.5278 | \-0.0973 | 0.1326 | 0.5856 |
| A3 | — | 0.7576 \[0.7404, 0.7745\] | 0.7124 | 0.7034 | 0.7566 | 0.5892 | 0.5410 | 0.0152 | 0.1084 | 0.5057 |

&nbsp;

DoTA-dev and DoTA-CAP-dev columns are different clip sets (D14): never compare them.

Position: `p_T2` DoTA-dev 0.8187 \[0.7997, 0.8377\], DoTA-CAP-dev 0.8220 \[0.8026, 0.8415\]; `p_CAP` (in-domain) 0.852; `t/N` (monotone) DoTA-dev 0.5663 \[0.5465, 0.5863\], DoTA-CAP-dev 0.5573 \[0.5380, 0.5765\].

* G6 Δ Spearman(A2 \- A0) on DoTA-CAP-dev: \+0.2544 \[+0.2172, \+0.2905\] — **this gain co-moves with the position prior**  
* G6 Δ Spearman(A3 \- A1) on DoTA-CAP-dev: \+0.2299 \[+0.2018, \+0.2579\] — **this gain co-moves with the position prior**

### Macro per accident-share bin

| arm | set | \<30 | 30-50 | 50-70 | \>70 |
| :---- | :---- | :---- | :---- | :---- | :---- |
| A0 | DoTA-dev | 0.6785 \[0.6503, 0.7059\] | 0.6858 \[0.6503, 0.7199\] | 0.6673 \[0.6152, 0.7198\] | 0.6680 \[0.5444, 0.7815\] |
| A0 | DoTA-CAP-dev (n/1397) | 0.6739 \[0.6413, 0.7046\] | 0.6931 \[0.6549, 0.7297\] | 0.6512 \[0.5912, 0.7142\] | 0.6326 \[0.4887, 0.7669\] |
| A1 | DoTA-dev | 0.6671 \[0.6411, 0.6924\] | 0.6687 \[0.6358, 0.7021\] | 0.6193 \[0.5678, 0.6696\] | 0.6291 \[0.5136, 0.7417\] |
| A1 | DoTA-CAP-dev (n/1397) | 0.6693 \[0.6401, 0.6966\] | 0.6605 \[0.6279, 0.6950\] | 0.6174 \[0.5583, 0.6788\] | 0.5934 \[0.4653, 0.7122\] |
| A2 | DoTA-CAP-dev (n/1397) | 0.7405 \[0.7149, 0.7656\] | 0.7716 \[0.7400, 0.8029\] | 0.7512 \[0.7017, 0.8019\] | 0.7697 \[0.6608, 0.8633\] |
| A3 | DoTA-CAP-dev (n/1397) | 0.7537 \[0.7283, 0.7778\] | 0.7687 \[0.7414, 0.7958\] | 0.7425 \[0.6828, 0.8006\] | 0.7534 \[0.6319, 0.8607\] |

&nbsp;