# K5f / K5dfi / L0f / L0dfi trained on MSAD-full — KIM_RUN.md §41

**PRELIMINARY (n = 1).** Seed 2024; one seed per row. Both kernels (3, 9) are printed; nothing here picks a kernel or a row (§41 R2 / R6). Every row is trained on MSAD; no Nexar-trained row is compared.

## MSAD-traffic (140 videos, macro over 19, base rate 0.0316)

| row | AUC macro | micro | micro, equal video | AP | clip AUC |
|---|---|---|---|---|---|
| K5dfiM3 | 0.6772 [0.5535, 0.7866] | 0.9652 | 0.9466 | 0.4378 | 0.9796 |
| K5dfiM9 | 0.7007 [0.5840, 0.8054] | 0.9643 | 0.9443 | 0.4322 | 0.9821 |
| K5fM3 | 0.6393 [0.5162, 0.7501] | 0.9628 | 0.9433 | 0.4195 | 0.9788 |
| K5fM9 | 0.6437 [0.5279, 0.7514] | 0.9612 | 0.9397 | 0.4075 | 0.9788 |
| L0dfiM3 | 0.6721 [0.5495, 0.7876] | 0.9662 | 0.9459 | 0.4419 | 0.9758 |
| L0dfiM9 | 0.6802 [0.5588, 0.7936] | 0.9663 | 0.9455 | 0.4347 | 0.9750 |
| L0fM3 | 0.6967 [0.5821, 0.8061] | 0.9663 | 0.9462 | 0.4480 | 0.9746 |
| L0fM9 | 0.6879 [0.5612, 0.8071] | 0.9622 | 0.9412 | 0.4215 | 0.9688 |

Rulers (no pixel): clip oracle micro 0.9771, length ruler micro 0.9639, `t/N` macro 0.5874 [0.4213, 0.7525].

| contrast (a - b) | a | b | delta macro [95 %] | |
|---|---|---|---|---|
| motion + KIM, DVS off (k3) | K5fM3 | L0fM3 | -0.0574 [-0.1949, +0.0742] | not separated |
| motion + KIM, DVS ignore (k3) | K5dfiM3 | L0dfiM3 | +0.0051 [-0.1509, +0.1498] | not separated |
| DVS ignore on KIM (k3) | K5dfiM3 | K5fM3 | +0.0379 [+0.0191, +0.0595] | above |
| DVS ignore on plain CLIP (k3) | L0dfiM3 | L0fM3 | -0.0246 [-0.0467, -0.0011] | below |
| motion + KIM, DVS off (k9) | K5fM9 | L0fM9 | -0.0442 [-0.1454, +0.0611] | not separated |
| motion + KIM, DVS ignore (k9) | K5dfiM9 | L0dfiM9 | +0.0206 [-0.1020, +0.1459] | not separated |
| DVS ignore on KIM (k9) | K5dfiM9 | K5fM9 | +0.0570 [+0.0317, +0.0856] | above |
| DVS ignore on plain CLIP (k9) | L0dfiM9 | L0fM9 | -0.0077 [-0.0406, +0.0297] | not separated |
| kernel 9 - 3, K5f (printed, never a selection) | K5fM9 | K5fM3 | +0.0044 [-0.0229, +0.0348] | not separated |
| kernel 9 - 3, K5dfi (printed, never a selection) | K5dfiM9 | K5dfiM3 | +0.0235 [-0.0046, +0.0538] | not separated |
| kernel 9 - 3, L0f (printed, never a selection) | L0fM9 | L0fM3 | -0.0088 [-0.0608, +0.0470] | not separated |
| kernel 9 - 3, L0dfi (printed, never a selection) | L0dfiM9 | L0dfiM3 | +0.0080 [-0.0290, +0.0461] | not separated |

## MSAD-full (240 videos, macro over 96, base rate 0.2300)

| row | AUC macro | micro | micro, equal video | AP | clip AUC |
|---|---|---|---|---|---|
| K5dfiM3 | 0.7793 [0.7323, 0.8241] | 0.9070 | 0.8698 | 0.7190 | 0.9710 |
| K5dfiM9 | 0.7787 [0.7288, 0.8252] | 0.9041 | 0.8635 | 0.7094 | 0.9699 |
| K5fM3 | 0.7666 [0.7179, 0.8136] | 0.9073 | 0.8685 | 0.7171 | 0.9706 |
| K5fM9 | 0.7474 [0.6938, 0.7983] | 0.9042 | 0.8612 | 0.7054 | 0.9682 |
| L0dfiM3 | 0.7183 [0.6657, 0.7700] | 0.8891 | 0.8443 | 0.6697 | 0.9645 |
| L0dfiM9 | 0.7221 [0.6647, 0.7769] | 0.8948 | 0.8473 | 0.6674 | 0.9649 |
| L0fM3 | 0.7252 [0.6715, 0.7767] | 0.8920 | 0.8459 | 0.6716 | 0.9640 |
| L0fM9 | 0.7187 [0.6581, 0.7777] | 0.8974 | 0.8503 | 0.6840 | 0.9619 |

Rulers (no pixel): clip oracle micro 0.8943, length ruler micro 0.6466, `t/N` macro 0.7090 [0.6407, 0.7750].

| contrast (a - b) | a | b | delta macro [95 %] | |
|---|---|---|---|---|
| motion + KIM, DVS off (k3) | K5fM3 | L0fM3 | +0.0414 [-0.0063, +0.0885] | not separated |
| motion + KIM, DVS ignore (k3) | K5dfiM3 | L0dfiM3 | +0.0610 [+0.0135, +0.1076] | above |
| DVS ignore on KIM (k3) | K5dfiM3 | K5fM3 | +0.0127 [+0.0050, +0.0209] | above |
| DVS ignore on plain CLIP (k3) | L0dfiM3 | L0fM3 | -0.0069 [-0.0184, +0.0046] | not separated |
| motion + KIM, DVS off (k9) | K5fM9 | L0fM9 | +0.0287 [-0.0185, +0.0755] | not separated |
| motion + KIM, DVS ignore (k9) | K5dfiM9 | L0dfiM9 | +0.0566 [+0.0090, +0.1030] | above |
| DVS ignore on KIM (k9) | K5dfiM9 | K5fM9 | +0.0313 [+0.0207, +0.0426] | above |
| DVS ignore on plain CLIP (k9) | L0dfiM9 | L0fM9 | +0.0035 [-0.0130, +0.0212] | not separated |
| kernel 9 - 3, K5f (printed, never a selection) | K5fM9 | K5fM3 | -0.0191 [-0.0374, -0.0019] | below |
| kernel 9 - 3, K5dfi (printed, never a selection) | K5dfiM9 | K5dfiM3 | -0.0006 [-0.0164, +0.0144] | not separated |
| kernel 9 - 3, L0f (printed, never a selection) | L0fM9 | L0fM3 | -0.0065 [-0.0344, +0.0203] | not separated |
| kernel 9 - 3, L0dfi (printed, never a selection) | L0dfiM9 | L0dfiM3 | +0.0038 [-0.0191, +0.0262] | not separated |

## TAD (100 videos, macro over 60, base rate 0.0724)

| row | AUC macro | micro | micro, equal video | AP | clip AUC |
|---|---|---|---|---|---|
| K5dfiM3 | 0.7228 [0.6696, 0.7748] | 0.8760 | 0.6962 | 0.3075 | 0.8471 |
| K5dfiM9 | 0.7509 [0.6985, 0.8021] | 0.8777 | 0.6998 | 0.3067 | 0.8496 |
| K5fM3 | 0.7168 [0.6618, 0.7691] | 0.8780 | 0.6940 | 0.3060 | 0.8425 |
| K5fM9 | 0.7417 [0.6877, 0.7943] | 0.8772 | 0.6965 | 0.2992 | 0.8463 |
| L0dfiM3 | 0.7603 [0.7078, 0.8092] | 0.8934 | 0.7194 | 0.3134 | 0.8800 |
| L0dfiM9 | 0.7646 [0.7020, 0.8231] | 0.8892 | 0.7109 | 0.2949 | 0.8662 |
| L0fM3 | 0.7676 [0.7130, 0.8181] | 0.8916 | 0.7157 | 0.3090 | 0.8738 |
| L0fM9 | 0.7569 [0.6895, 0.8192] | 0.8845 | 0.6995 | 0.2764 | 0.8617 |

Rulers (no pixel): clip oracle micro 0.9226, length ruler micro 0.8968, `t/N` macro 0.5872 [0.5126, 0.6611].

| contrast (a - b) | a | b | delta macro [95 %] | |
|---|---|---|---|---|
| motion + KIM, DVS off (k3) | K5fM3 | L0fM3 | -0.0508 [-0.0992, -0.0052] | below |
| motion + KIM, DVS ignore (k3) | K5dfiM3 | L0dfiM3 | -0.0375 [-0.0797, +0.0028] | not separated |
| DVS ignore on KIM (k3) | K5dfiM3 | K5fM3 | +0.0060 [-0.0051, +0.0168] | not separated |
| DVS ignore on plain CLIP (k3) | L0dfiM3 | L0fM3 | -0.0073 [-0.0167, +0.0019] | not separated |
| motion + KIM, DVS off (k9) | K5fM9 | L0fM9 | -0.0152 [-0.0648, +0.0366] | not separated |
| motion + KIM, DVS ignore (k9) | K5dfiM9 | L0dfiM9 | -0.0137 [-0.0602, +0.0330] | not separated |
| DVS ignore on KIM (k9) | K5dfiM9 | K5fM9 | +0.0092 [-0.0005, +0.0190] | not separated |
| DVS ignore on plain CLIP (k9) | L0dfiM9 | L0fM9 | +0.0078 [-0.0139, +0.0311] | not separated |
| kernel 9 - 3, K5f (printed, never a selection) | K5fM9 | K5fM3 | +0.0249 [-0.0145, +0.0647] | not separated |
| kernel 9 - 3, K5dfi (printed, never a selection) | K5dfiM9 | K5dfiM3 | +0.0281 [-0.0073, +0.0643] | not separated |
| kernel 9 - 3, L0f (printed, never a selection) | L0fM9 | L0fM3 | -0.0107 [-0.0564, +0.0309] | not separated |
| kernel 9 - 3, L0dfi (printed, never a selection) | L0dfiM9 | L0dfiM3 | +0.0043 [-0.0305, +0.0366] | not separated |

## How to read this

* Micro on MSAD measures clip classification (oracle in the rulers); read macro first.
* Never write "the better kernel": it would be chosen on the set it is read on.
* TAD micro is clip classification (oracle 0.9226); TAD macro is the localization read.
