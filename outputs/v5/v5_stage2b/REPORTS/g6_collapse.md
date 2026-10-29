# S1-G6 + §2.4 B13 G0: **FAIL**

Entropy bound 0.999 log 49 = 3.8879; moved = some query below 0.95 log 49 = 3.6972 (**required**, B13 G0).

| run | finite | entropy per query | did not leave the mean | moved | logit std per query |
|---|---|---|---|---|---|
| R5q/s2024 | {'mil': True, 'smil': True} | 3.733, 3.779, 3.714, 3.843, 3.763, 3.822, 3.863, 3.789 | no | no | 0.554, 0.580, 0.575, 0.297, 0.530, 0.359, 0.233, 0.462 |
| R5dq/s2024 | {'mil': True, 'smil': True} | 3.739, 3.745, 3.695, 3.813, 3.730, 3.753, 3.810, 3.792 | no | yes | 0.545, 0.578, 0.599, 0.376, 0.568, 0.494, 0.370, 0.453 |
