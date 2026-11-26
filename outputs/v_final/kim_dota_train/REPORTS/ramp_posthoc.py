import glob, sys
import numpy as np
O = "outputs/v_final"
ROWS = {"K5dfi": f"{O}/kim_dvs_ignore/K5dfi/eval/{{s}}/K5/s2024/scores",
        "K5dfiDoTA": f"{O}/kim_dota_train/K5dfiDoTA/eval/{{s}}/K5/s2024/scores"}
BINS = 5
def curves(path, normal_only):
    out = []
    for f in sorted(glob.glob(path + "/*.npz")):
        z = np.load(f); s, g = z["score"], z["gt"]
        if normal_only and g.sum() > 0: continue
        if len(s) < BINS: continue
        b = np.array_split(s, BINS); out.append([x.mean() for x in b])
    return np.array(out)
for s in ("dad", "tad"):
    for name, p in ROWS.items():
        c = curves(p.format(s=s), True)
        if len(c) == 0: print(s, name, "no files"); continue
        m = c.mean(0); late = (c[:, -1] > c[:, 0]).mean()
        print(f"{s:4s} {name:10s} n={len(c):3d} mean score by fifth: {np.round(m,3)}  share last>first {late:.2f}")
