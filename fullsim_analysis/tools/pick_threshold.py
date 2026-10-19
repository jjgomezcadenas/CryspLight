"""Pick a trigger threshold from a trigger-scan table and write it into a front-end
configuration. Rule (agreed 18 Oct 2026): the lowest threshold of the given mode whose
511 keV efficiency is at least 0.995 and whose dark trigger rate is below 1 kHz at 95%
confidence (the upper limit of the count of dark triggers).

    python3 fullsim_analysis/tools/pick_threshold.py <trigger_scan.csv> <tag> <mode> <config.toml> <key>
"""
import re
import sys

import pandas as pd

csv, tag, mode, cfg, key = sys.argv[1:6]
d = pd.read_csv(csv)
d = d[(d.tag == tag) & (d["mode"] == mode)].sort_values("threshold")
ok = d[(d.efficiency_511 >= 0.995) & (d.dark_khz_ul95 < 1.0)]
if ok.empty:
    sys.exit(f"no threshold of {tag} {mode} satisfies the rule")
r = ok.iloc[0]
s = open(cfg).read()
s, n = re.subn(rf"^{key} = [\d.]+.*$",
               f"{key} = {r.threshold}   # from the trigger scan of 19 Oct: 511 keV {r.efficiency_511:.3f}, "
               f"200-300 keV {r.efficiency_200_300:.3f}, dark {r.dark_khz:.2f} kHz (below {r.dark_khz_ul95:.2f} at 95%)", s, flags=re.M)
assert n == 1, f"{key} not found in {cfg}"
open(cfg, "w").write(s)
print(f"{tag} {mode}: threshold {r.threshold} ({r.get('threshold_pe', r.threshold)} pe), 511 keV {r.efficiency_511:.3f}, "
      f"200-300 keV {r.efficiency_200_300:.3f}, dark {r.dark_khz:.2f} kHz -> {cfg}")
