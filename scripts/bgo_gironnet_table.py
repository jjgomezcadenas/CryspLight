#!/usr/bin/env python3
"""bgo_gironnet_table.py: BGO light yield and decay time between 190 K and room temperature,
from Gironnet et al., NIM A 594 (2008) 358 (papers/tau_dependence_BGO.txt).

    python3 scripts/bgo_gironnet_table.py

Decay time: the three-level model of their Eq. (1) with their fitted parameters
(k1 = 7.5e3 /s, k2 = 4.5e5 /s, Delta = 6.4 meV, K = 1.3e8 /s, DeltaE = 105 meV). It gives
0.44 us at 295 K and 2.08 us at 200 K, against their measured 0.43 +- 0.08 and 2.2 +- 0.2 us.

Light: their Fig. 1 (alpha excitation, 241Am), digitised by eye from the linear-scale inset
(reading error about +-1 in the peak-integral units of the figure). The stored array has a
plateau of about 50.3 below 100 K and 16.4 at 295 K (interpolated), a ratio of 3.07 between
the plateau and room temperature; the text of the paper quotes about 3.5 (their 23 700 ph/MeV
at 6 K over the 6900 ph/MeV of Moszynski et al. at 295 K is 3.43). The array therefore does
not reproduce the plateau-to-295 K ratio of the text, and the two anchors cannot both be
used: the table is normalised to 6900 ph/MeV at 295 K only, and a column scaled to the
8000 ph/MeV adopted by the full simulation is added for convenience.

Digitisation uncertainty on the ratio used by the simulation, 250 K over 295 K: 1.69 from the
array; the text (half of the maximum at 250 K, a factor of about 3.5 between 6 K and 295 K)
implies about 1.75. Scaled to 8000 ph/MeV at room temperature this is 13 500 to 14 000
ph/MeV at 250 K; the simulation adopts 13 500 (data/bgo_250k.toml). Writes
data/bgo_gironnet_T.csv.
"""
from pathlib import Path

import numpy as np

KB = 8.617e-5                                   # eV/K
K1, K2, D, K, DE = 7.5e3, 4.5e5, 6.4e-3, 1.3e8, 0.105
T_DIG = np.array([50, 76, 100, 125, 150, 176, 201, 226, 241, 250, 260, 266, 275, 285, 296, 300, 308])
L_DIG = np.array([50.6, 50.1, 50.3, 45.9, 42.8, 40.3, 37.7, 33.1, 28.9, 27.7, 24.9, 23.7, 22.0,
                  19.4, 16.1, 15.4, 11.7])
Y295 = 6900.0           # Moszynski et al., the reference of the paper (gamma excitation)
Y295_SIM = 8000.0       # the room-temperature reference adopted by the full simulation
OUT = Path(__file__).resolve().parents[1] / "data" / "bgo_gironnet_T.csv"


def tau_us(T):
    x = np.exp(-D / (KB * T))
    return 1e6 / ((K1 + K2 * x) / (1 + x) + K * np.exp(-DE / (KB * T)))


def light(T):
    return np.interp(T, T_DIG, L_DIG)


if __name__ == "__main__":
    plateau = L_DIG[T_DIG <= 100].mean()
    print(f"digitised array: plateau {plateau:.1f} below 100 K, {light(295):.1f} at 295 K, "
          f"ratio {plateau / light(295):.2f} (text of the paper: about 3.5); "
          f"250 K / 295 K = {light(250) / light(295):.3f} (text: about 1.75)")
    rows = ["T_K,T_C,L_over_L295,yield_ph_per_MeV_ref6900,yield_ph_per_MeV_ref8000,tau_us"]
    for T in [190, 196, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 295]:
        r = light(T) / light(295)
        rows.append(f"{T},{T - 273.15:.1f},{r:.2f},{Y295 * r:.0f},{Y295_SIM * r:.0f},{tau_us(T):.2f}")
    OUT.write_text("\n".join(rows) + "\n")
    print("\n".join(rows)); print("written", OUT)
