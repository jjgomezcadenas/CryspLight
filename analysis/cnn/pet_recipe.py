#!/usr/bin/env python3
"""PET-simulation resolution recipe from the selection fits (tracked source).

Parameterizes the first-interaction position response per coordinate as a
two-Gaussian mixture with SCENARIO-INDEPENDENT shapes (sigma1, sigma2 frozen
to the no-selection fits) and scenario-dependent core weight:

    delta ~ f_core * N(0, sigma1) + (1 - f_core) * N(0, sigma2)

Scenarios: no selection, and the reconstructability selection at 80% / 60%
single-photon acceptance (the acceptance is an efficiency factor per detected
511 keV photon; a coincidence pays it twice).

Reads results/cnn/selection_fits/selection_fits.json; writes
pet_resolution_recipe.json next to it and prints the table.

Usage: python3 analysis/cnn/pet_recipe.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FITS = HERE.parent / "results" / "cnn" / "selection_fits"
SCENARIOS = [("none", 1.0), ("80%", 0.8), ("60%", 0.6)]


def main():
    fits = json.loads((FITS / "selection_fits.json").read_text())
    recipe = {"model": "delta ~ f_core*N(0,sigma1) + (1-f_core)*N(0,sigma2), "
                       "per coordinate, mm",
              "acceptance_is_per_single_photon_efficiency": True,
              "source": "CryspLight window samples; shapes frozen to the "
                        "no-selection two-Gaussian fits",
              "scanners": {}}
    for crystal, label in (("csitl", "CsI(Tl)"), ("bgo", "BGO")):
        sc = {}
        for name, eff in SCENARIOS:
            coords = {}
            for lab in "xyz":
                shape = fits[crystal]["none"][lab]
                r = (shape["A2_over_A1"] if name == "none"
                     else fits[crystal][name][lab]["fixed_shape"]["A2_over_A1"])
                coords[lab] = {"sigma1_mm": shape["sigma1_mm"],
                               "sigma2_mm": shape["sigma2_mm"],
                               "f_core": round(1.0 / (1.0 + r), 4)}
            sc[name] = {"efficiency": eff, "coords": coords}
        recipe["scanners"][label] = sc
    (FITS / "pet_resolution_recipe.json").write_text(json.dumps(recipe, indent=2))

    for label, sc in recipe["scanners"].items():
        print(f"\n{label}")
        print(f"{'scenario':>9s} {'eff':>5s}"
              + "".join(f" | {c}: sig1 sig2 f_core" for c in "xyz"))
        for name, v in sc.items():
            row = f"{name:>9s} {v['efficiency']:5.2f}"
            for lab in "xyz":
                c = v["coords"][lab]
                row += (f" | {c['sigma1_mm']:5.2f} {c['sigma2_mm']:5.2f} "
                        f"{c['f_core']:6.3f}")
            print(row)
    print(f"\nwritten: {FITS / 'pet_resolution_recipe.json'}")


if __name__ == "__main__":
    main()
