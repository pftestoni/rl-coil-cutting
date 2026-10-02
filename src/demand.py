import numpy as np
from dataclasses import dataclass

@dataclass
class DemandConfig:
    model: str  # "negbin" | "bernoulli_lognormal"
    group_probs: list
    mean_units_by_group: list
    nb_dispersion_k: float
    noise_sigma_frac: float
    cap_fraction_of_capacity_mm: float
    target_utilization: float
    tolerance: float

class DemandGenerator:
    def __init__(self, cfg: DemandConfig, seed_groups: int, seed_demands: int,
                 n_skus: int, widths_mm: np.ndarray, capacity_sheets_per_day: int, coil_width_mm: int):
        self.cfg = cfg
        self.rng_groups = np.random.default_rng(seed_groups)
        self.rng_dem = np.random.default_rng(seed_demands)
        self.n = n_skus
        self.widths = widths_mm.astype(np.int32)
        self.cap_sheets = capacity_sheets_per_day
        self.coil = coil_width_mm
        self.groups = self.rng_groups.choice(len(cfg.group_probs), size=self.n, p=np.array(cfg.group_probs))
        base_cap_mm = float(capacity_sheets_per_day) * float(coil_width_mm)
        self.daily_target_mm = int(round(base_cap_mm * float(cfg.target_utilization)))

    def _neg_bin(self, mean: float, k: float):
        if mean <= 0:
            return 0
        p = k / (k + mean)
        r = k
        return self.rng_dem.negative_binomial(r, p)

    def sample_day_units(self) -> np.ndarray:
        units = np.zeros(self.n, dtype=np.int32)
        if self.cfg.model == "negbin":
            for i in range(self.n):
                mu = float(self.cfg.mean_units_by_group[self.groups[i]])
                u = self._neg_bin(mu, float(self.cfg.nb_dispersion_k))
                if u > 0 and self.cfg.noise_sigma_frac > 0:
                    noise = self.rng_dem.normal(1.0, self.cfg.noise_sigma_frac)
                    u = max(0, int(round(u * max(0.0, noise))))
                units[i] = u
        else:
            for i in range(self.n):
                mu = float(self.cfg.mean_units_by_group[self.groups[i]])
                units[i] = self.rng_dem.poisson(mu)

        total_mm = int(np.dot(units, self.widths))

        lower_bound = int(round(self.daily_target_mm * (1.0 - float(self.cfg.tolerance))))
        upper_bound = int(round(self.daily_target_mm * (1.0 + float(self.cfg.tolerance))))

        if total_mm > upper_bound and total_mm > 0:
            alpha = upper_bound / total_mm
            scaled = np.floor(units * alpha).astype(np.int32)
            resid_mm = upper_bound - int(np.dot(scaled, self.widths))
            if resid_mm > 0:
                order = np.argsort(-self.widths)
                for idx in order:
                    if resid_mm >= self.widths[idx]:
                        scaled[idx] += 1
                        resid_mm -= int(self.widths[idx])
                        if resid_mm <= 0:
                            break
            units = scaled
            total_mm = int(np.dot(units, self.widths))

        attempts = 0
        while total_mm < lower_bound and attempts < 1000:
            idx = int(self.rng_dem.integers(0, self.n))
            units[idx] += 1
            total_mm += int(self.widths[idx])
            if total_mm > upper_bound:
                units[idx] -= 1
                total_mm -= int(self.widths[idx])
            attempts += 1

        return units
