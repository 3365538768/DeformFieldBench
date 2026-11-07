"""Cross-engine result contract; engine adapters must produce real measurements.

No analytic placeholder is presented as a successfully calibrated simulator.
"""
from dataclasses import dataclass
from typing import Protocol
import numpy as np


@dataclass
class SimulationResult:
    time_seconds: np.ndarray
    positions_m: np.ndarray
    contact_force_n: np.ndarray
    kinetic_energy_j: np.ndarray
    strain_energy_j: np.ndarray
    metadata: dict

    def validate(self):
        t = self.time_seconds
        if t.ndim != 1 or len(t) < 2 or not np.all(np.diff(t)>0):
            raise ValueError("Simulation timestamps must increase in SI seconds")
        if self.positions_m.ndim != 3 or self.positions_m.shape[0] != len(t) or self.positions_m.shape[-1] != 3:
            raise ValueError("Positions must be [T,N,3] metres")
        if self.contact_force_n.shape != (len(t), 3):
            raise ValueError("Contact force must be [T,3] newtons")
        for a in (t,self.positions_m,self.contact_force_n,self.kinetic_energy_j,self.strain_energy_j):
            if not np.isfinite(a).all():
                raise ValueError("Nonfinite simulator output")
        for name in ("engine", "commit", "dt_seconds", "grid_or_mesh", "boundary_mode", "length_scale_m"):
            if name not in self.metadata:
                raise ValueError(f"Missing simulator provenance {name}")
        return self


class EngineAdapter(Protocol):
    def run(self, scene: dict, material: dict, interaction: dict) -> SimulationResult: ...


CALIBRATION_CASES = (
    "uniaxial_force_displacement", "small_strain_cantilever", "free_oscillation_decay",
    "drop_rebound", "mass_volume_conservation", "dt_grid_convergence", "contact_penetration",
    "bonded_two_material_interface")
