import numpy as np


class GridCellEncoder:
    """Multi-module grid-cell encoding of a 2D position.

    Each cell behaves like one cell of a hexagonal grid-cell module
    (Solstad et al. 2006): three plane waves at 60-degree offsets summed
    and normalized to [0, 1]. Cells differ in:
      - scale (lattice spacing) — geometric progression from min_scale to max_scale
      - orientation — random in [0, pi/3) (lattice has 60-degree symmetry)
      - phase offset — random translation within one period

    Shapes are vectorized so encode() returns all activations in one shot.
    """

    def __init__(self, num_cells, min_scale=2.0, max_scale=40.0, seed=0):
        self.num_cells = int(num_cells)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        rng = np.random.default_rng(seed)

        if self.num_cells <= 0:
            raise ValueError(f"num_cells must be > 0, got {self.num_cells}")

        if self.num_cells == 1:
            self.scales = np.array([self.max_scale], dtype=np.float64)
        else:
            self.scales = np.geomspace(
                self.min_scale, self.max_scale, self.num_cells
            ).astype(np.float64)

        self.orientations = rng.uniform(0.0, np.pi / 3.0, size=self.num_cells)
        # phase offset in arena coords, uniform within one lattice spacing
        self.phases = np.stack(
            [
                rng.uniform(-self.scales, self.scales),
                rng.uniform(-self.scales, self.scales),
            ],
            axis=1,
        )  # (N, 2)

        # Wave-vector magnitude that gives lattice spacing equal to `scale`
        # for a hexagonal lattice formed by three plane waves at 60 deg.
        k_mag = 4.0 * np.pi / (np.sqrt(3.0) * self.scales)  # (N,)
        wave_vecs = np.zeros((self.num_cells, 3, 2), dtype=np.float64)
        for i in range(3):
            angle = self.orientations + i * np.pi / 3.0
            wave_vecs[:, i, 0] = k_mag * np.cos(angle)
            wave_vecs[:, i, 1] = k_mag * np.sin(angle)
        self.wave_vecs = wave_vecs  # (N, 3, 2)

    def encode(self, x, y):
        """Return per-cell activations in [0, 1] for the 2D position (x, y)."""
        pos = np.array([float(x), float(y)], dtype=np.float64)
        rel = pos[None, :] - self.phases  # (N, 2)
        dots = np.einsum("nik,nk->ni", self.wave_vecs, rel)  # (N, 3)
        g = (2.0 / 3.0) * np.cos(dots).sum(axis=1)  # range [-1, 2]
        return ((g + 1.0) / 3.0).astype(np.float32)  # range [0, 1]
