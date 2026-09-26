"""Scenario loading and run-directory layout (everything lives on D:)."""
import json
from pathlib import Path

HYDROSIM_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = HYDROSIM_ROOT.parent
RUNS_ROOT = HYDROSIM_ROOT / "runs"


class Scenario:
    def __init__(self, path):
        self.path = Path(path).resolve()
        with open(self.path, encoding="utf-8") as f:
            self.cfg = json.load(f)
        self.name = self.cfg["name"]
        self.run_dir = RUNS_ROOT / self.name
        self.scenario_dir = self.run_dir / "scenario"
        self.delft3d_dir = self.run_dir / "delft3d"
        self.sph_dir = self.run_dir / "sph"
        self.results_dir = self.run_dir / "results"
        self.viewer_dir = self.run_dir / "viewer"
        for d in (self.scenario_dir, self.delft3d_dir, self.sph_dir, self.results_dir, self.viewer_dir):
            d.mkdir(parents=True, exist_ok=True)

    def __getitem__(self, key):
        return self.cfg[key]

    def input_path(self, key):
        return PROJECT_ROOT / self.cfg["inputs"][key]

    @property
    def terrain_file(self):
        return self.scenario_dir / "terrain.npz"

    @property
    def meta_file(self):
        return self.scenario_dir / "scenario_meta.json"

    def load_meta(self):
        with open(self.meta_file, encoding="utf-8") as f:
            return json.load(f)
