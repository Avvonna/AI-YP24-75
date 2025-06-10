import pickle
from pathlib import Path
from typing import Any, Optional

import joblib
from app.configs import ModelConfigUnion
from app.schemas import (
    ComparisonResult,
    ExperimentMetrics,
    ExperimentRecord,
    TickerHistory,
)


class ExperimentManager:
    def __init__(self):
        self.experiments: dict[str, ExperimentRecord] = {}
        self.current_experiment: Optional[str] = None

    def save(self, name: str, model: Any, config: ModelConfigUnion, metrics: ExperimentMetrics, history: TickerHistory):
        record = ExperimentRecord(
            name=name,
            model=model,
            config=config.model_dump(),
            metrics=metrics,
            training_data=history
        )
        self.experiments[name] = record
        self.current_experiment = name

    def get_all(self) -> list[str]:
        return list(self.experiments.keys())

    def get(self, name: str) -> Optional[ExperimentRecord]:
        return self.experiments.get(name)

    def compare(self, names: list[str]) -> ComparisonResult:
        existing = set(self.get_all())
        found = [n for n in names if n in existing]
        missing = list(set(names) - existing)

        return ComparisonResult(
            experiments=[self.experiments[n] for n in found],
            missing_experiments=missing
        )

    def save_to_file(self, name: str, path: str, fmt="pickle"):
        path = Path(path)
        model_data = self.experiments.get(name)
        if fmt == "pickle":
            with open(path, "wb") as f:
                pickle.dump(model_data, f)
        elif fmt == "joblib":
            joblib.dump(model_data, path)

    def load_from_file(self, name: str, path: str, fmt="pickle"):
        path = Path(path)
        if fmt == "pickle":
            with open(path, "rb") as f:
                data = pickle.load(f)
        elif fmt == "joblib":
            data = joblib.load(path)
        self.experiments[name] = data
        self.current_experiment = name
