import pickle
from pathlib import Path

import joblib


class ExperimentManager:
    def __init__(self):
        self.experiments = {}
        self.current_experiment = None

    def save(self, name: str, model, config, metrics, history):
        self.experiments[name] = {
            "model": model,
            "config": config.dict(),
            "metrics": metrics,
            "training_history": history["values"],
            "training_dates": history["dates"]
        }
        self.current_experiment = name

    def get(self, name: str):
        return self.experiments.get(name)

    def compare(self, names: list[str]):
        existing = set(self.experiments)
        found = [n for n in names if n in existing]
        missing = list(set(names) - existing)

        results = {
            "experiments": {
                n: {
                    "metrics": self.experiments[n]["metrics"],
                    "config": self.experiments[n]["config"],
                    "training_history": {
                        "dates": self.experiments[n]["training_dates"],
                        "values": self.experiments[n]["training_history"]
                    }
                } for n in found
            },
            "missing_experiments": {"count": len(missing), "names": missing} if missing else None
        }

        return results

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
