import pandas as pd
from server.dataset_factory import DatasetFactory
from server.evaluator import Evaluator
from server.reward import compute, compute_stats
from server import specialists

SPECIALIST_MAP = {
    "cleaner":   specialists.cleaner.run,
    "augmenter": specialists.augmenter.run,
    "balancer":  specialists.balancer.run,
    "relabeler": specialists.relabeler.run,
    "validator": specialists.validator.run,
}

MAX_BUDGET = 10


class DataCentricEnvironment:
    def __init__(self):
        self.factory = DatasetFactory()
        self.evaluator = Evaluator()
        self.df = None
        self.target_accuracy = None
        self.budget = MAX_BUDGET
        self.current_accuracy = 0.0
        self.episode_step = 0
        self.done = False
        self.difficulty = "easy"
        self._episode_count = 0

    def reset(self, difficulty=None):
        self.difficulty = difficulty or self._next_difficulty()
        self.df, self.target_accuracy = self.factory.generate(self.difficulty)
        self.budget = MAX_BUDGET
        self.episode_step = 0
        self.done = False
        self.current_accuracy = self.evaluator.evaluate(self.df)
        return self._observation()

    def step(self, action: dict):
        if self.done:
            return {"error": "Episode is done. Call /reset to start a new episode."}

        agent_name = action.get("agent")
        if agent_name not in SPECIALIST_MAP:
            return {"error": f"Unknown agent: {agent_name}. Available: {list(SPECIALIST_MAP.keys())}"}

        prev_accuracy = self.current_accuracy
        prev_stats = compute_stats(self.df)

        # Execute specialist with a timeout via threading (cross-platform)
        import threading
        relabeler_used = (agent_name == "relabeler")
        result_holder = {}
        error_holder = {}

        def _run():
            try:
                result_holder["result"] = SPECIALIST_MAP[agent_name](self.df, action)
            except Exception as e:
                error_holder["error"] = str(e)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=10)  # 10-second hard limit per step

        if t.is_alive():
            return {"error": "Specialist tool exceeded 10-second time limit."}
        if "error" in error_holder:
            return {"error": f"Specialist error: {error_holder['error']}"}

        result = result_holder["result"]
        self.df = result["df"]
        tool_log = result["log"]

        # Budget cost (relabeler costs 2)
        budget_cost = 2 if relabeler_used else 1
        self.budget = max(0, self.budget - budget_cost)
        self.episode_step += 1

        # Evaluate
        self.current_accuracy = self.evaluator.evaluate(self.df)
        new_stats = compute_stats(self.df)

        # Reward
        reward = compute(
            prev_accuracy, self.current_accuracy,
            prev_stats, new_stats,
            action, self.episode_step, MAX_BUDGET,
            self.budget, self.target_accuracy,
            relabeler_used
        )

        # Done conditions
        self.done = (self.current_accuracy >= self.target_accuracy) or (self.budget <= 0)

        return {
            "observation": self._observation(),
            "reward": reward,
            "done": self.done,
            "tool_log": tool_log,
            "info": {
                "prev_accuracy": round(prev_accuracy, 4),
                "new_accuracy": round(self.current_accuracy, 4),
                "target_accuracy": self.target_accuracy,
                "budget_remaining": self.budget,
                "episode_step": self.episode_step,
                "success": self.current_accuracy >= self.target_accuracy,
            }
        }

    def state(self):
        return self._observation()

    def _observation(self):
        stats = compute_stats(self.df) if self.df is not None else {}
        return {
            "current_accuracy": round(self.current_accuracy, 4),
            "target_accuracy": self.target_accuracy,
            "budget_remaining": self.budget,
            "difficulty": self.difficulty,
            "dataset_stats": {
                "n_rows": len(self.df) if self.df is not None else 0,
                "n_cols": len(self.df.columns) if self.df is not None else 0,
                "missing_pct": round(stats.get("missing_pct", 0), 4),
                "balance_ratio": round(stats.get("balance_ratio", 0), 4),
            },
            "available_tools": list(SPECIALIST_MAP.keys()),
        }

    def _next_difficulty(self):
        # Simple curriculum: start easy, increase over episodes
        self._episode_count += 1
        if self._episode_count < 20:
            return "easy"
        elif self._episode_count < 50:
            return "medium"
        return "hard"
