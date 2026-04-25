"""
server/environment.py

Core environment logic. Implements the query/apply pattern:

  query_cleaner      → CleanerAgent analyzes dataset, returns ranked recommendations
  query_augmenter    → AugmenterAgent suggests synthetic rows
  query_balancer     → BalancerAgent suggests resampling
  query_validator    → ValidatorAgent checks business rules (costs 2 budget)
  query_analyst      → AnalystAgent gives holistic diagnosis + action plan (costs 2 budget)
  apply              → Applies a pending recommendation by rec_id

The LLM acts as orchestrator: it queries agents for information, then
decides which recommendation to apply. This forces deliberate planning.
"""

import pandas as pd
from server.dataset_factory import DatasetFactory
from server.evaluator import Evaluator
from server.reward import compute, compute_stats
from server.specialist_agents import (
    CleanerAgent, AugmenterAgent, BalancerAgent, ValidatorAgent, AnalystAgent
)

MAX_BUDGET = 12  # slightly higher budget because query steps also cost budget

QUERY_ACTIONS = {
    "query_cleaner",
    "query_augmenter",
    "query_balancer",
    "query_validator",
    "query_analyst",
}

QUERY_COSTS = {
    "query_cleaner":   1,
    "query_augmenter": 1,
    "query_balancer":  1,
    "query_validator": 2,  # expensive — business rule checks
    "query_analyst":   2,  # expensive — full holistic diagnosis
}


class DataCentricEnvironment:

    def __init__(self):
        self.factory = DatasetFactory()
        self.evaluator = Evaluator()
        self.agents = {
            "cleaner":   CleanerAgent(),
            "augmenter": AugmenterAgent(),
            "balancer":  BalancerAgent(),
            "validator": ValidatorAgent(),
            "analyst":   AnalystAgent(),
        }
        self._episode_count = 0
        self._reset_state()

    def _reset_state(self):
        self.df: pd.DataFrame = None
        self.target_accuracy: float = None
        self.budget: int = MAX_BUDGET
        self.current_accuracy: float = 0.0
        self.episode_step: int = 0
        self.done: bool = False
        self.difficulty: str = "easy"
        # Pending recommendations: rec_id → {rec dict + agent name}
        self.pending_recs: dict = {}
        self.last_query_result: dict = {}
        self.applied_rec_ids: set = set()

    def reset(self, difficulty: str = None) -> dict:
        self._reset_state()
        self.difficulty = difficulty or self._next_difficulty()
        self.df, self.target_accuracy = self.factory.generate(self.difficulty)
        self.budget = MAX_BUDGET
        self.current_accuracy = self.evaluator.evaluate(self.df)
        return self._observation()

    def step(self, action: dict) -> dict:
        if self.done:
            return {"error": "Episode is done. Call /reset to start a new episode."}
        if self.df is None:
            return {"error": "Environment not initialized. Call /reset first."}

        action_type = action.get("action", "")

        # ── Query actions ──────────────────────────────────────────────────────
        if action_type in QUERY_ACTIONS:
            return self._handle_query(action_type, action)

        # ── Apply action ───────────────────────────────────────────────────────
        elif action_type == "apply":
            return self._handle_apply(action)

        # ── Unknown action ─────────────────────────────────────────────────────
        else:
            valid = list(QUERY_ACTIONS) + ["apply"]
            return {
                "error": (
                    f"Unknown action type: '{action_type}'. "
                    f"Valid actions: {valid}"
                )
            }

    # ── Query handler ──────────────────────────────────────────────────────────

    def _handle_query(self, action_type: str, action: dict) -> dict:
        cost = QUERY_COSTS[action_type]
        prev_stats = compute_stats(self.df)
        prev_accuracy = self.current_accuracy

        # Run the appropriate agent
        if action_type == "query_cleaner":
            result = self.agents["cleaner"].query(self.df)
        elif action_type == "query_augmenter":
            target_class = action.get("target_class", None)
            result = self.agents["augmenter"].query(self.df, target_class)
        elif action_type == "query_balancer":
            result = self.agents["balancer"].query(self.df)
        elif action_type == "query_validator":
            result = self.agents["validator"].query(self.df)
        elif action_type == "query_analyst":
            result = self.agents["analyst"].query(self.df)
        else:
            result = {}

        # Register returned recommendations
        new_rec_ids = []
        for rec in result.get("recommendations", []):
            rec_id = rec["id"]
            self.pending_recs[rec_id] = {
                "rec": rec,
                "agent": result.get("agent", "unknown"),
            }
            new_rec_ids.append(rec_id)

        self.last_query_result = result
        self.budget = max(0, self.budget - cost)
        self.episode_step += 1

        # Reward for a query step (no data changed)
        new_stats = compute_stats(self.df)
        reward = compute(
            prev_accuracy=prev_accuracy,
            new_accuracy=self.current_accuracy,  # unchanged
            prev_stats=prev_stats,
            new_stats=new_stats,
            action=action,
            steps_taken=self.episode_step,
            max_steps=MAX_BUDGET,
            budget_remaining=self.budget,
            target_accuracy=self.target_accuracy,
            step_type="query",
            n_recs_returned=len(new_rec_ids),
        )

        self.done = self.budget <= 0

        return {
            "observation": self._observation(),
            "reward": reward,
            "done": self.done,
            "query_result": result,
            "new_recommendation_ids": new_rec_ids,
            "info": {
                "action_type": "query",
                "agent_queried": action_type.replace("query_", ""),
                "budget_cost": cost,
                "budget_remaining": self.budget,
                "n_recommendations": len(new_rec_ids),
                "episode_step": self.episode_step,
            },
        }

    # ── Apply handler ──────────────────────────────────────────────────────────

    def _handle_apply(self, action: dict) -> dict:
        rec_id = action.get("rec_id", "")

        if not rec_id:
            return {"error": "apply action requires 'rec_id'. Example: {\"action\": \"apply\", \"rec_id\": \"clean_abc123\"}"}

        if rec_id not in self.pending_recs:
            available = list(self.pending_recs.keys())
            return {
                "error": (
                    f"rec_id '{rec_id}' not found in pending recommendations. "
                    f"Available: {available}. Query an agent first."
                )
            }

        if rec_id in self.applied_rec_ids:
            return {"error": f"Recommendation '{rec_id}' has already been applied."}

        entry = self.pending_recs[rec_id]
        agent_name = entry["agent"]
        rec = entry["rec"]

        prev_accuracy = self.current_accuracy
        prev_stats = compute_stats(self.df)

        # Execute the recommendation
        import threading
        result_holder = {}
        error_holder = {}

        def _run():
            try:
                agent = self.agents[agent_name]
                df_out, log = agent.apply(self.df, rec)
                result_holder["df"] = df_out
                result_holder["log"] = log
            except Exception as e:
                error_holder["error"] = str(e)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=10)

        if t.is_alive():
            return {"error": "Apply operation exceeded 10-second time limit."}
        if "error" in error_holder:
            return {"error": f"Apply error: {error_holder['error']}"}

        self.df = result_holder["df"]
        tool_log = result_holder["log"]
        self.applied_rec_ids.add(rec_id)

        # Apply costs 0 additional budget (cost was paid at query time)
        self.episode_step += 1

        # Re-evaluate
        self.current_accuracy = self.evaluator.evaluate(self.df)
        new_stats = compute_stats(self.df)

        # Reward
        reward = compute(
            prev_accuracy=prev_accuracy,
            new_accuracy=self.current_accuracy,
            prev_stats=prev_stats,
            new_stats=new_stats,
            action=action,
            steps_taken=self.episode_step,
            max_steps=MAX_BUDGET,
            budget_remaining=self.budget,
            target_accuracy=self.target_accuracy,
            step_type="apply",
            n_recs_returned=0,
        )

        self.done = (self.current_accuracy >= self.target_accuracy) or (self.budget <= 0)

        return {
            "observation": self._observation(),
            "reward": reward,
            "done": self.done,
            "tool_log": tool_log,
            "info": {
                "action_type": "apply",
                "rec_id": rec_id,
                "agent": agent_name,
                "rec_type": rec.get("type", "?"),
                "prev_accuracy": round(prev_accuracy, 4),
                "new_accuracy": round(self.current_accuracy, 4),
                "target_accuracy": self.target_accuracy,
                "budget_remaining": self.budget,
                "episode_step": self.episode_step,
                "success": self.current_accuracy >= self.target_accuracy,
            },
        }

    def state(self) -> dict:
        return self._observation()

    def _observation(self) -> dict:
        stats = compute_stats(self.df) if self.df is not None else {}

        # Summarize pending recs without exposing full internal state
        pending_summary = {}
        for rec_id, entry in self.pending_recs.items():
            if rec_id in self.applied_rec_ids:
                continue  # hide already-applied recs
            rec = entry["rec"]
            pending_summary[rec_id] = {
                "agent": entry["agent"],
                "type": rec.get("type", "?"),
                "priority": rec.get("priority", "?"),
                "reason": rec.get("reason", ""),
            }

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
            "pending_recommendations": pending_summary,
            "last_query_result": self.last_query_result,
            "available_actions": (
                "query_cleaner | query_augmenter | query_balancer | "
                "query_validator (cost 2) | query_analyst (cost 2) | "
                "apply {rec_id}"
            ),
        }

    def _next_difficulty(self) -> str:
        self._episode_count += 1
        if self._episode_count < 20:
            return "easy"
        elif self._episode_count < 50:
            return "medium"
        return "hard"
