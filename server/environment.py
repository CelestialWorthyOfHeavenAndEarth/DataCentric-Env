"""
server/environment.py — Session-aware, thread-safe environment.

One DataCentricEnvironment instance per session (managed by SessionManager).
All actions go through AntiExploit checks before executing.
All events are structured-logged.
"""
import threading
import pandas as pd
from server.dataset_factory import DatasetFactory
from server.evaluator import Evaluator
from server.reward import compute, compute_stats
from server.anti_exploit import AntiExploit, ExploitDetected
from server.config import cfg
from server.logger import get_logger, log_event
from server.specialist_agents import (
    CleanerAgent, AugmenterAgent, BalancerAgent, ValidatorAgent, AnalystAgent
)

logger = get_logger("environment")

QUERY_COSTS = {
    "query_cleaner":   1,
    "query_augmenter": 1,
    "query_balancer":  1,
    "query_validator": 2,
    "query_analyst":   2,
}
QUERY_ACTIONS = set(QUERY_COSTS.keys())


class DataCentricEnvironment:

    def __init__(self, session_id: str, episode_count: int = 0):
        self.session_id = session_id
        self._episode_count = episode_count
        self.factory = DatasetFactory()
        self.evaluator = Evaluator()
        self.agents = {
            "cleaner":   CleanerAgent(),
            "augmenter": AugmenterAgent(),
            "balancer":  BalancerAgent(),
            "validator": ValidatorAgent(),
            "analyst":   AnalystAgent(),
        }
        self.anti_exploit = AntiExploit()
        self._lock = threading.Lock()
        self._reset_internal_state()

    def _reset_internal_state(self):
        self.df: pd.DataFrame = None
        self.golden_row_ids: set = set()
        self.target_accuracy: float = None
        self.budget: int = cfg.MAX_BUDGET
        self.current_accuracy: float = 0.0
        self.episode_step: int = 0
        self.done: bool = False
        self.difficulty: str = "easy"
        self.pending_recs: dict = {}
        self.applied_rec_ids: set = set()
        self.last_query_result: dict = {}
        self.anti_exploit.reset()
        # Metrics
        self.accuracy_history: list = []
        self.reward_history: list = []

    # ── Public interface ───────────────────────────────────────────────────────

    def _clean_df(self, df):
        """Strip metadata columns before passing to agents or evaluator."""
        drop_cols = [c for c in df.columns if c.startswith("_")]
        return df.drop(columns=drop_cols) if drop_cols else df

    def reset(self, difficulty: str = None) -> dict:
        with self._lock:
            self._episode_count += 1
            self._reset_internal_state()
            self.difficulty = difficulty or self._curriculum_difficulty()
            self.df, self.target_accuracy, self.golden_row_ids = self.factory.generate(self.difficulty)
            self.current_accuracy = self.evaluator.evaluate(self._clean_df(self.df))
            self.accuracy_history.append(self.current_accuracy)

            log_event(logger, "episode_reset",
                      session_id=self.session_id,
                      difficulty=self.difficulty,
                      initial_accuracy=round(self.current_accuracy, 4),
                      target_accuracy=self.target_accuracy,
                      n_rows=len(self.df))
            return self._observation()

    def step(self, action: dict) -> dict:
        with self._lock:
            if self.done:
                return self._error("Episode is done. Call /reset to start a new episode.")
            if self.df is None:
                return self._error("Environment not initialized. Call /reset first.")

            # Anti-exploit check FIRST
            try:
                self.anti_exploit.check(
                    action=action,
                    budget_remaining=self.budget,
                    pending_recs=self.pending_recs,
                    applied_rec_ids=self.applied_rec_ids,
                )
            except ExploitDetected as e:
                log_event(logger, "exploit_detected",
                          session_id=self.session_id,
                          rule=e.rule, detail=e.detail,
                          action=action)
                # Return minimum reward — don't crash, just penalize
                self.episode_step += 1
                self.budget = max(0, self.budget - 1)
                self.done = self.budget <= 0
                return {
                    "observation": self._observation(),
                    "reward": 0.001,
                    "done": self.done,
                    "exploit_detected": True,
                    "error": f"[{e.rule}] {e.detail}",
                    "info": {"episode_step": self.episode_step, "budget_remaining": self.budget},
                }

            action_type = action.get("action", "")

            if action_type in QUERY_ACTIONS:
                return self._handle_query(action_type, action)
            elif action_type == "apply":
                return self._handle_apply(action)
            else:
                valid = list(QUERY_ACTIONS) + ["apply"]
                return self._error(f"Unknown action '{action_type}'. Valid: {valid}")

    def state(self) -> dict:
        with self._lock:
            return self._observation()

    # ── Query handler ──────────────────────────────────────────────────────────

    def _handle_query(self, action_type: str, action: dict) -> dict:
        cost = QUERY_COSTS[action_type]
        prev_stats = compute_stats(self.df)

        if action_type == "query_cleaner":
            result = self.agents["cleaner"].query(self._clean_df(self.df))
        elif action_type == "query_augmenter":
            result = self.agents["augmenter"].query(self._clean_df(self.df), action.get("target_class"))
        elif action_type == "query_balancer":
            result = self.agents["balancer"].query(self._clean_df(self.df))
        elif action_type == "query_validator":
            result = self.agents["validator"].query(self._clean_df(self.df))
        elif action_type == "query_analyst":
            result = self.agents["analyst"].query(self._clean_df(self.df))
        else:
            result = {}

        new_rec_ids = []
        for rec in result.get("recommendations", []):
            rid = rec["id"]
            self.pending_recs[rid] = {"rec": rec, "agent": result.get("agent", "unknown")}
            new_rec_ids.append(rid)

        self.last_query_result = result
        self.budget = max(0, self.budget - cost)
        self.episode_step += 1
        new_stats = compute_stats(self.df)

        reward = compute(
            prev_accuracy=self.current_accuracy,
            new_accuracy=self.current_accuracy,
            prev_stats=prev_stats,
            new_stats=new_stats,
            action=action,
            steps_taken=self.episode_step,
            max_steps=cfg.MAX_BUDGET,
            budget_remaining=self.budget,
            target_accuracy=self.target_accuracy,
            step_type="query",
            n_recs_returned=len(new_rec_ids),
        )
        self.reward_history.append(reward)
        self.done = self.budget <= 0

        log_event(logger, "query_step",
                  session_id=self.session_id,
                  action=action_type, cost=cost,
                  n_recs=len(new_rec_ids),
                  budget_remaining=self.budget,
                  reward=reward)

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
        entry = self.pending_recs[rec_id]
        agent_name = entry["agent"]
        rec = entry["rec"]

        prev_accuracy = self.current_accuracy
        prev_stats = compute_stats(self.df)

        # Execute with timeout
        result_holder: dict = {}
        error_holder: dict = {}

        def _run():
            try:
                df_out, log_msg = self.agents[agent_name].apply(self._clean_df(self.df), rec)
                # Re-attach metadata columns from original df
                meta_cols = [c for c in self.df.columns if c.startswith("_")]
                for mc in meta_cols:
                    if mc not in df_out.columns:
                        df_out[mc] = self.df[mc].iloc[0] if len(self.df) > 0 else None
                result_holder["df"] = df_out
                result_holder["log"] = log_msg
            except Exception as e:
                error_holder["error"] = str(e)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=cfg.STEP_TIMEOUT_SECONDS)

        if t.is_alive():
            return self._error("Apply operation exceeded time limit.")
        if "error" in error_holder:
            return self._error(f"Apply error: {error_holder['error']}")

        new_df = result_holder["df"]
        tool_log = result_holder["log"]

        # Golden row integrity check — did the operation corrupt any golden rows?
        golden_penalty = self._check_golden_rows(self.df, new_df)

        self.df = new_df
        self.applied_rec_ids.add(rec_id)
        self.episode_step += 1
        self.current_accuracy = self.evaluator.evaluate(self._clean_df(self.df))
        self.accuracy_history.append(self.current_accuracy)
        new_stats = compute_stats(self.df)

        reward = compute(
            prev_accuracy=prev_accuracy,
            new_accuracy=self.current_accuracy,
            prev_stats=prev_stats,
            new_stats=new_stats,
            action=action,
            steps_taken=self.episode_step,
            max_steps=cfg.MAX_BUDGET,
            budget_remaining=self.budget,
            target_accuracy=self.target_accuracy,
            step_type="apply",
            n_recs_returned=0,
            golden_penalty=golden_penalty,
        )
        self.reward_history.append(reward)
        self.done = (self.current_accuracy >= self.target_accuracy) or (self.budget <= 0)

        log_event(logger, "apply_step",
                  session_id=self.session_id,
                  rec_id=rec_id, agent=agent_name,
                  rec_type=rec.get("type"),
                  prev_accuracy=round(prev_accuracy, 4),
                  new_accuracy=round(self.current_accuracy, 4),
                  target=self.target_accuracy,
                  golden_penalty=golden_penalty,
                  reward=reward,
                  success=self.done and self.current_accuracy >= self.target_accuracy)

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
                "golden_penalty": golden_penalty,
                "success": self.current_accuracy >= self.target_accuracy,
            },
        }

    # ── Golden row integrity check ─────────────────────────────────────────────

    def _check_golden_rows(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> float:
        """
        Returns a penalty in [0.0, 1.0] if golden rows were corrupted.
        0.0 = no corruption, 1.0 = all golden rows destroyed.
        """
        if not self.golden_row_ids:
            return 0.0

        # Golden rows are identified by index — check if they still exist and are clean
        feature_cols = [c for c in df_before.columns if c not in ("label", "_archetype")]
        corrupted = 0
        for idx in self.golden_row_ids:
            if idx not in df_after.index:
                corrupted += 1  # row was dropped
                continue
            if df_after.loc[idx, feature_cols].isnull().any():
                corrupted += 1  # golden row now has NaN
        return round(corrupted / max(len(self.golden_row_ids), 1), 4)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _observation(self) -> dict:
        stats = compute_stats(self.df) if self.df is not None else {}
        pending_summary = {
            rid: {
                "agent": entry["agent"],
                "type": entry["rec"].get("type", "?"),
                "priority": entry["rec"].get("priority", "?"),
                "reason": entry["rec"].get("reason", ""),
            }
            for rid, entry in self.pending_recs.items()
            if rid not in self.applied_rec_ids
        }
        return {
            "session_id": self.session_id,
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

    def _error(self, msg: str) -> dict:
        return {"error": msg, "session_id": self.session_id}

    def _curriculum_difficulty(self) -> str:
        if self._episode_count < cfg.CURRICULUM_MEDIUM_AFTER:
            return "easy"
        elif self._episode_count < cfg.CURRICULUM_HARD_AFTER:
            return "medium"
        return "hard"

    def episode_summary(self) -> dict:
        return {
            "session_id": self.session_id,
            "episode_count": self._episode_count,
            "accuracy_history": [round(a, 4) for a in self.accuracy_history],
            "reward_history": [round(r, 4) for r in self.reward_history],
            "mean_reward": round(sum(self.reward_history) / max(len(self.reward_history), 1), 4),
        }
