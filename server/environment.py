"""
server/environment.py — Session-aware environment (v0.5).

New in v0.5:
  1. Rollback action — undo last apply (real data engineers do this)
  2. Episode reasoning trace — running history of what the agent tried + effects
  3. Feature importance — returned after every apply so agent sees what the model learned
  4. Regression explanation — when accuracy drops, explains the likely cause
  5. Baseline comparison — agent always knows how far ahead of majority-class predictor it is
"""
import threading
import pandas as pd
import numpy as np

from server.dataset_registry import DatasetRegistry
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

_registry = DatasetRegistry()


class DataCentricEnvironment:

    def __init__(self, session_id: str, episode_count: int = 0):
        self.session_id = session_id
        self._episode_count = episode_count
        self.agents = {
            "cleaner":   CleanerAgent(),
            "augmenter": AugmenterAgent(),
            "balancer":  BalancerAgent(),
            "validator": ValidatorAgent(),
            "analyst":   AnalystAgent(),
        }
        self.anti_exploit = AntiExploit()
        self._lock = threading.Lock()
        self._reset_state()

    def _reset_state(self):
        self.train_df: pd.DataFrame = None
        self.holdout_df: pd.DataFrame = None
        self.domain_metadata: dict = {}
        self.evaluator: Evaluator = None
        self.target_accuracy: float = None
        self.initial_row_count: int = 0
        self.baseline_accuracy: float = 0.0  # majority-class predictor on holdout
        self.starting_accuracy: float = 0.0  # accuracy before ANY agent action
        self.budget: int = cfg.MAX_BUDGET
        self.current_accuracy: float = 0.0
        self.episode_step: int = 0
        self.done: bool = False
        self.difficulty: str = "easy"
        self.pending_recs: dict = {}
        self.applied_rec_ids: set = set()
        self.last_query_result: dict = {}
        self.last_feature_importance: dict = {}
        self.anti_exploit.reset()
        self.accuracy_history: list = []
        self.reward_history: list = []
        # Rollback: stack of (df_snapshot, accuracy) — last 3 states
        self._state_stack: list[tuple] = []
        # Reasoning trace: running log of every step
        self._episode_trace: list[dict] = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def reset(self, difficulty: str = None, seed: int = None) -> dict:
        with self._lock:
            self._episode_count += 1
            self._reset_state()
            self.difficulty = difficulty or self._curriculum_difficulty()

            self.train_df, self.holdout_df, self.domain_metadata = _registry.get(
                difficulty=self.difficulty, seed=seed
            )
            self.initial_row_count = len(self.train_df)
            self.evaluator = Evaluator(self.holdout_df)

            pub_baseline = self.domain_metadata.get("published_baseline", 0.80)
            self.target_accuracy = round(pub_baseline * 0.97, 4)
            self.baseline_accuracy = self.evaluator.baseline_accuracy()

            self.current_accuracy = self.evaluator.evaluate(self._clean_df(self.train_df))
            self.starting_accuracy = self.current_accuracy
            self.accuracy_history.append(self.current_accuracy)

            self._episode_trace.append({
                "step": 0,
                "type": "reset",
                "dataset": self.domain_metadata.get("display_name"),
                "accuracy": round(self.current_accuracy, 4),
                "baseline_accuracy": self.baseline_accuracy,
                "target_accuracy": self.target_accuracy,
            })

            log_event(logger, "episode_reset",
                      session_id=self.session_id,
                      dataset=self.domain_metadata.get("display_name"),
                      difficulty=self.difficulty,
                      initial_accuracy=round(self.current_accuracy, 4),
                      target_accuracy=self.target_accuracy,
                      baseline_accuracy=self.baseline_accuracy,
                      published_baseline=pub_baseline,
                      n_train=len(self.train_df),
                      n_holdout=len(self.holdout_df))
            return self._observation()

    def step(self, action: dict) -> dict:
        with self._lock:
            if self.done:
                return self._error("Episode done. Call /reset.")
            if self.train_df is None:
                return self._error("Not initialized. Call /reset first.")

            # Rollback action — no anti-exploit check needed
            action_type = action.get("action", "")
            if action_type == "rollback":
                return self._handle_rollback()

            try:
                self.anti_exploit.check(
                    action=action,
                    budget_remaining=self.budget,
                    pending_recs=self.pending_recs,
                    applied_rec_ids=self.applied_rec_ids,
                )
            except ExploitDetected as e:
                log_event(logger, "exploit_detected", session_id=self.session_id,
                          rule=e.rule, detail=e.detail)
                self.episode_step += 1
                self.budget = max(0, self.budget - 1)
                self.done = self.budget <= 0
                self._episode_trace.append({
                    "step": self.episode_step,
                    "type": "exploit_blocked",
                    "rule": e.rule,
                    "detail": e.detail,
                })
                return {
                    "observation": self._observation(),
                    "reward": 0.001,
                    "done": self.done,
                    "exploit_detected": True,
                    "error": f"[{e.rule}] {e.detail}",
                    "info": {"episode_step": self.episode_step, "budget_remaining": self.budget},
                }

            if action_type in QUERY_ACTIONS:
                return self._handle_query(action_type, action)
            elif action_type == "apply":
                return self._handle_apply(action)
            else:
                return self._error(f"Unknown action '{action_type}'. Valid: {list(QUERY_ACTIONS) + ['apply', 'rollback']}")

    def state(self) -> dict:
        with self._lock:
            return self._observation()

    # ── Rollback ───────────────────────────────────────────────────────────────

    def _handle_rollback(self) -> dict:
        """Undo the last apply operation. Costs 1 budget. Max 3 rollbacks per episode."""
        rollbacks_used = sum(1 for e in self._episode_trace if e["type"] == "rollback")
        if rollbacks_used >= 3:
            return self._error("Maximum 3 rollbacks per episode reached.")
        if not self._state_stack:
            return self._error("Nothing to roll back. No apply operations have been made yet.")

        prev_df, prev_accuracy = self._state_stack.pop()
        self.train_df = prev_df
        self.current_accuracy = prev_accuracy
        self.accuracy_history.append(self.current_accuracy)
        self.budget = max(0, self.budget - 1)
        self.episode_step += 1
        self.done = self.budget <= 0

        self._episode_trace.append({
            "step": self.episode_step,
            "type": "rollback",
            "accuracy_after_rollback": round(self.current_accuracy, 4),
            "note": "Last apply undone. Dataset restored to previous state.",
        })

        log_event(logger, "rollback", session_id=self.session_id,
                  accuracy_after=round(self.current_accuracy, 4))

        return {
            "observation": self._observation(),
            "reward": 0.3,  # small penalty for indecision, but not zero
            "done": self.done,
            "rollback": True,
            "accuracy_after_rollback": round(self.current_accuracy, 4),
            "info": {
                "episode_step": self.episode_step,
                "budget_remaining": self.budget,
                "rollbacks_remaining": 3 - rollbacks_used - 1,
                "note": "Dataset restored to state before last apply.",
            },
        }

    # ── Query handler ──────────────────────────────────────────────────────────

    def _handle_query(self, action_type: str, action: dict) -> dict:
        cost = QUERY_COSTS[action_type]
        prev_stats = compute_stats(self.train_df)
        clean = self._clean_df(self.train_df)
        meta = self.domain_metadata

        if action_type == "query_cleaner":
            result = self.agents["cleaner"].query(clean, meta)
        elif action_type == "query_augmenter":
            result = self.agents["augmenter"].query(clean, action.get("target_class"), meta)
        elif action_type == "query_balancer":
            result = self.agents["balancer"].query(clean, meta)
        elif action_type == "query_validator":
            result = self.agents["validator"].query(clean, meta)
        elif action_type == "query_analyst":
            result = self.agents["analyst"].query(clean, meta)
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
        new_stats = compute_stats(self.train_df)

        reward, decomp = compute(
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

        agent_name = action_type.replace("query_", "")
        self._episode_trace.append({
            "step": self.episode_step,
            "type": "query",
            "agent": agent_name,
            "n_recs": len(new_rec_ids),
            "budget_cost": cost,
            "budget_remaining": self.budget,
            "reward": reward,
            "rec_ids": new_rec_ids,
        })

        log_event(logger, "query_step", session_id=self.session_id,
                  action=action_type, n_recs=len(new_rec_ids),
                  budget=self.budget, reward=reward)

        return {
            "observation": self._observation(),
            "reward": reward,
            "reward_decomposition": decomp,
            "done": self.done,
            "query_result": result,
            "new_recommendation_ids": new_rec_ids,
            "info": {
                "action_type": "query",
                "agent_queried": agent_name,
                "budget_cost": cost,
                "budget_remaining": self.budget,
                "n_recommendations": len(new_rec_ids),
                "episode_step": self.episode_step,
                "domain": self.domain_metadata.get("display_name"),
            },
        }

    # ── Apply handler ──────────────────────────────────────────────────────────

    def _handle_apply(self, action: dict) -> dict:
        rec_id = action.get("rec_id", "")
        entry = self.pending_recs[rec_id]
        agent_name = entry["agent"]
        rec = entry["rec"]

        prev_accuracy = self.current_accuracy
        prev_stats = compute_stats(self.train_df)
        prev_rows = len(self.train_df)
        meta = self.domain_metadata

        # Save state for rollback BEFORE applying
        self._state_stack.append((self.train_df.copy(), prev_accuracy))
        if len(self._state_stack) > 3:
            self._state_stack.pop(0)  # keep at most last 3

        result_holder: dict = {}
        error_holder: dict = {}

        def _run():
            try:
                clean = self._clean_df(self.train_df)
                df_out, log_msg = self.agents[agent_name].apply(clean, rec, meta)
                result_holder["df"] = df_out
                result_holder["log"] = log_msg
            except Exception as e:
                error_holder["error"] = str(e)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=cfg.STEP_TIMEOUT_SECONDS)

        if t.is_alive():
            self._state_stack.pop()  # failed — don't keep stale snapshot
            return self._error("Apply timed out.")
        if "error" in error_holder:
            self._state_stack.pop()
            return self._error(f"Apply error: {error_holder['error']}")

        new_df = result_holder["df"]
        tool_log = result_holder["log"]

        # Data integrity constraint: cannot delete more than 10% of rows
        new_rows = len(new_df)
        deletion_pct = max(0, (prev_rows - new_rows) / max(prev_rows, 1))
        if deletion_pct > 0.10:
            self._state_stack.pop()
            return self._error(
                f"Data integrity violation: would delete {deletion_pct:.1%} of training rows "
                f"(limit: 10%). Use targeted imputation instead of drop_rows."
            )

        self.train_df = new_df
        self.applied_rec_ids.add(rec_id)
        self.episode_step += 1

        # Full evaluation with feature importance + regression explanation
        eval_result = self.evaluator.evaluate_with_details(
            self._clean_df(self.train_df), prev_accuracy
        )
        self.current_accuracy = eval_result["accuracy"]
        self.last_feature_importance = eval_result.get("feature_importance", {})
        regression_explanation = eval_result.get("regression_explanation")

        self.accuracy_history.append(self.current_accuracy)
        new_stats = compute_stats(self.train_df)

        reward, decomp = compute(
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
        )
        self.reward_history.append(reward)
        self.done = (self.current_accuracy >= self.target_accuracy) or (self.budget <= 0)

        acc_delta = round(self.current_accuracy - prev_accuracy, 4)
        self._episode_trace.append({
            "step": self.episode_step,
            "type": "apply",
            "agent": agent_name,
            "rec_type": rec.get("type", "?"),
            "rec_id": rec_id,
            "accuracy_before": round(prev_accuracy, 4),
            "accuracy_after": round(self.current_accuracy, 4),
            "accuracy_delta": acc_delta,
            "effect": "improved" if acc_delta > 0.001 else ("hurt" if acc_delta < -0.001 else "neutral"),
            "reward": reward,
            "rows_before": prev_rows,
            "rows_after": new_rows,
        })

        log_event(logger, "apply_step", session_id=self.session_id,
                  rec_id=rec_id, agent=agent_name,
                  prev_acc=round(prev_accuracy, 4),
                  new_acc=round(self.current_accuracy, 4),
                  target=self.target_accuracy,
                  reward=reward,
                  success=self.current_accuracy >= self.target_accuracy)

        response = {
            "observation": self._observation(),
            "reward": reward,
            "reward_decomposition": decomp,
            "done": self.done,
            "tool_log": tool_log,
            "feature_importance": self.last_feature_importance,
            "info": {
                "action_type": "apply",
                "rec_id": rec_id,
                "agent": agent_name,
                "rec_type": rec.get("type", "?"),
                "prev_accuracy": round(prev_accuracy, 4),
                "new_accuracy": round(self.current_accuracy, 4),
                "accuracy_delta": acc_delta,
                "target_accuracy": self.target_accuracy,
                "published_baseline": self.domain_metadata.get("published_baseline"),
                "improvement_over_start": round(self.current_accuracy - self.starting_accuracy, 4),
                "improvement_over_majority_baseline": round(self.current_accuracy - self.baseline_accuracy, 4),
                "budget_remaining": self.budget,
                "episode_step": self.episode_step,
                "success": self.current_accuracy >= self.target_accuracy,
                "rollbacks_available": max(0, 3 - sum(1 for e in self._episode_trace if e["type"] == "rollback")),
                "data_integrity": {
                    "rows_before": prev_rows,
                    "rows_after": new_rows,
                    "deletion_pct": round(deletion_pct, 4),
                },
            },
        }

        # Only include regression explanation when accuracy dropped
        if regression_explanation:
            response["regression_explanation"] = regression_explanation

        return response

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _clean_df(self, df):
        drop_cols = [c for c in df.columns if c.startswith("_")]
        return df.drop(columns=drop_cols) if drop_cols else df

    def _observation(self) -> dict:
        stats = compute_stats(self.train_df) if self.train_df is not None else {}
        pending_summary = {
            rid: {
                "agent": entry["agent"],
                "type": entry["rec"].get("type", "?"),
                "priority": entry["rec"].get("priority", "?"),
                "reason": entry["rec"].get("reason", ""),
                "domain_informed": entry["rec"].get("domain_informed", False),
            }
            for rid, entry in self.pending_recs.items()
            if rid not in self.applied_rec_ids
        }
        meta = self.domain_metadata

        # Compact trace — last 5 steps for context without overwhelming the prompt
        recent_trace = self._episode_trace[-5:] if self._episode_trace else []

        return {
            "session_id": self.session_id,

            # What the agent is working on
            "dataset": {
                "name": meta.get("display_name", "Unknown"),
                "domain": meta.get("domain", "generic"),
                "description": meta.get("description", ""),
                "known_issues": meta.get("known_issues", []),
                "published_baseline": meta.get("published_baseline"),
            },

            # Current state
            "current_accuracy": round(self.current_accuracy, 4),
            "target_accuracy": self.target_accuracy,
            "accuracy_gap": round(max(0, self.target_accuracy - self.current_accuracy), 4),
            "budget_remaining": self.budget,
            "difficulty": self.difficulty,

            # Comparisons — what does this number actually mean?
            "benchmarks": {
                "majority_class_baseline": self.baseline_accuracy,
                "starting_accuracy": round(self.starting_accuracy, 4),
                "improvement_over_start": round(self.current_accuracy - self.starting_accuracy, 4),
                "improvement_over_baseline": round(self.current_accuracy - self.baseline_accuracy, 4),
                "published_baseline": meta.get("published_baseline"),
            },

            "dataset_stats": {
                "n_train_rows": len(self.train_df) if self.train_df is not None else 0,
                "n_holdout_rows": len(self.holdout_df) if self.holdout_df is not None else 0,
                "n_cols": len(self.train_df.columns) if self.train_df is not None else 0,
                "missing_pct": round(stats.get("missing_pct", 0), 4),
                "balance_ratio": round(stats.get("balance_ratio", 0), 4),
            },

            # Feature importance from last evaluation
            "feature_importance": self.last_feature_importance,

            # Episodic memory — what has the agent tried so far?
            "episode_trace": recent_trace,

            "pending_recommendations": pending_summary,
            "last_query_result": self.last_query_result,
            "available_actions": (
                "query_cleaner | query_augmenter | query_balancer | "
                "query_validator (cost 2) | query_analyst (cost 2) | "
                "apply {rec_id} | rollback (undo last apply, max 3/episode)"
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
            "full_trace": self._episode_trace,
        }
