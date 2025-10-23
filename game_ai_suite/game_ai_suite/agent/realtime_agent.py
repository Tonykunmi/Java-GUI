from dataclasses import dataclass
from typing import Callable, Any, Dict, Optional
import time
import threading


@dataclass
class AgentConfig:
    decision_hz: float = 10.0
    max_queue_size: int = 1


class RealTimeAgent:
    """
    A minimal real-time agent loop that ingests observations, makes decisions via a user-provided
    policy function, and emits actions at a steady rate.
    """

    def __init__(
        self,
        policy_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        config: Optional[AgentConfig] = None,
    ) -> None:
        self.policy_fn = policy_fn
        self.config = config or AgentConfig()
        self._latest_obs: Optional[Dict[str, Any]] = None
        self._action_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def set_action_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        self._action_callback = callback

    def submit_observation(self, obs: Dict[str, Any]) -> None:
        self._latest_obs = obs

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run_loop(self) -> None:
        dt = 1.0 / max(self.config.decision_hz, 1e-6)
        next_time = time.perf_counter()
        while not self._stop.is_set():
            now = time.perf_counter()
            if now < next_time:
                time.sleep(max(0.0, next_time - now))
                continue
            next_time += dt

            obs = self._latest_obs
            if obs is None:
                continue

            try:
                action = self.policy_fn(obs)
            except Exception as e:
                action = {"error": str(e)}

            if self._action_callback:
                try:
                    self._action_callback(action)
                except Exception:
                    pass
