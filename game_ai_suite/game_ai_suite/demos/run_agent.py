from game_ai_suite.agent import RealTimeAgent, AgentConfig
import time


def simple_policy(obs):
    # Example policy: mirror input into an action
    value = obs.get("value", 0)
    return {"action": "throttle", "amount": value * 2}


def on_action(a):
    print("Action:", a)


def main():
    agent = RealTimeAgent(policy_fn=simple_policy, config=AgentConfig(decision_hz=5))
    agent.set_action_callback(on_action)
    agent.start()

    try:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 3.0:
            agent.submit_observation({"value": (time.perf_counter() - t0)})
            time.sleep(0.05)
    finally:
        agent.stop()


if __name__ == "__main__":
    main()
