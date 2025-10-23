from game_ai_suite.rl import train_npc, load_npc, run_npc_episode


def main():
    print("Training NPC (short run)...")
    model_path = train_npc(total_timesteps=2000)
    print("Saved:", model_path)
    model = load_npc()
    reward = run_npc_episode(model, render=False)
    print("Episode reward:", reward)


if __name__ == "__main__":
    main()
