import numpy as np
from collections import deque


class PerPromptStatTracker:
    def __init__(self, global_std=False):
        self.global_std = global_std
        self.stats = {}
        self.history_prompts = set()

    # exp reward is for rwr
    def update(self, prompts, rewards, exp=False):
        prompts = np.array(prompts)
        rewards = np.array(rewards, dtype=np.float64)
        unique = np.unique(prompts)
        advantages = np.empty_like(rewards) * 0.0
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = []
            self.stats[prompt].extend(prompt_rewards)
            self.history_prompts.add(hash(prompt))  # Add hash of prompt to history_prompts
        for prompt in unique:
            self.stats[prompt] = np.stack(self.stats[prompt])
            prompt_rewards = rewards[prompts == prompt]  # Fix: Recalculate prompt_rewards for each prompt
            mean = np.mean(self.stats[prompt], axis=0, keepdims=True)
            if self.global_std:
                std = np.std(rewards, axis=0, keepdims=True) + 1e-4  # Use global std of all rewards
            else:
                std = np.std(self.stats[prompt], axis=0, keepdims=True) + 1e-4
            advantages[prompts == prompt] = (prompt_rewards - mean) / std
        return advantages

    def get_stats(self):
        avg_group_size = sum(len(v) for v in self.stats.values()) / len(self.stats) if self.stats else 0
        history_prompts = len(self.history_prompts)
        return avg_group_size, history_prompts

    def clear(self):
        self.stats = {}

    def get_mean_of_top_rewards(self, top_percentage):
        if not self.stats:
            return 0.0

        assert 0 <= top_percentage <= 100

        per_prompt_top_means = []
        for prompt_rewards in self.stats.values():
            if isinstance(prompt_rewards, list):
                rewards = np.array(prompt_rewards)
            else:
                rewards = prompt_rewards

            if rewards.size == 0:
                continue

            if top_percentage == 100:
                per_prompt_top_means.append(np.mean(rewards))
                continue

            lower_bound_percentile = 100 - top_percentage
            threshold = np.percentile(rewards, lower_bound_percentile)

            top_rewards = rewards[rewards >= threshold]

            if top_rewards.size > 0:
                per_prompt_top_means.append(np.mean(top_rewards))

        if not per_prompt_top_means:
            return 0.0

        return np.mean(per_prompt_top_means)
    
    def state_dict(self) -> dict:
        """
        将内部状态序列化为可保存的字典。
        注意：self.stats 中可能包含 numpy 数组，先转为 list 保证兼容性。
        """
        stats_serializable = {}
        for prompt, rewards in self.stats.items():
            if isinstance(rewards, np.ndarray):
                stats_serializable[prompt] = rewards.tolist()
            elif isinstance(rewards, list):
                stats_serializable[prompt] = rewards  # 已经是 list
            else:
                stats_serializable[prompt] = rewards  # 其他情况，尝试直接保存
        return {
            "stats": stats_serializable,
            "history_prompts": list(self.history_prompts),  # set 转 list 以便序列化
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """
        从保存的字典恢复内部状态。
        """
        # 恢复 stats，将 list 转回 numpy array（如果原先是 array）
        raw_stats = state_dict.get("stats", {})
        self.stats = {}
        for prompt, rewards in raw_stats.items():
            if isinstance(rewards, list):
                # 尝试转换为 ndarray，但保留为 list 也可（外部使用时会处理）
                self.stats[prompt] = np.array(rewards, dtype=np.float64)
            else:
                self.stats[prompt] = rewards

        # 恢复 history_prompts
        raw_history = state_dict.get("history_prompts", [])
        self.history_prompts = set(raw_history)


def main():
    tracker = PerPromptStatTracker()
    prompts = ["a", "b", "a", "c", "b", "a"]
    rewards = [1, 2, 3, 4, 5, 6]
    advantages = tracker.update(prompts, rewards)
    print("Advantages:", advantages)
    avg_group_size, history_prompts = tracker.get_stats()
    print("Average Group Size:", avg_group_size)
    print("History Prompts:", history_prompts)
    tracker.clear()
    print("Stats after clear:", tracker.stats)


if __name__ == "__main__":
    main()
