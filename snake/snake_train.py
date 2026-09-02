"""
snake_train.py - Browser build of Multi-Agent PPO Snake training.
Faithful port of NullTronPrime/Multi-agent-PPO-training-in-snake/Snaketrier.py
running in the browser via Pyodide. Keeps the ORIGINAL algorithm, reward
shaping, PPO math, adaptive entropy/clip and curriculum - only the runtime that
a browser cannot do is adapted: torch/autograd -> explicit numpy gradients,
numba -> numpy, pygame/threads/CSV -> a single loop yielding to a JS canvas.
Scale is reduced so it trains live in a browser (adjust CONFIG to train more).
"""
import numpy as np
import random

CONFIG = {
    'num_updates': 120,
    'rollout_steps': 600,
    'num_agents': 1,
    'grid': 16,
}

seed = 42
random.seed(seed)
np.random.seed(seed)

# ------------------------ Hyperparameters (original) ------------------------ #
batch_size = 128
initial_entropy_coef = 0.05
min_entropy_coef = 0.001
entropy_decay = 0.9999
initial_clip_epsilon = 0.2
min_clip_epsilon = 0.05

state_dim = 16
action_dim = 4
lr = 2.5e-4
gamma = 0.99
gae_lambda = 0.95
ppo_epochs = 10
grad_clip = 0.5
hidden = 128


# ------------------------ State computation (original, minus numba) ------------------------ #
def compute_state(snake, food_x, food_y, grid_w, grid_h, dir_index):
    head_x, head_y = snake[0]
    state = np.zeros(16, dtype=np.float32)
    state[0] = head_x / grid_w
    state[1] = head_y / grid_h
    state[2] = food_x / grid_w
    state[3] = food_y / grid_h
    state[4] = (food_x - head_x) / grid_w
    state[5] = (food_y - head_y) / grid_h
    state[6] = 1.0 if (head_y - 1 < 0 or (head_x, head_y - 1) in snake[1:]) else 0.0
    state[7] = 1.0 if (head_y + 1 >= grid_h or (head_x, head_y + 1) in snake[1:]) else 0.0
    state[8] = 1.0 if (head_x - 1 < 0 or (head_x - 1, head_y) in snake[1:]) else 0.0
    state[9] = 1.0 if (head_x + 1 >= grid_w or (head_x + 1, head_y) in snake[1:]) else 0.0
    state[10] = len(snake) / (grid_w * grid_h)
    if dir_index == 0:
        state[11:15] = np.array([1, 0, 0, 0], dtype=np.float32)
    elif dir_index == 1:
        state[11:15] = np.array([0, 1, 0, 0], dtype=np.float32)
    elif dir_index == 2:
        state[11:15] = np.array([0, 0, 1, 0], dtype=np.float32)
    elif dir_index == 3:
        state[11:15] = np.array([0, 0, 0, 1], dtype=np.float32)
    state[15] = 1.0 if len(snake) > 1 else 0.0
    return state


# ------------------------ PPO network (original architecture, numpy) ------------------------ #
class PPOAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=hidden, rng=None):
        rng = rng if rng is not None else np.random.default_rng(seed)
        lim = 1.0 / np.sqrt(state_dim)
        self.W1 = rng.uniform(-lim, lim, (state_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        lim = 1.0 / np.sqrt(hidden_dim)
        self.W2 = rng.uniform(-lim, lim, (hidden_dim, hidden_dim))
        self.b2 = np.zeros(hidden_dim)
        self.Wa = rng.uniform(-lim, lim, (hidden_dim, hidden_dim))
        self.ba = np.zeros(hidden_dim)
        self.Wo = rng.uniform(-lim, lim, (hidden_dim, action_dim))
        self.bo = np.zeros(action_dim)
        self.Wc = rng.uniform(-lim, lim, (hidden_dim, 1))
        self.bc = np.zeros(1)

        # Adam state
        self.opt_t = 0
        self.m = {}
        self.v = {}
        for name, p in self._params().items():
            self.m[name] = np.zeros_like(p)
            self.v[name] = np.zeros_like(p)

    def _params(self):
        return {
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2,
            'Wa': self.Wa, 'ba': self.ba,
            'Wo': self.Wo, 'bo': self.bo,
            'Wc': self.Wc, 'bc': self.bc,
        }

    def forward(self, x):
        h1 = np.maximum(x @ self.W1 + self.b1, 0.0)
        h2 = np.maximum(h1 @ self.W2 + self.b2, 0.0)
        ha = np.maximum(h2 @ self.Wa + self.ba, 0.0)
        logits = ha @ self.Wo + self.bo
        la = logits - logits.max()
        e = np.exp(la)
        probs = e / e.sum()
        value = float((h2 @ self.Wc + self.bc)[0])
        return probs, value

    def adam_step(self, grads):
        self.opt_t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        params = self._params()
        for name in params:
            g = grads[name]
            self.m[name] = b1 * self.m[name] + (1 - b1) * g
            self.v[name] = b2 * self.v[name] + (1 - b2) * (g * g)
            mhat = self.m[name] / (1 - b1 ** self.opt_t)
            vhat = self.v[name] / (1 - b2 ** self.opt_t)
            params[name] -= lr * mhat / (np.sqrt(vhat) + eps)


# ------------------------ Snake Environment (original) ------------------------ #
class SnakeEnv:
    def __init__(self, grid_w=16, grid_h=16):
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.reset()

    def reset(self):
        self.snake = [(self.grid_w // 2, self.grid_h // 2)]
        self.direction = 3
        self.spawn_food()
        self.done = False
        self.steps_without_food = 0
        self.max_steps_without_food = 2 * (self.grid_w + self.grid_h)
        return self.get_state()

    def spawn_food(self):
        while True:
            self.food = (random.randint(0, self.grid_w - 1), random.randint(0, self.grid_h - 1))
            if self.food not in self.snake:
                break

    def step(self, action):
        if action == 0 and self.direction != 1:
            self.direction = 0
        elif action == 1 and self.direction != 0:
            self.direction = 1
        elif action == 2 and self.direction != 3:
            self.direction = 2
        elif action == 3 and self.direction != 2:
            self.direction = 3

        delta = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}[self.direction]
        head = self.snake[0]
        new_head = (head[0] + delta[0], head[1] + delta[1])

        if (new_head[0] < 0 or new_head[0] >= self.grid_w or
                new_head[1] < 0 or new_head[1] >= self.grid_h or
                new_head in self.snake):
            self.done = True
            reward = -10.0
            return self.get_state(), reward, self.done, {"score": len(self.snake)}

        self.snake.insert(0, new_head)
        self.steps_without_food += 1

        if new_head == self.food:
            reward = 10.0 + len(self.snake) * 0.1
            self.spawn_food()
            self.steps_without_food = 0
        else:
            self.snake.pop()
            prev_dist = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
            curr_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
            reward = 0.1 if curr_dist < prev_dist else -0.1

        if len(self.snake) > 10:
            min_x = min(p[0] for p in self.snake)
            max_x = max(p[0] for p in self.snake)
            min_y = min(p[1] for p in self.snake)
            max_y = max(p[1] for p in self.snake)
            area = (max_x - min_x + 1) * (max_y - min_y + 1)
            efficiency = len(self.snake) / max(area, 1)
            reward += efficiency * 0.5

        adj = sum(1 for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                  if (head[0] + dx, head[1] + dy) in self.snake[1:])
        if adj >= 2 and not self.done:
            reward += 0.2

        if self.steps_without_food >= self.max_steps_without_food:
            self.done = True
            reward = -5.0

        return self.get_state(), reward, self.done, {"score": len(self.snake)}

    def get_state(self):
        return compute_state(self.snake, self.food[0], self.food[1],
                             self.grid_w, self.grid_h, self.direction)


# ------------------------ PPO components (original) ------------------------ #
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self):
        self.__init__()


def compute_gae(rewards, values, dones, gamma, gae_lambda):
    next_value = 0
    advantages = []
    returns = []
    for step in reversed(range(len(rewards))):
        if step == len(rewards) - 1 or dones[step]:
            next_value = 0
        delta = rewards[step] + gamma * next_value * (1 - dones[step]) - values[step]
        advantage = delta + gamma * gae_lambda * (1 - dones[step]) * (advantages[0] if advantages else 0)
        advantages.insert(0, advantage)
        returns.insert(0, advantage + values[step])
        next_value = values[step]
    return returns, advantages


def adjust_environment_difficulty(env, update):
    if update < 1000:
        env.max_steps_without_food = 100
    elif update < 5000:
        env.max_steps_without_food = 60
    else:
        env.max_steps_without_food = 40


def _forward_batch(agent, x):
    h1 = np.maximum(x @ agent.W1 + agent.b1, 0.0)
    h2 = np.maximum(h1 @ agent.W2 + agent.b2, 0.0)
    ha = np.maximum(h2 @ agent.Wa + agent.ba, 0.0)
    logits = ha @ agent.Wo + agent.bo
    la = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(la)
    probs = e / e.sum(axis=1, keepdims=True)
    values = (h2 @ agent.Wc + agent.bc).ravel()
    return h1, h2, ha, probs, values


def _backward_ppo(agent, s_b, actions_b, old_logp, returns_b, adv_b, clip_epsilon, entropy_coef):
    x = s_b
    h1, h2, ha, probs, values = _forward_batch(agent, x)
    N = x.shape[0]

    logp_a = np.log(probs[np.arange(N), actions_b] + 1e-10)
    ratio = np.exp(logp_a - old_logp)
    surr1 = ratio * adv_b
    surr2 = np.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * adv_b
    # active = min(surr1, surr2) == surr1  (surr2 is clipped/constant otherwise)
    active = surr1 <= surr2
    # dL/dz (policy, includes 1/N from .mean()) = active * adv * ratio * (p - onehot) / N
    dlogits_pol = active[:, None] * (adv_b * ratio)[:, None] * (probs - np.eye(4)[actions_b]) / N

    # ---- entropy term: d(-ent*H_mean)/dz = +ent/N * p*(log p + H) ----
    H = -(probs * np.log(probs + 1e-10)).sum(1)
    dlogits_ent = entropy_coef / N * probs * (np.log(probs + 1e-10) + H[:, None])

    dlogits = dlogits_pol + dlogits_ent

    dWo = ha.T @ dlogits
    dbo = dlogits.sum(0)
    dha = (dlogits @ agent.Wo.T) * (ha > 0)

    # ---- critic: d/dv of 0.5*mean((v-ret)^2) = (v-ret)/N ----
    dv = (values - returns_b) / N
    dWc = h2.T @ dv[:, None]
    dbc = dv.sum(0)

    # ---- shared backprop ----
    dh2 = (dha @ agent.Wa.T + dv[:, None] @ agent.Wc.T) * (h2 > 0)
    dWa = h2.T @ dha
    dba = dha.sum(0)
    dW2 = h1.T @ dh2
    db2 = dh2.sum(0)
    dh1 = (dh2 @ agent.W2.T) * (h1 > 0)
    dW1 = x.T @ dh1
    db1 = dh1.sum(0)

    # clip_grad_norm
    grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2,
             'Wa': dWa, 'ba': dba, 'Wo': dWo, 'bo': dbo,
             'Wc': dWc, 'bc': dbc}
    total = np.sqrt(sum(np.sum(g * g) for g in grads.values()))
    if total > grad_clip:
        scale = grad_clip / (total + 1e-6)
        for k in grads:
            grads[k] *= scale
    return grads


# ------------------------ Main training (headless + browser) ------------------------ #
class Trainer:
    """Incremental PPO trainer. `one_update()` processes a single rollout + PPO
    pass and returns a JSON-serializable snapshot (perfect for driving Pyodide
    a step at a time from the browser). `train()` is a headless convenience loop."""

    def __init__(self, env, num_updates=None, rollout_steps=None):
        self.envs = env if isinstance(env, list) else [env]
        self.num_updates = num_updates or CONFIG['num_updates']
        self.rollout_steps = rollout_steps or CONFIG['rollout_steps']
        self.rng = np.random.default_rng(seed)
        self.agent = PPOAgent(state_dim, action_dim)
        self.buffer = RolloutBuffer()
        self.clip_epsilon = initial_clip_epsilon
        self.episode_rewards = [0.0] * len(self.envs)
        self.episode_count = 0
        self.states = [e.reset() for e in self.envs]
        self.best_reward = -float('inf')
        self.update = 0

    def _snapshot(self, entropy_coef):
        env0 = self.envs[0]
        return {
            'update': self.update,
            'done': self.update >= self.num_updates,
            'snake': [[int(x), int(y)] for x, y in env0.snake],
            'food': [int(env0.food[0]), int(env0.food[1])],
            'direction': env0.direction,
            'score': len(env0.snake),
            'episodes': self.episode_count,
            'best_reward': round(self.best_reward, 2),
            'entropy': round(entropy_coef, 4),
            'clip': round(self.clip_epsilon, 4),
        }

    def one_update(self):
        if self.update >= self.num_updates:
            return None
        update = self.update
        entropy_coef = max(initial_entropy_coef * (entropy_decay ** update), min_entropy_coef)
        if update % 100 == 0 and update > 0:
            self.clip_epsilon = max(self.clip_epsilon * 0.95, min_clip_epsilon)
        for e in self.envs:
            adjust_environment_difficulty(e, update)

        for _ in range(self.rollout_steps):
            for i, env in enumerate(self.envs):
                state = self.states[i]
                probs, value = self.agent.forward(state)
                action = self.rng.choice(action_dim, p=probs)
                log_prob = float(np.log(probs[action] + 1e-10))

                next_state, reward, done, _ = env.step(int(action))
                self.episode_rewards[i] += reward

                self.buffer.states.append(state)
                self.buffer.actions.append(int(action))
                self.buffer.log_probs.append(log_prob)
                self.buffer.rewards.append(reward)
                self.buffer.dones.append(done)
                self.buffer.values.append(value)

                self.states[i] = next_state
                if done or (len(env.snake) >= env.grid_w * env.grid_h):
                    self.episode_count += 1
                    if self.episode_rewards[i] > self.best_reward:
                        self.best_reward = self.episode_rewards[i]
                    self.states[i] = env.reset()
                    self.episode_rewards[i] = 0.0

        states_a = np.array(self.buffer.states, dtype=np.float64)
        actions_a = np.array(self.buffer.actions, dtype=np.int64)
        old_logp = np.array(self.buffer.log_probs, dtype=np.float64)

        returns, advantages = compute_gae(self.buffer.rewards, self.buffer.values,
                                          self.buffer.dones, gamma, gae_lambda)
        returns = np.array(returns)
        adv = np.array(advantages)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        dataset_size = states_a.shape[0]
        for _epoch in range(ppo_epochs):
            permutation = np.random.permutation(dataset_size)
            for i in range(0, dataset_size, batch_size):
                indices = permutation[i:i + batch_size]
                grads = _backward_ppo(self.agent, states_a[indices], actions_a[indices],
                                      old_logp[indices], returns[indices], adv[indices],
                                      self.clip_epsilon, entropy_coef)
                self.agent.adam_step(grads)

        self.buffer.clear()
        self.update += 1
        return self._snapshot(entropy_coef)

    def train(self, callback):
        while True:
            snap = self.one_update()
            if snap is None:
                return self.best_reward
            callback(snap)


def train(env, callback):
    """Headless convenience: runs the full schedule, calling callback(snapshot)
    after every update. Returns the best episode reward."""
    return Trainer(env).train(callback)


# ------------------------ Step-by-step trainer (browser animation) ------------------------ #
class BrowserTrainer:
    """Drives the exact same algorithm one environment step at a time, so a
    browser page can render the snake moving and dying between steps (like the
    original pygame trainer). Every `rollout_steps` steps it runs the PPO pass."""

    def __init__(self, env, rollout_steps=CONFIG['rollout_steps'], ppo_reps=ppo_epochs):
        self.env = env
        self.rollout_steps = rollout_steps
        self.ppo_reps = ppo_reps
        self.rng = np.random.default_rng(seed)
        self.agent = PPOAgent(state_dim, action_dim)
        self.buffer = RolloutBuffer()
        self.state = env.reset()
        self.steps_this_rollout = 0
        self.update = 0
        self.ep_reward = 0.0
        self.ep_len = 0
        self.episodes = 0
        self.best = -float('inf')
        self.total_steps = 0
        self.clip_epsilon = initial_clip_epsilon
        self.last_entropy = initial_entropy_coef

    def _snapshot(self, died):
        return {
            'snake': [[int(x), int(y)] for x, y in self.env.snake],
            'food': [int(self.env.food[0]), int(self.env.food[1])],
            'direction': self.env.direction,
            'score': len(self.env.snake),
            'died': died,
            'episodes': self.episodes,
            'best_reward': round(self.best, 2),
            'update': self.update,
            'steps': self.total_steps,
            'entropy': round(self.last_entropy, 4),
        }

    def step(self):
        probs, value = self.agent.forward(self.state)
        action = self.rng.choice(action_dim, p=probs)
        log_prob = float(np.log(probs[action] + 1e-10))

        next_state, reward, done, _ = self.env.step(int(action))
        self.ep_reward += reward
        self.ep_len += 1

        self.buffer.states.append(self.state)
        self.buffer.actions.append(int(action))
        self.buffer.log_probs.append(log_prob)
        self.buffer.rewards.append(reward)
        self.buffer.dones.append(done)
        self.buffer.values.append(value)

        self.state = next_state
        self.steps_this_rollout += 1
        self.total_steps += 1

        died = False
        if done or len(self.env.snake) >= self.env.grid_w * self.env.grid_h:
            died = True
            self.episodes += 1
            if self.ep_reward > self.best:
                self.best = self.ep_reward
            self.state = self.env.reset()
            self.ep_reward = 0.0
            self.ep_len = 0

        if self.steps_this_rollout >= self.rollout_steps:
            self._policy_update()
            self.steps_this_rollout = 0
            self.buffer.clear()

        return self._snapshot(died)

    def _policy_update(self):
        entropy_coef = max(initial_entropy_coef * (entropy_decay ** self.update), min_entropy_coef)
        self.last_entropy = entropy_coef
        if self.update % 100 == 0 and self.update > 0:
            self.clip_epsilon = max(self.clip_epsilon * 0.95, min_clip_epsilon)

        states_a = np.array(self.buffer.states, dtype=np.float64)
        actions_a = np.array(self.buffer.actions, dtype=np.int64)
        old_logp = np.array(self.buffer.log_probs, dtype=np.float64)

        returns, advantages = compute_gae(self.buffer.rewards, self.buffer.values,
                                          self.buffer.dones, gamma, gae_lambda)
        returns = np.array(returns)
        adv = np.array(advantages)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        dataset_size = states_a.shape[0]
        for _epoch in range(self.ppo_reps):
            permutation = np.random.permutation(dataset_size)
            for i in range(0, dataset_size, batch_size):
                indices = permutation[i:i + batch_size]
                grads = _backward_ppo(self.agent, states_a[indices], actions_a[indices],
                                      old_logp[indices], returns[indices], adv[indices],
                                      self.clip_epsilon, entropy_coef)
                self.agent.adam_step(grads)

        self.update += 1

