import random
import time
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from Gato_models.d4rl_infos import REF_MIN_SCORE, REF_MAX_SCORE, D4RL_DATASET_STATS
from Gato_models.tokenizers import inverse_tokenize_act_values



def evaluate_dt(model, device, context_len, env, rtg_target, rtg_scale,
                    num_eval_ep=10, max_test_ep_len=1000,
                    state_mean=None, state_std=None, render=False):

    eval_batch_size = 1  # required for forward pass

    results = {}
    total_reward = 0
    total_timesteps = 0

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    if state_mean is None:
        state_mean = torch.zeros((state_dim,)).to(device)
    else:
        state_mean = torch.from_numpy(state_mean).to(device)

    if state_std is None:
        state_std = torch.ones((state_dim,)).to(device)
    else:
        state_std = torch.from_numpy(state_std).to(device)

    # same as timesteps used for training the transformer
    # also, crashes if device is passed to arange()
    timesteps = torch.arange(start=0, end=max_test_ep_len, step=1)
    timesteps = timesteps.repeat(eval_batch_size, 1).to(device)

    model.eval()

    with torch.no_grad():

        for _ in range(num_eval_ep):

            # zeros place holders
            actions = torch.zeros((eval_batch_size, max_test_ep_len, act_dim),
                                dtype=torch.float32, device=device)
            states = torch.zeros((eval_batch_size, max_test_ep_len, state_dim),
                                dtype=torch.float32, device=device)
            rewards_to_go = torch.zeros((eval_batch_size, max_test_ep_len, 1),
                                dtype=torch.float32, device=device)

            # init episode
            running_state = env.reset()
            running_reward = 0
            running_rtg = rtg_target / rtg_scale

            for t in range(max_test_ep_len):

                total_timesteps += 1

                # add state in placeholder and normalize
                states[0, t] = torch.from_numpy(running_state).to(device)
                states[0, t] = (states[0, t] - state_mean) / state_std

                # calcualate running rtg and add it in placeholder
                running_rtg = running_rtg - (running_reward / rtg_scale)
                rewards_to_go[0, t] = running_rtg

                if t < context_len:
                    _, act_preds, _ = model.forward(timesteps[:,:context_len],
                                                states[:,:context_len],
                                                actions[:,:context_len],
                                                rewards_to_go[:,:context_len])
                    act = act_preds[0, t].detach()
                else:
                    _, act_preds, _ = model.forward(timesteps[:,t-context_len+1:t+1],
                                                states[:,t-context_len+1:t+1],
                                                actions[:,t-context_len+1:t+1],
                                                rewards_to_go[:,t-context_len+1:t+1])
                    act = act_preds[0, -1].detach()

                running_state, running_reward, done, _ = env.step(act.cpu().numpy())

                # add action in placeholder
                actions[0, t] = act

                total_reward += running_reward

                if render:
                    env.render()
                if done:
                    break

    results['eval/avg_reward'] = total_reward / num_eval_ep
    results['eval/avg_ep_len'] = total_timesteps / num_eval_ep

    return results


def evaluate_gato(model, device, context_len, env,
                  rtg_scale, eval_data,
                  num_eval_ep=1, max_test_ep_len=1000,
                  state_mean=None, state_std=None, render=False):

# ------------------------------------------------------------------------------------------------------------------

    # 初始化
    eval_batch_size = 1
    pro_timesteps, pro_states, pro_actions, pro_rewards, pro_mask = eval_data

    results = {}
    total_reward = 0
    total_timesteps = 0

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # 归一化状态
    if state_mean is None:
        state_mean = torch.zeros((state_dim,)).to(device)
    else:
        state_mean = torch.from_numpy(state_mean).to(device)
    if state_std is None:
        state_std = torch.ones((state_dim,)).to(device)
    else:
        state_std = torch.from_numpy(state_std).to(device)

    pro_timesteps = pro_timesteps.unsqueeze(dim=0).to(device)  # T -> B x T  B=1
    pro_states = pro_states.unsqueeze(dim=0).to(device)  # T x state_dim -> B x T x state_dim B=1
    pro_actions = pro_actions.unsqueeze(dim=0).to(device)  # T x act_dim -> B x T x act_dim B=1
    prompt_length = pro_timesteps.shape[1]

# ------------------------------------------------------------------------------------------------------------------
    # 评估
    model.eval()
    with torch.no_grad():
        for _ in range(num_eval_ep):

            # zeros place holders
            actions = torch.zeros((eval_batch_size, max_test_ep_len + prompt_length, act_dim),
                                  dtype=torch.float32, device=device)
            states = torch.zeros((eval_batch_size, max_test_ep_len + prompt_length, state_dim),
                                 dtype=torch.float32, device=device)
            timesteps = torch.arange(start=0, end=max_test_ep_len + prompt_length, step=1).unsqueeze(dim=0)

            actions[:, :prompt_length, :] = pro_actions
            states[:, :prompt_length, :] = pro_states

            # init episode
            running_state = env.reset()
            running_reward = 0

            for t in range(max_test_ep_len):

                total_timesteps += 1

                # add state in placeholder and normalize
                states[0, t+context_len] = torch.from_numpy(running_state).to(device)
                states[0, t+context_len] = (states[0, t+context_len] - state_mean) / state_std

                action_preds = model.forward(timesteps[:, t:t+context_len],
                                             states[:, t:t+context_len],
                                             actions[:, t:t+context_len],
                                             )

                # 得到 action 空间中的 action
                act = inverse_tokenize_act_values(action_preds[0].detach())
                assert act in env.action_space

                running_state, running_reward, done, _ = env.step(act.cpu().numpy())

                # add action in placeholder
                actions[0, t] = act
                total_reward += running_reward

                if render:
                    env.render()
                if done:
                    break

    results['eval/avg_reward'] = total_reward / num_eval_ep
    results['eval/avg_ep_len'] = total_timesteps / num_eval_ep

    return results
