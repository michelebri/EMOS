import os
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional

import imageio
import numpy as np
import torch
import tqdm

from habitat import logger
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay
from habitat.utils.visualizations.utils import (
    observations_to_image,
    overlay_frame,
)
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.rl.ppo.evaluator import Evaluator, pause_envs
from habitat_baselines.rl.multi_agent.episode_results import (
    append_episode_result,
)
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    get_action_space_info,
    inference_mode,
    is_continuous_action_space,
)
from habitat_baselines.utils.info_dict import extract_scalars_from_info
from habitat_mas.utils.models import ToolCallCrashError


class _StreamingDiskVideo:
    """Write evaluation frames immediately instead of retaining them in RAM."""

    def __init__(self, video_dir: str, fps: int, stream_name: str):
        self.video_dir = video_dir
        self.fps = fps
        self.stream_name = stream_name
        self.frame_count = 0
        self._writer = None
        self._temp_path: Optional[str] = None

    def append(self, frame: np.ndarray) -> None:
        if self._writer is None:
            os.makedirs(self.video_dir, exist_ok=True)
            self._temp_path = os.path.join(
                self.video_dir,
                f".{self.stream_name}-{uuid.uuid4().hex}.mp4",
            )
            self._writer = imageio.get_writer(
                self._temp_path,
                fps=self.fps,
                quality=5,
            )
        self._writer.append_data(frame)
        self.frame_count += 1

    def finish(self, video_name: str) -> str:
        if self._writer is None or self._temp_path is None:
            return ""
        self._writer.close()
        self._writer = None
        final_path = os.path.join(
            self.video_dir,
            video_name.replace(" ", "_").replace("\n", "_")[:251] + ".mp4",
        )
        os.replace(self._temp_path, final_path)
        self._temp_path = None
        self.frame_count = 0
        logger.info(f"Video created: {final_path}")
        return final_path


def _evaluation_video_name(
    episode_id: str,
    checkpoint_index: int,
    metrics: Dict[str, float],
    keys_to_include: Optional[List[str]],
) -> str:
    if keys_to_include:
        metric_keys = [
            key
            for key in metrics
            if any(fragment in key for fragment in keys_to_include)
        ]
    else:
        metric_keys = list(metrics.keys())
    metric_suffix = "-".join(
        f"{key}={metrics[key]:.2f}" for key in metric_keys
    )
    return f"episode={episode_id}-ckpt={checkpoint_index}-{metric_suffix}"


class HabitatMASEvaluator(Evaluator):
    """
    Evaluator for Habitat environments.
    """

    def evaluate_agent(
        self,
        agent,
        envs,
        config,
        checkpoint_index,
        step_id,
        writer,
        device,
        obs_transforms,
        env_spec,
        rank0_keys,
    ):
        observations = envs.reset()
        observations = envs.post_step(observations)
        batch = batch_obs(observations, device=device)
        batch = apply_obs_transforms_batch(batch, obs_transforms)  # type: ignore

        action_shape, discrete_actions = get_action_space_info(
            agent.actor_critic.policy_action_space
        )

        current_episode_reward = torch.zeros(envs.num_envs, 1, device="cpu")

        test_recurrent_hidden_states = torch.zeros(
            (
                config.habitat_baselines.num_environments,
                *agent.actor_critic.hidden_state_shape,
            ),
            device=device,
        )

        hidden_state_lens = agent.actor_critic.hidden_state_shape_lens
        action_space_lens = agent.actor_critic.policy_action_space_shape_lens

        prev_actions = torch.zeros(
            config.habitat_baselines.num_environments,
            *action_shape,
            device=device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        not_done_masks = torch.zeros(
            config.habitat_baselines.num_environments,
            *agent.masks_shape,
            device=device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode
        ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0)

        episode_results_path = os.environ.get("EMOS_EPISODE_RESULTS_PATH")
        if not episode_results_path:
            episode_results_path = os.path.join(
                os.path.dirname(str(config.habitat_baselines.video_dir)),
                "episode_results.jsonl",
            )
        print(f"[EP_RESULT_FILE] {episode_results_path}")

        current_episode_key = None
        episode_state = {
            "steps": 0,
            "instantaneous_pddl_success": None,
            "ever_pddl_success": False,
        }

        def _episode_key(episode_info):
            return (
                str(episode_info.scene_id),
                str(episode_info.episode_id),
            )

        def _episode_tokens() -> Dict[str, Any]:
            get_usage = getattr(
                agent.actor_critic, "get_episode_token_usage", None
            )
            empty_usage = {
                "group_discussion": None,
                "execution": None,
                "execution_by_agent": {},
                "total": None,
            }
            if get_usage is None:
                return empty_usage
            try:
                return get_usage()
            except Exception as token_error:
                empty_usage["error"] = (
                    f"{type(token_error).__name__}: {token_error}"
                )
                return empty_usage

        def _write_episode_result(
            episode_info,
            crash_type=None,
            crash_message=None,
        ) -> Dict[str, Any]:
            key = _episode_key(episode_info)
            token_breakdown = _episode_tokens()
            record = {
                "scene_id": key[0],
                "episode_id": key[1],
                "eval_index": ep_eval_count[key] + 1,
                "instantaneous_pddl_success": episode_state[
                    "instantaneous_pddl_success"
                ],
                "ever_pddl_success": bool(
                    episode_state["ever_pddl_success"]
                ),
                "crash_type": crash_type,
                "steps": int(episode_state["steps"]),
                "tokens": token_breakdown.get("total"),
                "token_breakdown": token_breakdown,
                "model": os.environ.get(
                    "HABITAT_LLM_MODEL", "gpt-4o"
                ),
            }
            if crash_message:
                record["crash_message"] = str(crash_message)
            append_episode_result(episode_results_path, record)
            return record

        def _crash_type(error: Exception) -> str:
            if isinstance(error, ToolCallCrashError):
                return "tool_call_crash"
            if (
                isinstance(error, ValueError)
                and "Cannot find matching entity" in str(error)
            ):
                return "entity_resolution_crash"
            return f"policy_{type(error).__name__.lower()}"

        if len(config.habitat_baselines.eval.image_option) > 0:
            os.makedirs(config.habitat_baselines.image_dir, exist_ok=True)

        video_options = config.habitat_baselines.eval.video_option
        stream_video_to_disk = "disk" in video_options
        buffer_video = any(option != "disk" for option in video_options)
        if stream_video_to_disk:
            os.makedirs(config.habitat_baselines.video_dir, exist_ok=True)
            disk_video_streams = [
                _StreamingDiskVideo(
                    config.habitat_baselines.video_dir,
                    config.habitat_baselines.video_fps,
                    f"env-{env_idx}",
                )
                for env_idx in range(
                    config.habitat_baselines.num_environments
                )
            ]
            disk_video_streams_fourth = (
                [
                    _StreamingDiskVideo(
                        config.habitat_baselines.video_dir,
                        config.habitat_baselines.video_fps,
                        f"env-{env_idx}-fourth",
                    )
                    for env_idx in range(
                        config.habitat_baselines.num_environments
                    )
                ]
                if config.habitat_baselines.eval.generate_fourth_rgb
                else None
            )
            pending_disk_frames: List[Optional[np.ndarray]] = [
                None
                for _ in range(config.habitat_baselines.num_environments)
            ]
            pending_disk_frames_fourth: List[Optional[np.ndarray]] = [
                None
                for _ in range(config.habitat_baselines.num_environments)
            ]
        else:
            disk_video_streams = None
            disk_video_streams_fourth = None
            pending_disk_frames = []
            pending_disk_frames_fourth = []

        if buffer_video:
            # Add the first frame of the episode to the video.
            rgb_frames: List[List[np.ndarray]] = [
                [
                    observations_to_image(
                        {k: v[env_idx] for k, v in batch.items() if
                             k != "agent_0_fourth_rgb" and k != "agent_1_fourth_rgb"}, {}, config,
                        0,
                    )
                ]
                for env_idx in range(config.habitat_baselines.num_environments)
            ]
            if config.habitat_baselines.eval.generate_fourth_rgb:
                rgb_frames_fourth: List[List[np.ndarray]] = [
                    [
                        observations_to_image(
                            {k: v[env_idx] for k, v in batch.items()if
                                 k == "agent_0_fourth_rgb"}, {}, config, 0,
                        )
                    ]
                    for env_idx in range(config.habitat_baselines.num_environments)
                ]
        else:
            rgb_frames_fourth = None
            rgb_frames = None

        if len(video_options) > 0:
            os.makedirs(config.habitat_baselines.video_dir, exist_ok=True)

        if stream_video_to_disk:
            for env_idx in range(
                config.habitat_baselines.num_environments
            ):
                initial_frame = observations_to_image(
                    {
                        k: v[env_idx]
                        for k, v in batch.items()
                        if k != "agent_0_fourth_rgb"
                        and k != "agent_1_fourth_rgb"
                    },
                    {},
                    config,
                    0,
                )
                disk_video_streams[env_idx].append(initial_frame)
                if disk_video_streams_fourth is not None:
                    initial_fourth = observations_to_image(
                        {
                            k: v[env_idx]
                            for k, v in batch.items()
                            if k == "agent_0_fourth_rgb"
                        },
                        {},
                        config,
                        0,
                    )
                    disk_video_streams_fourth[env_idx].append(initial_fourth)

        number_of_eval_episodes = config.habitat_baselines.test_episode_count
        evals_per_ep = config.habitat_baselines.eval.evals_per_ep
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(envs.number_of_episodes)
        else:
            total_num_eps = sum(envs.number_of_episodes)
            # if total_num_eps is negative, it means the number of evaluation episodes is unknown
            if total_num_eps < number_of_eval_episodes and total_num_eps > 1:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps
            else:
                assert evals_per_ep == 1
        assert (
            number_of_eval_episodes > 0
        ), "You must specify a number of evaluation episodes with test_episode_count"
        envs_text_context = {}
        pbar = tqdm.tqdm(total=number_of_eval_episodes * evals_per_ep)
        agent.eval()
        cur_ep_id = -1
        while (
            len(stats_episodes) < (number_of_eval_episodes * evals_per_ep)
            and envs.num_envs > 0
        ):
            current_episodes_info = envs.current_episodes()

            # If all prev_actions are zero, meaning this is the start of an episode
            # Then collect the context of the episode
            episode_key = _episode_key(current_episodes_info[0])
            if episode_key != current_episode_key:
                current_episode_key = episode_key
                episode_state = {
                    "steps": 0,
                    "instantaneous_pddl_success": None,
                    "ever_pddl_success": False,
                }
                cur_ep_id = current_episodes_info[0].episode_id
                start_accounting = getattr(
                    agent.actor_critic, "start_episode_accounting", None
                )
                if start_accounting is not None:
                    start_accounting(cur_ep_id)
                print("===============================================================================")
                print("=================================Episode ID====================================")
                print("Current Episode ID: ", cur_ep_id)
                print("=================================Episode ID====================================")
                print("===============================================================================")
                envs_text_context = envs.call(["get_task_text_context"] * envs.num_envs)
                if 'pddl_text_goal' in batch:
                    envs_pddl_text_goal_np = batch['pddl_text_goal'].cpu().numpy()
                    for i in range(envs.num_envs):
                        pddl_text_goal_np = envs_pddl_text_goal_np[i, ...]
                        envs_text_context[i]['pddl_text_goal'] = ''.join(str(pddl_text_goal_np, encoding='UTF-8'))
                
                for i in range(envs.num_envs):
                    # also add the debug/ logging info to the text context for convenience
                    envs_text_context[i]['episode_id'] = current_episodes_info[i].episode_id

            space_lengths = {}
            n_agents = len(config.habitat.simulator.agents)
            if n_agents > 1:
                space_lengths = {
                    "index_len_recurrent_hidden_states": hidden_state_lens,
                    "index_len_prev_actions": action_space_lens,
                }
            try:
                with inference_mode():
                    action_data = agent.actor_critic.act(
                        batch,
                        test_recurrent_hidden_states,
                        prev_actions,
                        not_done_masks,
                        deterministic=False,
                        envs_text_context=envs_text_context,
                        **space_lengths,
                    )
                    if action_data.should_inserts is None:
                        test_recurrent_hidden_states = (
                            action_data.rnn_hidden_states
                        )
                        prev_actions.copy_(action_data.actions)  # type: ignore
                    else:
                        agent.actor_critic.update_hidden_state(
                            test_recurrent_hidden_states,
                            prev_actions,
                            action_data,
                        )
            except Exception as crash:
                kind = _crash_type(crash)
                record = _write_episode_result(
                    current_episodes_info[0], kind, str(crash)
                )
                key = _episode_key(current_episodes_info[0])
                ep_eval_count[key] += 1
                stats_episodes[(key, ep_eval_count[key])] = {
                    "reward": 0.0,
                    # A policy crash is always a benchmark failure. The last
                    # instantaneous value remains available in the JSONL.
                    "pddl_success": 0.0,
                    "ever_pddl_success": float(
                        record["ever_pddl_success"]
                    ),
                    "num_steps": float(record["steps"]),
                    "tokens": float(record["tokens"] or 0),
                    "llm_crash": 1.0,
                    "tool_call_crash": float(
                        kind == "tool_call_crash"
                    ),
                    "entity_resolution_crash": float(
                        kind == "entity_resolution_crash"
                    ),
                }
                pbar.update()
                print(
                    f"[LLM_CRASH:{kind}] "
                    f"scene_id={record['scene_id']} "
                    f"episode_id={record['episode_id']} "
                    f"pddl_success={record['instantaneous_pddl_success']} "
                    f"ever_pddl_success={record['ever_pddl_success']} "
                    f"steps={record['steps']} tokens={record['tokens']} "
                    f"reason={str(crash)[:200]}"
                )
                current_episode_key = None
                current_episode_reward.zero_()
                if len(stats_episodes) >= (
                    number_of_eval_episodes * evals_per_ep
                ):
                    continue

                observations = envs.reset()
                observations = envs.post_step(observations)
                batch = batch_obs(observations, device=device)
                batch = apply_obs_transforms_batch(
                    batch, obs_transforms
                )  # type: ignore
                not_done_masks = torch.zeros(
                    config.habitat_baselines.num_environments,
                    *agent.masks_shape,
                    device=device,
                    dtype=torch.bool,
                )
                prev_actions.zero_()
                test_recurrent_hidden_states.zero_()
                cur_ep_id = -1
                continue

            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            if is_continuous_action_space(env_spec.action_space):
                # Clipping actions to the specified limits
                step_data = [
                    np.clip(
                        a.numpy(),
                        env_spec.action_space.low,
                        env_spec.action_space.high,
                    )
                    for a in action_data.env_actions.cpu()
                ]
            else:
                step_data = [a.item() for a in action_data.env_actions.cpu()]

            try:
                outputs = envs.step(step_data)
            except Exception as crash:
                _write_episode_result(
                    current_episodes_info[0],
                    f"environment_{type(crash).__name__.lower()}",
                    str(crash),
                )
                # A failed vector-environment worker cannot be assumed safe to
                # reset, but the episode record is already durable.
                raise
            episode_state["steps"] += 1

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            instantaneous_success = infos[0].get("pddl_success")
            if instantaneous_success is not None:
                try:
                    instantaneous_success = bool(
                        float(instantaneous_success)
                    )
                except (TypeError, ValueError):
                    instantaneous_success = bool(instantaneous_success)
                episode_state[
                    "instantaneous_pddl_success"
                ] = instantaneous_success
                episode_state["ever_pddl_success"] = bool(
                    episode_state["ever_pddl_success"]
                    or instantaneous_success
                )

            # Persist the terminal metric before visualization, batching, or
            # aggregate-statistics code can fail.
            completed_records = {}
            if dones[0]:
                completed_records[0] = _write_episode_result(
                    current_episodes_info[0]
                )

            # Note that `policy_infos` represents the information about the
            # action BEFORE `observations` (the action used to transition to
            # `observations`).
            policy_infos = agent.actor_critic.get_extra(
                action_data, infos, dones
            )
            for i in range(len(policy_infos)):
                infos[i].update(policy_infos[i])

            observations = envs.post_step(observations)
            batch = batch_obs(  # type: ignore
                observations,
                device=device,
            )
            batch = apply_obs_transforms_batch(batch, obs_transforms)  # type: ignore

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            ).repeat(1, *agent.masks_shape)

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes_info = envs.current_episodes()
            envs_to_pause = []
            n_envs = envs.num_envs
            for i in range(n_envs):
                if (
                    ep_eval_count[_episode_key(next_episodes_info[i])]
                    == evals_per_ep
                ):
                    envs_to_pause.append(i)

                # Exclude the keys from `_rank0_keys` from displaying in the video
                disp_info = {
                    k: v for k, v in infos[i].items() if k not in rank0_keys
                }

                if len(video_options) > 0:
                    if (
                        stream_video_to_disk
                        and pending_disk_frames[i] is not None
                    ):
                        disk_video_streams[i].append(
                            pending_disk_frames[i]
                        )
                        pending_disk_frames[i] = None
                        if (
                            disk_video_streams_fourth is not None
                            and pending_disk_frames_fourth[i] is not None
                        ):
                            disk_video_streams_fourth[i].append(
                                pending_disk_frames_fourth[i]
                            )
                            pending_disk_frames_fourth[i] = None
                    frame_id = (
                        disk_video_streams[i].frame_count
                        if stream_video_to_disk
                        else len(rgb_frames[i])
                    )
                    # TODO move normalization / channel changing out of the policy and undo it here
                    frame = observations_to_image(
                        {k: v[i] for k, v in batch.items()if
                         k != "agent_0_fourth_rgb" and k != "agent_1_fourth_rgb"}, disp_info,
                        config, frame_id,
                        episode_id=current_episodes_info[i].episode_id,
                    )
                    if config.habitat_baselines.eval.generate_fourth_rgb:
                        frame_fourth = observations_to_image(
                            {k: v[i] for k, v in batch.items() if
                             k == "agent_0_fourth_rgb"}, infos[i],
                            config, frame_id,
                            episode_id=current_episodes_info[i].episode_id,
                        )
                    if not not_done_masks[i].any().item():
                        # The last frame corresponds to the first frame of the next episode
                        # but the info is correct. So we use a black frame
                        final_frame = observations_to_image(
                            {k: v[i] * 0.0 for k, v in batch.items()if
                             k != "agent_0_fourth_rgb" and k != "agent_1_fourth_rgb"},
                            disp_info, config,
                            frame_id=frame_id,
                            episode_id=current_episodes_info[i].episode_id,
                        )
                        if config.habitat_baselines.eval.generate_fourth_rgb:
                            final_frame_fourth = observations_to_image(
                                {k: v[i] for k, v in batch.items() if
                                 k == "agent_0_fourth_rgb"}, infos[i],
                                config, frame_id,
                                episode_id=current_episodes_info[i].episode_id,
                            )
                        final_frame = overlay_frame(final_frame, disp_info)
                        if stream_video_to_disk:
                            disk_video_streams[i].append(final_frame)
                        if buffer_video:
                            rgb_frames[i].append(final_frame)
                            # The starting frame of the next episode will be the final element.
                            rgb_frames[i].append(frame)
                        if config.habitat_baselines.eval.generate_fourth_rgb:
                            final_frame_fourth = overlay_frame(final_frame_fourth, infos[i])
                            if disk_video_streams_fourth is not None:
                                disk_video_streams_fourth[i].append(
                                    final_frame_fourth
                                )
                            if buffer_video:
                                rgb_frames_fourth[i].append(final_frame_fourth)
                                rgb_frames_fourth[i].append(frame_fourth)
                    else:
                        frame = overlay_frame(frame, disp_info)
                        if stream_video_to_disk:
                            disk_video_streams[i].append(frame)
                        if buffer_video:
                            rgb_frames[i].append(frame)
                        if config.habitat_baselines.eval.generate_fourth_rgb:
                            frame_fourth = overlay_frame(frame_fourth, infos[i])
                            if disk_video_streams_fourth is not None:
                                disk_video_streams_fourth[i].append(
                                    frame_fourth
                                )
                            if buffer_video:
                                rgb_frames_fourth[i].append(frame_fourth)

                # episode ended
                if not not_done_masks[i].any().item():
                    pbar.update()
                    episode_stats = {
                        "reward": current_episode_reward[i].item()
                    }
                    episode_stats.update(extract_scalars_from_info(infos[i]))
                    episode_stats.setdefault("llm_crash", 0.0)
                    episode_stats.setdefault("tool_call_crash", 0.0)
                    episode_stats.setdefault(
                        "entity_resolution_crash", 0.0
                    )
                    episode_stats["ever_pddl_success"] = float(
                        episode_state["ever_pddl_success"]
                    )
                    episode_stats.setdefault(
                        "num_steps", float(episode_state["steps"])
                    )
                    current_episode_reward[i] = 0
                    k = _episode_key(current_episodes_info[i])
                    record = completed_records[i]
                    episode_stats["tokens"] = float(
                        record["tokens"] or 0
                    )
                    ep_eval_count[k] += 1
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[(k, ep_eval_count[k])] = episode_stats
                    print(
                        f"[EP_RESULT] scene_id={record['scene_id']} "
                        f"episode_id={record['episode_id']} "
                        "pddl_success="
                        f"{record['instantaneous_pddl_success']} "
                        f"ever_pddl_success={record['ever_pddl_success']} "
                        f"crash_type=None steps={record['steps']} "
                        f"tokens={record['tokens']}"
                    )

                    # clear the prev_actions and recurrent_hidden_states
                    prev_actions[i] = 0
                    test_recurrent_hidden_states[i] = 0

                    if len(video_options) > 0:
                        video_metrics = extract_scalars_from_info(disp_info)
                        episode_video_id = (
                            f"{current_episodes_info[i].episode_id}_"
                            f"{ep_eval_count[k]}"
                        )
                        if stream_video_to_disk:
                            if disk_video_streams_fourth is not None:
                                fourth_name = _evaluation_video_name(
                                    f"{episode_video_id}_fourth",
                                    checkpoint_index,
                                    video_metrics,
                                    config.habitat_baselines.eval_keys_to_include_in_name,
                                )
                                disk_video_streams_fourth[i].finish(
                                    fourth_name
                                )
                            video_name = _evaluation_video_name(
                                episode_video_id,
                                checkpoint_index,
                                video_metrics,
                                config.habitat_baselines.eval_keys_to_include_in_name,
                            )
                            disk_video_streams[i].finish(video_name)

                        non_disk_options = [
                            option
                            for option in video_options
                            if option != "disk"
                        ]
                    if buffer_video:
                        if config.habitat_baselines.eval.generate_fourth_rgb:
                            generate_video(
                                video_option=non_disk_options,
                                video_dir=config.habitat_baselines.video_dir,
                                images=rgb_frames_fourth[i][:-1],
                                episode_id=f"{current_episodes_info[i].episode_id}_{ep_eval_count[k]}_fourth",
                                checkpoint_idx=checkpoint_index,
                                metrics=extract_scalars_from_info(disp_info),
                                fps=config.habitat_baselines.video_fps,
                                tb_writer=writer,
                                keys_to_include_in_name=config.habitat_baselines.eval_keys_to_include_in_name,
                            )
                        generate_video(
                            video_option=non_disk_options,
                            video_dir=config.habitat_baselines.video_dir,
                            # Since the final frame is the start frame of the next episode.
                            images=rgb_frames[i][:-1],
                            episode_id=f"{current_episodes_info[i].episode_id}_{ep_eval_count[k]}",
                            checkpoint_idx=checkpoint_index,
                            metrics=extract_scalars_from_info(disp_info),
                            fps=config.habitat_baselines.video_fps,
                            tb_writer=writer,
                            keys_to_include_in_name=config.habitat_baselines.eval_keys_to_include_in_name,
                        )

                        # Since the starting frame of the next episode is the final frame.
                        if config.habitat_baselines.eval.generate_fourth_rgb:
                            rgb_frames_fourth[i] = rgb_frames_fourth[i][-1:]
                        rgb_frames[i] = rgb_frames[i][-1:]

                    if stream_video_to_disk:
                        # The environment has already advanced. Keep only its
                        # first frame in RAM until we know another episode
                        # will actually be evaluated.
                        pending_disk_frames[i] = frame
                        if disk_video_streams_fourth is not None:
                            pending_disk_frames_fourth[i] = frame_fourth

                    gfx_str = infos[i].get(GfxReplayMeasure.cls_uuid, "")
                    if gfx_str != "":
                        write_gfx_replay(
                            gfx_str,
                            config.habitat.task,
                            current_episodes_info[i].episode_id,
                        )

            retained_env_indices = [
                index
                for index in range(envs.num_envs)
                if index not in envs_to_pause
            ]
            not_done_masks = not_done_masks.to(device=device)
            (
                envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = pause_envs(
                envs_to_pause,
                envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )
            if stream_video_to_disk and envs_to_pause:
                disk_video_streams = [
                    disk_video_streams[index]
                    for index in retained_env_indices
                ]
                pending_disk_frames = [
                    pending_disk_frames[index]
                    for index in retained_env_indices
                ]
                if disk_video_streams_fourth is not None:
                    disk_video_streams_fourth = [
                        disk_video_streams_fourth[index]
                        for index in retained_env_indices
                    ]
                    pending_disk_frames_fourth = [
                        pending_disk_frames_fourth[index]
                        for index in retained_env_indices
                    ]

            # We pause the statefull parameters in the policy.
            # We only do this if there are envs to pause to reduce the overhead.
            # In addition, HRL policy requires the solution_actions to be non-empty, and
            # empty list of envs_to_pause will raise an error.
            if any(envs_to_pause):
                agent.actor_critic.on_envs_pause(envs_to_pause)

        pbar.close()
        assert (
            len(ep_eval_count) >= number_of_eval_episodes
        ), f"Expected {number_of_eval_episodes} episodes, got {len(ep_eval_count)}."

        aggregated_stats = {}
        all_ks = set()
        for ep in stats_episodes.values():
            all_ks.update(ep.keys())
        for stat_key in all_ks:
            aggregated_stats[stat_key] = np.mean(
                [v[stat_key] for v in stats_episodes.values() if stat_key in v]
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        writer.add_scalar(
            "eval_reward/average_reward", aggregated_stats["reward"], step_id
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)
