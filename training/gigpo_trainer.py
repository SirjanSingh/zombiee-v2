"""GiGPOTrainer — TRL GRPOTrainer subclass that adds step-level advantage.

Phase 2 of `.planning2/11_RESEARCH_FINDINGS_AND_REVISED_PLAN.md`.

What this class does
--------------------
Inherits everything from `trl.GRPOTrainer` (TRL 0.15.2) and overrides
`_prepare_inputs` to inject a step-level advantage term computed by
`training.gigpo.compose_gigpo_advantages` AFTER GRPO's standard
within-prompt-group normalisation.

Final advantage tensor sent to the PPO clipped objective:

    advantages = (r - group_mean) / (group_std + eps)        # GRPO part
               + step_advantage_w * step_norm_within_cluster(r_step)

The episode-level part is exactly what stock GRPOTrainer already
computes. The step-level part is the GiGPO contribution: cluster
generations by anchor state, normalise step rewards inside each cluster.

The hook point
--------------
We do NOT touch the reward function pipeline, the generation loop, the
ref-model logps, or the PPO loss. Only `_prepare_inputs` is overridden,
and the override is a copy of the parent method (TRL 0.15.2) with a
single new block inserted just before the per-process slice. This keeps
the diff against TRL minimal — when we eventually upgrade past 0.15.2
we re-port the override.

Side-channel data flow
----------------------
GiGPO needs per-generation (anchor_key, step_reward). The reward function
already has both — it runs the env, knows the first-action state, and
records `model_step_raws[0]`. The cleanest way to pass these out to the
trainer without changing TRL's `reward_funcs(prompts, completions)`
signature is a shared mutable list:

    side_channel = []  # populated by reward_fn, consumed by trainer

    reward_fn = create_reward_fn(..., gigpo_side_channel=side_channel)
    trainer = GiGPOTrainer(..., gigpo_side_channel=side_channel, ...)

The reward function clears the list at the start of each call and
appends one (anchor, step_reward) tuple per (prompt, completion) in the
exact same order as the rewards list it returns. The trainer reads the
list immediately after `gather(rewards_per_func)`.

Multi-GPU caveat
----------------
On a single-GPU run (our V100 DGX setup), the side-channel data is
populated and consumed on the same process — no synchronisation needed.
On a multi-GPU run, the side-channel data would need to be gathered
across processes alongside `rewards_per_func`. We document this as a
known limitation and assert single-process at construction time. Lifting
the assertion requires wrapping the side-channel in `gather_object`.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Union

import torch
import torch.nn as nn

from training.gigpo import compose_gigpo_advantages

logger = logging.getLogger("survivecity_v2.gigpo_trainer")


def make_gigpo_trainer_class():
    """Return GiGPOTrainer as a subclass of trl.GRPOTrainer.

    Wrapped in a factory so this module can be imported without trl
    installed (e.g. on a doc-build / lint machine). The trl import only
    happens when the factory is called.
    """
    from trl import GRPOTrainer
    from accelerate.utils import broadcast_object_list, gather, gather_object
    from trl.data_utils import (
        apply_chat_template,
        is_conversational,
        maybe_apply_chat_template,
    )
    from trl.models.utils import unwrap_model_for_generation
    from trl.trainer.utils import pad

    class GiGPOTrainer(GRPOTrainer):
        """GRPOTrainer + GiGPO step-level advantage injection.

        Extra constructor kwargs:
          gigpo_side_channel: mutable list shared with the reward fn.
              Reward fn appends (anchor_key, step_reward) per generation
              in the order it returns rewards. This trainer reads from
              the list inside _prepare_inputs and clears it before the
              next reward fn call.
          step_advantage_w: weight on the step-level advantage. Paper
              default 1.0; insensitive in [0.4, 1.2]. Pass 0.0 to
              effectively run vanilla GRPO via this class.
          gigpo_remove_std: if True (default, paper convention), step
              advantage is mean-subtract only. If False, full z-score.
          gigpo_log_every: log clustering diagnostics every N steps
              (default 1 — log every step; matches our existing
              [metrics] cadence).
        """

        def __init__(
            self,
            *args,
            gigpo_side_channel: Optional[list] = None,
            step_advantage_w: float = 1.0,
            gigpo_remove_std: bool = True,
            gigpo_log_every: int = 1,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.gigpo_side_channel = gigpo_side_channel
            self.step_advantage_w = float(step_advantage_w)
            self.gigpo_remove_std = bool(gigpo_remove_std)
            self.gigpo_log_every = max(1, int(gigpo_log_every))
            self._gigpo_step_counter = 0

            if self.gigpo_side_channel is None:
                logger.warning(
                    "GiGPOTrainer constructed with gigpo_side_channel=None — "
                    "step-level advantage will be disabled, behaviour reduces "
                    "to vanilla GRPOTrainer."
                )

            # Multi-GPU guard: we haven't implemented the cross-process
            # side-channel gather. Refuse to silently produce a wrong gradient.
            if (
                self.accelerator is not None
                and self.accelerator.num_processes > 1
                and self.gigpo_side_channel is not None
            ):
                raise RuntimeError(
                    f"GiGPOTrainer currently supports single-process only "
                    f"(got num_processes={self.accelerator.num_processes}). "
                    f"Multi-GPU support requires gathering anchor_obs across "
                    f"processes alongside rewards_per_func — TODO. For now, "
                    f"run with a single CUDA device or pass step_advantage_w=0 "
                    f"+ gigpo_side_channel=None to fall back to GRPO."
                )

            logger.info(
                f"GiGPOTrainer armed: step_advantage_w={self.step_advantage_w} "
                f"remove_std={self.gigpo_remove_std} "
                f"side_channel={'on' if self.gigpo_side_channel is not None else 'off'}"
            )

        # ------------------------------------------------------------------
        # The only override
        # ------------------------------------------------------------------
        def _prepare_inputs(
            self, inputs: dict[str, Union[torch.Tensor, Any]]
        ) -> dict[str, Union[torch.Tensor, Any]]:
            """Copy of GRPOTrainer._prepare_inputs (TRL 0.15.2) with the
            GiGPO step-advantage injection just before the per-process slice.

            If you upgrade TRL past 0.15.2, re-port this method against the
            new upstream version. The injection block is clearly marked.
            """
            # IMPORTANT: clear the side-channel BEFORE calling the reward funcs.
            # We do this here (rather than inside the reward fn) so that even
            # if the user's reward fn forgets to clear, the contract holds:
            # one _prepare_inputs call = one fresh side channel.
            if self.gigpo_side_channel is not None:
                self.gigpo_side_channel.clear()

            device = self.accelerator.device
            prompts = [x["prompt"] for x in inputs]
            prompts_text = [
                maybe_apply_chat_template(example, self.processing_class)["prompt"]
                for example in inputs
            ]
            prompt_inputs = self.processing_class(
                prompts_text, return_tensors="pt", padding=True,
                padding_side="left", add_special_tokens=False,
            )
            prompt_inputs = super(GRPOTrainer, self)._prepare_inputs(prompt_inputs)
            prompt_ids = prompt_inputs["input_ids"]
            prompt_mask = prompt_inputs["attention_mask"]

            if self.max_prompt_length is not None:
                prompt_ids = prompt_ids[:, -self.max_prompt_length:]
                prompt_mask = prompt_mask[:, -self.max_prompt_length:]

            # Generate completions
            if self.args.use_vllm:
                if self.state.global_step != self._last_loaded_step:
                    self._move_model_to_vllm()
                    self._last_loaded_step = self.state.global_step
                all_prompts_text = gather_object(prompts_text)
                if self.accelerator.is_main_process:
                    outputs = self.llm.generate(
                        all_prompts_text, sampling_params=self.sampling_params,
                        use_tqdm=False,
                    )
                    completion_ids = [
                        out.token_ids
                        for completions in outputs
                        for out in completions.outputs
                    ]
                else:
                    completion_ids = [None] * len(all_prompts_text)
                completion_ids = broadcast_object_list(completion_ids, from_process=0)
                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                completion_ids = completion_ids[process_slice]
                completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
                completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
                prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            else:
                with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                    prompt_completion_ids = unwrapped_model.generate(
                        prompt_ids, attention_mask=prompt_mask,
                        generation_config=self.generation_config,
                    )
                prompt_length = prompt_ids.size(1)
                prompt_ids = prompt_completion_ids[:, :prompt_length]
                completion_ids = prompt_completion_ids[:, prompt_length:]

            # Mask everything after the first EOS
            is_eos = completion_ids == self.processing_class.eos_token_id
            eos_idx = torch.full(
                (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device,
            )
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(
                is_eos.size(0), -1,
            )
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            logits_to_keep = completion_ids.size(1)

            with torch.inference_mode():
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, prompt_completion_ids,
                        attention_mask, logits_to_keep,
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model, prompt_completion_ids,
                            attention_mask, logits_to_keep,
                        )

            # Decode completions for the reward funcs
            completions_text = self.processing_class.batch_decode(
                completion_ids, skip_special_tokens=True,
            )
            if is_conversational(inputs[0]):
                completions = []
                for prompt, completion in zip(prompts, completions_text):
                    bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                    completions.append([{"role": "assistant", "content": bootstrap + completion}])
            else:
                completions = completions_text

            rewards_per_func = torch.zeros(
                len(prompts), len(self.reward_funcs), device=device,
            )
            for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)
            ):
                if isinstance(reward_func, nn.Module):
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [
                            apply_chat_template(x, reward_processing_class)["text"]
                            for x in messages
                        ]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        texts, return_tensors="pt", padding=True,
                        padding_side="right", add_special_tokens=False,
                    )
                    reward_inputs = super(GRPOTrainer, self)._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
                else:
                    keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                    reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                    output_reward_func = reward_func(
                        prompts=prompts, completions=completions, **reward_kwargs,
                    )
                    rewards_per_func[:, i] = torch.tensor(
                        output_reward_func, dtype=torch.float32, device=device,
                    )

            # Gather rewards across processes
            rewards_per_func = gather(rewards_per_func)
            rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

            # GRPO group-relative normalisation (episode advantage)
            mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
            std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
                self.num_generations, dim=0,
            )
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(
                self.num_generations, dim=0,
            )
            advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

            # ==================================================================
            # GIGPO STEP-LEVEL INJECTION — the one substantive override.
            #
            # Add the cluster-normalised step-level advantage to GRPO's
            # episode advantage. Skipped silently if the side channel is
            # unwired, the weight is 0, or the size doesn't match (which
            # would indicate the reward fn forgot to record per-generation
            # data).
            # ==================================================================
            self._gigpo_step_counter += 1
            n_local = rewards.shape[0]
            if (
                self.gigpo_side_channel is not None
                and self.step_advantage_w > 0
                and len(self.gigpo_side_channel) == n_local
            ):
                anchor_keys = [t[0] for t in self.gigpo_side_channel]
                step_rewards = torch.tensor(
                    [t[1] for t in self.gigpo_side_channel],
                    dtype=advantages.dtype, device=advantages.device,
                )
                # Prompt index: which GRPO group this generation belongs to.
                # In GRPOTrainer, completions are laid out as
                # [p0_g0, p0_g1, ..., p0_g{G-1}, p1_g0, ...], so
                # prompt_idx[i] = i // num_generations.
                prompt_index = [i // self.num_generations for i in range(n_local)]
                combined, diag = compose_gigpo_advantages(
                    episode_advantages=advantages,
                    step_rewards=step_rewards,
                    anchor_keys=anchor_keys,
                    prompt_index=prompt_index,
                    step_advantage_w=self.step_advantage_w,
                    remove_std=self.gigpo_remove_std,
                )
                advantages = combined
                if self._gigpo_step_counter % self.gigpo_log_every == 0:
                    self._metrics.setdefault("gigpo/n_clusters", []).append(diag["n_clusters"])
                    self._metrics.setdefault("gigpo/mean_cluster_size", []).append(diag["mean_size"])
                    self._metrics.setdefault("gigpo/singleton_frac", []).append(diag["singleton_frac"])
                    self._metrics.setdefault("gigpo/step_adv_std", []).append(diag["step_adv_std"])
                    logger.info(
                        f"[gigpo] step={self.state.global_step} "
                        f"n_clusters={diag['n_clusters']} "
                        f"mean_size={diag['mean_size']} "
                        f"singleton_frac={diag['singleton_frac']} "
                        f"step_adv_std={diag['step_adv_std']:.4f}"
                    )
            elif self.step_advantage_w > 0 and self.gigpo_side_channel is not None:
                # Size mismatch → reward fn didn't populate correctly. Don't
                # silently fall back; tell the operator.
                logger.warning(
                    f"[gigpo] side-channel size {len(self.gigpo_side_channel)} "
                    f"!= local rewards {n_local}. Skipping step injection this "
                    f"call. Reward fn must append exactly one (anchor, step_r) "
                    f"per generation."
                )
            # ==================================================================
            # END GiGPO injection
            # ==================================================================

            # Per-process slice
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            advantages = advantages[process_slice]

            # Standard GRPO metrics
            reward_per_func = rewards_per_func.mean(0)
            for i, reward_func in enumerate(self.reward_funcs):
                if isinstance(reward_func, nn.Module):
                    reward_func_name = reward_func.config._name_or_path.split("/")[-1]
                else:
                    reward_func_name = reward_func.__name__
                self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())
            self._metrics["reward"].append(rewards.mean().item())
            self._metrics["reward_std"].append(std_grouped_rewards.mean().item())

            return {
                "prompt_ids": prompt_ids,
                "prompt_mask": prompt_mask,
                "completion_ids": completion_ids,
                "completion_mask": completion_mask,
                "ref_per_token_logps": ref_per_token_logps,
                "advantages": advantages,
            }

    return GiGPOTrainer
