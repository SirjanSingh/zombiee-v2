"""Closed-loop trainer (run 8): advantage math, rollouts, and a tiny-model end-to-end smoke.

No downloads: the smoke test builds a 2-layer Qwen2 from a config and a byte-level
tokenizer in memory.
"""

import json
import math
import random
import statistics as S

import pytest

torch = pytest.importorskip("torch")

from training import train_closed_loop as T  # noqa: E402


def approx(a, b, tol=1e-3):
    return all(abs(x - y) < tol for x, y in zip(a, b)) and len(a) == len(b)


# ---------------------------------------------------------------------------
# Advantage math on hand-made trajectories
# ---------------------------------------------------------------------------

def test_discounted_returns():
    assert approx(T.discounted_returns([0, 0, 1], 0.5), [0.25, 0.5, 1.0])
    assert T.discounted_returns([], 0.9) == []


def test_grpo_group_normalised_and_broadcast():
    trajs = [
        {"group": 0, "rewards": [0, 1], "anchors": ["a", "b"]},      # R = 1
        {"group": 0, "rewards": [0, 0, 3], "anchors": ["a", "c", "d"]},  # R = 3
        {"group": 1, "rewards": [1], "anchors": ["a"]},              # identical returns in group 1
        {"group": 1, "rewards": [1], "anchors": ["a"]},
        {"group": 2, "rewards": [5], "anchors": ["z"]},              # singleton group
    ]
    adv, _ = T.compute_advantages(trajs, "grpo")
    sd = math.sqrt(2)                       # sample std of [1, 3]
    assert approx(adv[0], [-1 / (sd + 1e-4)] * 2)
    assert approx(adv[1], [1 / (sd + 1e-4)] * 3)
    assert adv[2] == [0.0] and adv[3] == [0.0] and adv[4] == [0.0]


def test_gigpo_adds_anchor_step_advantage_within_group_only():
    g = 0.5
    trajs = [
        {"group": 0, "rewards": [0, 1], "anchors": ["s0", "s"]},   # G = [0.5, 1]
        {"group": 0, "rewards": [0, 0], "anchors": ["s0", "s"]},   # G = [0, 0]
        {"group": 1, "rewards": [0, 7], "anchors": ["s0", "s"]},   # same keys, other group
        {"group": 1, "rewards": [0, 7], "anchors": ["s0", "s"]},
    ]
    adv, diag = T.compute_advantages(trajs, "gigpo", gamma=g, step_w=1.0)
    ep = 0.5 / (math.sqrt(0.5) + 1e-4)      # GRPO part for returns [1, 0]
    assert approx(adv[0], [ep + 0.25, ep + 0.5])
    assert approx(adv[1], [-ep - 0.25, -ep - 0.5])
    # group 1 has identical rollouts: zero GRPO and zero step advantage, and its
    # anchors did not leak into group 0's clusters (else the means above would move)
    assert approx(adv[2], [0, 0]) and approx(adv[3], [0, 0])
    assert diag["n_clusters"] == 4


def test_gigpo_step_weight_zero_equals_grpo():
    trajs = [{"group": 0, "rewards": [0, r], "anchors": ["x", "y"]} for r in (0.0, 1.0, 2.0)]
    a1, _ = T.compute_advantages(trajs, "gigpo", gamma=0.9, step_w=0.0)
    a2, _ = T.compute_advantages(trajs, "grpo")
    assert all(approx(x, y) for x, y in zip(a1, a2))


def test_gagpo_values_gae_recursion_and_group_standardisation():
    g, lam = 0.5, 0.5
    trajs = [
        {"group": 0, "rewards": [0, 1], "anchors": ["a", "b"]},    # G = [0.5, 1]
        {"group": 0, "rewards": [0, -1], "anchors": ["a", "c"]},   # G = [-0.5, -1]
        {"group": 1, "rewards": [2], "anchors": ["b"]},            # G = [2]; shares "b" across groups
    ]
    V = T.gagpo_values(trajs, g)
    assert V == pytest.approx({"a": 0.0, "b": 1.5, "c": -1.0})
    raw, diag = T.gagpo_advantages(trajs, g, lam, standardize=False)
    # traj 0: d1 = 1 + 0 - 1.5 = -0.5 ; d0 = 0 + 0.5*1.5 - 0 = 0.75 ; A0 = 0.75 + 0.25*(-0.5)
    assert approx(raw[0], [0.625, -0.5])
    # traj 1: d1 = -1 - (-1) = 0 ; d0 = 0 + 0.5*(-1) - 0 = -0.5
    assert approx(raw[1], [-0.5, 0.0])
    # traj 2: d0 = 2 - 1.5
    assert approx(raw[2], [0.5])
    assert diag["n_states"] == 3
    adv, _ = T.compute_advantages(trajs, "gagpo", gamma=g, lam=lam)
    g0 = adv[0] + adv[1]
    assert abs(S.fmean(g0)) < 1e-6 and abs(S.stdev(g0) - 1) < 1e-3
    assert adv[2] == [0.0]                  # singleton group
    m, sd = S.fmean([0.625, -0.5, -0.5, 0.0]), S.stdev([0.625, -0.5, -0.5, 0.0])
    assert approx(adv[0], [(0.625 - m) / (sd + 1e-4), (-0.5 - m) / (sd + 1e-4)])


def test_gagpo_lambda_zero_is_one_step_td():
    trajs = [{"group": 0, "rewards": [0, 0, 1], "anchors": ["a", "b", "c"]},
             {"group": 0, "rewards": [0, 0, 0], "anchors": ["a", "b", "d"]}]
    raw, _ = T.gagpo_advantages(trajs, 0.9, 0.0, standardize=False)
    V = T.gagpo_values(trajs, 0.9)
    for tr, a in zip(trajs, raw):
        for t in range(3):
            v_next = V[tr["anchors"][t + 1]] if t < 2 else 0.0
            assert a[t] == pytest.approx(tr["rewards"][t] + 0.9 * v_next - V[tr["anchors"][t]])


def test_group_return_stats_zero_variance_fraction():
    trajs = [{"group": 0, "rewards": [1]}, {"group": 0, "rewards": [1]},
             {"group": 1, "rewards": [0]}, {"group": 1, "rewards": [2]}]
    st = T.group_return_stats(trajs)
    assert st["zero_var_group_frac"] == 0.5
    assert st["return_mean"] == 1.0


# ---------------------------------------------------------------------------
# Rollouts on the real env (scripted completions, no model)
# ---------------------------------------------------------------------------

class CharTok:
    eos_token_id = 1

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [2 + (ord(c) % 50) for c in text]}


def _rollouts(reward_mode, horizon=12, g=3, k=2):
    from survivecity_v2_env.balance import get_balance
    from training.scenarios import build_scenarios, replay_to
    import copy
    bal = get_balance("v3-rc1")
    sc = build_scenarios(1, seed=3, balance=bal, a0_healthy=True, prefix_k=k)[0]
    env, obs, ok = replay_to(sc["seed"], sc["t"], sc["mix"], balance=bal, a0_healthy=True)
    assert ok
    rs = [T.Rollout(copy.deepcopy(env), copy.deepcopy(obs), seed=sc["seed"], group=0, balance=bal,
                    teammate="camp", k=k, horizon=horizon, reward_mode=reward_mode) for _ in range(g)]
    info = T.collect_rollouts(rs, T.make_scripted_generator(CharTok(), k))
    return rs, info, sc


def test_rollouts_graded_terminal_only_and_deterministic():
    rs, info, sc = _rollouts("graded")
    assert info["rounds"] >= 1
    for r in rs:
        assert r.decisions and all(d["parsed"] for d in r.decisions)
        assert all(d["reward"] == 0.0 for d in r.decisions[:-1])
        assert -1.0 <= r.decisions[-1]["reward"] <= 2.0
        assert r.env._episode.step_count <= r.t_end and 0 <= r.a0_life() <= r.horizon
        assert r.decisions[0]["t"] == sc["t"]
        assert r.decisions[0]["completion_ids"][-1] == CharTok.eos_token_id
    # same start state + same scripted actions -> identical rollouts (zero variance)
    assert len({tuple(d["reward"] for d in r.decisions) for r in rs}) == 1
    assert len({tuple(d["anchor"] for d in r.decisions) for r in rs}) == 1


def test_rollouts_env_reward_sums_to_cumulative_delta():
    rs, _, _ = _rollouts("env", horizon=10, g=1)
    r = rs[0]
    total = sum(d["reward"] for d in r.decisions)
    assert len(r.decisions) >= 2
    assert total == pytest.approx(r._cum() - r._cum_start, abs=1e-6)
    assert any(d["reward"] != 0.0 for d in r.decisions)


# ---------------------------------------------------------------------------
# Tiny-model end to end (no download)
# ---------------------------------------------------------------------------

def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def tiny_model_and_tokenizer():
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    vocab = {ch: i for i, ch in enumerate(_bytes_to_unicode().values())}
    vocab["<|endoftext|>"] = 256
    vocab["<|im_end|>"] = 257
    tk = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>",
                                               pad_token="<|endoftext|>")
    tok.add_special_tokens({"eos_token": "<|im_end|>", "pad_token": "<|endoftext|>"})
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(vocab_size=258, hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                   max_position_embeddings=512, eos_token_id=257, pad_token_id=256,
                                   bos_token_id=256, tie_word_embeddings=True)
    model = transformers.Qwen2ForCausalLM(cfg)
    model.generation_config.eos_token_id = 257
    model.generation_config.pad_token_id = 256
    return model, tok


def _args(tmp_path, **kw):
    argv = ["--output-dir", str(tmp_path / "out"), "--allow-cpu", "--max-steps", "2",
            "--batch-starts", "2", "--group-size", "2", "--horizon", "6", "--prefix-actions", "2",
            "--num-scenarios", "3", "--max-new-tokens", "6", "--max-prompt-tokens", "48",
            "--gen-batch-size", "3", "--micro-batch-size", "3", "--save-steps", "1",
            "--save-total-limit", "1", "--lr", "1e-2", "--lora-r", "4", "--lora-alpha", "8"]
    for k, v in kw.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return T.parse_args(argv)


@pytest.mark.parametrize("estimator", ["grpo", "gigpo", "gagpo"])
def test_end_to_end_two_steps_tiny_model(tmp_path, estimator):
    pytest.importorskip("peft")
    base, tok = tiny_model_and_tokenizer()
    # init adapter: a LoRA saved to disk, so the reference is the frozen init adapter
    from peft import get_peft_model
    init = get_peft_model(base, T.lora_config(4, 8))
    with torch.no_grad():
        for n, p in init.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.02)      # non-zero init adapter, distinct from the base model
    init.save_pretrained(str(tmp_path / "init"))
    base = init.unload()

    model, ref_mode = T.prepare_policy(base, init_adapter=str(tmp_path / "init"), ref_adapter=None,
                                       lora_r=4, lora_alpha=8, grad_ckpt=True)
    assert ref_mode == "adapter"
    ref_before = {n: p.detach().clone() for n, p in model.named_parameters() if ".ref." in n}
    assert ref_before and all(not p.requires_grad for n, p in model.named_parameters() if ".ref." in n)

    seen_training = []
    orig_generate = model.generate

    def spy_generate(*a, **k):
        seen_training.append(model.training)
        return orig_generate(*a, **k)
    model.generate = spy_generate

    args = _args(tmp_path, adv_estimator=estimator, beta=0.1)
    rows = T.train(args, model, tok, ref_mode)

    assert len(rows) == 2
    assert seen_training and not any(seen_training), "generation must run in eval mode"
    assert model.training, "update leaves the model in train mode"
    lines = (tmp_path / "out" / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 2
    row = json.loads(lines[-1])
    for key in ("return_mean", "return_std", "zero_var_group_frac", "kl", "loss", "a0_life_mean",
                "parse_rate", "tokens_per_s", "wall_time_s", "n_decisions"):
        assert key in row, key
    assert row["n_rollouts"] == 4 and row["n_decisions"] >= 4
    assert all(math.isfinite(r["loss"]) for r in rows)
    assert rows[0]["kl"] == pytest.approx(0.0, abs=1e-6)   # policy == reference at step 1
    # save-total-limit 1: only the last checkpoint is kept, and it holds only 'default'
    ckpts = sorted(p.name for p in (tmp_path / "out").iterdir() if p.name.startswith("checkpoint-"))
    assert ckpts == ["checkpoint-2"]
    saved = list((tmp_path / "out" / "checkpoint-2").glob("adapter_model.*"))
    assert saved and not (tmp_path / "out" / "checkpoint-2" / "ref").exists()
    # the reference adapter stayed frozen
    for n, p in model.named_parameters():
        if n in ref_before:
            assert torch.equal(p.detach(), ref_before[n]), n
    assert (tmp_path / "out" / "run_config.json").exists()


def test_policy_update_ppo_epochs_and_reinforce_fresh_lora(tmp_path):
    pytest.importorskip("peft")
    base, tok = tiny_model_and_tokenizer()
    model, ref_mode = T.prepare_policy(base, init_adapter=None, ref_adapter=None,
                                       lora_r=4, lora_alpha=8, grad_ckpt=False)
    assert ref_mode == "base"
    rng = random.Random(0)
    samples = [{"prompt_ids": [rng.randrange(256) for _ in range(rng.randint(5, 12))],
                "completion_ids": [rng.randrange(256) for _ in range(rng.randint(1, 5))] + [257],
                "advantage": rng.uniform(-1, 1)} for _ in range(5)]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    for loss in ("ppo", "reinforce"):
        st = T.policy_update(model, opt, scaler, samples, ref_mode=ref_mode, beta=0.1, loss_type=loss,
                             ppo_epochs=2, micro_batch=2, pad_id=256)
        assert math.isfinite(st["loss"]) and st["grad_norm"] > 0
    changed = any(not torch.equal(p.detach(), before[n]) for n, p in model.named_parameters() if n in before)
    assert changed


def test_completion_logps_match_padding_free_forward():
    base, tok = tiny_model_and_tokenizer()
    base.eval()
    s1 = {"prompt_ids": [5, 6, 7, 8, 9], "completion_ids": [10, 11, 257]}
    s2 = {"prompt_ids": [20, 21], "completion_ids": [22, 257]}
    with torch.no_grad():
        lp, mask = T.completion_logps(base, [s1, s2], pad_id=256, logits_key=T.logits_kwarg(base))
        for j, s in enumerate((s1, s2)):
            ids = torch.tensor([s["prompt_ids"] + s["completion_ids"]])
            logits = base(input_ids=ids).logits[0].float()
            P, C = len(s["prompt_ids"]), len(s["completion_ids"])
            want = torch.log_softmax(logits[P - 1:P + C - 1], -1).gather(
                -1, torch.tensor(s["completion_ids"])[:, None]).squeeze(-1)
            assert torch.allclose(lp[j, :C], want, atol=1e-4)
            assert mask[j].sum().item() == C
