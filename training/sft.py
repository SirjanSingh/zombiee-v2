"""LoRA SFT warm-start (W7): teach Qwen the camp planner before RL.

Loss is on the completion tokens only (the JSON action array + EOS); prompt
tokens are masked. Prompts are tokenized as raw text with no chat template,
exactly like TRL 0.15 treats our string prompts during GRPO. The LoRA config
matches training/train.py (r, alpha, q/k/v/o, dropout 0), so the saved adapter
loads with `python -m training.train --warmstart-from <out-dir>`.

  CUDA_VISIBLE_DEVICES=5 python -m training.sft --data data/sft_camp_v3.jsonl \
      --output-dir checkpoints/sft_camp_v3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger("survivecity_v2.sft")


def load_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def tokenize_example(tok, prompt: str, completion: str, max_len: int) -> dict:
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    c_ids = tok(completion, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    ids = (p_ids + c_ids)[-max_len:]
    n_prompt = max(0, len(ids) - len(c_ids))
    labels = [-100] * n_prompt + ids[n_prompt:]
    return {"input_ids": ids, "labels": labels}


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        import torch
        n = max(len(b["input_ids"]) for b in batch)
        ids = [b["input_ids"] + [self.pad_id] * (n - len(b["input_ids"])) for b in batch]
        lab = [b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]
        att = [[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab),
                "attention_mask": torch.tensor(att)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--output-dir", default="checkpoints/sft_camp_v3")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--per-device-batch-size", type=int, default=4)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--max-len", type=int, default=1792)
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--holdout", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=-1, help="for smoke tests")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    cuda = torch.cuda.is_available()
    logger.info(f"Device: {'cuda ' + torch.cuda.get_device_name(0) if cuda else 'cpu'}")
    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    rows = load_rows(args.data)
    random.Random(args.seed).shuffle(rows)
    hold, train_rows = rows[:args.holdout], rows[args.holdout:]
    train = [tokenize_example(tok, r["prompt"], r["completion"], args.max_len) for r in train_rows]
    evals = [tokenize_example(tok, r["prompt"], r["completion"], args.max_len) for r in hold]
    lens = sorted(len(x["input_ids"]) for x in train)
    logger.info(f"train {len(train)} / holdout {len(evals)}; tokens median {lens[len(lens) // 2]} max {lens[-1]}")

    dtype = torch.float16 if cuda else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.output_dir, num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps, learning_rate=args.lr,
        lr_scheduler_type="cosine", warmup_ratio=0.03, logging_steps=10,
        eval_strategy="steps" if evals else "no", eval_steps=50,
        save_strategy="no", fp16=cuda, report_to=["tensorboard"],
        remove_unused_columns=False, seed=args.seed, dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train,
                      eval_dataset=evals or None, data_collator=Collator(tok.pad_token_id))
    trainer.train()
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    logger.info(f"Saved LoRA adapter to {args.output_dir}")

    # Quick held-out check: exact-plan match and parse rate, greedy decoding.
    from training.inference import parse_actions
    model.eval()
    tok.padding_side = "left"
    n = min(40, len(hold))
    exact = parsed = 0
    for i in range(0, n, 8):
        chunk = hold[i:i + 8]
        enc = tok([r["prompt"] for r in chunk], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=160, do_sample=False, pad_token_id=tok.pad_token_id)
        for r, text in zip(chunk, tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)):
            got = parse_actions(text, agent_id=0, max_actions=5)
            want = parse_actions(r["completion"], agent_id=0, max_actions=5)
            parsed += len(got) == 5
            exact += [a["action_type"] for a in got] == [a["action_type"] for a in want]
    logger.info(f"holdout ({n}): parse {parsed / n:.0%}, exact action-type plan match {exact / n:.0%}")


if __name__ == "__main__":
    main()
