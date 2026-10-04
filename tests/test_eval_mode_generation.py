"""run_in_eval_mode: generation runs in eval mode (KV cache on), training mode is restored."""

import pytest

torch = pytest.importorskip("torch")

from training.gigpo_trainer import run_in_eval_mode


def test_eval_mode_during_fn_and_restored_after():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.5))
    model.train()
    seen = run_in_eval_mode(model, lambda: model.training)
    assert seen is False
    assert model.training is True


def test_train_mode_restored_on_exception():
    model = torch.nn.Linear(2, 2)
    model.train()

    def boom():
        raise RuntimeError("x")

    with pytest.raises(RuntimeError):
        run_in_eval_mode(model, boom)
    assert model.training is True


def test_eval_model_stays_eval():
    model = torch.nn.Linear(2, 2)
    model.eval()
    run_in_eval_mode(model, lambda: None)
    assert model.training is False
