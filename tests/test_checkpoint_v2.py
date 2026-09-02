import random

import numpy as np
import torch

from echo_research.engine.checkpoint import load_checkpoint, save_checkpoint
from echo_research.models import Pix2PixResearchModel


def test_v2_checkpoint_restores_training_metadata(tmp_path):
    model = Pix2PixResearchModel("binary", base_channels=4)
    gen_opt = torch.optim.Adam(model.generator.parameters(), lr=1e-4)
    disc_opt = torch.optim.Adam(model.discriminator.parameters(), lr=2.5e-5)
    gen_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(gen_opt)
    disc_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(disc_opt)
    path = tmp_path / "last.pt"
    config = {"data": {"task": "binary"}, "model": {"base_channels": 4, "num_classes": 4}}
    history = [{"epoch": 1, "val_dice": 0.8}]

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    # Populate both Adam optimizers so restore covers tensor-valued optimizer state.
    gen_opt.zero_grad()
    next(model.generator.parameters()).sum().backward()
    gen_opt.step()
    disc_opt.zero_grad()
    next(model.discriminator.parameters()).sum().backward()
    disc_opt.step()
    gen_sched.step(0.8)
    disc_sched.step(0.8)
    save_checkpoint(
        path,
        model,
        gen_opt,
        disc_opt,
        epoch=1,
        best_metric=0.8,
        config=config,
        gen_sched=gen_sched,
        disc_sched=disc_sched,
        history=history,
        global_step=12,
        no_improve=3,
    )
    assert path.exists()
    assert not path.with_suffix(".pt.tmp").exists()

    expected_random = (random.random(), float(np.random.random()), float(torch.rand(())))
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)

    restored = Pix2PixResearchModel("binary", base_channels=4)
    restored_gen_opt = torch.optim.Adam(restored.generator.parameters(), lr=9e-4)
    restored_disc_opt = torch.optim.Adam(restored.discriminator.parameters(), lr=9e-4)
    restored_gen_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(restored_gen_opt)
    restored_disc_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(restored_disc_opt)
    checkpoint = load_checkpoint(
        path,
        restored,
        restored_gen_opt,
        restored_disc_opt,
        map_location="cpu",
        gen_sched=restored_gen_sched,
        disc_sched=restored_disc_sched,
        restore_rng=True,
    )
    assert checkpoint["checkpoint_version"] == 2
    assert checkpoint["epoch"] == 1
    assert checkpoint["global_step"] == 12
    assert checkpoint["no_improve"] == 3
    assert checkpoint["history"] == history
    assert checkpoint["gen_scheduler"] is not None
    assert checkpoint["disc_scheduler"] is not None
    assert restored_gen_opt.state
    assert restored_disc_opt.state
    assert restored_gen_opt.param_groups[0]["lr"] == gen_opt.param_groups[0]["lr"]
    assert restored_disc_opt.param_groups[0]["lr"] == disc_opt.param_groups[0]["lr"]
    actual_random = (random.random(), float(np.random.random()), float(torch.rand(())))
    assert actual_random == expected_random
