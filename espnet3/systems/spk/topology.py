"""Fail closed and prove the explicitly configured multi-node training world."""

import json
from datetime import datetime, timezone
from pathlib import Path

from lightning.pytorch.callbacks import Callback


class TrainingTopologyProof(Callback):
    """Record advancing ranks without introducing additional collectives."""

    def on_train_start(self, trainer, pl_module):
        """Reject a partial world or a changed effective batch."""
        assert trainer.world_size == 16
        assert trainer.num_nodes == 2 and trainer.num_devices == 8
        assert trainer.accumulate_grad_batches == 1
        assert trainer.train_dataloader.batch_size == 140
        sampler = trainer.train_dataloader.sampler
        assert sampler.num_replicas == 16 and sampler.shuffle

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Keep one tiny rank receipt per node-local process after updates."""
        step = trainer.global_step
        if step not in (1, 2, 30) and step % 100:
            return
        record = {
            "rank": trainer.global_rank,
            "local_rank": trainer.local_rank,
            "world": trainer.world_size,
            "step": step,
            "epoch": trainer.current_epoch,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        directory = Path(trainer.default_root_dir) / "rank_proofs"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"rank{trainer.global_rank}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record) + "\n")
        temporary.replace(path)
        if step in (1, 2, 30) or step % 2000 == 0:
            print("TRAIN_RANK_PROOF " + json.dumps(record), flush=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Expose completed smoke validation across the full training world."""
        if trainer.global_step == 30:
            assert "valid/eer" in trainer.callback_metrics
            print(f"TRAIN_VALIDATION_PROOF rank={trainer.global_rank} step=30", flush=True)
