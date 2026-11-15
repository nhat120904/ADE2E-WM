"""The unmodified NAVSIM v2 devkit run_training.py, with the patches below (training mechanics only; model, loss, labels
and recipe unchanged). Used by scripts/hgo/cg_drivor_train.sbatch; same Hydra arguments as run_training.py.

1. DrivoRAgent.compute_loss returns a dict of losses; the devkit's AgentLightningModule logs a scalar. Each entry is
   logged and dict["loss"] is optimized, as in DrivoR's own fork of the devkit
   (third_party/DrivoR/navsim/planning/training/agent_lightning_module.py). Validation is the devkit's (losses on the val
   scenes, monitoring only).
2. DrivoR's 'best' checkpoint monitors val/score_epoch, which only the fork's validation step logs: dropped; the last
   checkpoint is saved (and the one used).
3. DRIVOR_RAY_THREADS (default 8, DrivoR's value): threads of the Ray pool that runs the online PDM scoring of the
   proposals (nuPlan's worker_map splits the batch into that many chunks).
4. Timing: every 50 training steps, mean seconds per step spent in PDM scoring (compute_score) and waiting for the
   data loader.
5. Periodic checkpoint every DRIVOR_CKPT_EVERY training steps (periodic-*.ckpt, latest only), besides DrivoR's epoch-end
   last.ckpt (the checkpoint used for evaluation).
6. Resume, when DRIVOR_RUN_DIR is set: the run's identity (all Hydra arguments + sha256 of the training scene filter
   and of the log lists) is written to DRIVOR_RUN_DIR/run_meta.json on the first start; a restart with a different
   identity stops. With the same identity, Lightning resumes from the newest checkpoint under DRIVOR_RUN_DIR (weights,
   optimizer, scheduler, step counter). The identity files (DRIVOR_ID_FILES) include the agent / training configs and
   the agent + label source, so edited defaults or code also refuse a resume. A mid-epoch resume restarts the data
   order of that epoch (Lightning 2.2 does not save the sampler state; the exact sample order is not restored) and, on
   its own, overshoots the step budget (CPU test of the follow-up review: 17 instead of 16 steps): the job therefore sets
   trainer max_steps = EPOCHS * ceil(N / batch).
7. DRIVOR_EXPECT_N: the training set the loader built (after every scene filter) must have exactly this many samples;
   DRIVOR_EXPECT_STEPS: the run must end at exactly this optimizer step.
"""

import glob
import hashlib
import json
import os
import runpy
import sys
import time

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from navsim.planning.training.agent_lightning_module import AgentLightningModule
import navsim.planning.utils.multithreading.worker_ray_no_torch as wr
from ade2e_wm.agents.drivor.drivor_agent import DrivoRAgent

T = {"score": 0.0, "data": 0.0, "step": 0, "t_end": None}


def _step(self, batch, logging_prefix):
    if logging_prefix == "train" and T["t_end"] is not None:
        T["data"] += time.time() - T["t_end"]
    features, targets = batch
    prediction = self.agent.forward(features)
    loss = self.agent.compute_loss(features, targets, prediction)
    if not isinstance(loss, dict):
        self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    for k, v in loss.items():
        self.log(f"{logging_prefix}/{k}", float(v) if isinstance(v, (int, float)) else v.detach().float(),
                 on_step=logging_prefix == "train", on_epoch=True, sync_dist=True)
    return loss["loss"]


def on_train_batch_end(self, outputs, batch, batch_idx):
    T["step"] += 1
    if T["step"] % 50 == 0:
        print(f"[timing] steps {T['step']}: per step pdm_score {T['score'] / 50:.3f} s, wait_data {T['data'] / 50:.3f} s",
              flush=True)
        T["score"] = T["data"] = 0.0
    T["t_end"] = time.time()


def on_train_epoch_start(self):
    T["t_end"] = None


AgentLightningModule._step = _step
AgentLightningModule.on_train_batch_end = on_train_batch_end
AgentLightningModule.on_train_epoch_start = on_train_epoch_start

_compute_score = DrivoRAgent.compute_score


def compute_score(self, *a, **k):
    t = time.time()
    r = _compute_score(self, *a, **k)
    if self.training:
        T["score"] += time.time() - t
    return r


DrivoRAgent.compute_score = compute_score

_callbacks = DrivoRAgent.get_training_callbacks


def get_training_callbacks(self):
    cbs = [cb for cb in _callbacks(self) if getattr(cb, "monitor", None) != "val/score_epoch"]
    every = int(os.environ.get("DRIVOR_CKPT_EVERY", "0"))
    if every > 0:
        cbs.append(ModelCheckpoint(filename="periodic-{epoch}-{step}", every_n_train_steps=every, save_top_k=1))
    return cbs


DrivoRAgent.get_training_callbacks = get_training_callbacks


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def resume_checkpoint():
    """None, or the newest checkpoint of this run under DRIVOR_RUN_DIR; stops on an identity mismatch."""
    run_dir = os.environ.get("DRIVOR_RUN_DIR")
    if not run_dir:
        return None
    meta = {"argv": sys.argv[1:],
            "files": {p: sha256(p) for p in os.environ.get("DRIVOR_ID_FILES", "").split(",") if p}}
    path = os.path.join(run_dir, "run_meta.json")
    os.makedirs(run_dir, exist_ok=True)
    if os.path.exists(path):
        old = json.load(open(path))
        if old != meta:
            raise SystemExit(f"resume refused: {path} differs from this run's arguments / manifests\n"
                             f"old: {json.dumps(old)}\nnew: {json.dumps(meta)}")
    else:
        json.dump(meta, open(path, "w"), indent=1)
    ckpts = glob.glob(os.path.join(run_dir, "**", "checkpoints", "*.ckpt"), recursive=True)
    ckpt = max(ckpts, key=os.path.getmtime) if ckpts else None
    print(f"[resume] {ckpt or 'none: fresh start'}", flush=True)
    return ckpt


_fit = pl.Trainer.fit


def fit(self, model, train_dataloaders=None, val_dataloaders=None, datamodule=None, ckpt_path=None):
    n = len(train_dataloaders.dataset)
    if "DRIVOR_EXPECT_N" in os.environ and n != int(os.environ["DRIVOR_EXPECT_N"]):
        raise SystemExit(f"training set has {n} samples, expected {os.environ['DRIVOR_EXPECT_N']}")
    print(f"[data] training samples {n}, steps per epoch {len(train_dataloaders)}", flush=True)
    out = _fit(self, model, train_dataloaders, val_dataloaders, datamodule, ckpt_path or resume_checkpoint())
    want = os.environ.get("DRIVOR_EXPECT_STEPS")
    print(f"[done] global_step {self.global_step} (expected {want})", flush=True)
    if want and self.global_step != int(want):
        raise SystemExit(f"run ended at optimizer step {self.global_step}, expected {want}")
    return out


pl.Trainer.fit = fit

_ray_init = wr.RayDistributedNoTorch.__init__


def ray_init(self, *a, **k):
    if "DRIVOR_RAY_THREADS" in os.environ:
        k["threads_per_node"] = int(os.environ["DRIVOR_RAY_THREADS"])
    _ray_init(self, *a, **k)


wr.RayDistributedNoTorch.__init__ = ray_init

if __name__ == "__main__":
    runpy.run_path(os.path.join(os.environ["NAVSIM_DEVKIT_ROOT"], "navsim/planning/script/run_training.py"),
                   run_name="__main__")
