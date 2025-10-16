# Changes: Resume pre-check, Resume policy, GPU Temperature Monitor

## Summary

This document describes all changes added to the repository to handle three related areas:

1. Prevent loading the model when resuming from a backup if the backup already completed as many epochs as configured.
2. A conservative "resume policy" that detects dataset/config differences and resets in-memory training progress and optimizer/EMA state (with optional physical deletion of optimizer/ema folders).
3. A GPU temperature monitor that pauses/resumes training when GPU temperatures cross configured thresholds; UI controls were added to expose these settings.

These changes are designed to be safe (wrapped with exception suppression where appropriate), informative (clear UI message and focus for users), and minimally invasive.

---

## Commit title (short)

Prevent loading model when resume epochs already completed; add resume-policy and GPU temp-control UI

---

## Detailed description (what was changed and why)

### 1) Resume pre-check (prevent pointless model load)

Behavioral change:
- When the user configured `continue_last_backup = true`, the trainer previously loaded the model from the backup and only later discovered that the backup's saved epoch was already >= the configured `epochs`. This resulted in wasted time and confusing behavior (appeared like "trainer refuses to continue").

New behavior:
- Before loading a model from the found backup, the trainer reads `<backup>/meta.json` and examines `train_progress.epoch`. If `config.epochs <= saved_epoch`, the trainer stops startup early (does NOT load the model) and communicates a clear message telling the user to increase `epochs` or disable `continue_last_backup`.

Why:
- Avoids unnecessary model load and provides a clear, early explanation to the user.

Files changed:
- `modules/trainer/GenericTrainer.py`
  - Added a pre-load check that reads `meta.json` (if found) and early-returns with a clear message when the configured total epochs are not greater than the saved epoch in the backup.

### 2) Resume policy for dataset/config differences

Feature:
- A `resume_policy` utility was added to detect differences between the current run configuration (concepts, samples) and the contents stored in a backup. When differences are detected, the default conservative policy performs the following actions:
  - Reset `model.train_progress` in memory.
  - Clear `model.optimizer_state_dict` and `model.ema_state_dict` in memory.
  - Optionally delete physical `optimizer/` and `ema/` directories under the backup (default: delete_physical=True).
  - Log all decisions and actions to `workspace/run/logs/resume_policy.log`.

Why:
- If the dataset or sampling config changed, continuing from previous optimizer/EMA state is often incorrect and can lead to poor or broken training behavior. The policy makes the resume operation safer by removing stale optimizer/EMA state and resetting counters to force a fresh start.

Files added/changed:
- `modules/util/resume_policy.py` (new)
  - Implements dataset comparisons (file hashes), a detection function, and `apply_reset_policy()` that resets in-memory state and optionally deletes physical optimizer/ema artifacts and writes log entries.
- `modules/trainer/GenericTrainer.py` (modified)
  - Calls `resume_policy.dataset_changed()` and `resume_policy.apply_reset_policy()` after loading model (guarded with try/except so it cannot fail startup).

Notes:
- The policy is conservative and non-blocking: failures inside the policy are caught and logged but do not crash the trainer startup.
- Current default deletes optimizer/ema directories (destructive); a dry-run or confirmation UI could be added on request.

### 3) GPU temperature monitor and UI

Feature:
- A GPU temperature monitor was added which, when enabled, reads GPU temps and pauses training if they exceed a user-configurable `max_temp`, resuming only when temps fall to `cool_to`.

Why:
- Protects GPUs from overheating and provides a mechanism to automatically pause training if thermal conditions become unsafe or throttled.

Files added/changed:
- `modules/util/gpu_temp_monitor.py` (new)
  - Attempts to read temps using `pynvml`, falling back to parsing `nvidia-smi` output if necessary. Exposes `pause_if_overtemp_if_needed(config, callbacks)` that can be called from the training loop and will block (sleep, with logging) until temperatures fall below `cool_to`.
- `modules/util/config/TrainConfig.py` (modified)
  - Added default values for UI/persistence: `gpu_temp_control_enabled`, `gpu_temp_max` (default 70°C), and `gpu_temp_cool_to` (default 60°C).
- `modules/ui/TrainingTab.py` (modified)
  - Added UI controls (switch + 2 entries) to configure the GPU temp monitor.
- `modules/trainer/GenericTrainer.py` (modified)
  - Calls `gpu_temp_monitor.pause_if_overtemp_if_needed(self.config, self.callbacks)` after `train_progress.next_step()` in the training loop (guarded with try/except so it cannot break training).

---

## Files added
- `modules/util/resume_policy.py`
- `modules/util/gpu_temp_monitor.py`
- `docs/RESUME_GPU_changes.md` (this file)

## Files modified
- `modules/trainer/GenericTrainer.py`
- `modules/ui/TrainUI.py` (safety: skip train() when start() returned early; show modal and focus Epochs)
- `modules/ui/TrainingTab.py` (save epochs entry reference + focus method + GPU UI controls)
- `modules/util/config/TrainConfig.py` (GPU defaults)

---

