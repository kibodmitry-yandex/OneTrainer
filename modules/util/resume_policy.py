import hashlib
import json
import os
import shutil
import time
from typing import Dict

from modules.util.TrainProgress import TrainProgress


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_changed(cfg, backup_path: str) -> Dict[str, bool]:
    """Сравнить текущие файлы концептов/samples с копиями из бэкапа (если они есть).
    Возвращает словарь {'concepts': bool, 'samples': bool}.
    """
    changes = {"concepts": False, "samples": False}

    try:
        # concepts: only treat as changed if both current and backup files exist and differ.
        # If the backup does not contain the concepts file, be conservative and treat it as unchanged
        # to avoid accidental optimizer/EMA resets when backup meta is incomplete.
        cur_concepts = os.path.abspath(cfg.concept_file_name)
        bak_concepts = os.path.join(backup_path, "onetrainer_config", "concepts.json")
        if os.path.isfile(bak_concepts) and os.path.isfile(cur_concepts):
            changes["concepts"] = sha256_of_file(cur_concepts) != sha256_of_file(bak_concepts)
        else:
            # do not assume a change when backup metadata is missing
            changes["concepts"] = False

        # samples: same conservative policy as for concepts
        cur_samples = os.path.abspath(cfg.sample_definition_file_name)
        bak_samples = os.path.join(backup_path, "onetrainer_config", "samples.json")
        if os.path.isfile(bak_samples) and os.path.isfile(cur_samples):
            changes["samples"] = sha256_of_file(cur_samples) != sha256_of_file(bak_samples)
        else:
            changes["samples"] = False

    except Exception:
        # on unexpected errors be conservative: do not trigger reset of optimizer/EMA
        try:
            changes["concepts"] = False
            changes["samples"] = False
        except Exception:
            pass

    return changes


def apply_reset_policy(model, backup_path: str, cfg, callbacks, workspace_dir: str, delete_physical: bool = True):
    actions = []

    # save log
    logdir = os.path.join(workspace_dir, "logs")
    os.makedirs(logdir, exist_ok=True)
    logpath = os.path.join(logdir, "resume_policy.log")

    # reset progress in memory
    try:
        old = (model.train_progress.epoch, model.train_progress.epoch_step, model.train_progress.global_step)
        model.train_progress = TrainProgress()
        actions.append(f"reset_progress from {old} to (0,0,0)")
    except Exception as e:
        actions.append(f"failed_reset_progress: {e}")

    # reset optimizer and ema in memory
    try:
        model.optimizer_state_dict = None
        actions.append("cleared optimizer_state_dict in memory")
    except Exception as e:
        actions.append(f"failed_clear_optimizer_state: {e}")

    try:
        model.ema_state_dict = None
        actions.append("cleared ema_state_dict in memory")
    except Exception as e:
        actions.append(f"failed_clear_ema_state: {e}")

    # optionally delete physical folders in backup
    if delete_physical and backup_path and os.path.isdir(backup_path):
        try:
            optp = os.path.join(backup_path, "optimizer")
            if os.path.isdir(optp):
                shutil.rmtree(optp)
                actions.append(f"deleted {optp}")
        except Exception as e:
            actions.append(f"failed_delete_optimizer_folder: {e}")

        try:
            emap = os.path.join(backup_path, "ema")
            if os.path.isdir(emap):
                shutil.rmtree(emap)
                actions.append(f"deleted {emap}")
        except Exception as e:
            actions.append(f"failed_delete_ema_folder: {e}")

    # write log
    try:
        with open(logpath, "a", encoding="utf-8") as f:
            f.write(f"{time.asctime()}: applied resume reset policy: {'; '.join(actions)}\n")
    except Exception:
        pass

    # callbacks status
    for a in actions:
        try:
            callbacks.on_update_status(a)
        except Exception:
            pass

    return actions
