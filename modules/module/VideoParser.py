import os
import json
import sys
from pathlib import Path
import tkinter as tk
from PIL import Image
from typing import Any
import logging
import customtkinter as ctk

# Optional heavy deps: import at module load time if available so static
# analysis sees the names. If not installed, set to None and guard uses.
try:
    import cv2
except Exception:
    cv2 = None


# Safe image reader: uses numpy.fromfile + cv2.imdecode to support Windows
# unicode / special-character paths and to avoid noisy OpenCV warnings when
# files are being written concurrently. Retries a few times before falling
# back to cv2.imread.
def _safe_imread(path, flags=None, _retries=3, _backoff=0.05):
    try:
        import numpy as _np
        from pathlib import Path as _Path

        p = _Path(path)
        # quick existence check
        if not p.exists():
            logger.debug(f"_safe_imread: not exists: {p}")
            return None
        # Try a few times in case file is being written
        for attempt in range(_retries):
            try:
                size = p.stat().st_size
            except Exception:
                size = 0
            if size == 0:
                logger.debug(
                    f"_safe_imread: zero size for {p} (attempt {attempt+1}/{_retries})"
                )
                try:
                    import time

                    time.sleep(_backoff)
                except Exception:
                    pass
                continue
            try:
                data = _np.fromfile(str(p), dtype=_np.uint8)
            except Exception as e:
                logger.debug(f"_safe_imread: numpy.fromfile failed for {p}: {e}")
                try:
                    import time

                    time.sleep(_backoff)
                except Exception:
                    pass
                continue
            if data.size == 0:
                logger.debug(
                    f"_safe_imread: read zero bytes for {p} (attempt {attempt+1}/{_retries})"
                )
                try:
                    import time

                    time.sleep(_backoff)
                except Exception:
                    pass
                continue
            try:
                if cv2 is not None:
                    im_flag = flags if flags is not None else (cv2.IMREAD_COLOR)
                    img = cv2.imdecode(data, im_flag)
                    if img is not None:
                        return img
                else:
                    # cannot decode without cv2; break to fallback path
                    pass
            except Exception as e:
                logger.debug(f"_safe_imread: imdecode failed for {p}: {e}")
            try:
                import time

                time.sleep(_backoff)
            except Exception:
                pass
        # final fallback to cv2.imread (may emit warnings)
        if cv2 is None:
            logger.error(f"_safe_imread: cv2 not available to read {path}")
            return None
        try:
            return (
                cv2.imread(str(path)) if flags is None else cv2.imread(str(path), flags)
            )
        except Exception as e:
            logger.error(f"_safe_imread: cv2.imread fallback failed for {path}: {e}")
            return None
    except Exception as e:
        try:
            logger.error(f"_safe_imread: unexpected error for {path}: {e}")
        except Exception:
            pass
        return None


try:
    import imageio_ffmpeg as imageio_ffmpeg
except Exception:
    imageio_ffmpeg = None

# Determine portable resampling constants to avoid static-analysis issues
try:
    Resampling = getattr(Image, "Resampling", None)
    if Resampling is not None:
        LANCZOS_CONST = Resampling.LANCZOS
        BICUBIC_CONST = Resampling.BICUBIC
    else:
        LANCZOS_CONST = getattr(Image, "LANCZOS", None)
        BICUBIC_CONST = getattr(Image, "BICUBIC", None)
except Exception:
    LANCZOS_CONST = None
    BICUBIC_CONST = None

SETTINGS_PATH = (
    Path(__file__).resolve().parents[2] / "workspace" / "masking_tool_settings.json"
)

# For debugging: when True, do NOT use system PATH ffmpeg; force checking package/vendored installs
SKIP_SYSTEM_FFMPEG_CHECK = False

# Shared logger
logger = logging.getLogger("video_parser")


class VideoParserWindow(ctk.CTkToplevel):
    # ------------------------------
    # Sidecar and crop helpers
    # ------------------------------
    def _read_parsing_sidecar(self, out_dir: Path) -> dict:
        sidecar = Path(out_dir) / "parsing.json"
        if not sidecar.exists():
            return {"files": []}
        try:
            with open(sidecar, "r", encoding="utf-8") as fp:
                obj = json.load(fp)
            if not isinstance(obj, dict):
                return {"files": []}
            if "files" not in obj or not isinstance(obj["files"], list):
                obj["files"] = []
            return obj
        except Exception as e:
            try:
                logger.error(f"[parsing_sidecar] read error: {e}")
            except Exception:
                pass
            return {"files": []}

    def _get_cached_sidecar(self, out_dir: Path, reload: bool = False) -> dict:
        """Return cached parsing.json for out_dir. If reload=True or cache miss, read from disk.

        Default behaviour: avoid re-reading parsing.json multiple times during a session.
        Writes should update both disk and this cache.
        """
        try:
            key = str(Path(out_dir))
        except Exception:
            key = None
        if key is None:
            return {"files": []}
        if (not reload) and key in getattr(self, "_parsing_sidecar_cache", {}):
            return self._parsing_sidecar_cache.get(key, {"files": []})
        # read from disk and populate cache
        obj = self._read_parsing_sidecar(out_dir)
        try:
            self._parsing_sidecar_cache[key] = obj
        except Exception:
            pass
        return obj

    def _atomic_write_json(self, path: Path, obj: dict):
        try:
            import tempfile

            fd, tmp_path = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fp:
                    json.dump(obj, fp, indent=2, ensure_ascii=False)
                    fp.flush()
                    os.fsync(fp.fileno())
                try:
                    os.replace(tmp_path, str(path))
                except PermissionError as perr:
                    # On Windows this can happen if the target is read-only or locked by another process.
                    # Try to make target writable and retry; if that fails, fall back to non-atomic write below.
                    try:
                        import stat

                        if os.path.exists(path):
                            try:
                                os.chmod(str(path), stat.S_IWRITE | stat.S_IREAD)
                            except Exception:
                                pass
                            try:
                                os.remove(str(path))
                            except Exception:
                                pass
                        os.replace(tmp_path, str(path))
                    except Exception:
                        # If still failing, don't treat as fatal: log at debug and let fallback write handle it.
                        try:
                            logger.debug(
                                f"[parsing_sidecar] atomic write failed (replace): {perr}"
                            )
                        except Exception:
                            pass
            finally:
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
        except Exception as e:
            try:
                # Lower severity: many failures here are transient (file locks / antivirus). Keep debug log.
                logger.debug(f"[parsing_sidecar] atomic write failed: {e}")
            except Exception:
                pass
            # Fallback: best-effort direct write (non-atomic)
            try:
                with open(path, "w", encoding="utf-8") as fp:
                    json.dump(obj, fp, indent=2, ensure_ascii=False)
            except Exception:
                # If even this fails, there's not much we can do here; swallow to keep UI responsive.
                try:
                    logger.debug(f"[parsing_sidecar] fallback write also failed: {e}")
                except Exception:
                    pass

    def _merge_and_write_sidecar(
        self, out_dir: Path, new_files_meta: list, selected_name: str | None
    ):
        # Use cached sidecar to avoid re-reading file frequently
        existing = self._get_cached_sidecar(out_dir, reload=False)
        old_by_name = {}
        for it in existing.get("files", []):
            if isinstance(it, dict) and "relpath" in it:
                old_by_name[it["relpath"]] = it
        merged = []
        for item in new_files_meta:
            name = item.get("relpath")
            prev = old_by_name.get(name, {})
            merged_item = {}
            merged_item.update(prev)
            merged_item.update(item)
            # keep existing crop if present
            if "crop" in prev and "crop" not in merged_item:
                merged_item["crop"] = prev["crop"]
            # selection
            try:
                merged_item["selected"] = name == selected_name
            except Exception:
                pass
            merged.append(merged_item)
        sidecar = Path(out_dir) / "parsing.json"
        written = {"files": merged}
        self._atomic_write_json(sidecar, written)
        # update cache to match just-written content
        try:
            key = str(Path(out_dir))
            self._parsing_sidecar_cache[key] = written
        except Exception:
            pass

    # ------------------------------
    # In-memory dataset/nav state helpers
    # ------------------------------
    def _get_dataset_key(self, out_dir: Path | None = None) -> str | None:
        try:
            od = (
                out_dir
                if out_dir is not None
                else getattr(self, "_current_output_dir", None)
            )
            if not od:
                return None
            return str(Path(od))
        except Exception:
            return None

    def _ensure_sidecar_entries_for_files(self, out_dir: Path):
        """Ensure in-memory sidecar has entries for every file in self._files_order."""
        try:
            key = self._get_dataset_key(out_dir)
            if key is None:
                return
            obj = self._parsing_sidecar_cache.get(key) or {"files": []}
            by_name = {}
            for it in obj.get("files", []) or []:
                if isinstance(it, dict) and "relpath" in it:
                    by_name[it["relpath"]] = it
            changed = False
            for name in getattr(self, "_files_order", []) or []:
                if name not in by_name:
                    (obj.setdefault("files", [])).append({"relpath": name})
                    changed = True
            if changed:
                self._parsing_sidecar_cache[key] = obj
        except Exception:
            pass

    def _set_selected_in_memory(self, out_dir: Path, name: str | None):
        """Update in-memory sidecar's selected flags without re-reading from disk."""
        try:
            key = self._get_dataset_key(out_dir)
            if key is None:
                return
            obj = self._parsing_sidecar_cache.get(key) or {"files": []}
            for it in obj.get("files", []) or []:
                try:
                    if name and it.get("relpath") == name:
                        it["selected"] = True
                    else:
                        if "selected" in it:
                            it.pop("selected", None)
                except Exception:
                    pass
            self._parsing_sidecar_cache[key] = obj
        except Exception:
            pass

    def _write_sidecar_now(self, out_dir: Path):
        """Write current in-memory sidecar to disk atomically (no re-read)."""
        try:
            key = self._get_dataset_key(out_dir)
            if key is None:
                return
            sidecar_path = Path(out_dir) / "parsing.json"
            obj = self._parsing_sidecar_cache.get(key) or {"files": []}
            self._atomic_write_json(sidecar_path, obj)
        except Exception:
            pass

    def _write_sidecar_async(self, out_dir: Path):
        """Schedule sidecar write on a background thread to avoid UI stalls."""
        try:
            import threading

            def _worker():
                try:
                    self._write_sidecar_async(out_dir)
                except Exception:
                    pass

            t = threading.Thread(target=_worker, daemon=True)
            t.start()
        except Exception:
            # fallback to sync write
            try:
                self._write_sidecar_async(out_dir)
            except Exception:
                pass

    def _get_file_crop(self, out_dir: Path, filename: str):
        obj = self._get_cached_sidecar(out_dir, reload=False)
        for it in obj.get("files", []):
            if it.get("relpath") == filename:
                return it.get("crop")
        return None

    def _update_file_crop(self, out_dir: Path, filename: str, crop: dict):
        obj = self._get_cached_sidecar(out_dir, reload=False)
        files = obj.get("files", [])
        found = False
        for it in files:
            if it.get("relpath") == filename:
                it["crop"] = crop
                found = True
                break
        if not found:
            files.append({"relpath": filename, "crop": crop})
        obj["files"] = files
        # write and update cache
        try:
            sidecar_path = Path(out_dir) / "parsing.json"
            self._atomic_write_json(sidecar_path, obj)
            try:
                key = str(Path(out_dir))
                self._parsing_sidecar_cache[key] = obj
            except Exception:
                pass
        except Exception:
            pass
        # refresh navigator colors immediately when crop changes
        try:
            self._refresh_nav_colors()
        except Exception:
            pass

    def _validate_and_fix_crop(self, image_w: int, image_h: int, crop: dict) -> dict:
        # Determine orientation
        if image_w == image_h:
            orientation = "square"
        elif image_w > image_h:
            orientation = "horizontal"
        else:
            orientation = "vertical"
        # Allowed square size by orientation
        size_px_allowed = (
            image_h
            if orientation == "horizontal"
            else (image_w if orientation == "vertical" else image_w)
        )
        # Read current
        try:
            x = int(crop.get("x_px", 0))
        except Exception:
            x = 0
        try:
            y = int(crop.get("y_px", 0))
        except Exception:
            y = 0
        size_px = size_px_allowed
        # Clamp within bounds and axis
        max_x = max(0, image_w - size_px)
        max_y = max(0, image_h - size_px)
        if orientation == "horizontal":
            y = 0
            x = max(0, min(max_x, x))
        elif orientation == "vertical":
            x = 0
            y = max(0, min(max_y, y))
        else:
            x, y = 0, 0
        norm_x = 0.0 if max_x == 0 else x / max_x
        norm_y = 0.0 if max_y == 0 else y / max_y
        return {
            "version": 1,
            "orientation": orientation,
            "image_w": image_w,
            "image_h": image_h,
            "size_px": size_px,
            "x_px": x,
            "y_px": y,
            "x_norm": norm_x,
            "y_norm": norm_y,
        }

    def _update_parsing_sidecar(self):
        """Create or update parsing.json with metadata for PNG files in the dataset."""
        import json
        from pathlib import Path
        import os

        out_dir = getattr(self, "_current_output_dir", None)
        if not out_dir or not Path(out_dir).exists():
            try:
                vp = Path(self.file_path_var.get())
                if vp and vp.exists():
                    candidate = vp.parent / vp.stem
                    if candidate.exists():
                        out_dir = candidate
                    else:
                        return
                else:
                    return
            except Exception:
                return
        files = sorted(Path(out_dir).glob("*.jpg"))
        data = []
        for f in files:
            try:
                stat = f.stat()
                item = {
                    "relpath": f.name,
                    "size_bytes": stat.st_size,
                    "width": None,
                    "height": None,
                    "created": stat.st_ctime,
                    "modified": stat.st_mtime,
                }
                try:
                    import cv2

                    img = _safe_imread(f)
                    if img is not None:
                        item["height"], item["width"] = img.shape[:2]
                except Exception as e:
                    print(f"[parsing_sidecar] Failed to read image {f}: {e}")
                data.append(item)
            except Exception as e:
                print(f"[parsing_sidecar] Failed to collect metadata for {f}: {e}")
        sidecar = Path(out_dir) / "parsing.json"
        try:
            selected_name = None
            try:
                sel = self.file_listbox.curselection()
                if sel and len(sel) > 0:
                    selected_name = self.file_listbox.get(sel[0])
            except Exception:
                selected_name = None
            if selected_name is None and files:
                try:
                    selected_name = files[0].name
                except Exception:
                    selected_name = None
            # Merge with existing to preserve crop
            if files:
                self._merge_and_write_sidecar(out_dir, data, selected_name)
            else:
                self._atomic_write_json(sidecar, {"files": []})
        except Exception as e:
            print(f"[parsing_sidecar] Failed to write sidecar: {e}")

    def _perform_selection_commit(self):
        """Write current selection into parsing.json immediately.

        This is separated so selection writes can be debounced by
        scheduling/cancelling the after job stored in
        self._selection_commit_job.
        """
        try:
            # clear the job id since we're executing it now
            try:
                self._selection_commit_job = None
            except Exception:
                pass
            out_dir = getattr(self, "_current_output_dir", None)
            if not out_dir or not Path(out_dir).exists():
                return
            sel = None
            try:
                sel_idx = self.file_listbox.curselection()
                if sel_idx and len(sel_idx) > 0:
                    sel = self.file_listbox.get(sel_idx[0])
            except Exception:
                sel = None
            # Recreate sidecar preserving crops but setting 'selected'
            files = sorted(Path(out_dir).glob("*.jpg"))
            data = []
            for f in files:
                try:
                    stat = f.stat()
                    item = {
                        "relpath": f.name,
                        "size_bytes": stat.st_size,
                        "width": None,
                        "height": None,
                        "created": stat.st_ctime,
                        "modified": stat.st_mtime,
                    }
                    try:
                        img = _safe_imread(f)
                        if img is not None:
                            item["height"], item["width"] = img.shape[:2]
                    except Exception:
                        pass
                    data.append(item)
                except Exception:
                    pass
            # Merge with existing to preserve crop data
            try:
                self._merge_and_write_sidecar(out_dir, data, sel)
            except Exception:
                pass
        except Exception:
            pass

    def __init__(self, master=None, keep_on_top: bool = False):
        super().__init__(master)
        # Ensure default for every_n is saved to config if missing
        if self._load_setting("video_every_n", None) is None:
            self._save_setting("video_every_n", 10)
        self.title("Video Parser")
        self.geometry(self._load_geometry())
        self.minsize(800, 800)
        try:
            if keep_on_top and master is not None:
                self.transient(master)
                self.lift()
        except Exception:
            pass
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<Configure>", self._on_configure)

        # --- UI Layout ---
        self.grid_rowconfigure(0, weight=0)  # top panel
        self.grid_rowconfigure(1, weight=1)  # bottom split
        self.grid_columnconfigure(0, weight=1)

        # Top panel for file operations
        top_panel = ctk.CTkFrame(
            self, corner_radius=8, border_width=2, border_color="#444"
        )
        top_panel.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        top_panel.grid_columnconfigure(0, weight=1)
        top_label = ctk.CTkLabel(top_panel, text="File", font=("Arial", 16, "bold"))
        top_label.grid(row=0, column=0, sticky="w", padx=8, pady=8)

        # Controls: Open, file path, Every N frames, Deduplication, Run
        controls_frame = ctk.CTkFrame(top_panel, fg_color="transparent")
        controls_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        controls_frame.grid_columnconfigure(0, weight=0)
        controls_frame.grid_columnconfigure(1, weight=1)
        controls_frame.grid_columnconfigure(2, weight=0)
        controls_frame.grid_columnconfigure(3, weight=0)
        controls_frame.grid_columnconfigure(4, weight=0)
        controls_frame.grid_columnconfigure(5, weight=0)

        self.open_btn = ctk.CTkButton(
            controls_frame, text="Open", width=80, command=self._on_open_file
        )
        self.open_btn.grid(row=0, column=0, padx=(0, 8), pady=4)

        self.file_path_var = ctk.StringVar(
            value=self._load_setting("video_last_file", "(no file)")
        )
        self.file_path_label = ctk.CTkLabel(
            controls_frame, textvariable=self.file_path_var, anchor="w", cursor="hand2"
        )
        self.file_path_label.grid(row=0, column=1, sticky="ew", padx=(0, 8))

        # Add click handler to open the folder
        def on_file_path_click(event):
            video_path = self.file_path_var.get()
            if not video_path or video_path == "(no file)":
                return
            from pathlib import Path
            import subprocess

            video_file = Path(video_path)
            # If there's a dataset (self._current_output_dir), open it; otherwise open the video folder
            target_dir = None
            if self._current_output_dir and Path(self._current_output_dir).exists():
                target_dir = Path(self._current_output_dir)
            elif video_file.exists():
                target_dir = video_file.parent
            if target_dir and target_dir.exists():
                # Open in Windows Explorer
                try:
                    subprocess.Popen(f'explorer "{str(target_dir)}"')
                except Exception:
                    pass

        # Clickable only when a file is selected
        def update_file_path_label_state(*args):
            video_path = self.file_path_var.get()
            if not video_path or video_path == "(no file)":
                self.file_path_label.configure(text_color="#888", cursor="arrow")
                self.file_path_label.unbind("<Button-1>")
            else:
                self.file_path_label.configure(text_color="#4af", cursor="hand2")
                self.file_path_label.bind("<Button-1>", on_file_path_click)

        self.file_path_var.trace_add("write", lambda *a: update_file_path_label_state())
        # Also refresh navigator whenever the selected video path changes. This
        # ensures that if the user programmatically sets the path (or selects a
        # different video elsewhere), the output dataset shown in the navigator
        # is updated accordingly. We clear the cached _current_output_dir so
        # _update_file_navigator will recompute it from the new video path.
        self.file_path_var.trace_add("write", lambda *a: self._on_file_path_changed())
        update_file_path_label_state()

        self.every_n_var = ctk.StringVar(
            value=str(self._load_setting("video_every_n", 10))
        )
        self.every_n_entry = ctk.CTkEntry(
            controls_frame, width=80, textvariable=self.every_n_var
        )
        self.every_n_entry.grid(row=0, column=2, padx=(0, 8))
        try:
            self.every_n_entry.configure(placeholder_text="Every N")
        except Exception:
            pass
        # bind validation and mouse wheel handlers
        try:
            self.every_n_entry.bind("<FocusOut>", lambda e: self._on_every_n_change())
            self.every_n_entry.bind("<MouseWheel>", lambda e: self._on_every_n_wheel(e))

            # validation: only digits, max 4 chars
            def validate_char(new_val):
                if new_val == "":
                    return True
                if len(new_val) > 4:
                    return False
                return new_val.isdigit()

            vcmd = (self.register(lambda s: validate_char(s)), "%P")
            try:
                # CTkEntry proxies to underlying tk.Entry
                self.every_n_entry.configure(validate="key", validatecommand=vcmd)
            except Exception:
                pass
        except Exception:
            pass

        self.dedup_var = ctk.BooleanVar(value=self._load_setting("video_dedup", False))
        self.dedup_btn = ctk.CTkButton(
            controls_frame,
            text="Deduplication",
            width=100,
            command=self._run_deduplication_only,
        )
        self.dedup_btn.grid(row=0, column=3, padx=(0, 8))
        # --- Dedup threshold ---
        threshold_default = 12
        threshold_cfg = self._load_setting("video_dedup_threshold", None)
        if threshold_cfg is None:
            self._save_setting("video_dedup_threshold", threshold_default)
            threshold_cfg = threshold_default
        self.dedup_threshold_var = ctk.StringVar(value=str(threshold_cfg))

        def show_dedup_threshold_dialog():
            win = tk.Toplevel(self)
            win.title("Deduplication Aggressiveness")
            win.geometry("340x180")
            win.grab_set()
            label = tk.Label(
                win,
                text="Aggressiveness threshold (0–32):\nHigher = more duplicates removed.\nRecommended: 12–16 for training.\n0 = off, 32 = max.",
                justify="left",
            )
            label.pack(padx=16, pady=(16, 4), anchor="w")
            entry_var = tk.StringVar(value=self.dedup_threshold_var.get())
            entry = tk.Entry(
                win, textvariable=entry_var, font=("Consolas", 14), width=6
            )
            entry.pack(padx=16, pady=(8, 4), anchor="w")

            def on_ok():
                try:
                    val = int(entry_var.get())
                except Exception:
                    val = threshold_default
                val = max(0, min(32, val))
                self.dedup_threshold_var.set(str(val))
                self._save_setting("video_dedup_threshold", val)
                entry_var.set(str(val))
                win.destroy()

            ok_btn = tk.Button(win, text="OK", command=on_ok)
            ok_btn.pack(pady=(8, 12))
            entry.bind("<Return>", lambda e: on_ok())
            entry.focus_set()

        help_btn = ctk.CTkButton(
            controls_frame,
            text="?",
            width=10,
            command=show_dedup_threshold_dialog,
        )
        help_btn.grid(row=0, column=5, padx=(0, 8))

        self.dedup_threshold_entry = ctk.CTkEntry(
            controls_frame, width=60, textvariable=self.dedup_threshold_var
        )
        self.dedup_threshold_entry.grid(row=0, column=4, padx=(0, 8))
        try:
            self.dedup_threshold_entry.configure(placeholder_text="Agg (0-32)")
        except Exception:
            pass
        try:
            self.dedup_threshold_entry.bind(
                "<FocusOut>", lambda e: self._on_dedup_threshold_change()
            )
            self.dedup_threshold_entry.bind(
                "<Return>", lambda e: self._on_dedup_threshold_change()
            )
        except Exception:
            pass

        # Dataset clear button
        def on_clear_dataset():
            import tkinter.messagebox as mb
            from pathlib import Path

            out_dir = self._current_output_dir
            if not out_dir or not Path(out_dir).exists():
                mb.showinfo("Clear dataset", "Frame folder not found.")
                return
            files = list(Path(out_dir).glob("*.jpg"))
            if not files:
                mb.showinfo("Clear dataset", "No files to delete.")
                return
            if mb.askyesno("Clear dataset", f"Delete {len(files)} frames?"):
                removed = 0
                for f in files:
                    try:
                        f.unlink()
                        removed += 1
                    except Exception:
                        pass
                    # Clear parsing.json if present
                try:
                    sidecar = Path(out_dir) / "parsing.json"
                    written = {"files": []}
                    self._atomic_write_json(sidecar, written)
                    try:
                        key = str(Path(out_dir))
                        self._parsing_sidecar_cache[key] = written
                    except Exception:
                        pass
                    # Restore structure using standard function (rebuild metadata)
                    try:
                        self._update_parsing_sidecar()
                    except Exception:
                        pass
                except Exception:
                    pass
                self._update_file_navigator(force=True)
                # mb.showinfo("Clear dataset", f"Deleted {removed} files.")

        clear_btn = ctk.CTkButton(
            controls_frame,
            text="✖",
            width=40,
            fg_color="#a00",
            hover_color="#c33",
            text_color="white",
            command=on_clear_dataset,
        )
        clear_btn.grid(row=0, column=6, padx=(0, 8))
        try:
            # Tooltip on hover for clarity
            tooltip_win: list[Any] = [None]

            def show_tooltip(event):
                try:
                    if tooltip_win[0] is not None:
                        return
                    win = tk.Toplevel(clear_btn)
                    win.wm_overrideredirect(True)
                    x = event.x_root + 10
                    y = event.y_root + 10
                    win.geometry(f"+{x}+{y}")
                    label = tk.Label(
                        win,
                        text="Clear dataset",
                        bg="#fff",
                        fg="#a00",
                        relief="solid",
                        borderwidth=1,
                        font=("Arial", 10),
                    )
                    label.pack()
                    tooltip_win[0] = win
                except Exception:
                    pass

            def hide_tooltip(event):
                try:
                    if tooltip_win[0] is not None:
                        tooltip_win[0].destroy()
                        tooltip_win[0] = None
                except Exception:
                    pass

            clear_btn.bind("<Enter>", show_tooltip)
            clear_btn.bind("<Leave>", hide_tooltip)
        except Exception:
            pass

        self.run_btn = ctk.CTkButton(
            controls_frame, text="Run", width=80, command=self._on_run
        )
        self.run_btn.grid(row=0, column=7, padx=(0, 8))

        # Bottom part: horizontal split (bottom panel)
        bottom_panel = ctk.CTkFrame(
            self, corner_radius=8, border_width=2, border_color="#444"
        )
        bottom_panel.grid(row=1, column=0, sticky="nsew", padx=8, pady=(4, 8))
        bottom_panel.grid_rowconfigure(0, weight=1)
        bottom_panel.grid_columnconfigure(0, weight=1)
        bottom_panel.grid_columnconfigure(1, weight=2)

        # File navigator (left)
        self.nav_frame = ctk.CTkFrame(
            bottom_panel, corner_radius=6, border_width=2, border_color="#666"
        )
        self.nav_frame.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)
        self.nav_frame.grid_rowconfigure(0, weight=0)
        self.nav_frame.grid_rowconfigure(1, weight=1)
        self.nav_frame.grid_columnconfigure(0, weight=1)
        nav_label = ctk.CTkLabel(
            self.nav_frame, text="File Navigator", font=("Arial", 14, "bold")
        )
        nav_label.grid(row=0, column=0, sticky="w", padx=8, pady=6)
        self.file_listbox = tk.Listbox(self.nav_frame, height=30)
        self.file_listbox.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)
        try:
            self.file_listbox.bind(
                "<<ListboxSelect>>", lambda e: self._on_listbox_select(e)
            )
        except Exception:
            pass
        # Global arrow-key navigation: allow Up/Down to switch images even when listbox not focused
        try:
            # bind_all ensures keys are captured regardless of focus
            self.bind_all("<Up>", lambda e: self._on_nav_up(e))
            self.bind_all("<Down>", lambda e: self._on_nav_down(e))
            # numpad arrows
            self.bind_all("<KP_Up>", lambda e: self._on_nav_up(e))
            self.bind_all("<KP_Down>", lambda e: self._on_nav_down(e))
            # Delete key: remove current image from disk and parsing.json
            try:
                self.bind_all("<Delete>", lambda e: self._on_delete_current(e))
                self.bind_all("<KP_Delete>", lambda e: self._on_delete_current(e))
            except Exception:
                pass
        except Exception:
            pass
        self._current_output_dir = None
        self._nav_update_job = None
        # Debounced job id for committing selection to parsing.json
        self._selection_commit_job = None
        # Short lock to prevent duplicate nav handling (ms)
        self._nav_locked = False
        self._processing_state = None

        # In-memory cache for parsing sidecars to avoid re-reading file repeatedly.
        # Keyed by output directory (string path) -> dict (parsed JSON structure)
        try:
            self._parsing_sidecar_cache = {}
        except Exception:
            self._parsing_sidecar_cache = {}

        # In-memory state for file navigator
        self._files_order: list[str] = []
        self._selected_name: str | None = None
        self._last_index: int = 0

        # Preview (right)
        preview_frame = ctk.CTkFrame(
            bottom_panel, corner_radius=6, border_width=2, border_color="#666"
        )
        preview_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=8)
        preview_frame.grid_rowconfigure(0, weight=0)
        preview_frame.grid_rowconfigure(1, weight=1)
        # reserve row for metadata under preview image
        preview_frame.grid_rowconfigure(2, weight=0)
        preview_frame.grid_columnconfigure(0, weight=1)
        # Header: title + frame size selector
        header_frame = ctk.CTkFrame(preview_frame, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        header_frame.grid_columnconfigure(0, weight=1)
        preview_label = ctk.CTkLabel(
            header_frame, text="Preview", font=("Arial", 14, "bold")
        )
        preview_label.grid(row=0, column=0, sticky="w")

        # Frame size selector (square sizes for SD and other models)
        sizes = [
            "256x256",
            "384x384",
            "512x512",
            "640x640",
            "768x768",
            "1024x1024",
            "1280x1280",
        ]
        # load default/previous value
        current_size = str(self._load_setting("video_frame_size", "1024x1024"))
        if current_size not in sizes:
            current_size = "1024x1024"
        self.video_frame_size_var = ctk.StringVar(value=current_size)

        # Ensure default is persisted if missing
        try:
            if self._load_setting("video_frame_size", None) is None:
                self._save_setting("video_frame_size", current_size)
        except Exception:
            pass

        def _on_frame_size_change(new_val):
            try:
                # save immediately and atomically
                self._save_setting("video_frame_size", new_val)
            except Exception:
                pass
            try:
                # reflect in UI metadata
                self._update_preview_image()
            except Exception:
                pass

        try:
            self.frame_size_menu = ctk.CTkOptionMenu(
                header_frame,
                values=sizes,
                variable=self.video_frame_size_var,
                command=_on_frame_size_change,
            )
            self.frame_size_menu.grid(row=0, column=1, sticky="e")
        except Exception:
            # fallback: use normal OptionMenu if CTkOptionMenu unavailable
            try:
                import tkinter as _tk

                self.frame_size_menu = _tk.OptionMenu(
                    header_frame,
                    self.video_frame_size_var,
                    *sizes,
                    command=_on_frame_size_change,
                )
                self.frame_size_menu.grid(row=0, column=1, sticky="e")
            except Exception:
                pass

        # Image preview area (Canvas-based with cropper)
        import tkinter as _tk

        self.preview_canvas = _tk.Canvas(
            preview_frame, bg="#2a2a2a", highlightthickness=0
        )
        self.preview_canvas.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        # Rendering and crop state
        self._disp_image = None  # PIL.Image after resize
        self._disp_photo = None  # ImageTk.PhotoImage
        self._disp_img_id = None  # canvas image id
        self._disp_w = 0
        self._disp_h = 0
        self._disp_x0 = 0
        self._disp_y0 = 0
        self._scale = 1.0
        self._img_w = 0
        self._img_h = 0
        self._crop = None  # dict with crop in original px
        self._crop_rect_id = None
        self._dragging = False
        self._drag_axis = None  # 'x' or 'y'
        self._drag_anchor = (
            None  # (anchor_x, anchor_y) in image px relative to crop top-left
        )
        # Mouse bindings
        self.preview_canvas.bind("<Button-1>", self._on_canvas_click)
        self.preview_canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.preview_canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

        # Metadata label (filename / dims / size)
        self.preview_meta_label = ctk.CTkLabel(
            preview_frame, text="", anchor="w", justify="left"
        )
        self.preview_meta_label.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))

        # Update preview when container or canvas resizes
        try:
            preview_frame.bind("<Configure>", lambda e: self._update_preview_image())
            self.preview_canvas.bind(
                "<Configure>", lambda e: self._update_preview_image()
            )
        except Exception:
            pass

        # --- Status bar ---
        self.status_var = tk.StringVar(value="No video file selected.")
        status_frame = ctk.CTkFrame(self, fg_color="#222", corner_radius=0)
        status_frame.grid(row=2, column=0, sticky="ew", padx=0, pady=0)
        status_label = ctk.CTkLabel(
            status_frame,
            textvariable=self.status_var,
            anchor="w",
            font=("Consolas", 12),
            text_color="#eee",
        )
        # Размещаем статус слева, справа блок кнопок (Crop, Send to MT)
        status_label.pack(side="left", fill="x", expand=True, padx=12, pady=2)
        # контейнер справа
        try:
            actions_frame = ctk.CTkFrame(status_frame, fg_color="transparent")
            actions_frame.pack(side="right", padx=8, pady=2)
        except Exception:
            actions_frame = status_frame
        # Кнопка Crop
        try:
            self.crop_btn = ctk.CTkButton(
                actions_frame, text="Crop", width=80, command=self._on_crop_all
            )
            self.crop_btn.pack(side="right", padx=6)
        except Exception:
            self.crop_btn = None
        # Кнопка Send to MT — только если окно открыто из MaskingTool
        self._has_maskingtool_parent = False
        try:
            m = self.master
            if (
                m is not None
                and hasattr(m, "_scan_directory")
                and hasattr(m, "_set_folder_text")
            ):
                self._has_maskingtool_parent = True
        except Exception:
            self._has_maskingtool_parent = False
        self.send_to_mt_btn = None
        if self._has_maskingtool_parent:
            try:
                self.send_to_mt_btn = ctk.CTkButton(
                    actions_frame,
                    text="Send to MT",
                    width=110,
                    command=self._on_send_to_masking_tool,
                )
                self.send_to_mt_btn.pack(side="right", padx=6)
            except Exception:
                self.send_to_mt_btn = None
        # Initial navigator update now that status_var exists
        try:
            self._update_file_navigator()
        except Exception:
            pass

    def _on_send_to_masking_tool(self):
        """Передать текущий датасет в MaskingTool и закрыть окно (только если запущено из MaskingTool)."""
        try:
            if not getattr(self, "_has_maskingtool_parent", False):
                return
            out_dir = getattr(self, "_current_output_dir", None)
            if not out_dir or not Path(out_dir).exists():
                return
            parent = self.master
            if parent is None:
                return
            # Установить директорию датасета в MaskingTool
            try:
                if hasattr(parent, "_set_folder_text"):
                    parent._set_folder_text(str(out_dir))
            except Exception:
                pass
            try:
                setattr(parent, "dataset_dir", str(out_dir))
            except Exception:
                pass
            # Запустить сканирование каталога
            try:
                import threading as _th

                if hasattr(parent, "_scan_directory"):
                    _th.Thread(
                        target=parent._scan_directory, args=(str(out_dir),), daemon=True
                    ).start()
            except Exception:
                pass
            # Закрыть окно видеопарсера
            try:
                self._on_close()
            except Exception:
                try:
                    self.destroy()
                except Exception:
                    pass
        except Exception:
            pass

    def _update_action_buttons(self):
        """Обновить доступность кнопок действий по наличию датасета и контексту родителя."""
        try:
            out_dir = getattr(self, "_current_output_dir", None)
            has_ds = bool(out_dir and Path(out_dir).exists())
        except Exception:
            has_ds = False
        try:
            if self.crop_btn is not None:
                self.crop_btn.configure(state=("normal" if has_ds else "disabled"))
        except Exception:
            pass
        try:
            if (
                getattr(self, "_has_maskingtool_parent", False)
                and self.send_to_mt_btn is not None
            ):
                self.send_to_mt_btn.configure(
                    state=("normal" if has_ds else "disabled")
                )
        except Exception:
            pass

    def _update_preview_image(self):
        """Load and display the selected image and draw square cropper with normalization and logging."""
        try:
            from modules.util.image_util import load_image
        except Exception:
            load_image = None

        try:
            out_dir = getattr(self, "_current_output_dir", None)
            if not out_dir or not Path(out_dir).exists():
                # nothing to preview
                self.preview_canvas.delete("all")
                self._disp_image = None
                self._disp_photo = None
                self.preview_meta_label.configure(text="")
                return

            sel = None
            try:
                sel_idx = self.file_listbox.curselection()
                if sel_idx and len(sel_idx) > 0:
                    sel = self.file_listbox.get(sel_idx[0])
            except Exception:
                sel = None

            if not sel:
                # choose first file if available
                try:
                    sel = self.file_listbox.get(0)
                except Exception:
                    sel = None

            if not sel:
                self.preview_canvas.delete("all")
                self.preview_meta_label.configure(text="")
                return

            file_path = Path(out_dir) / sel
            if not file_path.exists():
                self.preview_meta_label.configure(text="File not found")
                return

            # Load image (prefer load_image util), gracefully fallback to PIL
            img = None
            try:
                if load_image:
                    img = load_image(str(file_path), "RGB")
                else:
                    img = Image.open(str(file_path)).convert("RGB")
            except Exception:
                img = None

            if img is None:
                self.preview_canvas.delete("all")
                self.preview_meta_label.configure(text="Unable to open image")
                return

            # Compute target size for preview
            try:
                # prefer the canvas size if available
                w = max(64, min(4096, self.preview_canvas.winfo_width() or 512))
                h = max(64, min(4096, self.preview_canvas.winfo_height() or 512))
            except Exception:
                w, h = 512, 512
            # create a copy and thumbnail to maintain aspect ratio
            try:
                img_copy = img.copy()
                # Pillow versions differ: prefer Resampling enum if present
                resampling = getattr(Image, "Resampling", None)
                if resampling is not None:
                    resample_val = Image.Resampling.LANCZOS
                elif hasattr(Image, "LANCZOS"):
                    resample_val = Image.LANCZOS
                else:
                    resample_val = Image.BICUBIC
                img_copy.thumbnail((w, h), resample_val)
            except Exception:
                img_copy = img

            # Draw on canvas: center image
            self.preview_canvas.delete("all")
            self._disp_image = img_copy
            self._img_w, self._img_h = img.width, img.height
            self._disp_w, self._disp_h = img_copy.width, img_copy.height
            try:
                sx = self._disp_w / max(1, self._img_w)
                sy = self._disp_h / max(1, self._img_h)
                self._scale = min(sx, sy)
            except Exception:
                self._scale = 1.0
            cw = max(1, self.preview_canvas.winfo_width())
            ch = max(1, self.preview_canvas.winfo_height())
            self._disp_x0 = (cw - self._disp_w) // 2
            self._disp_y0 = (ch - self._disp_h) // 2
            try:
                from PIL import ImageTk

                self._disp_photo = ImageTk.PhotoImage(self._disp_image)
                # create image with a known tag so we can control stacking reliably
                try:
                    self._disp_img_id = self.preview_canvas.create_image(
                        self._disp_x0,
                        self._disp_y0,
                        anchor="nw",
                        image=self._disp_photo,
                        tags=("preview_img",),
                    )
                    # ensure the preview image stays below interactive overlays
                    try:
                        self.preview_canvas.tag_lower("preview_img")
                    except Exception:
                        pass
                except Exception:
                    # fallback to previous API if tagging failed
                    self._disp_img_id = self.preview_canvas.create_image(
                        self._disp_x0,
                        self._disp_y0,
                        anchor="nw",
                        image=self._disp_photo,
                    )
            except Exception:
                self._disp_photo = None
                self._disp_img_id = None

            # Restore or create crop
            # Восстановить кроп из сайдкара (кэш), чтобы после рестарта всё сохранилось
            crop = self._get_file_crop(out_dir, sel)
            if not crop:
                # default center according to orientation
                if img.width == img.height:
                    orientation = "square"
                    size_px = img.width
                    x_px, y_px = 0, 0
                elif img.width > img.height:
                    orientation = "horizontal"
                    size_px = img.height
                    x_px = max(0, (img.width - size_px) // 2)
                    y_px = 0
                else:
                    orientation = "vertical"
                    size_px = img.width
                    x_px = 0
                    y_px = max(0, (img.height - size_px) // 2)
                crop = {
                    "image_w": img.width,
                    "image_h": img.height,
                    "size_px": size_px,
                    "x_px": x_px,
                    "y_px": y_px,
                }
            crop = self._validate_and_fix_crop(img.width, img.height, crop)
            self._crop = crop
            # Persist immediately if not present
            try:
                self._update_file_crop(out_dir, sel, crop)
            except Exception:
                pass
            # Draw crop (ensure cropper is visible and saved)
            try:
                # If crop size is invalid (0) fix it
                if not self._crop or int(self._crop.get("size_px", 0)) <= 0:
                    self._crop = self._validate_and_fix_crop(
                        img.width, img.height, self._crop or {}
                    )
                # Persist again to ensure sidecar contains current geometry
                try:
                    self._update_file_crop(out_dir, sel, self._crop)
                except Exception:
                    pass
            except Exception:
                pass
            self._draw_cropper()

            # Metadata: name, dimensions and size
            try:
                stat = file_path.stat()
                dims = f"{img.width}x{img.height}"
                size = f"{stat.st_size} bytes"
                chosen_size = (
                    self.video_frame_size_var.get()
                    if getattr(self, "video_frame_size_var", None)
                    else ""
                )
                text = f"{sel}\n{dims} — {size}"
                if chosen_size:
                    text += f"\nRender size: {chosen_size}"
                self.preview_meta_label.configure(text=text)
            except Exception:
                try:
                    self.preview_meta_label.configure(text=sel)
                except Exception:
                    pass
            # minimal preview log (kept silent to avoid console spam)
            # preview loaded - no debug log to avoid log spam
        except Exception:
            # swallow errors to keep UI responsive
            pass

    def _draw_cropper(self):
        if not self._disp_image or not self._crop:
            return
        try:
            size_disp = int(round(self._crop["size_px"] * self._scale))
        except Exception:
            size_disp = 0
        try:
            x_disp = int(round(self._disp_x0 + self._crop["x_px"] * self._scale))
        except Exception:
            x_disp = self._disp_x0
        try:
            y_disp = int(round(self._disp_y0 + self._crop["y_px"] * self._scale))
        except Exception:
            y_disp = self._disp_y0
        # Prevent zero-size crop on display
        if size_disp <= 0:
            # fall back to displaying full available image area as square
            size_disp = min(self._disp_w, self._disp_h)
        x1, y1 = x_disp, y_disp
        x2, y2 = x_disp + size_disp, y_disp + size_disp
        # Clamp inside displayed image
        img_x1, img_y1 = self._disp_x0, self._disp_y0
        img_x2, img_y2 = self._disp_x0 + self._disp_w, self._disp_y0 + self._disp_h
        x1 = max(img_x1, min(img_x2 - size_disp, x1))
        y1 = max(img_y1, min(img_y2 - size_disp, y1))
        x2, y2 = x1 + size_disp, y1 + size_disp
        try:
            # Always recreate the rectangle to avoid z-order/coordinate glitches.
            try:
                if self._crop_rect_id is not None:
                    self.preview_canvas.delete(self._crop_rect_id)
            except Exception:
                pass
            # create visible orange square cropper (no fill)
            try:
                self._crop_rect_id = self.preview_canvas.create_rectangle(
                    x1,
                    y1,
                    x2,
                    y2,
                    outline="#ffa500",
                    width=3,
                    fill="",
                    tags=("cropper",),
                )
            except Exception:
                # fallback without tags
                try:
                    self._crop_rect_id = self.preview_canvas.create_rectangle(
                        x1, y1, x2, y2, outline="#ffa500", width=3, fill=""
                    )
                except Exception:
                    self._crop_rect_id = None
            # Ensure cropper is above the preview image. Use lift on item id as it's more reliable.
            try:
                if self._crop_rect_id is not None:
                    try:
                        self.preview_canvas.lift(self._crop_rect_id)
                    except Exception:
                        pass
                try:
                    self.preview_canvas.tag_raise("cropper")
                except Exception:
                    pass
            except Exception:
                pass
        except Exception:
            pass
        # Log visible cropper coords for debugging
        # no verbose logging here

    def _on_canvas_click(self, event):
        if not self._disp_image or not self._crop:
            return
        orient = self._crop.get("orientation", "square")
        self._drag_axis = (
            "x" if orient == "horizontal" else ("y" if orient == "vertical" else None)
        )
        self._dragging = True
        # Compute image coordinates for exact click position and store anchor
        try:
            ix = (event.x - self._disp_x0) / max(1e-9, self._scale)
            iy = (event.y - self._disp_y0) / max(1e-9, self._scale)
            # anchor is offset from crop top-left to clicked point
            try:
                anchor_x = ix - float(self._crop.get("x_px", 0))
            except Exception:
                anchor_x = 0.0
            try:
                anchor_y = iy - float(self._crop.get("y_px", 0))
            except Exception:
                anchor_y = 0.0
            self._drag_anchor = (anchor_x, anchor_y)
        except Exception:
            self._drag_anchor = None
        # update using exact click position (no centering jump)
        self._update_crop_from_mouse(event.x, event.y)
        # show current offset in status while user starts dragging
        try:
            c = self._crop or {}
            x_px = int(c.get("x_px", 0))
            y_px = int(c.get("y_px", 0))
            x_norm = float(c.get("x_norm", 0.0))
            y_norm = float(c.get("y_norm", 0.0))
            try:
                self._set_status(
                    f"Crop offset: x={x_px} y={y_px} (norm={x_norm:.3f},{y_norm:.3f})"
                )
            except Exception:
                pass
        except Exception:
            pass

    def _on_canvas_drag(self, event):
        if not self._dragging:
            return
        self._update_crop_from_mouse(event.x, event.y)
        # update status with live offset while dragging
        try:
            c = self._crop or {}
            x_px = int(c.get("x_px", 0))
            y_px = int(c.get("y_px", 0))
            x_norm = float(c.get("x_norm", 0.0))
            y_norm = float(c.get("y_norm", 0.0))
            try:
                self._set_status(
                    f"Crop offset: x={x_px} y={y_px} (norm={x_norm:.3f},{y_norm:.3f})"
                )
            except Exception:
                pass
        except Exception:
            pass

    def _on_canvas_release(self, event):
        if self._dragging:
            self._update_crop_from_mouse(event.x, event.y)
        # restore normal dataset status after finishing drag
        try:
            # small attempt to update status to dataset summary
            try:
                self._update_status()
            except Exception:
                pass
        except Exception:
            pass
        self._dragging = False
        self._drag_axis = None
        # clear anchor
        try:
            self._drag_anchor = None
        except Exception:
            pass

    def _update_crop_from_mouse(self, mx: int, my: int):
        if not self._disp_image or not self._crop:
            return
        out_dir = getattr(self, "_current_output_dir", None)
        if not out_dir:
            return
        # map mouse to image px
        ix = (mx - self._disp_x0) / max(1e-9, self._scale)
        iy = (my - self._disp_y0) / max(1e-9, self._scale)
        size_px = self._crop["size_px"]
        max_x = max(0, self._img_w - size_px)
        max_y = max(0, self._img_h - size_px)
        x_px = self._crop["x_px"]
        y_px = self._crop["y_px"]
        # If we have an anchor (exact click offset), move crop preserving that offset
        try:
            if self._drag_anchor is not None:
                anchor_x, anchor_y = self._drag_anchor
                if self._drag_axis == "x":
                    new_x = int(round(ix - anchor_x))
                    x_px = max(0, min(max_x, new_x))
                elif self._drag_axis == "y":
                    new_y = int(round(iy - anchor_y))
                    y_px = max(0, min(max_y, new_y))
                else:
                    # free drag: preserve both offsets
                    new_x = int(round(ix - anchor_x))
                    new_y = int(round(iy - anchor_y))
                    x_px = max(0, min(max_x, new_x))
                    y_px = max(0, min(max_y, new_y))
            else:
                if self._drag_axis == "x":
                    x_px = int(round(ix - size_px / 2))
                    x_px = max(0, min(max_x, x_px))
                elif self._drag_axis == "y":
                    y_px = int(round(iy - size_px / 2))
                    y_px = max(0, min(max_y, y_px))
        except Exception:
            # fallback to previous behavior on error
            try:
                if self._drag_axis == "x":
                    x_px = int(round(ix - size_px / 2))
                    x_px = max(0, min(max_x, x_px))
                elif self._drag_axis == "y":
                    y_px = int(round(iy - size_px / 2))
                    y_px = max(0, min(max_y, y_px))
            except Exception:
                pass
        # update state
        self._crop["x_px"] = x_px
        self._crop["y_px"] = y_px
        self._crop["x_norm"] = 0.0 if max_x == 0 else x_px / max_x
        self._crop["y_norm"] = 0.0 if max_y == 0 else y_px / max_y
        # redraw
        self._draw_cropper()
        # persist
        try:
            sel_idx = self.file_listbox.curselection()
            if sel_idx and len(sel_idx) > 0:
                sel = self.file_listbox.get(sel_idx[0])
            else:
                sel = None
        except Exception:
            sel = None
        if sel:
            try:
                self._update_file_crop(Path(out_dir), sel, self._crop)
                try:
                    self._refresh_nav_colors()
                except Exception:
                    pass
            except Exception:
                pass

    def _refresh_nav_colors(self):
        """Color items in the file_listbox based on parsing.json crop state.

        - green: has crop in sidecar (valid crop)
        - orange: crop exists and covers whole image (square/no-op)
        - red: no crop entry
        """
        try:
            out_dir = getattr(self, "_current_output_dir", None)
            if not out_dir or not Path(out_dir).exists():
                return
            # obtain sidecar from in-memory cache (do not re-read file repeatedly)
            try:
                obj = self._get_cached_sidecar(out_dir, reload=False)
            except Exception:
                obj = {"files": []}
            crops_by_name = {}
            try:
                for it in obj.get("files", []):
                    if isinstance(it, dict) and "relpath" in it and "crop" in it:
                        crops_by_name[it["relpath"]] = it.get("crop")
            except Exception:
                crops_by_name = {}
            # iterate listbox items and set colors
            for i in range(self.file_listbox.size()):
                try:
                    fname = self.file_listbox.get(i)
                except Exception:
                    fname = None
                fg = "#ffffff"
                bg = None
                if fname and fname in crops_by_name:
                    crop = crops_by_name.get(fname) or {}
                    try:
                        size_px = int(crop.get("size_px", 0))
                        img_w = int(crop.get("image_w", 0))
                        img_h = int(crop.get("image_h", 0))
                    except Exception:
                        size_px = 0
                        img_w = 0
                        img_h = 0
                    if size_px > 0 and img_w == img_h == size_px:
                        bg = "#ff9900"  # orange
                        fg = "#000000"
                    else:
                        bg = "#2e8b57"  # green
                        fg = "#ffffff"
                else:
                    bg = "#a00"  # red
                    fg = "#ffffff"
                try:
                    self.file_listbox.itemconfig(i, {"bg": bg, "fg": fg})
                except Exception:
                    try:
                        # fallback: adjust select colors so at least selection reflects state
                        self.file_listbox.configure(
                            selectbackground=bg, selectforeground=fg
                        )
                    except Exception:
                        pass
        except Exception:
            pass

    def _on_open_file(self):
        try:
            from tkinter import filedialog

            p = filedialog.askopenfilename()
            if p:
                self.file_path_var.set(str(p))
                try:
                    self._save_setting("video_last_file", str(p))
                except Exception:
                    pass
                # Update navigator for the new target folder
                self._update_file_navigator(force=True)
        except Exception:
            pass

    def _on_file_path_changed(self):
        """Handle updates when the selected video file path changes.

        Clear any cached output directory so that the navigator will recompute
        the dataset folder from the new video path and refresh immediately.
        """
        try:
            # Clear cached output dir so _update_file_navigator recomputes it
            try:
                self._current_output_dir = None
            except Exception:
                pass
            try:
                # compute output dir and force reload of sidecar once on open
                try:
                    vp = Path(self.file_path_var.get())
                    if vp and vp.exists():
                        out_dir = vp.parent / vp.stem
                        if out_dir.exists():
                            try:
                                self._get_cached_sidecar(out_dir, reload=True)
                                # reset in-memory nav state for new dataset
                                try:
                                    self._files_order = []
                                    self._selected_name = None
                                    self._last_index = 0
                                except Exception:
                                    pass
                            except Exception:
                                pass
                except Exception:
                    pass
                self._update_file_navigator(force=True)
            except Exception:
                pass
            try:
                self._update_status()
            except Exception:
                pass
        except Exception:
            pass

    def _schedule_nav_update(self, delay_ms: int | None = None):
        """Immediate navigator update (debounce disabled by design)."""
        try:
            # Cancel any pending scheduled job and update immediately
            try:
                nav_job = getattr(self, "_nav_update_job", None)
                if nav_job is not None:
                    try:
                        self.after_cancel(nav_job)
                    except Exception:
                        pass
                    self._nav_update_job = None
            except Exception:
                pass
            self._update_file_navigator(force=True)
        except Exception:
            pass

    def _on_every_n_wheel(self, event):
        try:
            cur = int(self.every_n_var.get() or 0)
        except Exception:
            cur = 10
        delta = 1 if event.delta > 0 else -1
        new = max(1, min(9999, cur + delta))
        self.every_n_var.set(str(new))
        self._on_every_n_change()

    def _on_every_n_change(self):
        try:
            val = int(self.every_n_var.get())
            val = max(1, min(9999, val))
            self.every_n_var.set(str(val))
            self._save_setting("video_every_n", val)
        except Exception:
            pass

    def _on_dedup_toggle(self):
        try:
            v = bool(self.dedup_var.get())
            self._save_setting("video_dedup", v)
            # Save threshold as well when toggled
            try:
                t = int(self.dedup_threshold_var.get())
                t = max(0, min(32, t))
                self._save_setting("video_dedup_threshold", t)
                self.dedup_threshold_var.set(str(t))
            except Exception:
                pass
        except Exception:
            pass

    def _on_dedup_threshold_change(self):
        try:
            t = int(self.dedup_threshold_var.get())
        except Exception:
            t = self._load_setting("video_dedup_threshold", 12)
            if t is None:
                t = 12
            try:
                t = int(t)
            except Exception:
                t = 12
        t = max(0, min(32, t))
        self.dedup_threshold_var.set(str(t))
        try:
            self._save_setting("video_dedup_threshold", t)
        except Exception:
            pass

    def _on_run(self):
        import shutil
        import tkinter as tk
        from tkinter import messagebox

        logger = logging.getLogger("video_parser")
        logger.info(
            f"Run clicked: file={self.file_path_var.get()}, every_n={self.every_n_var.get()}"
        )

        def _find_ffmpeg_executable():
            # 1) check environment override
            env_path = os.environ.get("FFMPEG_BINARY") or os.environ.get("FFMPEG_PATH")
            if env_path:
                if shutil.which(env_path) or Path(env_path).exists():
                    return env_path
            # 2) check PATH (skip if debugging)
            if not SKIP_SYSTEM_FFMPEG_CHECK:
                ffmpeg_path = shutil.which("ffmpeg")
                if ffmpeg_path:
                    return ffmpeg_path
            # 3) check virtualenv scripts/bin
            venv = os.environ.get("VIRTUAL_ENV")
            if venv:
                candidate = (
                    Path(venv)
                    / ("Scripts" if os.name == "nt" else "bin")
                    / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
                )
                if candidate.exists():
                    return str(candidate)
            # also check next to python executable
            try:
                pyexe = Path(sys.executable)
                candidate = pyexe.parent / (
                    "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
                )
                if candidate.exists():
                    return str(candidate)
            except Exception:
                pass
            # 4) try imageio_ffmpeg package which can provide a binary
            try:
                import imageio_ffmpeg as iioff

                try:
                    exe = iioff.get_ffmpeg_exe()
                except Exception:
                    exe = None
                if exe and Path(exe).exists():
                    return exe
            except Exception:
                pass
            # not found
            return None

        def show_ffmpeg_alert():
            import sys, os

            venv = os.environ.get("VIRTUAL_ENV")
            if venv:
                pip_path = (
                    os.path.join(venv, "Scripts", "pip.exe")
                    if os.name == "nt"
                    else os.path.join(venv, "bin", "pip")
                )
            else:
                pip_path = (
                    sys.executable.replace("python.exe", "Scripts/pip.exe")
                    if os.name == "nt"
                    else sys.executable.replace("bin/python", "bin/pip")
                )
            if os.name == "nt":
                # PowerShell requires & before an executable path with quotes
                cmd = f'& "{pip_path}" install ffmpeg-python'
            else:
                cmd = f'"{pip_path}" install ffmpeg-python'
            win = tk.Toplevel(self)
            win.title("FFmpeg Required")
            win.geometry("520x220")
            win.grab_set()
            label = tk.Label(
                win,
                text="FFmpeg not found in system PATH or as Python package.\n\nPlease install ffmpeg and/or the Python package using the command below:",
                justify="left",
                wraplength=500,
            )
            label.pack(padx=16, pady=(16, 4), anchor="w")
            entry = tk.Entry(win, width=48, font=("Consolas", 12))
            entry.insert(0, cmd)
            entry.config(state="readonly")
            entry.pack(padx=16, pady=(0, 4), anchor="w", fill="x")

            def copy_cmd():
                win.clipboard_clear()
                win.clipboard_append(cmd)

            copy_btn = tk.Button(win, text="Copy", command=copy_cmd)
            copy_btn.pack(padx=16, pady=(0, 8), anchor="w")
            info = tk.Label(
                win, text="After installation, click OK to retry.", justify="left"
            )
            info.pack(padx=16, pady=(0, 8), anchor="w")

            def on_ok():
                try:
                    win.destroy()
                except Exception:
                    pass

            ok_btn = tk.Button(win, text="OK", command=on_ok)
            ok_btn.pack(pady=(0, 12))
            entry.bind("<Return>", lambda e: on_ok())
            entry.focus_set()

        ffmpeg_exe = _find_ffmpeg_executable()
        while not ffmpeg_exe:
            # if python wrapper exists, log info
            try:
                import ffmpeg as _ff

                logger.info(
                    "ffmpeg-python is installed but ffmpeg binary not found; please install binary"
                )
            except Exception:
                pass
            show_ffmpeg_alert()
            ffmpeg_exe = _find_ffmpeg_executable()
        self._ffmpeg_executable = ffmpeg_exe
        logger.info(f"ffmpeg binary located at: {self._ffmpeg_executable}")
        # --- New block: create folder and run ffmpeg ---
        import subprocess, tempfile, shutil, glob
        from pathlib import Path
        import shlex
        import numpy as np
        import cv2

        video_path = Path(self.file_path_var.get())
        if not video_path.exists():
            logger.error("Selected file does not exist.")
            messagebox.showerror("Error", "Selected file does not exist.")
            return
        output_dir = video_path.parent / video_path.stem
        output_dir.mkdir(exist_ok=True)
        self._current_output_dir = output_dir
        self._update_file_navigator(force=True)
        try:
            every_n = int(self.every_n_var.get())
        except Exception:
            every_n = 10
        # 1. Prepare for extraction (we will write restored frames directly to output_dir)
        self._update_status()

        # Disable buttons to prevent duplicate runs
        try:
            self.run_btn.configure(state="disabled")
        except Exception:
            pass
        try:
            self.open_btn.configure(state="disabled")
        except Exception:
            pass

        import threading

        def worker(out_dir: Path, every_n_local: int, dedup_threshold: int):
            # Background worker: run ffmpeg, parse -progress output, then optionally deduplicate
            try:
                try:
                    self._processing_state = "extracting"
                    self.after(0, lambda: self._set_status("Распаковка кадров: 0"))
                except Exception:
                    pass

                vf_expr = f"select=not(mod(n\\,{every_n_local}))"
                ffmpeg_cmd_local = [
                    str(self._ffmpeg_executable),
                    "-i",
                    str(video_path),
                    "-nostats",
                    "-progress",
                    "pipe:1",
                    "-vsync",
                    "vfr",
                    "-q:v",
                    "2",
                    "-vf",
                    vf_expr,
                    str(out_dir / "%06d.jpg"),
                ]

                logger.info(
                    "Extracting restored frames (background): %s",
                    " ".join(shlex.quote(str(x)) for x in ffmpeg_cmd_local),
                )

                try:
                    proc = subprocess.Popen(
                        ffmpeg_cmd_local,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        text=True,
                    )
                except Exception as e:
                    logger.error(f"Failed to start ffmpeg: {e}")
                    self.after(0, lambda: messagebox.showerror("ffmpeg error", str(e)))
                    return

                # estimate total expected frames
                total_expected = None
                try:
                    cap = cv2.VideoCapture(str(video_path))
                    if cap is not None:
                        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                        try:
                            cap.release()
                        except Exception:
                            pass
                        if fc > 0:
                            import math

                            total_expected = max(
                                0, math.ceil(fc / float(every_n_local))
                            )
                except Exception:
                    total_expected = None

                # parse progress from ffmpeg stdout
                try:
                    created_count = 0
                    stdout_it = proc.stdout if proc.stdout is not None else []
                    for raw in stdout_it:
                        line = raw.strip()
                        if not line:
                            continue
                        if line.startswith("frame="):
                            try:
                                frame = int(line.split("=", 1)[1])
                            except Exception:
                                frame = None
                            if frame is not None:
                                created_count = frame
                                try:
                                    if total_expected:
                                        self.after(
                                            0,
                                            lambda c=created_count, t=total_expected: self._set_status(
                                                f"Распаковка кадров: {c}/{t}"
                                            ),
                                        )
                                    else:
                                        self.after(
                                            0,
                                            lambda c=created_count: self._set_status(
                                                f"Распаковка кадров: {c}"
                                            ),
                                        )
                                except Exception:
                                    pass
                        elif (
                            line.startswith("progress=")
                            and line.split("=", 1)[1] == "end"
                        ):
                            break
                    retcode = proc.wait()
                    if retcode != 0:
                        logger.error(f"ffmpeg exited with code: {retcode}")
                        self.after(
                            0,
                            lambda: messagebox.showerror(
                                "ffmpeg error", f"ffmpeg exited with code {retcode}"
                            ),
                        )
                        return
                except Exception as e:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    logger.error(f"Failed while reading ffmpeg progress: {e}")
                    self.after(0, lambda: messagebox.showerror("ffmpeg error", str(e)))
                    return

                # Finalize: count created files as fallback and update status
                try:
                    out_files_local = sorted(out_dir.glob("*.jpg"))
                    created_count = len(out_files_local)
                    try:
                        self.after(
                            0,
                            lambda c=created_count, t=created_count: self._set_status(
                                f"Frames created: {c}/{t}"
                            ),
                        )
                    except Exception:
                        pass
                except Exception:
                    created_count = 0

                # Deduplication (always run after extraction)
                try:
                    self._processing_state = "dedup"
                    self.after(
                        0, lambda: self._set_status("Deduplication in progress...")
                    )
                except Exception:
                    pass
                logger.info("Starting deduplication...")

                def dhash(image, hash_size=8):
                    if cv2 is None:
                        # fallback simple hash using PIL to keep flow running
                        try:
                            from PIL import Image as _PILImage
                            import numpy as _np

                            pil_img = (
                                _PILImage.fromarray(image[..., ::-1])
                                if image is not None
                                else None
                            )
                            if pil_img is None:
                                return 0
                            pil_img = pil_img.convert("L").resize(
                                (hash_size + 1, hash_size)
                            )
                            arr = _np.array(pil_img)
                            diff = arr[:, 1:] > arr[:, :-1]
                            dh = 0
                            for v in diff.flatten():
                                dh = (dh << 1) | int(bool(v))
                            return dh
                        except Exception:
                            return 0
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    resized = cv2.resize(gray, (hash_size + 1, hash_size))
                    diff = resized[:, 1:] > resized[:, :-1]
                    dh = 0
                    for v in diff.flatten():
                        dh = (dh << 1) | int(v)
                    return dh

                def hamming(a, b):
                    x = a ^ b
                    return bin(x).count("1")

                hashes_local = []
                removed_local = 0
                out_files_local = sorted(out_dir.glob("*.jpg"))
                total_out_files = len(out_files_local)
                try:
                    self.after(
                        0,
                        lambda total=total_out_files: self._set_status(
                            f"Deduplication in progress: remaining {total} files."
                        ),
                    )
                except Exception:
                    pass

                DEDUP_THRESHOLD = max(0, min(32, int(dedup_threshold)))
                for f_local in out_files_local:
                    img_local = _safe_imread(
                        f_local, cv2.IMREAD_UNCHANGED if cv2 is not None else None
                    )
                    if img_local is None:
                        continue
                    h_local = dhash(img_local)
                    dup_local = False
                    for oh, ofp in hashes_local:
                        if hamming(h_local, oh) <= DEDUP_THRESHOLD:
                            logger.debug(
                                f"Removing duplicate {f_local} similar to {ofp}"
                            )
                            try:
                                f_local.unlink()
                                removed_local += 1
                            except Exception:
                                pass
                            dup_local = True
                            try:
                                # refresh navigator immediately without debounce
                                self.after(0, lambda: self._schedule_nav_update())
                            except Exception:
                                pass
                            try:
                                remaining = max(0, total_out_files - removed_local)
                                self.after(
                                    0,
                                    lambda r=remaining: self._set_status(
                                        f"Deduplication in progress: remaining {r} files."
                                    ),
                                )
                            except Exception:
                                pass
                            break
                    if not dup_local:
                        hashes_local.append((h_local, f_local))
                logger.info(f"Deduplication completed, removed {removed_local} files")
                try:
                    remaining = max(0, total_out_files - removed_local)
                    self.after(
                        0,
                        lambda rem=remaining, remd=removed_local: self._set_status(
                            f"Deduplication completed: {remd} removed, {rem} remaining."
                        ),
                    )
                except Exception:
                    pass

                # Финал: обновить навигатор и статус
                try:
                    self.after(0, lambda: self._update_file_navigator(force=True))
                except Exception:
                    pass
                try:
                    self.after(0, self._update_status)
                except Exception:
                    pass

            except Exception as e:
                logger.exception("Unexpected error in background worker")
                try:
                    self.after(0, lambda: messagebox.showerror("Error", str(e)))
                except Exception:
                    pass
            finally:
                try:
                    self.after(0, lambda: self.run_btn.configure(state="normal"))
                except Exception:
                    pass
                try:
                    self.after(0, lambda: self.open_btn.configure(state="normal"))
                except Exception:
                    pass
                try:
                    self._processing_state = None
                    self.after(0, self._update_status)
                except Exception:
                    pass

        # Запуск фонового потока
        try:
            dedup_threshold = int(self.dedup_threshold_var.get())
        except Exception:
            dedup_threshold = 12
        thread = threading.Thread(
            target=worker, args=(output_dir, every_n, dedup_threshold), daemon=True
        )
        thread.start()

    def _run_deduplication_only(self):
        """Run deduplication on existing files in the current output directory."""
        import threading
        import tkinter as tk
        from tkinter import messagebox
        from pathlib import Path

        logger = logging.getLogger("video_parser")

        out_dir = self._current_output_dir
        if not out_dir or not Path(out_dir).exists():
            messagebox.showerror(
                "Error", "No output directory found. Please run extraction first."
            )
            return

        try:
            dedup_threshold = int(self.dedup_threshold_var.get())
        except Exception:
            dedup_threshold = 12

        def dedup_worker():
            try:
                self._processing_state = "dedup"
                self.after(0, lambda: self._set_status("Deduplication in progress..."))
                logger.info("Starting deduplication on existing files...")

                def dhash(image, hash_size=8):
                    if cv2 is None:
                        try:
                            from PIL import Image as _PILImage
                            import numpy as _np

                            pil_img = (
                                _PILImage.fromarray(image[..., ::-1])
                                if image is not None
                                else None
                            )
                            if pil_img is None:
                                return 0
                            pil_img = pil_img.convert("L").resize(
                                (hash_size + 1, hash_size)
                            )
                            arr = _np.array(pil_img)
                            diff = arr[:, 1:] > arr[:, :-1]
                            dh = 0
                            for v in diff.flatten():
                                dh = (dh << 1) | int(bool(v))
                            return dh
                        except Exception:
                            return 0
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    resized = cv2.resize(gray, (hash_size + 1, hash_size))
                    diff = resized[:, 1:] > resized[:, :-1]
                    dh = 0
                    for v in diff.flatten():
                        dh = (dh << 1) | int(v)
                    return dh

                def hamming(a, b):
                    x = a ^ b
                    return bin(x).count("1")

                hashes_local = []
                removed_local = 0
                out_files_local = sorted(out_dir.glob("*.jpg"))
                total_out_files = len(out_files_local)
                try:
                    self.after(
                        0,
                        lambda total=total_out_files: self._set_status(
                            f"Deduplication in progress: remaining {total} files."
                        ),
                    )
                except Exception:
                    pass

                DEDUP_THRESHOLD = max(0, min(32, dedup_threshold))
                for f_local in out_files_local:
                    img_local = _safe_imread(
                        f_local, cv2.IMREAD_UNCHANGED if cv2 is not None else None
                    )
                    if img_local is None:
                        continue
                    h_local = dhash(img_local)
                    dup_local = False
                    for oh, ofp in hashes_local:
                        if hamming(h_local, oh) <= DEDUP_THRESHOLD:
                            logger.debug(
                                f"Removing duplicate {f_local} similar to {ofp}"
                            )
                            try:
                                f_local.unlink()
                                removed_local += 1
                            except Exception:
                                pass
                            dup_local = True
                            try:
                                self.after(0, lambda: self._schedule_nav_update())
                            except Exception:
                                pass
                            try:
                                remaining = max(0, total_out_files - removed_local)
                                self.after(
                                    0,
                                    lambda r=remaining: self._set_status(
                                        f"Deduplication in progress: remaining {r} files."
                                    ),
                                )
                            except Exception:
                                pass
                            break
                    if not dup_local:
                        hashes_local.append((h_local, f_local))
                logger.info(f"Deduplication completed, removed {removed_local} files")
                try:
                    remaining = max(0, total_out_files - removed_local)
                    self.after(
                        0,
                        lambda rem=remaining, remd=removed_local: self._set_status(
                            f"Deduplication completed: {remd} removed, {rem} remaining."
                        ),
                    )
                except Exception:
                    pass

                # Update navigator
                try:
                    self.after(0, lambda: self._schedule_nav_update())
                except Exception:
                    pass

            except Exception as e:
                logger.exception("Error during deduplication")
                try:
                    self.after(0, lambda: messagebox.showerror("Error", str(e)))
                except Exception:
                    pass
            finally:
                try:
                    self._processing_state = None
                    self.after(0, self._update_status)
                except Exception:
                    pass

        # Run in background thread
        thread = threading.Thread(target=dedup_worker, daemon=True)
        thread.start()

    def _update_file_navigator(self, force=False):
        """Обновить список JPG-файлов в навигаторе без повторного чтения сайдкара.

        Правила:
        - сайдкар читаем один раз при выборе датасета (инициализация)
        - всё состояние (список файлов, выделение) держим в памяти
        - сайдкар пишем параллельно/немедленно как историю, но не используем для логики
        - после удаления гарантированно переходим к следующему элементу (или последнему)
        """
        import glob
        from pathlib import Path

        # Определить целевую папку
        if self._current_output_dir is None:
            video_path = Path(self.file_path_var.get())
            if video_path.exists():
                self._current_output_dir = video_path.parent / video_path.stem
        out_dir = self._current_output_dir
        if not out_dir or not out_dir.exists():
            self.file_listbox.delete(0, tk.END)
            # Clear preview area when dataset folder is missing
            try:
                self.preview_canvas.delete("all")
            except Exception:
                pass
            try:
                self._disp_image = None
                self._disp_photo = None
                self._disp_img_id = None
            except Exception:
                pass
            try:
                self.preview_meta_label.configure(text="")
            except Exception:
                pass
            self.status_var.set("Dataset folder not found.")
            try:
                self._update_action_buttons()
            except Exception:
                pass
            return
        # Загрузить сайдкар в кэш при первом обращении (инициализация)
        key = None
        try:
            key = self._get_dataset_key(out_dir)
            sc_path = Path(out_dir) / "parsing.json"
            if key and key not in self._parsing_sidecar_cache and sc_path.exists():
                self._parsing_sidecar_cache[key] = self._read_parsing_sidecar(out_dir)
        except Exception:
            pass

        # Сформировать новый список файлов, упорядоченный по сайдкару
        files_on_disk = sorted(out_dir.glob("*.jpg"))
        disk_names = [f.name for f in files_on_disk]
        old_names = list(self._files_order)
        try:
            obj = self._get_cached_sidecar(out_dir, reload=False)
        except Exception:
            obj = {"files": []}
        sidecar_order = [
            it.get("relpath")
            for it in (obj.get("files", []) or [])
            if isinstance(it, dict) and it.get("relpath")
        ]
        # Оставляем только существующие на диске
        ordered_existing = [n for n in sidecar_order if n in disk_names]
        # Новые файлы, которых нет в сайдкаре
        new_missing = sorted([n for n in disk_names if n not in set(ordered_existing)])
        new_names = ordered_existing + new_missing
        # Если появились новые файлы — добавим заготовки в кэш (без перезаписи кропов)
        if new_missing:
            try:
                files_meta = obj.get("files", []) or []
                present = set(sidecar_order)
                for n in new_missing:
                    if n not in present:
                        files_meta.append({"relpath": n})
                obj["files"] = files_meta
                if key:
                    self._parsing_sidecar_cache[key] = obj
            except Exception:
                pass
        # Обновить in-memory список
        self._files_order = new_names

        # Определить текущий выбор
        chosen = None
        # 1) Предпочесть предыдущий UI-выбор, если он существует
        if self._selected_name and self._selected_name in new_names:
            chosen = self._selected_name
        # 2) Затем выбор из сайдкара (selected), если такой есть и файл существует
        if not chosen:
            try:
                for it in obj.get("files", []) or []:
                    if it.get("selected") and it.get("relpath") in new_names:
                        chosen = it.get("relpath")
                        break
            except Exception:
                pass
        # 3) Затем восстановление по последнему индексу
        if not chosen and new_names:
            idx = max(0, min(len(new_names) - 1, getattr(self, "_last_index", 0)))
            chosen = new_names[idx]
        # 4) Иначе первый
        if not chosen and new_names:
            chosen = new_names[0]

        # Перерисовать Listbox согласно in-memory списку
        try:
            self.file_listbox.delete(0, tk.END)
            for name in new_names:
                self.file_listbox.insert(tk.END, name)
        except Exception:
            pass

        # Установить выделение и запомнить индекс
        if chosen and new_names:
            try:
                idx = new_names.index(chosen)
                self._selected_name = chosen
                self._last_index = idx
                self.file_listbox.selection_clear(0, tk.END)
                self.file_listbox.selection_set(idx)
                self.file_listbox.see(idx)
                # проставить selected в памяти и записать сайдкар
                self._set_selected_in_memory(out_dir, chosen)
                self._write_sidecar_async(out_dir)
            except Exception:
                pass

        # Обновить превью/цвета/статус
        try:
            if self.file_listbox.size() == 0:
                try:
                    self.preview_canvas.delete("all")
                except Exception:
                    pass
                try:
                    self._disp_image = None
                    self._disp_photo = None
                    self._disp_img_id = None
                except Exception:
                    pass
                try:
                    self.preview_meta_label.configure(text="")
                except Exception:
                    pass
            else:
                try:
                    self._update_preview_image()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            self._refresh_nav_colors()
        except Exception:
            pass
        try:
            self._update_status()
        except Exception:
            pass
        try:
            self._update_action_buttons()
        except Exception:
            pass

    def _on_listbox_select(self, event=None):
        """Handler for immediate save of selected file to parsing.json."""
        try:
            sel = self.file_listbox.curselection()
            if not sel and self.file_listbox.size() > 0:
                try:
                    self.file_listbox.selection_set(0)
                except Exception:
                    pass
            # Без дебаунса: сразу выставляем selection в памяти и пишем сайдкар
            try:
                idx = None
                try:
                    cur = self.file_listbox.curselection()
                    if cur and len(cur) > 0:
                        idx = cur[0]
                except Exception:
                    idx = None
                if idx is not None and 0 <= idx < self.file_listbox.size():
                    name = self.file_listbox.get(idx)
                    self._selected_name = name
                    self._last_index = idx
                    od = getattr(self, "_current_output_dir", None)
                    if od:
                        self._set_selected_in_memory(Path(od), name)
                        self._write_sidecar_async(Path(od))
            except Exception:
                pass
        except Exception:
            pass
        # Ensure status bar shows current dataset info (safe call)
        try:
            if not getattr(self, "_processing_state", None):
                # call central updater which computes dataset size safely
                try:
                    self._update_status()
                except Exception:
                    pass
        except Exception:
            pass
        # update preview immediately
        try:
            self._update_preview_image()
        except Exception:
            pass
        try:
            # ensure navigator colors update responsively
            self._refresh_nav_colors()
        except Exception:
            pass
        try:
            self._refresh_nav_colors()
        except Exception:
            pass
        # temp_dir автоматически удалится
        # TODO: дедупликация

    def _select_listbox_index(self, idx: int):
        try:
            size = self.file_listbox.size()
            if size == 0:
                return
            idx = max(0, min(size - 1, idx))
            self.file_listbox.selection_clear(0, tk.END)
            self.file_listbox.selection_set(idx)
            self.file_listbox.see(idx)
            # Update in-memory selected state
            try:
                name = self.file_listbox.get(idx)
                self._selected_name = name
                self._last_index = idx
                od = getattr(self, "_current_output_dir", None)
                if od:
                    self._set_selected_in_memory(Path(od), name)
                    self._write_sidecar_async(Path(od))
            except Exception:
                pass
            # trigger selection handlers
            try:
                self._on_listbox_select()
            except Exception:
                pass
        except Exception:
            pass

    def _on_nav_up(self, event=None):
        try:
            # If the event originated in the listbox, let its native
            # binding handle the movement to avoid double-stepping.
            try:
                focused = None
                try:
                    focused = self.focus_get()
                except Exception:
                    focused = None
                if focused is self.file_listbox:
                    return None
            except Exception:
                pass
            # cancel pending jobs to avoid races which can reset selection
            try:
                sel_job = getattr(self, "_selection_commit_job", None)
                if sel_job is not None:
                    try:
                        self.after_cancel(sel_job)
                    except Exception:
                        pass
                    try:
                        self._selection_commit_job = None
                    except Exception:
                        pass
            except Exception:
                pass

            # short-circuit if nav locked
            try:
                if getattr(self, "_nav_locked", False):
                    return "break"
                self._nav_locked = True
                # release lock after short delay
                try:
                    self.after(80, lambda: setattr(self, "_nav_locked", False))
                except Exception:
                    try:
                        self._nav_locked = False
                    except Exception:
                        pass
            except Exception:
                pass
            cur = None
            try:
                sel = self.file_listbox.curselection()
                if sel and len(sel) > 0:
                    cur = sel[0]
            except Exception:
                cur = None
            if cur is None:
                new = 0
            else:
                new = max(0, cur - 1)
            self._select_listbox_index(new)
            return "break"
        except Exception:
            return None

    def _on_nav_down(self, event=None):
        try:
            # If the event originated in the listbox, let its native
            # binding handle the movement to avoid double-stepping.
            try:
                focused = None
                try:
                    focused = self.focus_get()
                except Exception:
                    focused = None
                if focused is self.file_listbox:
                    return None
            except Exception:
                pass
            # cancel pending jobs to avoid races which can reset selection
            try:
                sel_job = getattr(self, "_selection_commit_job", None)
                if sel_job is not None:
                    try:
                        self.after_cancel(sel_job)
                    except Exception:
                        pass
                    try:
                        self._selection_commit_job = None
                    except Exception:
                        pass
            except Exception:
                pass

            # short-circuit if nav locked
            try:
                if getattr(self, "_nav_locked", False):
                    return "break"
                self._nav_locked = True
                # release lock after short delay
                try:
                    self.after(80, lambda: setattr(self, "_nav_locked", False))
                except Exception:
                    try:
                        self._nav_locked = False
                    except Exception:
                        pass
            except Exception:
                pass
            cur = None
            try:
                sel = self.file_listbox.curselection()
                if sel and len(sel) > 0:
                    cur = sel[0]
            except Exception:
                cur = None
            size = self.file_listbox.size()
            if cur is None:
                new = 0
            else:
                new = min(size - 1, cur + 1)
            self._select_listbox_index(new)
            return "break"
        except Exception:
            return None

    def _on_delete_current(self, event=None):
        """Delete currently selected image (file + sidecar entry) after user confirmation."""
        try:
            sel_idx = self.file_listbox.curselection()
            if not sel_idx or len(sel_idx) == 0:
                return "break"
            sel = self.file_listbox.get(sel_idx[0])
        except Exception:
            return "break"
        # Check user preference: ask before deleting
        try:
            ask = bool(self._load_setting("video_parser_file_delete_ask", True))
        except Exception:
            ask = True

        def _confirm_delete_dialog(filename: str):
            """Show modal dialog asking to delete filename with checkbox 'Don't ask again'.

            Returns tuple (confirmed, dont_ask_again)
            """
            try:
                dlg = tk.Toplevel(self)
                dlg.title("Delete frame")
                dlg.geometry("420x140")
                dlg.resizable(False, False)
                dlg.transient(self)
                dlg.grab_set()
                # message
                lbl = tk.Label(
                    dlg, text=f"Delete {filename}?", anchor="w", justify="left"
                )
                lbl.pack(fill="x", padx=12, pady=(12, 6))
                # checkbox
                dont_var = tk.BooleanVar(value=False)
                cb = tk.Checkbutton(dlg, text="Do not ask again", variable=dont_var)
                cb.pack(anchor="w", padx=12, pady=(0, 12))
                # buttons
                btn_frame = tk.Frame(dlg)
                btn_frame.pack(fill="x", padx=12, pady=(0, 12))
                confirmed = {"v": False}

                def on_ok():
                    confirmed["v"] = True
                    try:
                        dlg.destroy()
                    except Exception:
                        pass

                def on_cancel():
                    confirmed["v"] = False
                    try:
                        dlg.destroy()
                    except Exception:
                        pass

                ok = tk.Button(btn_frame, text="OK", width=10, command=on_ok)
                ok.pack(side="right", padx=(0, 6))
                cancel = tk.Button(
                    btn_frame, text="Cancel", width=10, command=on_cancel
                )
                cancel.pack(side="right")
                # focus
                try:
                    ok.focus_set()
                except Exception:
                    pass
                dlg.wait_window()
                return bool(confirmed["v"]), bool(dont_var.get())
            except Exception:
                return False, False

        if ask:
            try:
                confirmed, dont_ask = _confirm_delete_dialog(sel)
                if dont_ask:
                    try:
                        # save preference: ask -> False
                        self._save_setting("video_parser_file_delete_ask", False)
                    except Exception:
                        pass
                if not confirmed:
                    return "break"
            except Exception:
                return "break"
        # perform deletion
        out_dir = getattr(self, "_current_output_dir", None)
        if not out_dir:
            return "break"
        try:
            # Удаляем файл с диска
            target = Path(out_dir) / sel
            cur_idx = None
            try:
                sel_idx = self.file_listbox.curselection()
                if sel_idx and len(sel_idx) > 0:
                    cur_idx = sel_idx[0]
            except Exception:
                cur_idx = None
            try:
                if target.exists():
                    target.unlink()
            except Exception as e:
                try:
                    logger.error(f"Failed to delete {target}: {e}")
                except Exception:
                    pass
            # Обновляем in-memory список и UI без пересборки из сайдкара
            try:
                if sel in self._files_order:
                    self._files_order.remove(sel)
            except Exception:
                pass
            try:
                # Удалить элемент из Listbox
                if cur_idx is not None:
                    self.file_listbox.delete(cur_idx)
            except Exception:
                pass
            # Рассчитать следующий индекс/имя
            if self._files_order:
                next_idx = (
                    0 if cur_idx is None else min(cur_idx, len(self._files_order) - 1)
                )
                next_name = self._files_order[next_idx]
                try:
                    self.file_listbox.selection_clear(0, tk.END)
                    self.file_listbox.selection_set(next_idx)
                    self.file_listbox.see(next_idx)
                except Exception:
                    pass
                # Обновить память и сайдкар
                self._selected_name = next_name
                self._last_index = next_idx
                try:
                    # Удалить запись о старом файле и выставить selected у нового
                    obj = self._get_cached_sidecar(out_dir, reload=False)
                    files_meta = [
                        f for f in obj.get("files", []) if f.get("relpath") != sel
                    ]
                    for m in files_meta:
                        try:
                            if m.get("relpath") == next_name:
                                m["selected"] = True
                            else:
                                m.pop("selected", None)
                        except Exception:
                            pass
                    obj["files"] = files_meta
                    key = self._get_dataset_key(out_dir)
                    if key:
                        self._parsing_sidecar_cache[key] = obj
                    self._write_sidecar_async(out_dir)
                except Exception:
                    pass
                try:
                    self._update_preview_image()
                except Exception:
                    pass
            else:
                # Нет файлов — очистить состояние и UI
                self._selected_name = None
                self._last_index = 0
                try:
                    obj = self._get_cached_sidecar(out_dir, reload=False)
                    obj["files"] = []
                    key = self._get_dataset_key(out_dir)
                    if key:
                        self._parsing_sidecar_cache[key] = obj
                    self._write_sidecar_async(out_dir)
                except Exception:
                    pass
                try:
                    self.preview_canvas.delete("all")
                    self.preview_meta_label.configure(text="")
                except Exception:
                    pass
            try:
                self._refresh_nav_colors()
            except Exception:
                pass
            try:
                self._update_status()
            except Exception:
                pass
        except Exception:
            pass
        return "break"

    def _update_status(self):
        from pathlib import Path

        video_path = self.file_path_var.get()
        if not video_path or video_path == "(no file)":
            self.status_var.set("No video file selected.")
            return
        out_dir = self._current_output_dir
        if not out_dir or not Path(out_dir).exists():
            self.status_var.set("Dataset folder not found.")
            return
        files = list(Path(out_dir).glob("*.jpg"))
        self.status_var.set(f"Dataset contains {len(files)} files.")

    def _set_status(self, text: str):
        """Safely set status text from any thread via after(0, ...).

        Prefer calling self.after(0, lambda: self._set_status(...)) when updating from a background thread.
        """
        try:
            # running on UI thread: set directly
            self.status_var.set(str(text))
        except Exception:
            try:
                # fallback: schedule on main loop
                self.after(0, lambda: self.status_var.set(str(text)))
            except Exception:
                pass

    # Bottom layout created in __init__

    def _settings_path(self):
        return SETTINGS_PATH

    def _load_setting(self, key, default=None):
        p = self._settings_path()
        if not p.exists():
            return default
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            meta = data.get("meta", {})
            return meta.get(key, default)
        except Exception:
            return default

    def _save_setting(self, key, value):
        """Save a single setting under meta atomically and merge with existing."""
        p = self._settings_path()
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        else:
            data = {}
        meta = data.get("meta", {}) or {}
        meta[key] = value
        data["meta"] = meta
        # atomic write
        try:
            import tempfile

            fd, tmp_path = tempfile.mkstemp(prefix=p.name, dir=str(p.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, str(p))
            finally:
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
        except Exception:
            pass

    def _load_geometry(self):
        p = self._settings_path()
        if not p.exists():
            return "800x800"
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            geom = data.get("video_parser_geometry")
            if geom:
                x = geom.get("x", 100)
                y = geom.get("y", 100)
                w = geom.get("w", 800)
                h = geom.get("h", 800)
                return f"{w}x{h}+{x}+{y}"
        except Exception:
            pass
        return "800x800"

    def _save_geometry(self):
        try:
            self.update_idletasks()
            x = int(self.winfo_x())
            y = int(self.winfo_y())
            w = int(self.winfo_width())
            h = int(self.winfo_height())
            geom = {"x": x, "y": y, "w": w, "h": h}
            p = self._settings_path()
            if p.exists():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    data = {}
            else:
                data = {}
            data["video_parser_geometry"] = geom
            # атомарная запись
            import tempfile

            fd, tmp_path = tempfile.mkstemp(prefix=p.name, dir=str(p.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, str(p))
            finally:
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
        except Exception:
            pass

    def _on_configure(self, event=None):
        self._save_geometry()

    def _on_close(self):
        try:
            # синхронно записать сайдкар при выходе, чтобы не потерять selected/crop
            out_dir = getattr(self, "_current_output_dir", None)
            if out_dir:
                self._write_sidecar_now(Path(out_dir))
        except Exception:
            pass
        self._save_geometry()
        self.destroy()

    def _on_crop_all(self):
        """Batch crop and resize all files listed in sidecar using stored crop settings.

        Rules:
        - Validate crop per file strictly; on any violation log error and skip the file
        - For horizontal images: crop size equals image_h; vertical: equals image_w; square: equals image_w==image_h
        - Offsets must keep the crop square fully inside image
        - x_norm/y_norm must match offsets within small tolerance
        - After crop+resize, update file on disk and update sidecar (width/height/size_bytes/modified)
        - Run in background thread; update status; refresh navigator at the end
        """
        from pathlib import Path
        import threading
        import time

        try:
            out_dir = getattr(self, "_current_output_dir", None)
            if not out_dir or not Path(out_dir).exists():
                try:
                    self._set_status("Dataset folder not found.")
                except Exception:
                    pass
                return
            # Read sidecar from cache
            obj = self._get_cached_sidecar(out_dir, reload=False)
            files_meta = obj.get("files", []) if isinstance(obj, dict) else []
            # Target size from settings (must exist); fallback with error log
            vf = str(self._load_setting("video_frame_size", "1024x1024"))
            try:
                tw, th = [int(x) for x in vf.lower().split("x", 1)]
            except Exception:
                tw, th = 1024, 1024
            if tw != th:
                # enforce square target
                m = min(tw, th)
                tw = th = m

            def validate_crop(meta: dict, img_w: int, img_h: int) -> tuple[bool, str]:
                crop = meta.get("crop") if isinstance(meta, dict) else None
                if not isinstance(crop, dict):
                    return False, "crop missing"
                errs = []
                orientation = crop.get("orientation")
                try:
                    size_px = int(crop.get("size_px", 0))
                    x_px = int(crop.get("x_px", 0))
                    y_px = int(crop.get("y_px", 0))
                    x_norm = float(crop.get("x_norm", 0.0))
                    y_norm = float(crop.get("y_norm", 0.0))
                except Exception:
                    return False, "crop has non-numeric values"

                if img_w <= 0 or img_h <= 0:
                    errs.append("invalid image dimensions")
                # Orientation checks
                if orientation == "horizontal":
                    if not (img_w > img_h):
                        errs.append(
                            f"orientation horizontal but image {img_w}x{img_h} not horizontal"
                        )
                    if size_px != img_h:
                        errs.append(
                            f"size_px must equal image_h ({img_h}) for horizontal, got {size_px}"
                        )
                elif orientation == "vertical":
                    if not (img_h > img_w):
                        errs.append(
                            f"orientation vertical but image {img_w}x{img_h} not vertical"
                        )
                    if size_px != img_w:
                        errs.append(
                            f"size_px must equal image_w ({img_w}) for vertical, got {size_px}"
                        )
                else:  # square
                    if not (img_w == img_h):
                        errs.append(
                            f"orientation square but image not square: {img_w}x{img_h}"
                        )
                    if size_px != img_w:
                        errs.append(
                            f"size_px must equal image side ({img_w}) for square, got {size_px}"
                        )

                # Bounds
                max_x = max(0, img_w - size_px)
                max_y = max(0, img_h - size_px)
                if x_px < 0 or x_px > max_x:
                    errs.append(f"x_px out of bounds (0..{max_x}): {x_px}")
                if y_px < 0 or y_px > max_y:
                    errs.append(f"y_px out of bounds (0..{max_y}): {y_px}")

                # Norm checks (with tolerance)
                tol = 1e-3
                exp_xn = 0.0 if max_x == 0 else x_px / max_x
                exp_yn = 0.0 if max_y == 0 else y_px / max_y
                if abs(exp_xn - x_norm) > tol:
                    errs.append(
                        f"x_norm mismatch: expected {exp_xn:.4f} got {x_norm:.4f}"
                    )
                if abs(exp_yn - y_norm) > tol:
                    errs.append(
                        f"y_norm mismatch: expected {exp_yn:.4f} got {y_norm:.4f}"
                    )

                if errs:
                    return False, "; ".join(errs)
                return True, ""

            def worker():
                from PIL import Image as PILImage

                logger = logging.getLogger("video_parser")
                total = len(files_meta)
                processed = 0
                changed = False
                for meta in files_meta:
                    rel = None
                    try:
                        rel = meta.get("relpath")
                    except Exception:
                        rel = None
                    if not rel:
                        continue
                    img_path = Path(out_dir) / rel
                    if not img_path.exists():
                        try:
                            logger.info(f"[crop] skip missing file: {rel}")
                        except Exception:
                            pass
                        continue
                    crop = meta.get("crop")
                    if not isinstance(crop, dict):
                        try:
                            logger.info(f"[crop] skip without crop: {rel}")
                        except Exception:
                            pass
                        continue
                    # Read image via PIL
                    try:
                        im = PILImage.open(str(img_path)).convert("RGB")
                        img_w, img_h = im.width, im.height
                    except Exception as e:
                        try:
                            logger.error(f"[crop] open failed {rel}: {e}")
                        except Exception:
                            pass
                        continue
                    ok, err = validate_crop(meta, img_w, img_h)
                    if not ok:
                        try:
                            logger.error(f"[crop] invalid crop for {rel}: {err}")
                        except Exception:
                            pass
                        continue
                    # Apply crop
                    try:
                        size_px = int(crop.get("size_px", 0))
                        x_px = int(crop.get("x_px", 0))
                        y_px = int(crop.get("y_px", 0))
                        box = (x_px, y_px, x_px + size_px, y_px + size_px)
                        cropped = im.crop(box)
                        # Resize to target square
                        resample = (
                            LANCZOS_CONST
                            or getattr(PILImage, "LANCZOS", None)
                            or (getattr(PILImage, "BICUBIC", None))
                        )
                        if resample is None:
                            resized = cropped.resize((tw, th))
                        else:
                            resized = cropped.resize((tw, th), resample)
                        # Save back
                        try:
                            resized.save(str(img_path), format="JPEG", quality=92)
                        except Exception:
                            # fallback without params
                            resized.save(str(img_path))
                        # Update sidecar meta and crop for this file
                        try:
                            st = img_path.stat()
                            meta["width"], meta["height"] = tw, th
                            meta["size_bytes"] = st.st_size
                            meta["modified"] = st.st_mtime
                            # обновить crop до нового квадратного состояния
                            meta["crop"] = {
                                "version": 1,
                                "orientation": "square",
                                "image_w": tw,
                                "image_h": th,
                                "size_px": th,
                                "x_px": 0,
                                "y_px": 0,
                                "x_norm": 0.0,
                                "y_norm": 0.0,
                            }
                            changed = True
                        except Exception:
                            pass
                    except Exception as e:
                        try:
                            logger.error(f"[crop] processing failed {rel}: {e}")
                        except Exception:
                            pass
                        continue
                    finally:
                        try:
                            im.close()
                        except Exception:
                            pass
                    processed += 1
                    try:
                        self.after(
                            0,
                            lambda p=processed, t=total: self._set_status(
                                f"Cropping {p}/{t}..."
                            ),
                        )
                    except Exception:
                        pass
                # Write sidecar if changed
                if changed:
                    try:
                        key = self._get_dataset_key(out_dir)
                        if key:
                            self._parsing_sidecar_cache[key] = obj
                        self._write_sidecar_async(out_dir)
                    except Exception:
                        pass
                # Refresh UI
                try:
                    self.after(0, lambda: self._schedule_nav_update())
                except Exception:
                    pass
                try:
                    self.after(0, lambda: self._set_status("Cropping done."))
                except Exception:
                    pass

            # Run in background
            t = threading.Thread(target=worker, daemon=True)
            t.start()
        except Exception:
            pass


# Для отладки: запускать как отдельное окно
if __name__ == "__main__":
    import sys
    import time
    import subprocess
    import logging
    from datetime import datetime
    from pathlib import Path

    args = sys.argv[1:]
    no_watcher = False
    for a in args:
        if a in ("--no-watcher", "--no-reload"):
            no_watcher = True

    repo_root = Path(__file__).resolve().parents[2]
    logs_dir = repo_root / "logs"
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = logs_dir / f"{timestamp}-video_parser.log"

    logger = logging.getLogger("video_parser")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)

    class StreamToLogger:
        def __init__(self, logger, level=logging.INFO):
            self.logger = logger
            self.level = level
            self._buf = ""

        def write(self, message):
            if message.strip() == "":
                return
            for line in message.rstrip().splitlines():
                self.logger.log(self.level, line)

        def flush(self):
            pass

    sys_stdout = sys.stdout
    sys_stderr = sys.stderr
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)

    def gather_py_files(root: Path):
        exclude_parts = (
            "venv",
            ".venv",
            "env",
            "__pycache__",
            "workspace-cache",
            ".git",
            "logs",
            "workspace",
        )
        files = []
        for p in root.rglob("*.py"):
            if any(part in p.parts for part in exclude_parts):
                continue
            files.append(p)
        return files

    def snapshot(files):
        return {str(p): p.stat().st_mtime for p in files}

    child = None
    root = None

    def start_child():
        global child
        cmd = [sys.executable, str(Path(__file__).resolve()), "--no-watcher"]
        child = subprocess.Popen(cmd)
        print(f"Started child pid={child.pid}")

    def stop_child():
        global child
        if child is None:
            return
        try:
            child.terminate()
            child.wait(timeout=5)
        except Exception:
            try:
                child.kill()
            except Exception:
                pass
        print(f"Stopped child pid={child.pid}")
        child = None

    if no_watcher:
        # run app directly (child process)
        try:
            logger.info(f"VideoParser standalone started, logging to {logfile}")
            ctk.set_appearance_mode("dark")
            root = ctk.CTk()
            root.withdraw()
            win = VideoParserWindow(master=root)
            win.mainloop()
        except Exception:
            import traceback

            logger.exception("Unhandled exception in VideoParser standalone")
            traceback.print_exc()
        finally:
            try:
                if root is not None:
                    root.destroy()
            except Exception:
                pass
            sys.stdout = sys_stdout
            sys.stderr = sys_stderr
            logger.info("VideoParser standalone exiting")
    else:
        # supervisor: spawn child and restart on .py changes
        files = gather_py_files(repo_root)
        last_snap = snapshot(files)
        start_child()
        try:
            while True:
                time.sleep(1.0)
                files = gather_py_files(repo_root)
                new_snap = snapshot(files)
                changed = False
                if set(new_snap.keys()) != set(last_snap.keys()):
                    changed = True
                else:
                    for k, v in new_snap.items():
                        if last_snap.get(k) != v:
                            changed = True
                            break
                if changed:
                    print("Change detected in .py files, restarting child...")
                    stop_child()
                    start_child()
                    last_snap = new_snap
                if child is not None:
                    ret = child.poll()
                    if ret is not None:
                        print(f"Child exited with code {ret}, restarting...")
                        start_child()
        except KeyboardInterrupt:
            print("Supervisor exiting on KeyboardInterrupt, stopping child...")
            stop_child()
