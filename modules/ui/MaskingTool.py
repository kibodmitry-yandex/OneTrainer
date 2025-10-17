import threading
import time
import os
import sys
import subprocess
from pathlib import Path
from tkinter import filedialog
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw, ImageOps

import customtkinter as ctk

# When this module is executed directly (python modules/ui/MaskingTool.py)
# the package imports like `modules.*` may fail because the repository root
# isn't on sys.path. Detect that and add the repo root (two parents up) so
# imports resolve the same as when running the app from project root.
try:
    from modules.util.ui import components
except ModuleNotFoundError:
    import sys
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from modules.util.ui import components


class MaskingTool(ctk.CTkToplevel):
    """Lightweight Masking Tool: navigator + square editor.

    Layout:
    - header (controls)
    - workspace split: navigator (left) + editor (right, square)
    """

    SUPPORTED_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff")

    def __init__(self, master):
        super().__init__(master)
        self.title("Masking Tool")
        # default geometry (may be overridden by saved settings)
        self.geometry("1000x700")


        self.dataset_dir = None
        self.originals: list[str] = []
        self.original_to_mask: dict[str, str | None] = {}
        self._preview_image = None
        # store last selected original (full path) to persist selection across restarts
        self._last_selected_file: str | None = None
        # for debounced atomic settings save
        self._settings_save_after_id = None
        # input holders for dataset folder UI (may be entry or clickable link)
        self.folder_entry = None
        self.folder_link = None
        # mask display style: 0=black (30%), 1=red (30%), 2=inverted+hue->green (30%)
        self.mask_style = 0
        self.style_buttons = []
        self._style_images_refs = []
        # mask file watcher
        self._mask_watch_thread = None
        self._mask_watch_stop_event = threading.Event()
        self._mask_watch_target = None
        self._mask_watch_last_mtime = None

        self._build_ui()

        # remember last appearance mode so we can refresh icons when theme changes
        try:
            self._last_appearance_mode = ctk.get_appearance_mode()
            # start a lightweight poll to detect theme changes and refresh header icons
            try:
                self.after(500, lambda: self._poll_appearance_mode())
            except Exception:
                pass
        except Exception:
            self._last_appearance_mode = None

        # after UI built and tasks processed, attempt to restore saved geometry
        try:
            self.update_idletasks()
        except Exception:
            pass
        try:
            self._load_window_settings()
        except Exception:
            pass

    def _build_ui(self):
        # top-level: workspace row + controls row
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)

        # navigator (left)
        left_frame = ctk.CTkFrame(self, corner_radius=5)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        left_frame.grid_rowconfigure(0, weight=0)
        left_frame.grid_rowconfigure(1, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)

        components.label(left_frame, 0, 0, "Files")
        self.originals_list = tk.Listbox(left_frame)
        self.originals_list.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)

        # editor (right) — square canvas
        editor_frame = ctk.CTkFrame(self, corner_radius=5)
        editor_frame.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)
        editor_frame.grid_rowconfigure(0, weight=1)
        editor_frame.grid_columnconfigure(0, weight=1)

        # header for editor: label + mask mini-preview on the right
        header = ctk.CTkFrame(editor_frame, fg_color="transparent")
        header.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        header.grid_columnconfigure(0, weight=1)
        # label without extra padding so preview can occupy full header height
        try:
            title_label = ctk.CTkLabel(header, text="Editor")
            title_label.grid(row=0, column=0, padx=0, pady=0, sticky="nw")
        except Exception:
            components.label(header, 0, 0, "Editor")
        # preview is a label (no padding) so it can be sized exactly
        self.mask_preview_btn = ctk.CTkLabel(header, text="", image=None)
        self.mask_preview_btn.grid(row=0, column=1, padx=0, pady=0, sticky="nsew")
        # style selector frame (three small buttons) placed to the right
        try:
            header.grid_columnconfigure(2, weight=0)
            style_frame = ctk.CTkFrame(header, fg_color="transparent")
            style_frame.grid(row=0, column=2, padx=0, pady=0, sticky='nsew')
            # create three label-buttons for styles; images are generated on header resize
            for i in range(3):
                lbl = ctk.CTkLabel(style_frame, text="", image=None, fg_color='transparent', cursor='hand2')
                lbl.grid(row=0, column=i, padx=2, pady=0)
                try:
                    lbl.bind('<Button-1>', (lambda ii: (lambda e: self._set_mask_style(ii)))(i))
                except Exception:
                    pass
                self.style_buttons.append(lbl)
                self._style_images_refs.append(None)
        except Exception:
            # if CTkLayout fails, ignore — style buttons are optional
            pass
        # keep explicit reference to header frame and image to avoid relying on event attributes
        self._header_frame = header
        self.mask_preview_btn_image_ref = None
        # allow header to notify when resized so we can make preview exactly header height
        try:
            header.bind('<Configure>', lambda e: self._on_header_configure(e))
        except Exception:
            pass
        try:
            # bind click to preview: create mask if absent
            try:
                self.mask_preview_btn.bind('<Button-1>', lambda e: self._on_mask_preview_click())
            except Exception:
                # CTkLabel may not support bind in some versions; attempt widget.bind
                try:
                    self.mask_preview_btn.bind('<Button-1>', lambda e: self._on_mask_preview_click())
                except Exception:
                    pass
        except Exception:
            pass

        self.editor_canvas = tk.Canvas(editor_frame, bg="black", highlightthickness=0)
        self.editor_canvas.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)

        # controls (header)
        controls = ctk.CTkFrame(self, corner_radius=0)
        controls.grid(row=1, column=0, columnspan=2, sticky="ew", padx=6, pady=(0,6))
        controls.grid_columnconfigure(0, weight=0)
        controls.grid_columnconfigure(1, weight=1)

        components.label(controls, 0, 0, "Dataset folder:")
        ui_state = getattr(self.master, "ui_state", None)
        if ui_state is not None:
            # integrated mode: use shared ui_state entry component
            self.folder_entry = components.entry(controls, 0, 1, ui_state, "__masking_tool_folder__")
        else:
            # standalone mode: show a clickable link label instead of an editable entry
            # this avoids relying on external ui_state and gives a simple, robust UX
            self.folder_entry = None
            try:
                # clickable label that opens the folder in system explorer
                self.folder_link = ctk.CTkLabel(controls, text='(no folder)', cursor='hand2', fg_color='transparent')
                self.folder_link.grid(row=0, column=1, padx=6, pady=6, sticky='w')
                try:
                    self.folder_link.bind('<Button-1>', lambda e: self._open_folder_in_explorer())
                except Exception:
                    # some CTk versions may not support bind; ignore if so
                    pass
            except Exception:
                # fallback to a non-clickable entry if label creation fails for any reason
                try:
                    self.folder_entry = ctk.CTkEntry(controls, width=400)
                    self.folder_entry.grid(row=0, column=1, padx=6, pady=6, sticky='new')
                except Exception:
                    # last-resort: create a plain tk.Entry
                    try:
                        self.folder_entry = tk.Entry(controls, width=50)
                        self.folder_entry.grid(row=0, column=1, padx=6, pady=6, sticky='new')
                    except Exception:
                        # give up quietly
                        self.folder_entry = None

        browse_btn = ctk.CTkButton(controls, text="Browse", command=self._pick_folder)
        browse_btn.grid(row=0, column=2, padx=6, pady=6)

        # interactions
        self.originals_list.bind("<<ListboxSelect>>", self._on_list_select)
        # keep editor square on resize and save geometry debounced
        self.bind("<Configure>", self._on_configure)
        self.after(100, self._adjust_editor_size)

        # ensure we save window settings on close
        try:
            self.protocol('WM_DELETE_WINDOW', self._on_close)
        except Exception:
            pass

    # --- folder link helpers ---
    def _set_folder_text(self, path: str | None):
        try:
            if path is None:
                text = '(no folder)'
            else:
                text = str(path)
            if getattr(self, 'folder_link', None) is not None:
                try:
                    self.folder_link.configure(text=text)
                except Exception:
                    pass
            elif getattr(self, 'folder_entry', None) is not None:
                try:
                    # try delete/insert API
                    delete = getattr(self.folder_entry, 'delete', None)
                    insert = getattr(self.folder_entry, 'insert', None)
                    if callable(delete) and callable(insert):
                        try:
                            delete(0, 'end')
                            insert(0, text)
                        except Exception:
                            pass
                    else:
                        # try textvariable or cget
                        try:
                            var = getattr(self.folder_entry, '_textvariable', None)
                            if var is not None:
                                try:
                                    var.set(text)
                                except Exception:
                                    pass
                            else:
                                try:
                                    self.folder_entry.configure(text=text)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            pass

    def _get_folder_text(self) -> str | None:
        try:
            if getattr(self, 'folder_link', None) is not None:
                t = self.folder_link.cget('text') or None
                if t == '(no folder)':
                    return None
                return t
            if getattr(self, 'folder_entry', None) is not None:
                try:
                    get = getattr(self.folder_entry, 'get', None)
                    if callable(get):
                        try:
                            return str(get())
                        except Exception:
                            pass
                    # fallback: try reading cget('text')
                    cget = getattr(self.folder_entry, 'cget', None)
                    if callable(cget):
                        try:
                            return str(cget('text'))
                        except Exception:
                            pass
                except Exception:
                    return None
        except Exception:
            return None

    def _open_folder_in_explorer(self):
        try:
            p = self._get_folder_text()
            if not p:
                return
            # cross-platform open
            try:
                if os.name == 'nt':
                    os.startfile(p)
                elif sys.platform == 'darwin':
                    subprocess.Popen(['open', p])
                else:
                    subprocess.Popen(['xdg-open', p])
            except Exception:
                # fallback: try using explorer on Windows explicitly
                try:
                    os.startfile(p)
                except Exception:
                    pass
        except Exception:
            pass

    # --- window settings persistence ---
    def _settings_path(self) -> Path:
        # store per-repo in workspace folder
        repo_root = Path(__file__).resolve().parents[2]
        workspace_dir = repo_root / 'workspace'
        try:
            workspace_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return workspace_dir / 'masking_tool_settings.json'

    def _schedule_save_settings(self, delay_ms: int = 500):
        # debounce saves to avoid excessive disk I/O
        try:
            if self._settings_save_after_id is not None:
                try:
                    self.after_cancel(self._settings_save_after_id)
                except Exception:
                    pass
            self._settings_save_after_id = self.after(delay_ms, self._save_window_settings_atomic)
        except Exception:
            # best-effort; ignore scheduling errors
            try:
                self._save_window_settings_atomic()
            except Exception:
                pass

    def _load_window_settings(self):
        import json
        p = self._settings_path()
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            return

        # geometry expected as dict: {x:int,y:int,w:int,h:int}
        # initialize defaults in case geometry block is absent or malformed
        x = 0
        y = 0
        w = 1000
        h = 700
        geom = data.get('geometry')
        if geom:
            try:
                x = int(geom.get('x', x))
                y = int(geom.get('y', y))
                w = int(geom.get('w', w))
                h = int(geom.get('h', h))
            except Exception:
                # ignore geometry errors but continue to restore other settings
                x = max(0, x)
                y = max(0, y)
                w = max(1000, w)
                h = max(700, h)

        # enforce minimums
        min_w, min_h = 800, 600
        w = max(w, min_w)
        h = max(h, min_h)

        # ensure window is not placed off-screen (left/top not negative)
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()

        x = max(0, min(x, max(0, screen_w - 50)))
        y = max(0, min(y, max(0, screen_h - 50)))

        # additionally clamp size to screen size
        w = min(w, screen_w)
        h = min(h, screen_h)

        try:
            # apply geometry
            self.geometry(f"{w}x{h}+{x}+{y}")
        except Exception:
            pass

        # restore dataset folder if present (support legacy key and new meta namespace)
        try:
            meta = data.get('meta') or {}
            dataset_dir = data.get('dataset_dir') or data.get('dataset') or meta.get('dataset_dir')
            # restore mask style if present
            try:
                ms = int(meta.get('mask_style', 0))
                self.mask_style = ms if ms in (0, 1, 2) else 0
            except Exception:
                self.mask_style = 0
            # restore last selected file if present
            try:
                last_file = meta.get('last_file')
                if last_file:
                    # store for use after directory scan finishes
                    try:
                        self._last_selected_file = str(last_file)
                    except Exception:
                        self._last_selected_file = None
                else:
                    self._last_selected_file = None
            except Exception:
                self._last_selected_file = None
            if dataset_dir:
                # set entry text (if entry exists) and trigger a scan in background
                try:
                    # prefer Path checks for cross-platform correctness
                    from pathlib import Path as _Path
                    if _Path(dataset_dir).is_dir():
                        try:
                            self._set_folder_text(dataset_dir)
                            # remember in-memory
                            try:
                                self.dataset_dir = str(dataset_dir)
                            except Exception:
                                pass
                        except Exception:
                            pass
                        # schedule scan after short delay to allow UI to settle
                        try:
                            threading.Thread(target=self._scan_directory, args=(dataset_dir,), daemon=True).start()
                        except Exception:
                            pass
                    else:
                        # still set the entry even if directory missing
                        try:
                            self._set_folder_text(dataset_dir)
                            try:
                                self.dataset_dir = str(dataset_dir)
                            except Exception:
                                pass
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            pass

    def _on_configure(self, event=None):
        # adjust editor square size immediately
        try:
            self._adjust_editor_size()
        except Exception:
            pass
        # schedule debounced settings save
        try:
            self._schedule_save_settings()
        except Exception:
            pass

    def _save_window_settings_atomic(self):
        """Atomically write current window geometry to settings JSON.

        Use root coordinates (including window decorations) and actual
        reported width/height. Writes via a temporary file and os.replace.
        """
        import json
        import os
        import tempfile

        try:
            # make sure geometry info is up-to-date
            try:
                self.update_idletasks()
            except Exception:
                pass

            # prefer client coords (winfo_x/winfo_y) to avoid accumulating
            # decoration offsets when saving/restoring. Using root coords
            # previously caused the window to shift by the decoration size
            # on every restart.
            try:
                x = int(self.winfo_x())
                y = int(self.winfo_y())
                w = int(self.winfo_width())
                h = int(self.winfo_height())
            except Exception:
                # fallback to geometry parsing
                geo = str(self.geometry() or '')
                parts = geo.split('+')
                size = parts[0] if parts else ''
                pos = parts[1:] if len(parts) > 1 else []
                w, h = (size.split('x') + [None])[:2]
                x = pos[0] if len(pos) > 0 else '0'
                y = pos[1] if len(pos) > 1 else '0'
                w = int(w) if w is not None else 1000
                h = int(h) if h is not None else 700
                x = int(x)
                y = int(y)

            # enforce minima and clamp to screen
            w = max(800, w)
            h = max(600, h)
            screen_w = self.winfo_screenwidth()
            screen_h = self.winfo_screenheight()
            w = min(w, screen_w)
            h = min(h, screen_h)

            x = max(0, min(x, max(0, screen_w - 50)))
            y = max(0, min(y, max(0, screen_h - 50)))

            geometry_dict = {'x': x, 'y': y, 'w': w, 'h': h}

            # also persist currently selected dataset folder (if any)
            folder_val = None
            try:
                folder_val = self._get_folder_text()
            except Exception:
                folder_val = None

            # construct final save dict explicitly to avoid type inference issues
            try:
                meta_block = {}
                if folder_val:
                    meta_block['dataset_dir'] = str(folder_val)
                # include mask style
                try:
                    meta_block['mask_style'] = int(getattr(self, 'mask_style', 0))
                except Exception:
                    pass
                # include last selected file (full path) if available
                try:
                    lf = getattr(self, '_last_selected_file', None)
                    if lf:
                        meta_block['last_file'] = str(lf)
                except Exception:
                    pass
                if meta_block:
                    final_save = {'geometry': geometry_dict, 'meta': meta_block}
                else:
                    final_save = {'geometry': geometry_dict}
            except Exception:
                # fallback
                final_save = {'geometry': geometry_dict}

            p = self._settings_path()
            # ensure directory exists
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

            # write to temp file then atomically replace
            try:
                fd, tmp_path = tempfile.mkstemp(prefix=p.name, dir=str(p.parent))
                try:
                    with os.fdopen(fd, 'w', encoding='utf-8') as f:
                        json.dump(final_save, f, indent=2, ensure_ascii=False)
                        f.flush()
                        os.fsync(f.fileno())
                    # atomic replace
                    os.replace(tmp_path, str(p))
                finally:
                    # if tmp still exists, remove
                    try:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                    except Exception:
                        pass
            except Exception:
                # fallback simple write
                try:
                    p.write_text(json.dumps(final_save, indent=2), encoding='utf-8')
                except Exception:
                    pass
        except Exception:
            # swallow any exception to avoid crashing UI
            pass

    def _on_close(self):
        try:
            # ensure latest geometry is saved atomically
            self._save_window_settings_atomic()
        except Exception:
            pass
        try:
            self._stop_mask_watcher()
        except Exception:
            pass
        try:
            self.destroy()
        except Exception:
            pass

    def _pick_folder(self):
        dir_path = filedialog.askdirectory()
        if not dir_path:
            return
        try:
            self._set_folder_text(dir_path)
        except Exception:
            pass
        # remember selected folder in memory
        try:
            self.dataset_dir = str(dir_path)
        except Exception:
            pass

        # schedule a debounced save of window+folder settings and also try immediate atomic save
        try:
            self._schedule_save_settings()
            try:
                # attempt immediate atomic save to reduce chance of loss
                self._save_window_settings_atomic()
            except Exception:
                pass
        except Exception:
            pass

        t = threading.Thread(target=self._scan_directory, args=(dir_path,), daemon=True)
        t.start()

    def _scan_directory(self, dir_path: str):
        # Build a file database according to dataloader conventions.
        # Rules taken from DataLoaderText2ImageMixin:
        # - mask postfixes: '-masklabel', '-condlabel' (ModifyPath uses postfixes)
        # - sample prompt sidecars: same stem + '.txt'
        # - manifests: .json, .jsonl

        mask_postfixes = ["-masklabel", "-condlabel"]

        p = Path(dir_path)
        all_files = [x for x in p.rglob("*") if x.is_file()]

        # classify by extension
        image_exts = set(ext.lower() for ext in [*"".split()])
        try:
            from modules.util.path_util import supported_image_extensions
            image_exts = {e.lower() for e in supported_image_extensions()}
        except Exception:
            image_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff'}

        images = [f for f in all_files if f.suffix.lower() in image_exts]
        texts = [f for f in all_files if f.suffix.lower() == '.txt']
        manifests = [f for f in all_files if f.suffix.lower() in ('.json', '.jsonl')]
        others = [f for f in all_files if f not in images and f not in texts and f not in manifests]

        # split images into originals and masks
        originals = []
        masks = []
        for im in images:
            stem = im.stem
            is_mask = False
            for postfix in mask_postfixes:
                if stem.endswith(postfix):
                    is_mask = True
                    break
            if is_mask:
                masks.append(im)
            else:
                originals.append(im)

        # map stems to files for quick lookup
        masks_by_stem = {m.stem: m for m in masks}
        texts_by_stem = {t.stem: t for t in texts}
        manifests_names = [m.name for m in manifests]

        db_rows = []
        unmatched = set([f.name for f in others])

        sorted_originals = sorted(originals, key=lambda x: x.name.lower())
        # reset mapping
        self.original_to_mask = {}
        for orig in sorted_originals:
            o_stem = orig.stem
            # look for mask files that are exactly orig_stem + postfix
            found_mask = None
            for postfix in mask_postfixes:
                candidate = o_stem + postfix
                if candidate in masks_by_stem:
                    found_mask = masks_by_stem[candidate]
                    break

            # look for text sidecar
            text_file = texts_by_stem.get(o_stem)

            db_rows.append((orig.name, found_mask.name if found_mask is not None else '', text_file.name if text_file is not None else ''))

            # remember mapping from original full path to mask full path (or None)
            try:
                self.original_to_mask[str(orig)] = str(found_mask) if found_mask is not None else None
            except Exception:
                pass

            # remove matched items from unmatched set
            if found_mask is not None and found_mask.name in unmatched:
                unmatched.discard(found_mask.name)
            if text_file is not None and text_file.name in unmatched:
                unmatched.discard(text_file.name)

        # any masks not matched to originals are unmatched (but list their names)
        for m in masks:
            if not any(m.name == row[1] for row in db_rows):
                unmatched.add(m.name)

        # any text sidecars not matched to originals
        for t in texts:
            if not any(t.name == row[2] for row in db_rows):
                unmatched.add(t.name)

        # any manifest files not already listed are separate
        manifests_list = [m.name for m in manifests]

        # save to memory (also populate self.originals with full paths for UI)
        self.originals = [str(p) for p in sorted_originals]
        self.db = {
            'rows': db_rows,
            'manifests': manifests_list,
            'unmatched': sorted(list(unmatched)),
        }

        # update UI on main thread and print summary table to log
        try:
            self.after(0, lambda: (self._update_lists(), self._log_database_summary()))
        except Exception:
            self._update_lists()
            self._log_database_summary()

    def _log_database_summary(self):
        # print a simple table: Original | Mask | Text
        try:
            print("\nMaskingTool: dataset summary")
            print("Original\tMask\tText")
            for orig, mask, text in self.db.get('rows', []):
                print(f"{orig}\t{mask or '-'}\t{text or '-'}")

            print("\nManifests:")
            for m in self.db.get('manifests', []):
                print(f"  {m}")

            print("\nUnmatched files:")
            for u in self.db.get('unmatched', []):
                print(f"  {u}")
            print("\n")
        except Exception:
            pass

    def _update_lists(self):
        self.originals_list.delete(0, "end")
        for o in self.originals:
            self.originals_list.insert("end", Path(o).name)
        if self.originals:
            # attempt to restore last selected file if it was recorded and still exists
            try:
                last = getattr(self, '_last_selected_file', None)
                if last and last in self.originals:
                    try:
                        idx = self.originals.index(last)
                        self.originals_list.selection_set(idx)
                        self._show_preview_on_canvas(self.originals[idx])
                        try:
                            self._update_mask_preview(self.originals[idx])
                        except Exception:
                            pass
                    except Exception:
                        # fallback to first
                        self.originals_list.selection_set(0)
                        self._show_preview_on_canvas(self.originals[0])
                        try:
                            self._update_mask_preview(self.originals[0])
                        except Exception:
                            pass
                else:
                    # default to first and persist that choice
                    self.originals_list.selection_set(0)
                    self._show_preview_on_canvas(self.originals[0])
                    try:
                        self._update_mask_preview(self.originals[0])
                    except Exception:
                        pass
                    try:
                        self._last_selected_file = self.originals[0]
                        # save immediately so it persists across crashes
                        try:
                            self._save_window_settings_atomic()
                        except Exception:
                            pass
                    except Exception:
                        pass
            except Exception:
                pass

    def _on_list_select(self, _evt=None):
        try:
            sel = self.originals_list.curselection()
            if not sel:
                return
            idx = sel[0]
            path = self.originals[idx]
            self._show_preview_on_canvas(path)
            try:
                self._update_mask_preview(path)
            except Exception:
                pass

            # persist this selection immediately
            try:
                self._last_selected_file = path
                try:
                    self._save_window_settings_atomic()
                except Exception:
                    pass
            except Exception:
                pass
        except Exception:
            pass

    def _update_mask_preview(self, original_path: str):
        """Update the mini mask preview button to show a thumbnail of the mask (if present),
        otherwise show a black square with a red crossed X. If a size is stored in
        self._mask_preview_target_size it will be used as the exact square size.
        """
        try:
            # determine size: prefer explicit target set by header configure
            size = getattr(self, '_mask_preview_target_size', None) or 72
            size = int(max(8, size))
            mask_path = self.original_to_mask.get(original_path)
            if mask_path and Path(mask_path).is_file():
                img = Image.open(mask_path).convert('RGBA')
                # create square thumbnail of exact size
                img.thumbnail((size, size), Image.Resampling.LANCZOS)
                # ensure square canvas
                canvas = Image.new('RGBA', (size, size), (0, 0, 0, 0))
                x = (size - img.width) // 2
                y = (size - img.height) // 2
                canvas.paste(img, (x, y))
                img = canvas
            else:
                # create default black with red cross exactly size x size
                img = Image.new('RGBA', (size, size), 'black')
                draw = ImageDraw.Draw(img)
                # draw red cross
                draw.line((0, 0, size, size), fill=(200, 30, 30, 255), width=max(1, size//20))
                draw.line((0, size, size, 0), fill=(200, 30, 30, 255), width=max(1, size//20))

            self._mask_preview_image = ImageTk.PhotoImage(img)
            try:
                self.mask_preview_btn.configure(image=self._mask_preview_image, text="")
            except Exception:
                # fallback: set image attribute directly
                try:
                    # keep a reference on self to avoid GC
                    self.mask_preview_btn_image_ref = self._mask_preview_image
                except Exception:
                    pass
        except Exception:
            pass

    def _start_mask_watcher(self, mask_path: str):
        try:
            mask_path = str(mask_path)
            # if already watching same file, leave running
            if getattr(self, '_mask_watch_target', None) == mask_path and getattr(self, '_mask_watch_thread', None) is not None and getattr(self, '_mask_watch_thread').is_alive():
                return
            # stop previous
            try:
                self._stop_mask_watcher()
            except Exception:
                pass
            self._mask_watch_target = mask_path
            try:
                self._mask_watch_last_mtime = os.path.getmtime(mask_path)
            except Exception:
                self._mask_watch_last_mtime = None
            self._mask_watch_stop_event.clear()

            def _watch_loop():
                while not self._mask_watch_stop_event.is_set():
                    try:
                        if not getattr(self, '_mask_watch_target', None):
                            break
                        try:
                            target = getattr(self, '_mask_watch_target', None)
                            if target is None:
                                m = None
                            else:
                                m = os.path.getmtime(target)
                        except Exception:
                            m = None
                        if m is not None and self._mask_watch_last_mtime is not None and m != self._mask_watch_last_mtime:
                            self._mask_watch_last_mtime = m
                            try:
                                target_copy = str(getattr(self, '_mask_watch_target', ''))
                                self.after(50, lambda p=target_copy: self._update_mask_preview_for_path(p))
                            except Exception:
                                pass
                        elif m is not None and self._mask_watch_last_mtime is None:
                            self._mask_watch_last_mtime = m
                        time.sleep(0.5)
                    except Exception:
                        time.sleep(1.0)

            t = threading.Thread(target=_watch_loop, daemon=True)
            self._mask_watch_thread = t
            t.start()
        except Exception:
            pass

    def _stop_mask_watcher(self):
        try:
            self._mask_watch_target = None
            self._mask_watch_last_mtime = None
            self._mask_watch_stop_event.set()
            if getattr(self, '_mask_watch_thread', None) is not None:
                try:
                    th = self._mask_watch_thread
                    if th is not None:
                        th.join(timeout=0.5)
                except Exception:
                    pass
            self._mask_watch_thread = None
            self._mask_watch_stop_event.clear()
        except Exception:
            pass

    def _update_mask_preview_for_path(self, mask_path: str):
        try:
            for orig, mp in list(self.original_to_mask.items()):
                try:
                    if mp == mask_path:
                        sel = self.originals_list.curselection()
                        if sel:
                            idx = sel[0]
                            cur = self.originals[idx]
                            if cur == orig:
                                self._update_mask_preview(orig)
                                return
                except Exception:
                    pass
        except Exception:
            pass

    def _on_mask_preview_click(self):
        """If mask exists: do nothing. If no mask, create a greyscale black mask with same geometry
        as currently selected original, save it as <stem>-masklabel.png, update in-memory DB and preview.
        """
        try:
            sel = self.originals_list.curselection()
            if not sel:
                return
            idx = sel[0]
            orig_path = self.originals[idx]
            mask_path = self.original_to_mask.get(orig_path)
            if mask_path and Path(mask_path).is_file():
                # mask already exists — nothing to do
                return

            # create mask file alongside original
            try:
                orig_p = Path(orig_path)
                stem = orig_p.stem
                mask_filename = stem + '-masklabel.png'
                mask_file = orig_p.with_name(mask_filename)
                # open original to get size
                img = Image.open(orig_path)
                w, h = img.size
                # create grayscale black image
                mask_img = Image.new('L', (w, h), 0)
                # save as PNG
                mask_img.save(mask_file)
                mask_path = str(mask_file)
            except Exception:
                return

            # update in-memory structures: original_to_mask and self.db rows
            try:
                self.original_to_mask[orig_path] = mask_path
            except Exception:
                pass

            try:
                # update db rows: find row with orig name and set mask column
                rows = self.db.get('rows', [])
                new_rows = []
                for r in rows:
                    if r[0] == Path(orig_path).name:
                        new_rows.append((r[0], Path(mask_path).name, r[2]))
                    else:
                        new_rows.append(r)
                self.db['rows'] = new_rows
            except Exception:
                pass

            # refresh preview and lists
            try:
                self._update_mask_preview(orig_path)
            except Exception:
                pass
            try:
                # ensure the file is included in unmatched/manifests logic if needed by a full rescan
                # we'll also trigger a lightweight UI update
                self._update_lists()
            except Exception:
                pass
        except Exception:
            pass

    def _on_header_configure(self, event=None):
        # Simplified and robust header configure handler to avoid nested try/except mismatches
        try:
            # compute target size from event or header frame
            h = getattr(self, '_mask_preview_target_size', None)
            if event is not None and getattr(event, 'height', None) is not None:
                try:
                    h = int(event.height)
                except Exception:
                    pass
            if h is None:
                try:
                    h = int(self._header_frame.winfo_height())
                except Exception:
                    h = getattr(self, '_mask_preview_target_size', 72)

            self._mask_preview_target_size = max(8, int(h))

            # update current preview
            try:
                sel = self.originals_list.curselection()
                if sel:
                    idx = sel[0]
                    path = self.originals[idx]
                    self._update_mask_preview(path)
                elif self.originals:
                    self._update_mask_preview(self.originals[0])
            except Exception:
                pass

            # regenerate style thumbnails sized to header height
            try:
                size = max(8, int(self._mask_preview_target_size))
                icon_size = size
                for i, btn in enumerate(getattr(self, 'style_buttons', [])):
                    try:
                        if i == 0:
                            img = Image.new('RGBA', (icon_size, icon_size), (0, 0, 0, int(255 * 0.3)))
                        elif i == 1:
                            img = Image.new('RGBA', (icon_size, icon_size), (200, 30, 30, int(255 * 0.3)))
                        else:
                            img = Image.new('RGBA', (icon_size, icon_size), (60, 180, 90, int(255 * 0.3)))

                        tkimg = ImageTk.PhotoImage(img)
                        try:
                            btn.configure(image=tkimg, text="")
                        except Exception:
                            pass
                        # keep reference
                        try:
                            self._style_images_refs[i] = tkimg
                        except Exception:
                            pass

                        # overlay checkmark on active style
                        try:
                            if getattr(self, 'mask_style', 0) == i:
                                # choose checkmark color contrasting the current CTk appearance
                                try:
                                    mode = ctk.get_appearance_mode()
                                except Exception:
                                    mode = self._last_appearance_mode or 'Dark'
                                if str(mode).lower().startswith('dark'):
                                    check_fill = (255, 255, 255, 255)
                                else:
                                    check_fill = (30, 30, 30, 255)

                                ck = Image.new('RGBA', (icon_size, icon_size), (0, 0, 0, 0))
                                d = ImageDraw.Draw(ck)
                                w = max(1, icon_size // 10)
                                d.line((icon_size * 0.2, icon_size * 0.55, icon_size * 0.45, icon_size * 0.8), fill=check_fill, width=w)
                                d.line((icon_size * 0.45, icon_size * 0.8, icon_size * 0.85, icon_size * 0.2), fill=check_fill, width=w)
                                combined = img.copy()
                                combined.paste(ck, (0, 0), ck)
                                tkc = ImageTk.PhotoImage(combined)
                                try:
                                    btn.configure(image=tkc)
                                except Exception:
                                    pass
                                try:
                                    self._style_images_refs[i] = tkc
                                except Exception:
                                    pass
                        except Exception:
                            pass
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception:
            pass

    def _poll_appearance_mode(self):
        """Lightweight poll to detect CTk appearance mode changes and refresh header icons."""
        try:
            try:
                mode = ctk.get_appearance_mode()
            except Exception:
                mode = None
            if mode is not None and mode != getattr(self, '_last_appearance_mode', None):
                try:
                    self._last_appearance_mode = mode
                    # force regenerate of header icons
                    try:
                        self._on_header_configure(None)
                    except Exception:
                        pass
                except Exception:
                    pass
        except Exception:
            pass
        try:
            # reschedule
            self.after(500, lambda: self._poll_appearance_mode())
        except Exception:
            pass

    def _adjust_editor_size(self):
        try:
            h = max(64, self.winfo_height() - 120)
            self.grid_columnconfigure(1, minsize=h)
            self.editor_canvas.config(width=h, height=h)
            if self._preview_image:
                sel = self.originals_list.curselection()
                if sel:
                    idx = sel[0]
                    self._show_preview_on_canvas(self.originals[idx])
        except Exception:
            pass

    def _set_mask_style(self, style_index: int):
        try:
            style_index = int(style_index)
        except Exception:
            return
        if style_index not in (0,1,2):
            return
        try:
            self.mask_style = style_index
        except Exception:
            pass
        # immediate save of settings
        try:
            self._save_window_settings_atomic()
        except Exception:
            pass
        # refresh header icons (so the checkmark updates immediately)
        try:
            try:
                self._on_header_configure(None)
            except Exception:
                pass
        except Exception:
            pass
        # refresh preview for currently selected original
        try:
            sel = self.originals_list.curselection()
            if sel:
                idx = sel[0]
                self._show_preview_on_canvas(self.originals[idx])
                self._update_mask_preview(self.originals[idx])
        except Exception:
            pass

    def _show_preview_on_canvas(self, path: str):
        try:
            # load original and prepare thumbnail that fits the editor canvas
            img = Image.open(path).convert('RGBA')
            cw = int(self.editor_canvas.cget('width'))
            ch = int(self.editor_canvas.cget('height'))
            size = max(10, min(cw, ch))
            img.thumbnail((size, size), Image.Resampling.LANCZOS)

            # create a full-size canvas image and paste the thumbnail centered
            canvas_img = Image.new('RGBA', (cw, ch), (0, 0, 0, 0))
            x = (cw - img.width) // 2
            y = (ch - img.height) // 2
            canvas_img.paste(img, (x, y), img)

            # if a mask exists for this original, load it, resize to thumbnail size
            # and overlay it so that white -> black opaque, black -> transparent
            try:
                mask_path = self.original_to_mask.get(path)
                if mask_path and Path(mask_path).is_file():
                    m = Image.open(mask_path).convert('L')
                    # resize mask to match thumbnail dimensions
                    m.thumbnail((img.width, img.height), Image.Resampling.LANCZOS)
                    try:
                        # build alpha at 30% of mask intensity
                        try:
                            alpha = m.point(lambda p: int(p * 0.50))
                        except Exception:
                            # fallback to simple scaling
                            alpha = Image.eval(m, lambda p: int(p * 0.50))

                        # choose overlay color per style
                        style = getattr(self, 'mask_style', 0) or 0
                        if style == 0:
                            color = (0, 0, 0)
                            final_alpha = alpha
                        elif style == 1:
                            color = (200, 30, 30)
                            final_alpha = alpha
                        else:
                            # style 2: green tint — use same alpha mapping as other styles (no inversion)
                            # so white -> opaque, black -> transparent, but color is green
                            final_alpha = alpha
                            color = (60, 180, 90)

                        overlay = Image.new('RGBA', m.size, color + (0,))
                        overlay.putalpha(final_alpha)
                        canvas_img.paste(overlay, (x, y), overlay)
                    except Exception:
                        # ignore mask overlay failures, still show original
                        pass
            except Exception:
                # ignore mask overlay failures, still show original
                pass

            self._preview_image = ImageTk.PhotoImage(canvas_img)
            self.editor_canvas.delete("all")
            cx = cw // 2
            cy = ch // 2
            self.editor_canvas.create_image(cx, cy, image=self._preview_image, anchor='center')
        except Exception as e:
            try:
                self.editor_canvas.delete("all")
                self.editor_canvas.create_text(10, 10, text=f"Error loading image: {e}", anchor='nw', fill='white')
            except Exception:
                pass


def _run_standalone(initial_dir: str | None = None):
    """Run MaskingTool in isolation for development.

    initial_dir: optional path to pre-open (useful for testing)
    """
    import logging
    import sys
    from datetime import datetime
    # prepare logfile
    repo_root = Path(__file__).resolve().parents[2]
    logs_dir = repo_root / 'logs'
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logfile = logs_dir / f"{timestamp}-masking_tool.log"

    # configure a basic logger that writes to both console and file
    logger = logging.getLogger('masking_tool')
    logger.setLevel(logging.DEBUG)
    # avoid duplicate handlers on repeated calls
    if not logger.handlers:
        fh = logging.FileHandler(logfile, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)

    # redirect stdout and stderr to logger
    class StreamToLogger:
        def __init__(self, logger, level=logging.INFO):
            self.logger = logger
            self.level = level
            self._buf = ''

        def write(self, message):
            # buffer partial lines
            if message.strip() == '':
                return
            for line in message.rstrip().splitlines():
                self.logger.log(self.level, line)

        def flush(self):
            pass

    sys_stdout = sys.stdout
    sys_stderr = sys.stderr
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)

    root = ctk.CTk()
    root.withdraw()
    try:
        logger.info(f"MaskingTool standalone started, logging to {logfile}")
        window = MaskingTool(root)
        if initial_dir:
            # schedule scan after window is shown
            window.after(100, lambda: window._scan_directory(initial_dir))
        window.mainloop()
    except Exception:
        import traceback
        logger.exception("Unhandled exception in MaskingTool standalone")
        traceback.print_exc()
    finally:
        try:
            root.destroy()
        except Exception:
            pass
        # restore stdout/stderr
        sys.stdout = sys_stdout
        sys.stderr = sys_stderr
        logger.info("MaskingTool standalone exiting")


if __name__ == '__main__':
    import sys
    import time
    import subprocess

    # simple arg parsing: --no-watcher runs the app directly; otherwise run supervisor
    args = sys.argv[1:]
    no_watcher = False
    initial_dir = None
    for a in args:
        if a in ('--no-watcher', '--no-reload'):
            no_watcher = True
        else:
            initial_dir = a

    if no_watcher:
        # run app directly (child process)
        _run_standalone(initial_dir)
    else:
        # supervisor: spawn child and restart on .py changes
        repo_root = Path(__file__).resolve().parents[2]

        def gather_py_files(root: Path):
            exclude_parts = ('venv', '.venv', 'env', '__pycache__', 'workspace-cache', '.git', 'logs', 'workspace')
            files = []
            for p in root.rglob('*.py'):
                if any(part in p.parts for part in exclude_parts):
                    continue
                files.append(p)
            return files

        def snapshot(files):
            return {str(p): p.stat().st_mtime for p in files}

        child = None

        def start_child():
            global child
            cmd = [sys.executable, str(Path(__file__).resolve()), '--no-watcher']
            if initial_dir:
                cmd.append(initial_dir)
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

        # initial snapshot
        files = gather_py_files(repo_root)
        last_snap = snapshot(files)

        start_child()

        try:
            while True:
                time.sleep(1.0)
                files = gather_py_files(repo_root)
                new_snap = snapshot(files)
                # detect added/removed/changed
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

                # reap child if it exited unexpectedly
                if child is not None:
                    ret = child.poll()
                    if ret is not None:
                        print(f"Child exited with code {ret}, restarting...")
                        start_child()

        except KeyboardInterrupt:
            print("Supervisor exiting on KeyboardInterrupt, stopping child...")
            stop_child()
