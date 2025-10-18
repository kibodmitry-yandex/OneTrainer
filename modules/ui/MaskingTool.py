import threading
import time
import os
import sys
import subprocess
from pathlib import Path
from tkinter import filedialog
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw, ImageOps
from PIL import ImageFilter, ImageChops
from io import BytesIO
from customtkinter import CTkImage

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


class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event):
        if self.tooltip:
            return
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry("+{}+{}".format(event.x_root + 10, event.y_root + 10))
        label = tk.Label(self.tooltip, text=self.text, background="yellow", relief="solid", borderwidth=1)
        label.pack()

    def hide_tooltip(self, event):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None


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
        self._cached_resized_img = None
        self._cached_img_params = None
        # cached original image size (width, height) corresponding to _cached_resized_img
        self._cached_original_size = None
        self._cached_overlay = None
        self._cached_overlay_params = None
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
        self.active_tool = None
        self.brush_size = 10
        self.brush_softness = 0.5
        self.eraser_size = 10
        self.eraser_softness = 0.5
        self.panel_x = 100
        self.panel_y = 100
        self.drawing_toolbar = None
        self.size_slider = None
        self.softness_slider = None
        self.tool_buttons = {}
        self.tool_tips = {}
        # per-mask undo/redo stacks (store PIL.Image copies)
        self._undo_stacks: dict[str, list[Image.Image]] = {}
        self._redo_stacks: dict[str, list[Image.Image]] = {}
        self.zoom_scale = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.pan_start_x = None
        self.pan_start_y = None
        self.handler = None
        self._drag_after_id = None
        # drawing state
        self._drawing = False
        # map mask_path -> in-memory PIL.Image (mode 'L') for live-preview while editing
        self._in_memory_mask_overrides: dict[str, Image.Image] = {}
        # current mask being edited (full path)
        self._current_edit_mask_path: str | None = None
        # last point in mask-image coordinates for stroke continuity
        self._last_draw_point: tuple[int, int] | None = None
        # stamp cache: key -> (diameter, softness, mode) -> PIL.Image (L)
        self._stamp_cache: dict[tuple[int, float, str], Image.Image] = {}
        # marker (cursor preview) state
        self._marker_image_id = None
        self._marker_tkimage = None
        # cache for resized stamp previews keyed by (diameter, softness, mode, preview_size)
        self._marker_stamp_cache = {}
        # spline tool state (in canvas coordinates)
        self._spline_points: list[dict] = []  # [{'x':float,'y':float,'smooth':bool}]
        self._spline_closed: bool = False
        self._spline_drag_index: int | None = None
        self._spline_canvas_ids: list[int] = []
        # enable debug printing for spline flows (temporary; can be disabled)
        # disabled by default
        self._spline_debug: bool = False
        self._build_ui()

        # bind hotkeys
        self.bind_all('<Key>', self._on_key_press)
        # ensure Enter (Return) is always delivered to our handler even if
        # generic Key bindings are missed in some platform/widget combos
        try:
            self.bind_all('<Return>', self._on_key_press)
            self.bind_all('<KP_Enter>', self._on_key_press)
        except Exception:
            pass
        # track Shift state reliably (fallback for platforms where event.state on Return may not include modifiers)
        try:
            self._shift_down = False
            self.bind_all('<KeyPress-Shift_L>', lambda e: setattr(self, '_shift_down', True))
            self.bind_all('<KeyRelease-Shift_L>', lambda e: setattr(self, '_shift_down', False))
            self.bind_all('<KeyPress-Shift_R>', lambda e: setattr(self, '_shift_down', True))
            self.bind_all('<KeyRelease-Shift_R>', lambda e: setattr(self, '_shift_down', False))
        except Exception:
            self._shift_down = False

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
        self._build_drawing_toolbar()

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

        # ensure canvas receives key events when user interacts with spline tool
        try:
            # bind Enter keys directly on canvas in addition to global bindings
            self.editor_canvas.bind('<Return>', lambda e: self._on_key_press(e))
            self.editor_canvas.bind('<KP_Enter>', lambda e: self._on_key_press(e))
        except Exception:
            pass

        # bind canvas for panning and drawing
        try:
            self.editor_canvas.bind('<Button-1>', self._on_canvas_press)
            self.editor_canvas.bind('<B1-Motion>', self._on_canvas_motion)
            self.editor_canvas.bind('<ButtonRelease-1>', self._on_canvas_release)
            # mouse move/leave for marker preview
            self.editor_canvas.bind('<Motion>', self._on_canvas_mouse_move)
            self.editor_canvas.bind('<Leave>', self._on_canvas_leave)
            # mouse wheel bindings:
            # - plain wheel: zoom centered at cursor
            # - Ctrl+wheel: adjust brush/eraser size
            # - Shift+wheel: adjust softness
            # Windows/macOS: <MouseWheel>; X11: <Button-4>/<Button-5>
            self.editor_canvas.bind('<MouseWheel>', self._on_canvas_zoom)
            self.editor_canvas.bind('<Control-MouseWheel>', self._on_canvas_adjust_size)
            self.editor_canvas.bind('<Shift-MouseWheel>', self._on_canvas_adjust_softness)
            # X11 support (Button-4 = up, Button-5 = down)
            self.editor_canvas.bind('<Button-4>', lambda e: self._on_canvas_zoom(e))
            self.editor_canvas.bind('<Button-5>', lambda e: self._on_canvas_zoom(e))
            self.editor_canvas.bind('<Control-Button-4>', lambda e: self._on_canvas_adjust_size(e))
            self.editor_canvas.bind('<Control-Button-5>', lambda e: self._on_canvas_adjust_size(e))
            self.editor_canvas.bind('<Shift-Button-4>', lambda e: self._on_canvas_adjust_softness(e))
            self.editor_canvas.bind('<Shift-Button-5>', lambda e: self._on_canvas_adjust_softness(e))
        except Exception:
            pass

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
            try:
                self.active_tool = meta.get('active_tool')
                self.brush_size = float(meta.get('brush_size', 10))
                self.brush_softness = float(meta.get('brush_softness', 0.5))
                self.eraser_size = float(meta.get('eraser_size', 10))
                self.eraser_softness = float(meta.get('eraser_softness', 0.5))
                self.panel_x = int(meta.get('panel_x', 100))
                self.panel_y = int(meta.get('panel_y', 100))
            except Exception:
                pass
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
                try:
                    meta_block['active_tool'] = self.active_tool
                    meta_block['brush_size'] = self.brush_size
                    meta_block['brush_softness'] = self.brush_softness
                    meta_block['eraser_size'] = self.eraser_size
                    meta_block['eraser_softness'] = self.eraser_softness
                    meta_block['panel_x'] = self.panel_x
                    meta_block['panel_y'] = self.panel_y
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
            # clear in-memory edits and undo/redo for previous selection so
            # memory persists only until switching images
            try:
                prev = getattr(self, '_last_selected_file', None)
                if prev and prev != self.originals[sel[0]]:
                    try:
                        prev_mask = self.original_to_mask.get(prev)
                        if prev_mask:
                            # drop in-memory override for previous mask
                            self._in_memory_mask_overrides.pop(prev_mask, None)
                            # clear stacks for previous mask
                            self._undo_stacks.pop(prev_mask, None)
                            self._redo_stacks.pop(prev_mask, None)
                    except Exception:
                        pass
            except Exception:
                pass
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
            # prefer in-memory override for thumbnail so undo/redo updates appear immediately
            img = None
            try:
                if mask_path and mask_path in getattr(self, '_in_memory_mask_overrides', {}):
                    m = self._in_memory_mask_overrides.get(mask_path)
                    if isinstance(m, Image.Image):
                        try:
                            img = m.convert('RGBA')
                        except Exception:
                            img = m.copy().convert('RGBA')
            except Exception:
                img = None
            if img is None:
                if mask_path and Path(mask_path).is_file():
                    img = Image.open(mask_path).convert('RGBA')
                else:
                    img = None
            if img is not None:
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

            self._mask_preview_image = CTkImage(img, size=(size, size))
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

                        tkimg = CTkImage(img, size=(icon_size, icon_size))
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
                                tkc = CTkImage(combined, size=(icon_size, icon_size))
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
        if not path or not Path(path).is_file():
            self.editor_canvas.delete("all")
            self.editor_canvas.create_text(10, 10, text="No image selected or file not found", anchor='nw', fill='white')
            return
        try:
            cw = int(self.editor_canvas.cget('width'))
            ch = int(self.editor_canvas.cget('height'))
            size = int(max(10, min(cw, ch)) * self.zoom_scale)

            # cache resized image
            image_mtime = Path(path).stat().st_mtime
            img_params = (path, image_mtime, size)
            if self._cached_img_params == img_params and self._cached_resized_img is not None:
                img = self._cached_resized_img
            else:
                img = Image.open(path).convert('RGBA')
                img = img.resize((size, size), Image.Resampling.LANCZOS)
                self._cached_resized_img = img
                self._cached_img_params = img_params
                # remember original size corresponding to this resized preview
                try:
                    orig = Image.open(path)
                    self._cached_original_size = orig.size
                except Exception:
                    self._cached_original_size = (img.width, img.height)

            # create a full-size canvas image and paste the thumbnail centered
            canvas_img = Image.new('RGBA', (cw, ch), (0, 0, 0, 0))
            scaled_size = img.width
            base_x = (cw - scaled_size) // 2
            base_y = (ch - scaled_size) // 2
            x = base_x + self.pan_offset_x
            y = base_y + self.pan_offset_y
            # clamp to prevent image from going outside canvas bounds
            x = max(cw - scaled_size, min(0, x))
            y = max(ch - scaled_size, min(0, y))
            # update pan_offset to clamped values
            self.pan_offset_x = x - base_x
            self.pan_offset_y = y - base_y
            canvas_img.paste(img, (x, y), img)

            # cache overlay
            mask_path = self.original_to_mask.get(path)
            mask_mtime = None
            if mask_path and Path(mask_path).is_file():
                mask_mtime = Path(mask_path).stat().st_mtime
            overlay_params = (mask_path, mask_mtime, size, self.mask_style)
            if self._cached_overlay_params == overlay_params and self._cached_overlay is not None:
                overlay = self._cached_overlay
            else:
                overlay = None
                if mask_path and Path(mask_path).is_file():
                    # prefer in-memory override if user is actively editing the mask
                    if mask_path in getattr(self, '_in_memory_mask_overrides', {}):
                        try:
                            m = self._in_memory_mask_overrides[mask_path]
                        except Exception:
                            m = Image.open(mask_path).convert('L')
                    else:
                        m = Image.open(mask_path).convert('L')
                    m = m.resize((size, size), Image.Resampling.LANCZOS)
                    try:
                        # build alpha at 30% of mask intensity
                        try:
                            alpha = m.point(lambda p: int(p * 0.50))
                        except Exception:
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
                            final_alpha = alpha
                            color = (60, 180, 90)

                        overlay = Image.new('RGBA', m.size, color + (0,))
                        overlay.putalpha(final_alpha)
                    except Exception:
                        pass
                self._cached_overlay = overlay
                self._cached_overlay_params = overlay_params

            if overlay:
                canvas_img.paste(overlay, (x, y), overlay)

            self._preview_image = ImageTk.PhotoImage(canvas_img)
            # remove only previous preview image(s) so transient marker images are preserved
            try:
                self.editor_canvas.delete('preview')
            except Exception:
                try:
                    self.editor_canvas.delete("all")
                except Exception:
                    pass
            cx = cw // 2
            cy = ch // 2
            # create preview image with tag so we can refresh it without removing marker
            try:
                self.editor_canvas.create_image(cx, cy, image=self._preview_image, anchor='center', tags=('preview',))
            except Exception:
                # fallback without tags
                self.editor_canvas.create_image(cx, cy, image=self._preview_image, anchor='center')
        except Exception as e:
            try:
                try:
                    self.editor_canvas.delete('preview')
                except Exception:
                    try:
                        self.editor_canvas.delete("all")
                    except Exception:
                        pass
                self.editor_canvas.create_text(10, 10, text=f"Error loading image: {e}", anchor='nw', fill='white', tags=('preview',))
            except Exception:
                pass

    def _build_drawing_toolbar(self):
        # create toolbar frame
        self.drawing_toolbar = ctk.CTkFrame(self, corner_radius=0, fg_color='gray20')
        self.drawing_toolbar.place(x=self.panel_x, y=self.panel_y)
        self.drawing_toolbar.configure(width=40)
        # handler
        self.handler = ctk.CTkFrame(self.drawing_toolbar, height=10, fg_color='gray30')
        self.handler.pack()
        self.handler.configure(width=40)
        self.handler.bind('<Button-1>', self._start_drag)
        self.handler.bind('<B1-Motion>', self._drag)
        # buttons frame
        buttons_frame = ctk.CTkFrame(self.drawing_toolbar, fg_color='transparent')
        buttons_frame.pack()
        buttons_frame.configure(width=40)
        # load icons
        self._load_icons()
        # buttons
        tools = [
            'zoom',
            'brush',
            'eraser',
            'spline',
            'undo',
            'redo',
            'invert_mask',
            'delete_mask'
        ]
        for tool in tools:
            tooltips = {
                'zoom': 'Zoom In/Out',
                'brush': 'Brush Tool',
                'eraser': 'Eraser Tool',
                'spline': 'Spline Tool',
                'undo': 'Undo',
                'redo': 'Redo',
                'invert_mask': 'Invert Mask',
                'delete_mask': 'Delete Mask'
            }
            if tool == 'zoom':
                btn = ctk.CTkButton(buttons_frame, width=40, height=40, text="", image=self.tool_icons[tool])
                btn.bind('<Button-1>', self._on_zoom_click)
                btn.bind('<Motion>', self._on_zoom_motion)
            elif tool == 'undo':
                btn = ctk.CTkButton(buttons_frame, width=40, height=40, text="", image=self.tool_icons[tool], command=self._undo)
            elif tool == 'redo':
                btn = ctk.CTkButton(buttons_frame, width=40, height=40, text="", image=self.tool_icons[tool], command=self._redo)
            elif tool == 'invert_mask':
                btn = ctk.CTkButton(buttons_frame, width=40, height=40, text="", image=self.tool_icons[tool], command=self._invert_mask)
            elif tool == 'delete_mask':
                btn = ctk.CTkButton(buttons_frame, width=40, height=40, text="", image=self.tool_icons[tool], command=self._delete_mask)
            else:
                btn = ctk.CTkButton(buttons_frame, width=40, height=40, text="", image=self.tool_icons[tool], command=lambda t=tool: self._set_active_tool(t))
            btn.pack(pady=2)
            tooltip = ToolTip(btn, tooltips.get(tool, ''))
            self.tool_tips[tool] = tooltip
            self.tool_buttons[tool] = btn
        # set initial active
        if self.active_tool:
            self._set_active_tool(self.active_tool, save=False)

    def _load_icons(self):
        from PIL import Image, ImageDraw
        self.tool_icons = {}
        icon_size = 24
        for tool in ['zoom', 'brush', 'eraser', 'spline', 'undo', 'redo', 'delete_mask']:
            img = Image.new('RGBA', (icon_size, icon_size), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            if tool == 'zoom':
                # Magnifying glass
                draw.ellipse((2, 2, 18, 18), outline='white', width=2)
                draw.line((14, 14, 20, 20), fill='white', width=2)
            elif tool == 'brush':
                # Brush circle
                draw.ellipse((4, 4, 20, 20), fill='white')
            elif tool == 'eraser':
                # Eraser square
                draw.rectangle((4, 4, 20, 20), fill='white')
            elif tool == 'spline':
                # Curve line
                draw.arc((2, 2, 22, 22), start=0, end=180, fill='white', width=2)
            elif tool == 'undo':
                # Left arrow
                draw.polygon([(12, 4), (4, 12), (12, 20), (12, 16), (20, 16), (20, 8), (12, 8)], fill='white')
            elif tool == 'redo':
                # Right arrow
                draw.polygon([(12, 4), (20, 12), (12, 20), (12, 16), (4, 16), (4, 8), (12, 8)], fill='white')
            elif tool == 'delete_mask':
                # Trash can
                draw.rectangle((8, 4, 16, 18), fill='white')
                draw.rectangle((6, 18, 18, 20), fill='white')
                draw.line((10, 6, 14, 6), fill='black', width=1)
            self.tool_icons[tool] = CTkImage(img, size=(icon_size, icon_size))
        # add invert_mask icon
        img = Image.new('RGBA', (icon_size, icon_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        # left half dark, right half light
        draw.rectangle((0, 0, icon_size//2, icon_size), fill=(40, 40, 40, 255))
        draw.rectangle((icon_size//2, 0, icon_size, icon_size), fill=(220, 220, 220, 255))
        draw.line((icon_size//2, 2, icon_size//2, icon_size-2), fill=(120, 120, 120, 255), width=1)
        # small arc arrow to indicate invert
        try:
            cx = icon_size * 3 // 4
            cy = icon_size // 4
            r = icon_size // 6
            draw.arc((cx - r, cy - r, cx + r, cy + r), start=200, end=20, fill=(10, 100, 180, 255), width=2)
            draw.polygon((cx + r - 1, cy, cx + r - 4, cy - 3, cx + r + 2, cy - 2), fill=(10, 100, 180, 255))
        except Exception:
            pass
        self.tool_icons['invert_mask'] = CTkImage(img, size=(icon_size, icon_size))

    def _on_zoom_click(self, event):
        width = event.widget.winfo_width() // 2
        if event.x < width:
            self._zoom_out()  # левая - уменьшить
        else:
            self._zoom_in()   # правая - увеличить

    def _on_zoom_motion(self, event):
        tooltip = self.tool_tips.get('zoom')
        if tooltip:
            width = event.widget.winfo_width() // 2
            if event.x < width:
                tooltip.text = "Zoom Out"  # левая - уменьшить
            else:
                tooltip.text = "Zoom In"   # правая - увеличить
            if tooltip.tooltip:
                for child in tooltip.tooltip.winfo_children():
                    if isinstance(child, tk.Label):
                        child.configure(text=tooltip.text)
                        break

    def _zoom_in(self):
        old_scale = self.zoom_scale
        self.zoom_scale = min(5.0, self.zoom_scale * 1.2)
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        if self.zoom_scale != old_scale:
            self._update_current_preview()

    def _zoom_out(self):
        old_scale = self.zoom_scale
        self.zoom_scale = max(1.0, self.zoom_scale / 1.2)
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        if self.zoom_scale != old_scale:
            self._update_current_preview()

    def _update_current_preview(self):
        try:
            sel = self.originals_list.curselection()
            if sel:
                idx = sel[0]
                path = self.originals[idx]
                self._show_preview_on_canvas(path)
        except Exception:
            pass

    def _regenerate_overlay_from_inmemory(self, mask_path: str, size: int):
        """Generate and cache overlay (RGBA) from an in-memory mask for given preview size.
        This avoids repeatedly opening/resizing mask files during drawing.
        """
        try:
            if not mask_path:
                return
            # find in-memory mask, fall back to on-disk
            m_img = None
            try:
                m_img = self._in_memory_mask_overrides.get(mask_path)
            except Exception:
                m_img = None
            if m_img is None:
                if Path(mask_path).is_file():
                    try:
                        m_img = Image.open(mask_path).convert('L')
                    except Exception:
                        m_img = None

            if m_img is None:
                # clear cache for this overlay
                if getattr(self, '_cached_overlay_params', None) and self._cached_overlay_params[0] == mask_path:
                    self._cached_overlay = None
                    self._cached_overlay_params = None
                return

            # resize mask to preview size
            try:
                m = m_img.resize((size, size), Image.Resampling.LANCZOS)
            except Exception:
                m = m_img.copy()
                m = m.resize((size, size))

            # build alpha at 30% of mask intensity
            try:
                alpha = m.point(lambda p: int(p * 0.50))
            except Exception:
                alpha = Image.eval(m, lambda p: int(p * 0.50))

            style = getattr(self, 'mask_style', 0) or 0
            if style == 0:
                color = (0, 0, 0)
            elif style == 1:
                color = (200, 30, 30)
            else:
                color = (60, 180, 90)

            overlay = Image.new('RGBA', (size, size), color + (0,))
            overlay.putalpha(alpha)

            self._cached_overlay = overlay
            # store params (mask_path, mtime can be None for in-memory)
            try:
                mtime = None
                if Path(mask_path).is_file():
                    mtime = Path(mask_path).stat().st_mtime
            except Exception:
                mtime = None
            self._cached_overlay_params = (mask_path, mtime, size, self.mask_style)
        except Exception:
            pass

    # --- Painting helpers ---
    def _canvas_to_image_coords(self, cx: int, cy: int, orig_path: str) -> tuple[int, int] | None:
        """Convert canvas coordinates (cx,cy) to original image pixel coordinates for orig_path.
        Returns (ix,iy) in original image space or None if outside displayed image.
        """
        try:
            cw = int(self.editor_canvas.cget('width'))
            ch = int(self.editor_canvas.cget('height'))
            size = int(max(10, min(cw, ch)) * self.zoom_scale)
            # top-left of displayed image (as in _show_preview_on_canvas)
            base_x = (cw - size) // 2
            base_y = (ch - size) // 2
            x = base_x + self.pan_offset_x
            y = base_y + self.pan_offset_y
            rel_x = cx - x
            rel_y = cy - y
            if rel_x < 0 or rel_y < 0 or rel_x >= size or rel_y >= size:
                return None
            # map to original image size
            try:
                img = Image.open(orig_path)
                ow, oh = img.size
            except Exception:
                return None
            scale_x = ow / float(size)
            scale_y = oh / float(size)
            ix = int(rel_x * scale_x)
            iy = int(rel_y * scale_y)
            ix = max(0, min(ow - 1, ix))
            iy = max(0, min(oh - 1, iy))
            return (ix, iy)
        except Exception:
            return None

    def _on_canvas_mouse_move(self, event):
        """Show an inverted-color circular marker centered at cursor using cached preview.
        The marker is generated by sampling the cached resized preview under the stamp
        and inverting those pixels within the stamp shape so it remains visible.
        """
        # store last known canvas cursor position and render marker there
        try:
            cx = int(event.x)
            cy = int(event.y)
            self._last_cursor_canvas_pos = (cx, cy)
            self._cursor_on_canvas = True
        except Exception:
            self._last_cursor_canvas_pos = None
            self._cursor_on_canvas = False
        try:
            # Only show marker when brush or eraser is active
            if self.active_tool in ('brush', 'eraser'):
                self._render_marker_at_canvas(event.x, event.y)
            else:
                # remove any existing marker/outline when not in brush mode
                try:
                    if getattr(self, '_marker_image_id', None) is not None:
                        try:
                            self.editor_canvas.delete(self._marker_image_id)
                        except Exception:
                            pass
                        self._marker_image_id = None
                    try:
                        self.editor_canvas.delete('marker_outline')
                    except Exception:
                        pass
                except Exception:
                    pass
        except Exception:
            pass

    def _on_canvas_leave(self, event=None):
        # keep marker visible when cursor leaves canvas (so sliders/toolbars don't hide it)
        try:
            # simply mark that cursor is not on canvas but do not delete marker
            self._cursor_on_canvas = False
        except Exception:
            pass

    def _on_canvas_wheel(self, event):
        """Mouse wheel: change current tool size; Shift+wheel changes softness."""
        try:
            # Determine delta: on Windows event.delta is multiple of 120
            delta = 0
            try:
                delta = int(getattr(event, 'delta', 0))
            except Exception:
                # older X11 bindings use num/button
                if getattr(event, 'num', None) == 4:
                    delta = 120
                elif getattr(event, 'num', None) == 5:
                    delta = -120

            step_size = 1
            # when Shift is pressed, adjust softness in small increments
            if (event.state & 0x0001) != 0 or getattr(event, 'keysym', '') == 'Shift_L' or getattr(event, 'keysym', '') == 'Shift_R':
                # softness step
                amt = 0.05 if delta > 0 else -0.05
                # decide which tool
                if self.active_tool == 'eraser':
                    self.eraser_softness = max(0.0, min(1.0, self.eraser_softness + amt))
                    try:
                        if getattr(self, 'softness_slider', None) is not None:
                            self.softness_slider.set(self.eraser_softness if self.active_tool == 'eraser' else self.brush_softness)
                    except Exception:
                        pass
                else:
                    self.brush_softness = max(0.0, min(1.0, self.brush_softness + amt))
                    try:
                        if getattr(self, 'softness_slider', None) is not None:
                            self.softness_slider.set(self.brush_softness)
                    except Exception:
                        pass
            else:
                # size step depends on delta direction
                amt = 1 if delta > 0 else -1
                if self.active_tool == 'eraser':
                    self.eraser_size = max(1, min(200, int(self.eraser_size + amt)))
                    try:
                        if getattr(self, 'size_slider', None) is not None:
                            self.size_slider.set(self.eraser_size)
                    except Exception:
                        pass
                else:
                    self.brush_size = max(1, min(200, int(self.brush_size + amt)))
                    try:
                        if getattr(self, 'size_slider', None) is not None:
                            self.size_slider.set(self.brush_size)
                    except Exception:
                        pass

            # refresh marker to reflect changes
            try:
                if getattr(self, '_last_cursor_canvas_pos', None) is not None:
                    cx, cy = self._last_cursor_canvas_pos
                    self._render_marker_at_canvas(cx, cy)
            except Exception:
                pass
        except Exception:
            pass

    # --- Spline helpers ---
    def _render_spline_preview(self):
        """Render spline preview (points and smoothed curve) on the canvas."""
        try:
            # Only show spline when spline tool is active
            if getattr(self, 'active_tool', None) != 'spline':
                try:
                    for cid in list(getattr(self, '_spline_canvas_ids', []) or []):
                        try:
                            self.editor_canvas.delete(cid)
                        except Exception:
                            pass
                    self._spline_canvas_ids = []
                except Exception:
                    pass
                return
            # remove previous ids
            try:
                for cid in list(getattr(self, '_spline_canvas_ids', []) or []):
                    try:
                        self.editor_canvas.delete(cid)
                    except Exception:
                        pass
                self._spline_canvas_ids = []
            except Exception:
                pass

            pts = self._spline_points
            if not pts:
                return

            # draw points
            for i, p in enumerate(pts):
                x = p['x']
                y = p['y']
                r = 4
                cid = self.editor_canvas.create_oval(x-r, y-r, x+r, y+r, fill='white' if not p.get('smooth') else 'yellow', outline='black')
                self._spline_canvas_ids.append(cid)

            # collect ids of smooth curves so we can raise them above straight segments after drawing
            smooth_curve_ids = []

            # Draw segments so that the curve passes through points marked as smooth.
            # Approach: find contiguous runs of smooth interior points and draw a single
            # smooth polyline through start..end (including surrounding endpoints). Any
            # edges not covered by smooth runs are drawn as straight lines.
            n = len(pts)
            all_smooth_flag = False
            debug = getattr(self, '_spline_debug', False)
            covered_edges = set()

            def add_edge(a_idx):
                # mark edge a_idx -> a_idx+1 (mod n for closed)
                if self._spline_closed:
                    covered_edges.add(a_idx % n)
                else:
                    covered_edges.add(a_idx)

            # Helper to collect coords from idx_start to idx_end inclusive (wrap if closed)
            def collect_coords(start, end):
                coords = []
                if self._spline_closed:
                    i = start
                    # walk circularly from start to end inclusive
                    while True:
                        p = pts[i % n]
                        coords.extend((p['x'], p['y']))
                        if (i % n) == (end % n):
                            break
                        i += 1
                else:
                    for ii in range(start, end + 1):
                        p = pts[ii]
                        coords.extend((p['x'], p['y']))
                return coords

            # find runs of consecutive smooth interior points and draw Catmull-Rom
            def sample_catmull(points_xy, samples_per_seg=12):
                # points_xy: list of (x,y) through which the curve should pass
                # returns flat list of coords sampled along Catmull-Rom
                if not points_xy or len(points_xy) < 2:
                    return []
                # Use centripetal Catmull-Rom parameterization to avoid cusps
                sampled = []
                m = len(points_xy)

                def dist(a, b):
                    dx = a[0] - b[0]
                    dy = a[1] - b[1]
                    return (dx*dx + dy*dy) ** 0.5

                # parametrize points with centripetal scheme (alpha=0.5)
                alpha = 0.5
                def tj(ti, pi, pj):
                    return ti + (dist(pi, pj) ** alpha)

                for i in range(m - 1):
                    # determine control points p0,p1,p2,p3 with clamped endpoints
                    p1 = points_xy[i]
                    p2 = points_xy[i+1]
                    p0 = points_xy[i-1] if i-1 >= 0 else p1
                    p3 = points_xy[i+2] if i+2 < m else p2

                    t0 = 0.0
                    t1 = tj(t0, p0, p1)
                    t2 = tj(t1, p1, p2)
                    t3 = tj(t2, p2, p3)

                    # sample parameter u between t1..t2
                    for s in range(samples_per_seg):
                        u = t1 + (s / float(samples_per_seg)) * (t2 - t1)
                        # basis functions of centripetal Catmull-Rom (using barycentric interpolation)
                        A1x = (t1 - u) / (t1 - t0) * p0[0] + (u - t0) / (t1 - t0) * p1[0] if (t1 - t0) != 0 else p1[0]
                        A1y = (t1 - u) / (t1 - t0) * p0[1] + (u - t0) / (t1 - t0) * p1[1] if (t1 - t0) != 0 else p1[1]
                        A2x = (t2 - u) / (t2 - t1) * p1[0] + (u - t1) / (t2 - t1) * p2[0] if (t2 - t1) != 0 else p2[0]
                        A2y = (t2 - u) / (t2 - t1) * p1[1] + (u - t1) / (t2 - t1) * p2[1] if (t2 - t1) != 0 else p2[1]
                        A3x = (t3 - u) / (t3 - t2) * p2[0] + (u - t2) / (t3 - t2) * p3[0] if (t3 - t2) != 0 else p3[0]
                        A3y = (t3 - u) / (t3 - t2) * p2[1] + (u - t2) / (t3 - t2) * p3[1] if (t3 - t2) != 0 else p3[1]

                        B1x = (t2 - u) / (t2 - t0) * A1x + (u - t0) / (t2 - t0) * A2x if (t2 - t0) != 0 else A2x
                        B1y = (t2 - u) / (t2 - t0) * A1y + (u - t0) / (t2 - t0) * A2y if (t2 - t0) != 0 else A2y
                        B2x = (t3 - u) / (t3 - t1) * A2x + (u - t1) / (t3 - t1) * A3x if (t3 - t1) != 0 else A3x
                        B2y = (t3 - u) / (t3 - t1) * A2y + (u - t1) / (t3 - t1) * A3y if (t3 - t1) != 0 else A3y

                        Cx = (t2 - u) / (t2 - t1) * B1x + (u - t1) / (t2 - t1) * B2x if (t2 - t1) != 0 else B2x
                        Cy = (t2 - u) / (t2 - t1) * B1y + (u - t1) / (t2 - t1) * B2y if (t2 - t1) != 0 else B2y
                        sampled.extend((Cx, Cy))
                # append final point explicitly
                sampled.extend((points_xy[-1][0], points_xy[-1][1]))
                return sampled

            if n >= 2:
                if self._spline_closed:
                    # treat closed: linearize the circular array to avoid splitting runs
                    smooth_flags = [bool(p.get('smooth')) for p in pts]
                    # no smooth points -> nothing to do
                    if not any(smooth_flags):
                        pass
                    else:
                        # if all points are smooth, draw a single Catmull-Rom through all
                        if all(smooth_flags):
                            # For a closed spline where every point is smooth we must
                            # sample the closed Catmull-Rom curve without duplicating
                            # the wrap-around segment. The previous approach built a
                            # sequence with repeated indices ([n-1,0,...,n-1,0]) and
                            # could produce overlapping segments for last->first.
                            # Implement a closed sampler that samples each segment
                            # p_i -> p_{i+1} using circular indexing and concatenates
                            # samples without duplicating the seam.
                            def sample_catmull_closed(points_xy, samples_per_seg=16):
                                m = len(points_xy)
                                if m < 2:
                                    return []
                                sampled = []

                                def dist(a, b):
                                    dx = a[0] - b[0]
                                    dy = a[1] - b[1]
                                    return (dx*dx + dy*dy) ** 0.5

                                alpha = 0.5
                                def tj(ti, pi, pj):
                                    return ti + (dist(pi, pj) ** alpha)

                                # iterate over each segment p1->p2 (i .. i+1) and sample it
                                for i_seg in range(m):
                                    p0 = points_xy[(i_seg - 1) % m]
                                    p1 = points_xy[i_seg]
                                    p2 = points_xy[(i_seg + 1) % m]
                                    p3 = points_xy[(i_seg + 2) % m]

                                    t0 = 0.0
                                    t1 = tj(t0, p0, p1)
                                    t2 = tj(t1, p1, p2)
                                    t3 = tj(t2, p2, p3)

                                    # sample samples_per_seg points along the interval [t1, t2)
                                    # (exclude the endpoint to avoid duplication between segments)
                                    for s in range(samples_per_seg):
                                        u = t1 + (s / float(samples_per_seg)) * (t2 - t1)
                                        A1x = (t1 - u) / (t1 - t0) * p0[0] + (u - t0) / (t1 - t0) * p1[0] if (t1 - t0) != 0 else p1[0]
                                        A1y = (t1 - u) / (t1 - t0) * p0[1] + (u - t0) / (t1 - t0) * p1[1] if (t1 - t0) != 0 else p1[1]
                                        A2x = (t2 - u) / (t2 - t1) * p1[0] + (u - t1) / (t2 - t1) * p2[0] if (t2 - t1) != 0 else p2[0]
                                        A2y = (t2 - u) / (t2 - t1) * p1[1] + (u - t1) / (t2 - t1) * p2[1] if (t2 - t1) != 0 else p2[1]
                                        A3x = (t3 - u) / (t3 - t2) * p2[0] + (u - t2) / (t3 - t2) * p3[0] if (t3 - t2) != 0 else p3[0]
                                        A3y = (t3 - u) / (t3 - t2) * p2[1] + (u - t2) / (t3 - t2) * p3[1] if (t3 - t2) != 0 else p3[1]

                                        B1x = (t2 - u) / (t2 - t0) * A1x + (u - t0) / (t2 - t0) * A2x if (t2 - t0) != 0 else A2x
                                        B1y = (t2 - u) / (t2 - t0) * A1y + (u - t0) / (t2 - t0) * A2y if (t2 - t0) != 0 else A2y
                                        B2x = (t3 - u) / (t3 - t1) * A2x + (u - t1) / (t3 - t1) * A3x if (t3 - t1) != 0 else A3x
                                        B2y = (t3 - u) / (t3 - t1) * A2y + (u - t1) / (t3 - t1) * A3y if (t3 - t1) != 0 else A3y

                                        Cx = (t2 - u) / (t2 - t1) * B1x + (u - t1) / (t2 - t1) * B2x if (t2 - t1) != 0 else B2x
                                        Cy = (t2 - u) / (t2 - t1) * B1y + (u - t1) / (t2 - t1) * B2y if (t2 - t1) != 0 else B2y
                                        sampled.extend((Cx, Cy))

                                # append explicit final closing point (first point) to finish polyline
                                sampled.extend((points_xy[0][0], points_xy[0][1]))
                                return sampled

                            seq = [(pts[i]['x'], pts[i]['y']) for i in range(n)]
                            coords = sample_catmull_closed(seq, samples_per_seg=16)
                            if coords:
                                try:
                                    cid = self.editor_canvas.create_line(*coords, fill='cyan', width=2)
                                    self._spline_canvas_ids.append(cid)
                                    smooth_curve_ids.append(cid)
                                except Exception:
                                    pass
                            # mark all edges covered so no straight segments are drawn
                            try:
                                covered_edges = set(range(n))
                            except Exception:
                                pass
                            all_smooth_flag = True
                        else:
                            # find a pivot where point is NOT smooth so we can scan linearly
                            pivot = 0
                            for i in range(n):
                                if not smooth_flags[i]:
                                    pivot = i
                                    break
                            # scan linearized sequence starting after pivot
                            runs = []
                            i = (pivot + 1) % n
                            scanned = 0
                            while scanned < n:
                                if not smooth_flags[i]:
                                    i = (i + 1) % n
                                    scanned += 1
                                    continue
                                # build run
                                run = []
                                while scanned < n and smooth_flags[i]:
                                    run.append(i)
                                    i = (i + 1) % n
                                    scanned += 1
                                if run:
                                    runs.append(run)
                            # for each distinct run, include neighbor before and after and draw
                            for run in runs:
                                start = (run[0] - 1) % n
                                end = (run[-1] + 1) % n
                                seq_indices = [start] + list(run) + [end]
                                seq = [(pts[idx]['x'], pts[idx]['y']) for idx in seq_indices]
                                coords = sample_catmull(seq, samples_per_seg=16)
                                if coords:
                                    # compute consecutive edge indices for this seq
                                    seq_edges = []
                                    for ei in range(len(seq_indices) - 1):
                                        seq_edges.append(seq_indices[ei])
                                    # if any of these edges already covered, skip drawing to avoid duplicates
                                    if any((e % n) in covered_edges for e in seq_edges):
                                        # still mark edges even if skipping draw, to be consistent
                                        for edge_idx in seq_edges:
                                            add_edge(edge_idx)
                                    else:
                                        try:
                                            cid = self.editor_canvas.create_line(*coords, fill='cyan', width=2)
                                            self._spline_canvas_ids.append(cid)
                                            smooth_curve_ids.append(cid)
                                        except Exception:
                                            pass
                                        for edge_idx in seq_edges:
                                            add_edge(edge_idx)
                else:
                    i = 1
                    while i <= n - 2:
                        if not pts[i].get('smooth'):
                            i += 1
                            continue
                        # found run of smooth points starting at i
                        start_run = i
                        end_run = i
                        j = i + 1
                        while j <= n - 2 and pts[j].get('smooth'):
                            end_run = j
                            j += 1
                        start = start_run - 1
                        end = end_run + 1
                        # collect sequence indices from start..end (inclusive)
                        seq_indices = list(range(start, end + 1))
                        seq = [(pts[k]['x'], pts[k]['y']) for k in seq_indices]
                        coords = sample_catmull(seq, samples_per_seg=16)
                        if coords:
                            try:
                                cid = self.editor_canvas.create_line(*coords, fill='cyan', width=2)
                                self._spline_canvas_ids.append(cid)
                                try:
                                    smooth_curve_ids.append(cid)
                                except Exception:
                                    pass
                            except Exception:
                                pass
                            for e in seq_indices[:-1]:
                                add_edge(e)
                        i = end_run + 1

            # If we drew an all-smooth closed spline, skip drawing straight edges/removal
            # to avoid any possible duplicates; smooth curve ids are already recorded.
            if all_smooth_flag:
                try:
                    for cid in smooth_curve_ids:
                        try:
                            self.editor_canvas.tag_raise(cid)
                        except Exception:
                            pass
                except Exception:
                    pass
                return

            # draw remaining straight edges
            if self._spline_closed:
                for i in range(n):
                    if i in covered_edges:
                        continue
                    a = pts[i]
                    b = pts[(i+1) % n]
                    try:
                        cid = self.editor_canvas.create_line(a['x'], a['y'], b['x'], b['y'], fill='cyan', width=2)
                        self._spline_canvas_ids.append(cid)
                    except Exception:
                        pass
            else:
                for i in range(n-1):
                    if i in covered_edges:
                        continue
                    a = pts[i]
                    b = pts[i+1]
                    try:
                        cid = self.editor_canvas.create_line(a['x'], a['y'], b['x'], b['y'], fill='cyan', width=2)
                        self._spline_canvas_ids.append(cid)
                    except Exception:
                        pass
            # finally raise smooth curves above straight edges so smoothed segments aren't visually occluded
            try:
                for cid in smooth_curve_ids:
                    try:
                        self.editor_canvas.tag_raise(cid)
                    except Exception:
                        pass
            except Exception:
                pass
            # Remove straight-line items that are redundant because they appear as
            # direct segments inside a smoothed curve (handles last->first double-draw)
            try:
                # collect smooth coords lists
                smooth_coords_list = []
                # optional debug: list straight segments before removal
                straight_segments_before = []
                if debug:
                    try:
                        for cid in list(self._spline_canvas_ids):
                            try:
                                c = self.editor_canvas.coords(cid) or []
                                if len(c) == 4:
                                    straight_segments_before.append((cid, (c[0], c[1], c[2], c[3])))
                            except Exception:
                                pass
                    except Exception:
                        pass

                # debug prints removed to keep console output clean
                for scid in list(smooth_curve_ids):
                    try:
                        c = self.editor_canvas.coords(scid) or []
                        # store as list of point tuples
                        pts_list = [(c[i], c[i+1]) for i in range(0, len(c)-1, 2)] if len(c) >= 4 else []
                        smooth_coords_list.append(pts_list)
                    except Exception:
                        pass

                def approx_equal_points(p, q, eps=None):
                    # p,q are (x,y) tuples; use Euclidean distance tolerance in pixels.
                    # Default eps scales with current zoom so larger previews tolerate larger sampling offsets.
                    try:
                        if eps is None:
                            z = max(1.0, float(getattr(self, 'zoom_scale', 1.0)))
                            eps = max(1.5, 1.0 * z) * 1.5
                        dx = float(p[0]) - float(q[0])
                        dy = float(p[1]) - float(q[1])
                        return (dx*dx + dy*dy) ** 0.5 <= float(eps)
                    except Exception:
                        return False

                # iterate over current spline canvas ids and remove straight ones that are contained
                for cid in list(self._spline_canvas_ids):
                    try:
                        if cid in smooth_curve_ids:
                            continue
                        coords = self.editor_canvas.coords(cid) or []
                        # only consider simple straight segments (4 values)
                        if len(coords) != 4:
                            continue
                        seg_a = (coords[0], coords[1])
                        seg_b = (coords[2], coords[3])
                        redundant = False
                        for pts_list in smooth_coords_list:
                            if not pts_list:
                                continue
                            # scan consecutive pairs in smooth curve
                            for i in range(len(pts_list) - 1):
                                p1 = pts_list[i]
                                p2 = pts_list[i+1]
                                if (approx_equal_points(p1, seg_a) and approx_equal_points(p2, seg_b)):
                                    redundant = True
                                    break
                                # also allow reversed direction
                                if (approx_equal_points(p1, seg_b) and approx_equal_points(p2, seg_a)):
                                    redundant = True
                                    break
                            if redundant:
                                break
                        if redundant:
                            try:
                                self.editor_canvas.delete(cid)
                            except Exception:
                                pass
                            try:
                                self._spline_canvas_ids.remove(cid)
                            except Exception:
                                pass
                    except Exception:
                        pass
            except Exception:
                pass
            # If the closed spline was entirely smooth, aggressively remove any
            # remaining simple straight segments to avoid visual duplicates.
            try:
                if all_smooth_flag:
                    for cid in list(self._spline_canvas_ids):
                        try:
                            coords = self.editor_canvas.coords(cid) or []
                            if len(coords) == 4:
                                try:
                                    self.editor_canvas.delete(cid)
                                except Exception:
                                    pass
                                try:
                                    self._spline_canvas_ids.remove(cid)
                                except Exception:
                                    pass
                        except Exception:
                            pass
            except Exception:
                pass
        except Exception:
            pass

    def _hit_test_point(self, x, y, tol=6):
        """Return index of point under (x,y) or None."""
        try:
            # scale tolerance with zoom so hit feels consistent
            z = max(1.0, float(getattr(self, 'zoom_scale', 1.0)))
            tol_adj = int(max(4, tol * z))
            for i, p in enumerate(self._spline_points):
                dx = p['x'] - x
                dy = p['y'] - y
                if (dx*dx + dy*dy) <= (tol_adj * tol_adj):
                    return i
        except Exception:
            pass
        return None

    def _hit_test_segment(self, x, y, tol=6):
        """Return start-index of segment under (x,y) or None.

        For closed splines the segment n-1 refers to edge (n-1)->0.
        """
        try:
            pts = self._spline_points
            n = len(pts)
            if n < 2:
                return None
            # scale tolerance with zoom
            z = max(1.0, float(getattr(self, 'zoom_scale', 1.0)))
            tol_adj = float(max(4.0, tol * z))
            import math

            def point_seg_dist(px, py, x1, y1, x2, y2):
                # distance from P to segment AB
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0 and dy == 0:
                    return math.hypot(px - x1, py - y1)
                t = ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)
                t = max(0.0, min(1.0, t))
                proj_x = x1 + t * dx
                proj_y = y1 + t * dy
                return math.hypot(px - proj_x, py - proj_y)

            # consider segments; if closed include last->first
            seg_indices = range(n) if getattr(self, '_spline_closed', False) else range(n - 1)
            for i in seg_indices:
                a = pts[i]
                b = pts[(i+1) % n]
                d = point_seg_dist(x, y, a['x'], a['y'], b['x'], b['y'])
                if d <= tol_adj:
                    return i
        except Exception:
            pass
        return None

    def _insert_point_at(self, x, y, smooth=False):
        """Insert a new point at (x,y). If close to a segment, insert in-between; else append."""
        try:
            # try to find nearest segment
            pts = self._spline_points
            best_idx = None
            best_dist = 1e9
            n = len(pts)
            if n == 0:
                self._spline_points.append({'x':x,'y':y,'smooth':smooth})
                return len(self._spline_points)-1

            # consider segments; if closed, include last->first
            seg_range = range(n - 1)
            if getattr(self, '_spline_closed', False):
                seg_indices = range(n)
            else:
                seg_indices = range(n - 1)
            for i in seg_indices:
                x1, y1 = pts[i]['x'], pts[i]['y']
                x2, y2 = pts[(i+1) % n]['x'], pts[(i+1) % n]['y']
                # distance point-line
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0 and dy == 0:
                    continue
                t = ((x - x1)*dx + (y - y1)*dy) / float(dx*dx + dy*dy)
                t = max(0.0, min(1.0, t))
                px = x1 + t*dx
                py = y1 + t*dy
                d2 = (px - x)*(px - x) + (py - y)*(py - y)
                # scale insertion threshold with zoom
                z = max(1.0, float(getattr(self, 'zoom_scale', 1.0)))
                insert_tol = int(max(8, 16 * z))
                if d2 < best_dist and d2 <= (insert_tol * insert_tol):
                    best_dist = d2
                    best_idx = (i + 1) % (n + 0)  # insertion index between i and i+1
            if best_idx is not None:
                self._spline_points.insert(best_idx, {'x':x,'y':y,'smooth':smooth})
                return best_idx
            else:
                self._spline_points.append({'x':x,'y':y,'smooth':smooth})
                return len(self._spline_points)-1
        except Exception:
            return None

    def _on_canvas_zoom(self, event):
        """Zoom the preview centered at the canvas cursor position so the image point under cursor remains fixed."""
        try:
            # get current cursor canvas coords
            try:
                cx = int(getattr(event, 'x', 0))
                cy = int(getattr(event, 'y', 0))
            except Exception:
                return

            # determine wheel direction
            delta = 0
            try:
                delta = int(getattr(event, 'delta', 0))
            except Exception:
                if getattr(event, 'num', None) == 4:
                    delta = 120
                elif getattr(event, 'num', None) == 5:
                    delta = -120

            if delta == 0:
                return

            # compute scale factor (similar to existing zoom controls)
            old_scale = self.zoom_scale
            if delta > 0:
                new_scale = min(5.0, self.zoom_scale * 1.2)
            else:
                new_scale = max(1.0, self.zoom_scale / 1.2)
            if new_scale == old_scale:
                return

            # convert canvas cursor to image-relative coordinates before change
            cw = int(self.editor_canvas.cget('width'))
            ch = int(self.editor_canvas.cget('height'))
            size_before = int(max(10, min(cw, ch)) * old_scale)
            base_x = (cw - size_before) // 2
            base_y = (ch - size_before) // 2
            img_x = cx - (base_x + self.pan_offset_x)
            img_y = cy - (base_y + self.pan_offset_y)

            # normalize within image coordinates
            rel_x = 0.0
            rel_y = 0.0
            if size_before > 0:
                rel_x = img_x / float(size_before)
                rel_y = img_y / float(size_before)

            # apply new scale
            self.zoom_scale = new_scale

            # compute new size and new base
            size_after = int(max(10, min(cw, ch)) * self.zoom_scale)
            base_x_after = (cw - size_after) // 2
            base_y_after = (ch - size_after) // 2

            # compute new pan_offset so that the same image-relative point stays under cursor
            new_img_x = int(rel_x * size_after)
            new_img_y = int(rel_y * size_after)
            # desired top-left of image such that image point maps to cx,cy
            desired_x = cx - new_img_x
            desired_y = cy - new_img_y
            # convert to pan offset relative to base
            self.pan_offset_x = desired_x - base_x_after
            self.pan_offset_y = desired_y - base_y_after

            # clamp pan_offset so image stays within canvas bounds
            self.pan_offset_x = max(cw - size_after, min(0, self.pan_offset_x))
            self.pan_offset_y = max(ch - size_after, min(0, self.pan_offset_y))

            # refresh preview
            try:
                # refresh canvas preview for this original explicitly
                try:
                    self._show_preview_on_canvas(orig_path)
                except Exception:
                    try:
                        self._update_current_preview()
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception:
            pass

    def _on_canvas_adjust_size(self, event):
        """Adjust brush/eraser size with Ctrl+wheel or X11 control buttons."""
        try:
            delta = 0
            try:
                delta = int(getattr(event, 'delta', 0))
            except Exception:
                if getattr(event, 'num', None) == 4:
                    delta = 120
                elif getattr(event, 'num', None) == 5:
                    delta = -120

            amt = 1 if delta > 0 else -1
            if self.active_tool == 'eraser':
                self.eraser_size = max(1, min(200, int(self.eraser_size + amt)))
                try:
                    if getattr(self, 'size_slider', None) is not None:
                        self.size_slider.set(self.eraser_size)
                except Exception:
                    pass
            else:
                self.brush_size = max(1, min(200, int(self.brush_size + amt)))
                try:
                    if getattr(self, 'size_slider', None) is not None:
                        self.size_slider.set(self.brush_size)
                except Exception:
                    pass

            try:
                if getattr(self, '_last_cursor_canvas_pos', None) is not None:
                    cx, cy = self._last_cursor_canvas_pos
                    self._render_marker_at_canvas(cx, cy)
            except Exception:
                pass
        except Exception:
            pass

    def _on_canvas_adjust_softness(self, event):
        """Adjust brush/eraser softness with Shift+wheel or X11 shift buttons."""
        try:
            delta = 0
            try:
                delta = int(getattr(event, 'delta', 0))
            except Exception:
                if getattr(event, 'num', None) == 4:
                    delta = 120
                elif getattr(event, 'num', None) == 5:
                    delta = -120

            amt = 0.05 if delta > 0 else -0.05
            if self.active_tool == 'eraser':
                self.eraser_softness = max(0.0, min(1.0, self.eraser_softness + amt))
                try:
                    if getattr(self, 'softness_slider', None) is not None:
                        self.softness_slider.set(self.eraser_softness if self.active_tool == 'eraser' else self.brush_softness)
                except Exception:
                    pass
            else:
                self.brush_softness = max(0.0, min(1.0, self.brush_softness + amt))
                try:
                    if getattr(self, 'softness_slider', None) is not None:
                        self.softness_slider.set(self.brush_softness)
                except Exception:
                    pass

            try:
                if getattr(self, '_last_cursor_canvas_pos', None) is not None:
                    cx, cy = self._last_cursor_canvas_pos
                    self._render_marker_at_canvas(cx, cy)
            except Exception:
                pass
        except Exception:
            pass

    def _render_marker_at_canvas(self, cx: int, cy: int):
        """Render marker centered at given canvas coordinates (cx,cy).
        This is the extracted composition logic used by mouse move and other callers.
        """
        try:
            preview = getattr(self, '_cached_resized_img', None)
            if preview is None:
                return
            cw = int(self.editor_canvas.cget('width'))
            ch = int(self.editor_canvas.cget('height'))
            size = preview.width
            base_x = (cw - size) // 2
            base_y = (ch - size) // 2
            x = base_x + self.pan_offset_x
            y = base_y + self.pan_offset_y

            # cursor position relative to preview
            rel_x = int(cx) - x
            rel_y = int(cy) - y
            # clamp: if cursor outside preview, do not update marker but keep last
            if rel_x < 0 or rel_y < 0 or rel_x >= size or rel_y >= size:
                return

            diameter = int(self.brush_size if self.active_tool == 'brush' else self.eraser_size)
            softness = float(self.brush_softness if self.active_tool == 'brush' else self.eraser_softness)
            mode = 'brush' if self.active_tool == 'brush' else 'eraser'

            # compute scaled stamp size
            orig_size = getattr(self, '_cached_original_size', None)
            try:
                if orig_size and orig_size[0] > 0:
                    scale = preview.width / float(orig_size[0])
                else:
                    scale = 1.0
            except Exception:
                scale = 1.0

            stamp_px = max(1, int(max(1, diameter * scale)))
            key = (stamp_px, softness, mode, preview.width)
            stamp_preview = self._marker_stamp_cache.get(key)
            if stamp_preview is None:
                base_stamp = self._get_stamp(stamp_px, softness, mode)
                if base_stamp is None:
                    return
                stamp_preview = base_stamp
                self._marker_stamp_cache[key] = stamp_preview

            sx = int(rel_x)
            sy = int(rel_y)
            sw = stamp_preview.width
            sh = stamp_preview.height
            left = sx - sw // 2
            top = sy - sh // 2
            crop_left = max(0, left)
            crop_top = max(0, top)
            crop_right = min(size, left + sw)
            crop_bottom = min(size, top + sh)
            if crop_right <= crop_left or crop_bottom <= crop_top:
                return

            stamp_left = crop_left - left
            stamp_top = crop_top - top
            crop_box = (crop_left, crop_top, crop_right, crop_bottom)
            region = preview.crop(crop_box)
            stamp_region = stamp_preview.crop((stamp_left, stamp_top, stamp_left + (crop_right - crop_left), stamp_top + (crop_bottom - crop_top)))

            try:
                inverted = ImageChops.invert(region.convert('RGB'))
            except Exception:
                inverted = region.convert('RGB')
                inverted = ImageChops.invert(inverted)

            mask_alpha = stamp_region.convert('L')
            inverted_rgba = inverted.convert('RGBA')
            inverted_rgba.putalpha(mask_alpha)

            out_w = crop_right - crop_left
            out_h = crop_bottom - crop_top
            out = Image.new('RGBA', (out_w, out_h), (0, 0, 0, 0))
            out.paste(inverted_rgba, (0, 0), inverted_rgba)

            tkimg = ImageTk.PhotoImage(out)
            # keep marker (delete old marker image id only)
            try:
                if getattr(self, '_marker_image_id', None) is not None:
                    try:
                        self.editor_canvas.delete(self._marker_image_id)
                    except Exception:
                        pass
            except Exception:
                pass
            abs_x = x + crop_left
            abs_y = y + crop_top
            self._marker_tkimage = tkimg
            try:
                # tag marker so it won't be removed when refreshing preview
                self._marker_image_id = self.editor_canvas.create_image(abs_x, abs_y, image=self._marker_tkimage, anchor='nw', tags=('marker',))
            except Exception:
                self._marker_image_id = None
            # draw a thin vector outline matching the stamp diameter to indicate brush size
            try:
                # choose a high-contrast outline color based on the sampled region brightness
                try:
                    from PIL import ImageStat
                    gray = region.convert('L')
                    stat = ImageStat.Stat(gray)
                    avg = stat.mean[0] if getattr(stat, 'mean', None) else 128
                    outline_color = 'black' if avg > 127 else 'white'
                except Exception:
                    outline_color = 'white'

                # remove previous outline so we don't accumulate shapes
                try:
                    self.editor_canvas.delete('marker_outline')
                except Exception:
                    pass

                # compute canvas coordinates for circle center and radius
                try:
                    cx_canvas = x + sx
                    cy_canvas = y + sy
                    r = max(1, sw // 2)
                    left_oval = cx_canvas - r
                    top_oval = cy_canvas - r
                    right_oval = cx_canvas + r
                    bottom_oval = cy_canvas + r
                    # draw a thin outline on top of the marker
                    self.editor_canvas.create_oval(left_oval, top_oval, right_oval, bottom_oval,
                                                   outline=outline_color, width=1, tags=('marker_outline',))
                except Exception:
                    pass
            except Exception:
                pass
        except Exception:
            pass

    def _paint_at(self, mask_path: str, cx: int, cy: int, diameter: int, softness: float, mode: str = 'brush'):
        """Paint a circular brush on in-memory mask at image coords (cx,cy).
        mode: 'brush' adds intensity, 'eraser' subtracts intensity.
        softness: 0..1 where 0 is hard edge, >0 softer.
        """
        try:
            if mask_path not in self._in_memory_mask_overrides:
                return
            mask_img = self._in_memory_mask_overrides[mask_path]
            if not isinstance(mask_img, Image.Image):
                return
            d = max(1, int(diameter))
            # create brush alpha (L) as filled circle then blur according to softness
            brush = Image.new('L', (d, d), 0)
            bd = ImageDraw.Draw(brush)
            bd.ellipse((0, 0, d - 1, d - 1), fill=255)
            # apply gaussian blur proportional to softness
            try:
                if softness and softness > 0:
                    radius = max(0.5, softness * (d / 2.0))
                    brush = brush.filter(ImageFilter.GaussianBlur(radius=radius))
            except Exception:
                pass

            # compute paste box on mask
            ow, oh = mask_img.size
            left = int(cx - d // 2)
            top = int(cy - d // 2)
            right = left + d
            bottom = top + d

            # crop region intersection
            br_left = 0
            br_top = 0
            if left < 0:
                br_left = -left
                left = 0
            if top < 0:
                br_top = -top
                top = 0
            br_right = d - max(0, right - ow)
            br_bottom = d - max(0, bottom - oh)

            if br_right <= br_left or br_bottom <= br_top:
                return

            brush_region = brush.crop((br_left, br_top, br_right, br_bottom))
            # get pixel access objects
            mask_px = mask_img.load()
            brush_px = brush_region.load()
            bw, bh = brush_region.size
            for yy in range(bh):
                my = top + yy
                for xx in range(bw):
                    mx = left + xx
                    bval = brush_px[xx, yy]
                    if bval == 0:
                        continue
                    old = mask_px[mx, my]
                    if mode == 'brush':
                        new = min(255, int(old + bval))
                    else:
                        # eraser: subtract brush alpha
                        new = max(0, int(old - bval))
                    mask_px[mx, my] = new

            # store back
            self._in_memory_mask_overrides[mask_path] = mask_img
        except Exception:
            pass

    def _paint_line(self, mask_path: str, p0: tuple[int, int], p1: tuple[int, int]):
        """Paint a sequence of circles between p0 and p1 to form continuous stroke."""
        try:
            if p0 is None:
                p0 = p1
            x0, y0 = p0
            x1, y1 = p1
            dx = x1 - x0
            dy = y1 - y0
            dist = (dx * dx + dy * dy) ** 0.5
            diameter = int(self.brush_size if self.active_tool == 'brush' else self.eraser_size)
            # spacing = diameter / 4.0 ensures stamps overlap enough (half of radius)
            spacing = max(1.0, diameter / 4.0)
            import math
            if dist <= 0.0:
                steps = 1
            else:
                steps = max(1, int(math.ceil(dist / spacing)))

            stamp = self._get_stamp(diameter, self.brush_softness if self.active_tool == 'brush' else self.eraser_softness, ('brush' if self.active_tool == 'brush' else 'eraser'))
            if stamp is None:
                return

            # place stamps from just after p0 to p1 inclusive to avoid duplicating the initial stamp
            for i in range(1, steps + 1):
                t = i / float(steps)
                ix = int(round(x0 + dx * t))
                iy = int(round(y0 + dy * t))
                self._stamp_mask(mask_path, ix, iy, stamp, ('brush' if self.active_tool == 'brush' else 'eraser'))
        except Exception:
            pass

    def _get_stamp(self, diameter: int, softness: float, mode: str) -> Image.Image | None:
        """Return a cached stamp (L image) for given parameters, creating if necessary."""
        try:
            d = max(1, int(diameter))
            key = (d, float(softness), str(mode))
            if key in self._stamp_cache:
                return self._stamp_cache[key]
            # create precise circular stamp with optional linear falloff (softness)
            radius = d / 2.0
            brush = Image.new('L', (d, d), 0)
            px = brush.load()
            # softness: 0 => hard edge; >0 => linear falloff over softness*radius
            s = float(softness)
            inner = 0.0
            if s > 0:
                inner = max(0.0, radius * (1.0 - s))
            else:
                inner = radius
            cx = (d - 1) / 2.0
            cy = (d - 1) / 2.0
            for y in range(d):
                dy = y - cy
                for x in range(d):
                    dx = x - cx
                    dist = (dx*dx + dy*dy) ** 0.5
                    if dist <= inner:
                        val = 255
                    elif dist >= radius:
                        val = 0
                    else:
                        # linear falloff between inner and radius
                        t = (dist - inner) / (radius - inner) if (radius - inner) > 0 else 1.0
                        val = int(max(0, min(255, int(255 * (1.0 - t)))))
                    px[x, y] = val
            self._stamp_cache[key] = brush
            return brush
        except Exception:
            return None

    def _stamp_mask(self, mask_path: str, cx: int, cy: int, stamp: Image.Image, mode: str = 'brush'):
        """Stamp the given L-image onto in-memory mask at image coords (cx,cy)."""
        try:
            if mask_path not in self._in_memory_mask_overrides:
                return
            mask_img = self._in_memory_mask_overrides[mask_path]
            if not isinstance(mask_img, Image.Image):
                return
            d = stamp.width
            ow, oh = mask_img.size
            left = int(cx - d // 2)
            top = int(cy - d // 2)
            right = left + d
            bottom = top + d

            br_left = 0
            br_top = 0
            if left < 0:
                br_left = -left
                left = 0
            if top < 0:
                br_top = -top
                top = 0
            br_right = d - max(0, right - ow)
            br_bottom = d - max(0, bottom - oh)
            if br_right <= br_left or br_bottom <= br_top:
                return
            stamp_region = stamp.crop((br_left, br_top, br_right, br_bottom))
            mask_px = mask_img.load()
            stamp_px = stamp_region.load()
            bw, bh = stamp_region.size
            for yy in range(bh):
                my = top + yy
                for xx in range(bw):
                    mx = left + xx
                    sval = stamp_px[xx, yy]
                    if sval == 0:
                        continue
                    old = mask_px[mx, my]
                    if mode == 'brush':
                        new = min(255, int(old + sval))
                    else:
                        new = max(0, int(old - sval))
                    mask_px[mx, my] = new
            self._in_memory_mask_overrides[mask_path] = mask_img
        except Exception:
            pass

    def _on_key_press(self, event):
        # ignore hotkeys if focus is on a text input widget
        focused = self.focus_get()
        if focused and isinstance(focused, (tk.Entry, tk.Text, ctk.CTkEntry)):
            return
        step = 10  # pan step
        if event.char in ('-', '_'):
            self._zoom_out()
        elif event.char in ('+', '='):
            self._zoom_in()
        elif event.char in ('0', ')'):
            self.zoom_scale = 1.0
            self.pan_offset_x = 0
            self.pan_offset_y = 0
            self._update_current_preview()
        elif event.keysym == 'Left':
            if self.originals_list.curselection():
                self.pan_offset_x -= step
                self._update_current_preview()
        elif event.keysym == 'Right':
            if self.originals_list.curselection():
                self.pan_offset_x += step
                self._update_current_preview()
        elif event.keysym == 'Up':
            if self.originals_list.curselection():
                self.pan_offset_y -= step
                self._update_current_preview()
        elif event.keysym == 'Down':
            if self.originals_list.curselection():
                self.pan_offset_y += step
                self._update_current_preview()
        # spline keyboard controls
        if self.active_tool == 'spline':
            try:
                if event.keysym == 'Delete':
                    # clear spline
                    self._spline_points = []
                    self._spline_closed = False
                    try:
                        for cid in list(getattr(self, '_spline_canvas_ids', []) or []):
                            try:
                                self.editor_canvas.delete(cid)
                            except Exception:
                                pass
                        self._spline_canvas_ids = []
                    except Exception:
                        pass
                elif event.keysym in ('Return', 'KP_Enter'):
                    # Enter: if not closed -> close; if closed -> fill/erase
                    if not getattr(self, '_spline_closed', False):
                        # close if enough points
                        if len(self._spline_points) >= 3:
                            self._spline_closed = True
                            self._render_spline_preview()
                    else:
                        # debug hook removed for Return press
                        # fill polygon on mask: shift held -> erase, otherwise fill
                        sel = self.originals_list.curselection()
                        if sel:
                            idx = sel[0]
                            orig_path = self.originals[idx]
                            mask_path = self.original_to_mask.get(orig_path)
                            if mask_path is None:
                                return
                            # prepare polygon points in image coords
                            img_pts = []
                            for p in self._spline_points:
                                coords = self._canvas_to_image_coords(int(p['x']), int(p['y']), orig_path)
                                if coords is not None:
                                    img_pts.append(coords)
                            if not img_pts:
                                return
                            # draw polygon into mask
                            try:
                                mimg = None
                                if mask_path in self._in_memory_mask_overrides:
                                    mimg = self._in_memory_mask_overrides[mask_path]
                                else:
                                    if Path(mask_path).is_file():
                                        mimg = Image.open(mask_path).convert('L')
                                if mimg is None:
                                    return
                                # push current state so this polygon action can be undone
                                try:
                                    self._push_undo(mask_path, mimg.copy())
                                except Exception:
                                    pass
                                draw = ImageDraw.Draw(mimg)
                                poly = [(x, y) for (x, y) in img_pts]
                                # prefer explicit tracked flag _shift_down when available
                                shift_held = getattr(self, '_shift_down', None)
                                if shift_held is None:
                                    shift_held = (event.state & 0x1) != 0
                                if shift_held:
                                    # erase: fill with 0
                                    draw.polygon(poly, fill=0)
                                else:
                                    # fill with 255
                                    draw.polygon(poly, fill=255)
                                # store back
                                self._in_memory_mask_overrides[mask_path] = mimg
                                # regenerate overlay
                                try:
                                    cw = int(self.editor_canvas.cget('width'))
                                    size = int(max(10, min(cw, int(self.editor_canvas.cget('height')))) * self.zoom_scale)
                                    self._regenerate_overlay_from_inmemory(mask_path, size)
                                except Exception:
                                    pass
                                # fill/erase applied; preview refresh follows
                                # save modified mask to disk atomically so preview code
                                # which prefers on-disk mask will pick up changes
                                try:
                                    tmp = Path(mask_path).with_suffix('.tmp.mask.png')
                                    try:
                                        mimg.save(tmp)
                                        tmp.replace(mask_path)
                                    except Exception:
                                        # fallback to direct save
                                        try:
                                            mimg.save(mask_path)
                                        except Exception:
                                            pass
                                except Exception:
                                    pass
                                # clear cached overlay so it will be rebuilt from disk
                                try:
                                    self._cached_overlay = None
                                    self._cached_overlay_params = None
                                except Exception:
                                    pass
                                try:
                                    # ensure preview overlay and thumbnail update
                                    try:
                                        self._update_current_preview()
                                    except Exception:
                                        pass
                                    try:
                                        self._update_mask_preview(orig_path)
                                    except Exception:
                                        pass
                                except Exception:
                                    pass
                            except Exception:
                                pass
                            # After fill/erase, preserve the spline so user can continue editing.
                            # The Delete button (or Delete key) should be used to clear it explicitly.
                            try:
                                # regenerate preview but keep points
                                self._render_spline_preview()
                            except Exception:
                                pass
            except Exception:
                pass

    def _on_canvas_press(self, event):
        # if Ctrl pressed -> pan (but only when not editing spline)
        if (event.state & 0x4) and getattr(self, 'active_tool', None) != 'spline':
            self.pan_start_x = event.x
            self.pan_start_y = event.y
            return
        # spline tool: handle click to create or begin dragging a point (top-level to avoid nesting)
        if self.active_tool == 'spline':
            try:
                # ensure keyboard focus so Enter/Shift+Enter are delivered
                try:
                    self.editor_canvas.focus_set()
                except Exception:
                    pass
                cx = float(event.x)
                cy = float(event.y)
                # modifiers: check both state mask and keysym for reliability across platforms
                try:
                    ks = str(getattr(event, 'keysym', '')).lower()
                except Exception:
                    ks = ''
                shift = (event.state & 0x0001) != 0 or ks.startswith('shift')
                ctrl = (event.state & 0x0004) != 0 or ks.startswith('control') or ks.startswith('ctrl')
                # Alt detection: prefer explicit keysym checks; some platforms set low bits
                try:
                    ks = str(getattr(event, 'keysym', '')).lower()
                    alt = ks.startswith('alt') or ((event.state & 0x20000) != 0)
                except Exception:
                    alt = False

                # hit test existing point
                hit = self._hit_test_point(cx, cy)
                # if clicked the FIRST point and spline not closed and no SHIFT/CTRL -> close spline
                # (give closing precedence over deletion so first-point-close is reliable)
                try:
                    if hit == 0 and not getattr(self, '_spline_closed', False) and len(self._spline_points) >= 3 and not (shift or ctrl):
                        self._spline_closed = True
                        self._spline_drag_index = None
                        self._render_spline_preview()
                        return
                except Exception:
                    pass
                if hit is not None:
                    # modifier actions on click: Alt deletes, Shift softens, Ctrl hardens
                    if alt:
                        # delete point (but do NOT delete first point when closing is possible)
                        try:
                            print(f"[SPLINE] delete requested: hit={hit}, state={getattr(event,'state',None)}, keysym={getattr(event,'keysym',None)}")
                            if hit != 0 or getattr(self, '_spline_closed', False):
                                if 0 <= hit < len(self._spline_points):
                                    self._spline_points.pop(hit)
                        except Exception:
                            pass
                        self._spline_drag_index = None
                    elif shift:
                        # make soft
                        try:
                            self._spline_points[hit]['smooth'] = True
                        except Exception:
                            pass
                        # click should not start dragging — only moves when click-drag
                        self._spline_drag_index = hit
                    elif ctrl:
                        # make hard
                        try:
                            self._spline_points[hit]['smooth'] = False
                        except Exception:
                            pass
                        self._spline_drag_index = hit
                    else:
                        # plain click: start dragging this point
                        self._spline_drag_index = hit
                else:
                    # no hit — check if closed: when closed, only allow insertion
                    # if the click landed on a segment; otherwise ignore click.
                    smooth = shift
                    if getattr(self, '_spline_closed', False):
                        seg = self._hit_test_segment(cx, cy)
                        if seg is None:
                            # click in empty space on closed spline: ignore
                            self._spline_drag_index = None
                        else:
                            # insert into found segment
                            # insertion index returns index where new point is placed
                            idx = self._insert_point_at(cx, cy, smooth=smooth)
                            self._spline_drag_index = idx
                    else:
                        idx = self._insert_point_at(cx, cy, smooth=smooth)
                        self._spline_drag_index = idx
                # when clicking we always re-render
                self._render_spline_preview()
            except Exception:
                pass
            return
        # start drawing if brush/eraser active
        if self.active_tool in ('brush', 'eraser'):
            sel = self.originals_list.curselection()
            if not sel:
                return
            idx = sel[0]
            orig_path = self.originals[idx]
            mask_path = self.original_to_mask.get(orig_path)
            # if no mask exists, create one in same folder
            try:
                if not mask_path or not Path(mask_path).is_file():
                    orig_p = Path(orig_path)
                    mask_filename = orig_p.stem + '-masklabel.png'
                    mask_file = orig_p.with_name(mask_filename)
                    # create new black mask with same size as original
                    img = Image.open(orig_path)
                    w, h = img.size
                    new_mask = Image.new('L', (w, h), 0)
                    try:
                        new_mask.save(mask_file)
                    except Exception:
                        pass
                    mask_path = str(mask_file)
                    self.original_to_mask[orig_path] = mask_path
                    # update DB rows
                    try:
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

                # prepare in-memory mask copy for editing
                try:
                    if mask_path and Path(mask_path).is_file():
                        orig_mask = Image.open(mask_path).convert('L')
                    else:
                        # fallback: create new based on original size
                        img = Image.open(orig_path)
                        orig_mask = Image.new('L', img.size, 0)
                except Exception:
                    # last resort create tiny mask
                    orig_mask = Image.new('L', (10, 10), 0)

                self._in_memory_mask_overrides[mask_path] = orig_mask.copy()
                # push initial state for undo (so first undo restores original file state)
                try:
                    self._push_undo(mask_path, orig_mask.copy())
                except Exception:
                    pass
                self._current_edit_mask_path = mask_path
                # compute image-space coords for current mouse location
                pt = self._canvas_to_image_coords(event.x, event.y, orig_path)
                self._last_draw_point = pt
                self._drawing = True
                # ensure marker is visible at press location
                try:
                    self._render_marker_at_canvas(event.x, event.y)
                except Exception:
                    pass
                # apply initial stamp immediately
                try:
                    if pt is not None:
                        diameter = int(self.brush_size if self.active_tool == 'brush' else self.eraser_size)
                        stamp = self._get_stamp(diameter, self.brush_softness if self.active_tool == 'brush' else self.eraser_softness, ('brush' if self.active_tool == 'brush' else 'eraser'))
                        if stamp is not None:
                            self._stamp_mask(mask_path, pt[0], pt[1], stamp, ('brush' if self.active_tool == 'brush' else 'eraser'))
                            # regenerate small overlay quickly
                            try:
                                cw = int(self.editor_canvas.cget('width'))
                                size = int(max(10, min(cw, int(self.editor_canvas.cget('height')))) * self.zoom_scale)
                                self._regenerate_overlay_from_inmemory(mask_path, size)
                            except Exception:
                                pass
                except Exception:
                    pass
            except Exception:
                # swallow any error during start of drawing to avoid crashing UI
                pass

    def _on_canvas_motion(self, event):
        # panning
        if self.pan_start_x is not None:
            dx = event.x - self.pan_start_x
            dy = event.y - self.pan_start_y
            self.pan_offset_x += dx
            self.pan_offset_y += dy
            self.pan_start_x = event.x
            self.pan_start_y = event.y
            self._update_current_preview()
            return

        # spline drag motion: if dragging a spline point, update its position
        if self.active_tool == 'spline' and getattr(self, '_spline_drag_index', None) is not None:
            try:
                idx = self._spline_drag_index
                if idx is not None and 0 <= idx < len(self._spline_points):
                    # update point coords; do NOT change smooth flag during dragging
                    self._spline_points[idx]['x'] = float(event.x)
                    self._spline_points[idx]['y'] = float(event.y)
                    self._render_spline_preview()
            except Exception:
                pass
            return

        # drawing
        if self._drawing and self._current_edit_mask_path and self.active_tool in ('brush', 'eraser'):
            sel = self.originals_list.curselection()
            if not sel:
                return
            idx = sel[0]
            orig_path = self.originals[idx]
            # compute image-space coords
            pt = self._canvas_to_image_coords(event.x, event.y, orig_path)
            if pt is None:
                return
            last = self._last_draw_point
            diameter = int(self.brush_size if self.active_tool == 'brush' else self.eraser_size)
            # threshold: half of radius -> radius/2 = (diameter/2)/2 = diameter/4
            threshold = max(1, diameter / 4.0)
            do_stamp = False
            if last is None:
                do_stamp = True
            else:
                dx = pt[0] - last[0]
                dy = pt[1] - last[1]
                if (dx*dx + dy*dy) ** 0.5 >= threshold:
                    do_stamp = True

            if do_stamp:
                try:
                    # fill segment between last and pt with stamps spaced by spacing = diameter/4
                    spacing = max(1.0, diameter / 4.0)
                    import math
                    if last is None:
                        start_x, start_y = pt
                    else:
                        start_x, start_y = last
                    dx_seg = pt[0] - start_x
                    dy_seg = pt[1] - start_y
                    dist_seg = math.hypot(dx_seg, dy_seg)
                    if dist_seg <= 0.0:
                        # nothing to do
                        placed_any = False
                    else:
                        n = int(math.floor(dist_seg / spacing))
                        stamp = self._get_stamp(diameter, self.brush_softness if self.active_tool == 'brush' else self.eraser_softness, ('brush' if self.active_tool == 'brush' else 'eraser'))
                        placed_any = False
                        if stamp is not None and n > 0:
                            for k in range(1, n + 1):
                                t2 = (k * spacing) / dist_seg
                                sx = int(round(start_x + dx_seg * t2))
                                sy = int(round(start_y + dy_seg * t2))
                                self._stamp_mask(self._current_edit_mask_path, sx, sy, stamp, ('brush' if self.active_tool == 'brush' else 'eraser'))
                                placed_any = True
                        # if no intermediate stamps placed (distance smaller than spacing), place endpoint stamp
                        if not placed_any:
                            # place single stamp at endpoint
                            stamp = self._get_stamp(diameter, self.brush_softness if self.active_tool == 'brush' else self.eraser_softness, ('brush' if self.active_tool == 'brush' else 'eraser'))
                            if stamp is not None:
                                self._stamp_mask(self._current_edit_mask_path, pt[0], pt[1], stamp, ('brush' if self.active_tool == 'brush' else 'eraser'))
                        # regenerate overlay cache for current preview size so updates appear immediately
                        try:
                            cw = int(self.editor_canvas.cget('width'))
                            size = int(max(10, min(cw, int(self.editor_canvas.cget('height')))) * self.zoom_scale)
                            self._regenerate_overlay_from_inmemory(self._current_edit_mask_path, size)
                        except Exception:
                            pass
                        # update last stamp point to the last placed position (endpoint)
                        self._last_draw_point = pt
                except Exception:
                    pass
            # update preview
            try:
                # refresh canvas preview for this original explicitly
                try:
                    self._show_preview_on_canvas(orig_path)
                except Exception:
                    try:
                        self._update_current_preview()
                    except Exception:
                        pass
            except Exception:
                pass
            # update last cursor pos and re-render marker so it stays visible during drawing
            try:
                self._last_cursor_canvas_pos = (event.x, event.y)
                self._render_marker_at_canvas(event.x, event.y)
            except Exception:
                pass

    def _on_canvas_release(self, event):
        # finish pan if any
        if self.pan_start_x is not None:
            self.pan_start_x = None
            self.pan_start_y = None
            return

        # finish drawing: save mask to file immediately
        if self._drawing and self._current_edit_mask_path:
            try:
                mask_path = self._current_edit_mask_path
                img = self._in_memory_mask_overrides.get(mask_path)
                # regenerate overlay cache for current preview size before saving to disk
                try:
                    cw = int(self.editor_canvas.cget('width'))
                    size = int(max(10, min(cw, int(self.editor_canvas.cget('height')))) * self.zoom_scale)
                    self._regenerate_overlay_from_inmemory(mask_path, size)
                except Exception:
                    pass
                if isinstance(img, Image.Image):
                    # write to disk atomically
                    try:
                        tmp = Path(mask_path).with_suffix('.tmp.mask.png')
                        img.save(tmp)
                        tmp.replace(mask_path)
                    except Exception:
                        try:
                            img.save(mask_path)
                        except Exception:
                            pass
                # clear state
                self._in_memory_mask_overrides.pop(mask_path, None)
                self._current_edit_mask_path = None
                self._last_draw_point = None
                self._drawing = False
                # refresh caches and preview
                self._cached_overlay = None
                self._cached_overlay_params = None
                try:
                    self._update_current_preview()
                except Exception:
                    pass
                # update mask preview thumbnail
                try:
                    sel = self.originals_list.curselection()
                    if sel:
                        idx = sel[0]
                        self._update_mask_preview(self.originals[idx])
                except Exception:
                    pass
                # touch mask watcher mtime so other tools notice
                try:
                    os.utime(mask_path, None)
                except Exception:
                    pass
                # persist settings (last file etc.)
                try:
                    self._save_window_settings_atomic()
                except Exception:
                    pass
            except Exception:
                pass

        # spline release: finalize dragging a point
        if self.active_tool == 'spline':
            try:
                idx = getattr(self, '_spline_drag_index', None)
                # finalize dragging but DO NOT flip smooth automatically on release
                if idx is not None and 0 <= idx < len(self._spline_points):
                    # keep existing smooth setting
                    pass
                self._spline_drag_index = None
                self._render_spline_preview()
            except Exception:
                pass

    def _set_active_tool(self, tool, save=True):
        self.active_tool = tool
        for t, btn in self.tool_buttons.items():
            if t == tool:
                btn.configure(fg_color='blue')
            else:
                btn.configure(fg_color=['gray70', 'gray30'])
        if tool in ['brush', 'eraser']:
            self._show_sliders()
        else:
            self._hide_sliders()
        if save:
            self._save_window_settings_atomic()

    def _show_sliders(self):
        if self.size_slider is None:
            self.size_slider = ctk.CTkSlider(self, orientation='vertical', from_=1, to=200, width=10, height=200, command=self._on_size_change)
            self.softness_slider = ctk.CTkSlider(self, orientation='vertical', from_=0, to=1, width=10, height=200, command=self._on_softness_change)
        # place left and right of toolbar
        x = self.panel_x - 20
        y = self.panel_y
        self.size_slider.place(x=x, y=y)
        self.softness_slider.place(x=self.panel_x + 50, y=y)
        # set values
        if self.active_tool == 'brush':
            self.size_slider.set(self.brush_size)
            self.softness_slider.set(self.brush_softness)
        else:
            self.size_slider.set(self.eraser_size)
            self.softness_slider.set(self.eraser_softness)

    def _hide_sliders(self):
        if self.size_slider:
            self.size_slider.place_forget()
        if self.softness_slider:
            self.softness_slider.place_forget()

    def _on_size_change(self, value):
        if self.active_tool == 'brush':
            self.brush_size = value
        else:
            self.eraser_size = value
        self._save_window_settings_atomic()
        # update marker preview if cursor present
        try:
            pos = getattr(self, '_last_cursor_canvas_pos', None)
            if pos is not None:
                self._render_marker_at_canvas(pos[0], pos[1])
        except Exception:
            pass

    def _on_softness_change(self, value):
        if self.active_tool == 'brush':
            self.brush_softness = value
        else:
            self.eraser_softness = value
        self._save_window_settings_atomic()
        # update marker preview if cursor present
        try:
            pos = getattr(self, '_last_cursor_canvas_pos', None)
            if pos is not None:
                self._render_marker_at_canvas(pos[0], pos[1])
        except Exception:
            pass

    def _start_drag(self, event):
        self._drag_start_x = event.x_root - self.drawing_toolbar.winfo_x()
        self._drag_start_y = event.y_root - self.drawing_toolbar.winfo_y()

    def _drag(self, event):
        x = event.x_root - self._drag_start_x
        y = event.y_root - self._drag_start_y
        if self._drag_after_id:
            self.after_cancel(self._drag_after_id)
        self._drag_after_id = self.after(10, lambda: self._do_drag_update(x, y))

    def _do_drag_update(self, x, y):
        self.drawing_toolbar.place(x=x, y=y)
        self.panel_x = x
        self.panel_y = y
        # move sliders if visible
        if self.size_slider and self.size_slider.winfo_ismapped():
            self.size_slider.place(x=x-20, y=y)
            self.softness_slider.place(x=x+50, y=y)
        self._schedule_save_settings()

    def _undo(self):
        try:
            sel = self.originals_list.curselection()
            if sel:
                orig_path = self.originals[sel[0]]
            else:
                # fallback to last selected file (toolbar clicks may not leave a list selection)
                orig_path = getattr(self, '_last_selected_file', None)
                if not orig_path:
                    return
            mask_path = self.original_to_mask.get(orig_path)
            if not mask_path:
                return
            stack = self._undo_stacks.get(mask_path)
            if not stack:
                return
            # current image (if any) should go to redo
            try:
                cur = self._in_memory_mask_overrides.get(mask_path)
                if cur is not None:
                    self._redo_stacks.setdefault(mask_path, []).append(cur.copy())
                else:
                    # if no in-memory override exists, try to load the on-disk mask
                    try:
                        if Path(mask_path).is_file():
                            on_disk = Image.open(mask_path).convert('L')
                            self._redo_stacks.setdefault(mask_path, []).append(on_disk.copy())
                    except Exception:
                        pass
            except Exception:
                pass
            img = stack.pop()
            # if stack emptied, remove key to keep state tidy
            try:
                if not stack:
                    self._undo_stacks.pop(mask_path, None)
            except Exception:
                pass
            # set image as current in-memory override
            try:
                self._in_memory_mask_overrides[mask_path] = img.copy()
            except Exception:
                pass
            # also write restored image to disk atomically so on-disk preview picks it up
            try:
                if isinstance(img, Image.Image):
                    try:
                        # write to a temp file in the same directory then replace
                        dirp = Path(mask_path).parent
                        fd, tmp_path = tempfile.mkstemp(prefix=Path(mask_path).name, dir=str(dirp))
                        try:
                            with os.fdopen(fd, 'wb') as f:
                                img.save(f, format='PNG')
                        finally:
                            try:
                                os.replace(tmp_path, str(mask_path))
                            except Exception:
                                # last resort: try direct save
                                try:
                                    img.save(mask_path)
                                except Exception:
                                    pass
                    except Exception:
                        try:
                            img.save(mask_path)
                        except Exception:
                            pass
            except Exception:
                pass
            # refresh caches and previews
            try:
                self._cached_overlay = None
                self._cached_overlay_params = None
            except Exception:
                pass
            try:
                # regenerate overlay from in-memory so preview updates immediately
                cw = int(self.editor_canvas.cget('width'))
                size = int(max(10, min(cw, int(self.editor_canvas.cget('height')))) * self.zoom_scale)
                try:
                    self._regenerate_overlay_from_inmemory(mask_path, size)
                except Exception:
                    pass
            except Exception:
                pass
            try:
                self._update_current_preview()
            except Exception:
                pass
            try:
                self._update_mask_preview(orig_path)
            except Exception:
                pass
        except Exception:
            pass

    def _redo(self):
        try:
            sel = self.originals_list.curselection()
            if sel:
                orig_path = self.originals[sel[0]]
            else:
                orig_path = getattr(self, '_last_selected_file', None)
                if not orig_path:
                    return
            mask_path = self.original_to_mask.get(orig_path)
            if not mask_path:
                return
            stack = self._redo_stacks.get(mask_path)
            if not stack:
                return
            # current image goes back to undo
            try:
                cur = self._in_memory_mask_overrides.get(mask_path)
                if cur is not None:
                    self._undo_stacks.setdefault(mask_path, []).append(cur.copy())
                else:
                    # if no in-memory override, try to load the on-disk mask into undo
                    try:
                        if Path(mask_path).is_file():
                            on_disk = Image.open(mask_path).convert('L')
                            self._undo_stacks.setdefault(mask_path, []).append(on_disk.copy())
                    except Exception:
                        pass
            except Exception:
                pass
            img = stack.pop()
            try:
                # if stack emptied, remove key
                if not stack:
                    self._redo_stacks.pop(mask_path, None)
            except Exception:
                pass
            try:
                self._in_memory_mask_overrides[mask_path] = img.copy()
            except Exception:
                pass
            # also write restored image to disk atomically so on-disk preview picks it up
            try:
                if isinstance(img, Image.Image):
                    try:
                        dirp = Path(mask_path).parent
                        fd, tmp_path = tempfile.mkstemp(prefix=Path(mask_path).name, dir=str(dirp))
                        try:
                            with os.fdopen(fd, 'wb') as f:
                                img.save(f, format='PNG')
                        finally:
                            try:
                                os.replace(tmp_path, str(mask_path))
                            except Exception:
                                try:
                                    img.save(mask_path)
                                except Exception:
                                    pass
                    except Exception:
                        try:
                            img.save(mask_path)
                        except Exception:
                            pass
            except Exception:
                pass
            try:
                self._cached_overlay = None
                self._cached_overlay_params = None
            except Exception:
                pass
            try:
                # regenerate overlay from in-memory so preview updates immediately
                cw = int(self.editor_canvas.cget('width'))
                size = int(max(10, min(cw, int(self.editor_canvas.cget('height')))) * self.zoom_scale)
                try:
                    self._regenerate_overlay_from_inmemory(mask_path, size)
                except Exception:
                    pass
            except Exception:
                pass
            try:
                self._update_current_preview()
            except Exception:
                pass
            try:
                self._update_mask_preview(orig_path)
            except Exception:
                pass
        except Exception:
            pass

    def _push_undo(self, mask_path: str, img: Image.Image):
        """Push a copy of img onto the undo stack for mask_path and clear redo stack."""
        try:
            if not mask_path or img is None:
                return
            # ensure stacks exist
            st = self._undo_stacks.setdefault(mask_path, [])
            # store a copy
            try:
                st.append(img.copy())
            except Exception:
                st.append(img)
            # trim to reasonable history limit (e.g., 20)
            try:
                if len(st) > 20:
                    del st[0:len(st)-20]
            except Exception:
                pass
            # clear redo on new action
            try:
                self._redo_stacks.pop(mask_path, None)
            except Exception:
                pass
        except Exception:
            pass

    def _delete_mask(self):
        try:
            sel = self.originals_list.curselection()
            if not sel:
                # fallback to last selected
                orig_path = getattr(self, '_last_selected_file', None)
                if not orig_path:
                    return
            else:
                orig_path = self.originals[sel[0]]

            mask_path = self.original_to_mask.get(orig_path)
            if not mask_path:
                return

            # confirm with user
            try:
                from tkinter import messagebox
                ok = messagebox.askyesno('Delete mask', f'Delete mask file:\n{mask_path}?')
            except Exception:
                # best-effort: if messagebox not available, skip confirmation
                ok = True
            if not ok:
                return

            # attempt to remove file
            try:
                p = Path(mask_path)
                if p.exists():
                    try:
                        p.unlink()
                    except Exception:
                        # fallback: try os.remove
                        try:
                            os.remove(str(p))
                        except Exception:
                            pass
            except Exception:
                pass

            # clear in-memory override and stacks for this mask
            try:
                self._in_memory_mask_overrides.pop(mask_path, None)
            except Exception:
                pass
            try:
                self._undo_stacks.pop(mask_path, None)
            except Exception:
                pass
            try:
                self._redo_stacks.pop(mask_path, None)
            except Exception:
                pass

            # update original_to_mask mapping and db rows
            try:
                self.original_to_mask[orig_path] = None
            except Exception:
                pass
            try:
                rows = self.db.get('rows', [])
                new_rows = []
                for r in rows:
                    if r[0] == Path(orig_path).name:
                        new_rows.append((r[0], '', r[2]))
                    else:
                        new_rows.append(r)
                self.db['rows'] = new_rows
            except Exception:
                pass

            # refresh UI
            try:
                self._cached_overlay = None
                self._cached_overlay_params = None
            except Exception:
                pass
            try:
                # refresh canvas preview and thumbnail
                try:
                    self._show_preview_on_canvas(orig_path)
                except Exception:
                    try:
                        self._update_current_preview()
                    except Exception:
                        pass
                try:
                    self._update_mask_preview(orig_path)
                except Exception:
                    pass
                try:
                    self._update_lists()
                except Exception:
                    pass
            except Exception:
                pass
        except Exception:
            pass
    
    def _invert_mask(self):
        """Invert the currently selected mask (on-disk or in-memory override).
        The inversion is applied to the grayscale mask (L) where 0->255 and 255->0.
        After inversion, overlay cache and preview are refreshed and changes are
        written atomically to disk.
        """
        try:
            sel = self.originals_list.curselection()
            if not sel:
                return
            idx = sel[0]
            orig_path = self.originals[idx]
            mask_path = self.original_to_mask.get(orig_path)
            if not mask_path:
                # no mask to invert
                return

            # prefer in-memory mask if being edited
            mimg = None
            try:
                mimg = self._in_memory_mask_overrides.get(mask_path)
            except Exception:
                mimg = None

            # load from disk if needed
            if mimg is None:
                try:
                    if Path(mask_path).is_file():
                        mimg = Image.open(mask_path).convert('L')
                except Exception:
                    mimg = None

            if mimg is None:
                return

            # push undo before invert
            try:
                self._push_undo(mask_path, mimg.copy())
            except Exception:
                pass

            # invert
            try:
                # using point for fast inversion
                inv = mimg.point(lambda p: 255 - p)
            except Exception:
                try:
                    # fallback pixel-wise
                    inv = Image.eval(mimg, lambda p: 255 - p)
                except Exception:
                    return

            # store back to in-memory overrides so UI can show immediate change
            try:
                self._in_memory_mask_overrides[mask_path] = inv
            except Exception:
                pass

            # write to disk atomically
            try:
                tmp = Path(mask_path).with_suffix('.tmp.mask.png')
                inv.save(tmp)
                tmp.replace(mask_path)
            except Exception:
                try:
                    inv.save(mask_path)
                except Exception:
                    pass

            # clear cached overlay so it will be rebuilt
            try:
                self._cached_overlay = None
                self._cached_overlay_params = None
            except Exception:
                pass

            # regenerate overlay for current preview size if possible
            try:
                cw = int(self.editor_canvas.cget('width'))
                size = int(max(10, min(cw, int(self.editor_canvas.cget('height')))) * self.zoom_scale)
                self._regenerate_overlay_from_inmemory(mask_path, size)
            except Exception:
                pass

            # refresh previews
            try:
                self._update_current_preview()
            except Exception:
                pass
            try:
                self._update_mask_preview(orig_path)
            except Exception:
                pass

            # touch mtime for external watchers
            try:
                os.utime(mask_path, None)
            except Exception:
                pass

            # persist settings (last file etc.)
            try:
                self._save_window_settings_atomic()
            except Exception:
                pass
        except Exception:
            pass
        self.panel_x = x
        self.panel_y = y
        # move sliders if visible
        if self.size_slider and self.size_slider.winfo_ismapped():
            self.size_slider.place(x=x-30, y=y)
            self.softness_slider.place(x=x+50, y=y)
        self._schedule_save_settings()


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
