from abc import ABCMeta, abstractmethod

from modules.dataLoader.mixin.DataLoaderMgdsMixin import DataLoaderMgdsMixin

from mgds.MGDS import MGDS, TrainDataLoader
import weakref

import torch
from typing import Iterator

from modules.util import gpu_temp_monitor

# A weak-key mapping from DiskCache/SaveImage instances to (config, callbacks).
# Some vendored classes may disallow setting arbitrary attributes; register
# into this map as a fallback so the monkeypatch can still find the monitor
# tuple for an instance.
import weakref

_DISKCACHE_MONITOR_MAP = weakref.WeakKeyDictionary()
# Global fallback config/callbacks set by trainer at startup as a last-resort
# fallback if instance-specific registration fails.
_GLOBAL_TRAINER_CONFIG = None
_GLOBAL_TRAINER_CALLBACKS = None

def register_monitor_for_instance(inst, cfg, callbacks):
    """Register a monitor tuple for an mgds module instance (DiskCache/SaveImage).
    Attempts to set attributes on the instance first; if that fails, stores
    the tuple in a WeakKeyDictionary so monkeypatches can retrieve it later.
    """
    try:
        setattr(inst, '_monitor_config', cfg)
        setattr(inst, '_monitor_callbacks', callbacks)
        try:
            print(f"[REGISTER_MONITOR] set attrs on instance type={type(inst)} cfg_id={id(cfg) if cfg is not None else None}")
        except Exception:
            pass
        return
    except Exception as e_attr:
        try:
            _DISKCACHE_MONITOR_MAP[inst] = (cfg, callbacks)
            try:
                print(f"[REGISTER_MONITOR] stored in WeakKeyDictionary instance type={type(inst)} cfg_id={id(cfg) if cfg is not None else None}")
            except Exception:
                pass
            return
        except Exception as e_map:
            try:
                print(f"[REGISTER_MONITOR] FAILED to register monitor for instance type={type(inst)} cfg_id={id(cfg) if cfg is not None else None} attr_err={e_attr!r} map_err={e_map!r}")
            except Exception:
                pass
            return


def get_monitor_for_instance(inst):
    """Return (cfg, callbacks) for a registered instance, falling back to
    attributes if present or the WeakKeyDictionary mapping.
    """
    cfg = getattr(inst, '_monitor_config', None)
    cbs = getattr(inst, '_monitor_callbacks', None)
    if cfg is None:
        try:
            t = _DISKCACHE_MONITOR_MAP.get(inst)
            if t is not None:
                return t
        except Exception:
            pass
    return (cfg, cbs)


class BaseDataLoader(
    DataLoaderMgdsMixin,
    metaclass=ABCMeta,
):

    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
    ):
        super().__init__()

        self.train_device = train_device
        self.temp_device = temp_device

    class TempCheckedTrainDataLoader:
        """Iterator wrapper around a TrainDataLoader that calls the GPU temp monitor
        before yielding each item. This ensures temperature checks run for any
        code that iterates over the TrainDataLoader (caching, enumerating, saving).
        """

        def __init__(self, inner, cfg, callbacks):
            self._inner = inner
            self._cfg = cfg
            self._callbacks = callbacks

        def __iter__(self) -> Iterator:
            for item in self._inner:
                try:
                    gpu_temp_monitor.pause_if_overtemp_if_needed(self._cfg, self._callbacks)
                except Exception:
                    pass
                yield item

        def __getattr__(self, name):
            # delegate attribute access to inner loader
            return getattr(self._inner, name)

    def wrap_train_dataloader(self, dl):
        # prefer to wrap only if config present on self; allow None for compatibility
        cfg = getattr(self, 'config', None)
        callbacks = getattr(self, 'callbacks', None)
        return BaseDataLoader.TempCheckedTrainDataLoader(dl, cfg, callbacks)

    @abstractmethod
    def get_data_set(self) -> MGDS:
        pass

    @abstractmethod
    def get_data_loader(self) -> TrainDataLoader:
        pass


# Monkeypatch mgds internals at import time so we don't modify vendored packages on disk.
def _apply_mgds_monkeypatches():
    try:
        import concurrent.futures
        import mgds.pipelineModules.DiskCache as _dc_mod
        # Patch ThreadPoolExecutor.submit once: if an executor instance has attribute
        # '_before_submit_wrap' we wrap submitted callables so that the hook runs in the worker thread
        Executor = concurrent.futures.ThreadPoolExecutor

        if not hasattr(Executor, '_monkeypatched_submit_with_monitor'):
            orig_submit = Executor.submit

            def _submit_with_monitor(self, fn, *args, **kwargs):
                before = getattr(self, '_before_submit_wrap', None)
                if callable(before):
                    def _wrapped(*a, **k):
                        try:
                            before()
                        except Exception:
                            pass
                        return fn(*a, **k)

                    return orig_submit(self, _wrapped, *args, **kwargs)
                return orig_submit(self, fn, *args, **kwargs)

            Executor.submit = _submit_with_monitor
            try:
                setattr(Executor, '_monkeypatched_submit_with_monitor', True)
            except Exception:
                pass

        DiskCache = _dc_mod.DiskCache

        # wrap DiskCache.__refresh_cache so it sets executor._before_submit_wrap to call the GPU monitor
        refresh_name = '_DiskCache__refresh_cache'
        if hasattr(DiskCache, refresh_name) and not hasattr(DiskCache, '_monkeypatched_refresh'):
            orig_refresh = getattr(DiskCache, refresh_name)

            def patched_refresh(self, out_variation):
                global _CURRENT_DISKCACHE_MONITOR
                # set per-executor hook so that worker threads call the gpu temp check
                try:
                    exe = getattr(self, '_state').executor
                    def _before():
                        try:
                            from modules.util import gpu_temp_monitor
                            # prefer attributes on the instance, then fallback to the registry
                            cfg, cbs = get_monitor_for_instance(self)
                            # fallback to global trainer config if instance-specific not found
                            if cfg is None:
                                try:
                                    cfg = globals().get('_GLOBAL_TRAINER_CONFIG', None)
                                    cbs = globals().get('_GLOBAL_TRAINER_CALLBACKS', None)
                                    # print(f"GTM cfg_id={id(cfg) if cfg is not None else None}")
                                except Exception:
                                    pass
                            # import threading as _th
                            # has_attr = getattr(self, '_monitor_config', None) is not None or getattr(self, '_monitor_callbacks', None) is not None
                            # try:
                            #     in_map = (self in _DISKCACHE_MONITOR_MAP)
                            # except Exception:
                            #     in_map = False
                            # cfg_id = id(cfg) if cfg is not None else None
                            # enabled_val = getattr(cfg, 'gpu_temp_control_enabled', None) if cfg is not None else None
                            # print(f"GTM instance_id={id(self)} has_attr={has_attr} in_weakmap={in_map} cfg_id={cfg_id} enabled={enabled_val}\n")
                            # temps = gpu_temp_monitor.get_gpu_temps()
                            # print(f"GTM temps={temps}")
                            gpu_temp_monitor.pause_if_overtemp_if_needed(cfg, cbs)
                        except Exception:
                            pass

                    setattr(exe, '_before_submit_wrap', _before)
                except Exception:
                    exe = None

                # also set a global variable so patched tqdm can call monitor in the main thread loop
                try:
                    cfg_preview, cbs_preview = get_monitor_for_instance(self)
                    if cfg_preview is None:
                        # fallback to global trainer config
                        cfg_preview = globals().get('_GLOBAL_TRAINER_CONFIG', None)
                        cbs_preview = globals().get('_GLOBAL_TRAINER_CALLBACKS', None)
                    _CURRENT_DISKCACHE_MONITOR = (cfg_preview, cbs_preview)
                    # import threading as _th
                    # print(f"GTM={getattr(cfg_preview,'gpu_temp_control_enabled', None)}\n")
                except Exception:
                    _CURRENT_DISKCACHE_MONITOR = (None, None)

                try:
                    return orig_refresh(self, out_variation)
                finally:
                    try:
                        if exe is not None and hasattr(exe, '_before_submit_wrap'):
                            delattr(exe, '_before_submit_wrap')
                    except Exception:
                        pass
                    try:
                        _CURRENT_DISKCACHE_MONITOR = (None, None)
                    except Exception:
                        pass

            setattr(DiskCache, refresh_name, patched_refresh)
            try:
                setattr(DiskCache, '_monkeypatched_refresh', True)
            except Exception:
                pass
    except Exception:
        # Best-effort; don't crash import if mgds not available
        pass


def _apply_saveimage_monkeypatch():
    try:
        import mgds.pipelineModules.SaveImage as _si_mod
        if hasattr(_si_mod, 'SaveImage'):
            SaveImage = _si_mod.SaveImage
            if not hasattr(SaveImage, '_monkeypatched_start'):
                orig_start = getattr(SaveImage, 'start', None)

                def patched_start(self, variation: int):
                    # when saving debug images, tqdm is used inside the original start method
                    # we set a global monitor reference so that a small wrapper around tqdm iteration
                    # will call the gpu monitor before each iteration in this (worker/main) thread.
                    global _CURRENT_SAVEIMAGE_MONITOR
                    try:
                        # prefer attributes on the instance, then fallback to registry
                        from modules.dataLoader.BaseDataLoader import get_monitor_for_instance
                        _CURRENT_SAVEIMAGE_MONITOR = get_monitor_for_instance(self)
                    except Exception:
                        try:
                            _CURRENT_SAVEIMAGE_MONITOR = (getattr(self, '_monitor_config', None), getattr(self, '_monitor_callbacks', None))
                        except Exception:
                            _CURRENT_SAVEIMAGE_MONITOR = (None, None)

                    try:
                        if orig_start is not None:
                            return orig_start(self, variation)
                    finally:
                        try:
                            _CURRENT_SAVEIMAGE_MONITOR = (None, None)
                        except Exception:
                            pass

                SaveImage.start = patched_start
                try:
                    setattr(SaveImage, '_monkeypatched_start', True)
                except Exception:
                    pass
    except Exception:
        pass


# Apply monkeypatches eagerly
_apply_mgds_monkeypatches()
_apply_saveimage_monkeypatch()


# Patch tqdm iterator to call monitor when specific desc values are in use.
def _patch_tqdm_iter_for_monitor():
    try:
        from tqdm import tqdm as _tq
        if not hasattr(_tq, '_monkeypatched_iter_monitor'):
            orig_iter = _tq.__iter__

            def _iter_with_monitor(self):
                # call original iterator
                it = orig_iter(self)
                desc = getattr(self, 'desc', '')

                # decide if we need to run monitor before each step
                monitor_needed = False
                # we want temperature checks for caching, debug-image writing and sampling loops
                if desc == 'caching' or desc == 'sampling' or (isinstance(desc, str) and desc.startswith("writing debug images for '")):
                    monitor_needed = True

                if not monitor_needed:
                    for x in it:
                        yield x
                    return

                # if monitor is needed, attempt to retrieve current monitor tuple(s)
                from modules.util import gpu_temp_monitor
                for x in it:
                    try:
                        # prefer per-save monitor, then diskcache
                        m_cfg, m_cbs = (None, None)
                        try:
                            m_cfg, m_cbs = globals().get('_CURRENT_SAVEIMAGE_MONITOR', (None, None))
                        except Exception:
                            m_cfg, m_cbs = (None, None)
                        if m_cfg is None:
                            try:
                                m_cfg, m_cbs = globals().get('_CURRENT_DISKCACHE_MONITOR', (None, None))
                            except Exception:
                                m_cfg, m_cbs = (None, None)

                        # If per-save or per-diskcache monitor tuple not available, fall back
                        # to the global trainer config/callbacks if present so sampling code
                        # (which often runs outside DiskCache/SaveImage contexts) is covered.
                        if m_cfg is None:
                            try:
                                m_cfg = globals().get('_GLOBAL_TRAINER_CONFIG', None)
                                m_cbs = globals().get('_GLOBAL_TRAINER_CALLBACKS', None)
                                # print(f"GTM={id(m_cfg) if m_cfg is not None else None}")
                            except Exception:
                                pass

                        if m_cfg is not None:
                            try:
                                gpu_temp_monitor.pause_if_overtemp_if_needed(m_cfg, m_cbs)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    yield x

            _tq.__iter__ = _iter_with_monitor
            try:
                setattr(_tq, '_monkeypatched_iter_monitor', True)
            except Exception:
                pass
    except Exception:
        pass


_patch_tqdm_iter_for_monitor()
 
