import time
import subprocess
import os
import threading
import inspect

def _get_gpu_temps_pynvml():
    try:
        import pynvml
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        temps = []
        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            temps.append(pynvml.nvmlDeviceGetTemperature(h, 0))
        return temps
    except Exception:
        return None


def _get_gpu_temps_nvidia_smi():
    try:
        out = subprocess.check_output([
            "nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"
        ], universal_newlines=True)
        return [int(x.strip()) for x in out.splitlines() if x.strip()]
    except Exception:
        return None


def get_gpu_temps():
    t = _get_gpu_temps_pynvml()
    if t is not None:
        return t
    return _get_gpu_temps_nvidia_smi()


def pause_if_overtemp_if_needed(config, callbacks, logger_print=print):
    # config is expected to have attributes: gpu_temp_control_enabled, gpu_temp_max, gpu_temp_cool_to
    enabled = getattr(config, 'gpu_temp_control_enabled', False)
    # print(f"GTM={enabled}")
    # Support a debugging override: if environment variable ONETRAINER_FORCE_GPU_MONITOR is set,
    # treat the monitor as enabled regardless of config. This is useful to quickly verify
    # pause/resume behaviour while diagnosing registration issues.
    try:
        force_env = os.environ.get('ONETRAINER_FORCE_GPU_MONITOR', '')
        if force_env.lower() in ('1', 'true', 'yes'):
            # print(f"[GPU_MONITOR] FORCE ENABLED BY ENV ONETRAINER_FORCE_GPU_MONITOR={force_env}")
            # print(f"GTM={force_env}")
            enabled = True
    except Exception:
        pass

    if not enabled:
        return

    temps = get_gpu_temps()
    # print(f"GTM temps={temps}")
    if temps is None or len(temps) == 0:
        # cannot read temps
        return

    max_t = getattr(config, 'gpu_temp_max', 70)
    cool_to = getattr(config, 'gpu_temp_cool_to', 60)

    if any(t >= max_t for t in temps):
        # Notify status if possible, but don't fail if callbacks are not usable in this context
        try:
            if callbacks is not None:
                callbacks.on_update_status(f"GPU temp exceeded {max_t}C, pausing training until <= {cool_to}C")
        except Exception:
            try:
                logger_print(f"GPU temp exceeded {max_t}C, but callbacks.on_update_status failed")
            except Exception:
                pass

        try:
            logger_print(f"\nGPU temp exceeded {max_t}C, cool to {cool_to}C: {temps}\n")
        except Exception:
            pass
        # print(f"GTM detected overtemp {temps}; pausing until <= {cool_to}")

        # block until cooled; tolerate transient failures to read temps
        while True:
            time.sleep(1)
            temps = get_gpu_temps()
            if temps is None:
                # couldn't read temps right now; keep trying
                continue
            if all(t <= cool_to for t in temps):
                try:
                    if callbacks is not None:
                        callbacks.on_update_status(f"GPU cooled to {temps}, resuming training")
                except Exception:
                    try:
                        logger_print(f"GPU cooled to {temps} (callbacks failed to notify)")
                    except Exception:
                        pass

                try:
                    logger_print(f"GPU cooled to {temps}")
                except Exception:
                    pass
                # print(f"GTM resumed; temps={temps}")

                break
