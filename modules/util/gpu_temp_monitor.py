import time
import subprocess
import os

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
    if not enabled:
        return

    temps = get_gpu_temps()
    if temps is None or len(temps) == 0:
        # cannot read temps
        return

    max_t = getattr(config, 'gpu_temp_max', 70)
    cool_to = getattr(config, 'gpu_temp_cool_to', 60)

    if any(t >= max_t for t in temps):
        callbacks.on_update_status(f"GPU temp exceeded {max_t}C, pausing training until <= {cool_to}C")
        logger_print(f"GPU temp exceeded {max_t}C: {temps}")

        # block until cooled
        while True:
            time.sleep(1)
            temps = get_gpu_temps()
            if temps is None:
                break
            if all(t <= cool_to for t in temps):
                callbacks.on_update_status(f"GPU cooled to {temps}, resuming training")
                logger_print(f"GPU cooled to {temps}")
                break
