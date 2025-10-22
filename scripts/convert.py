#!/usr/bin/env python3
"""
Convert OneTrainer model artifacts (backup/internal safetensors or final safetensors)
into generator-compatible legacy safetensors.

Usage:
    python convert.py path/to/input.safetensors [--out path/to/output.safetensors]

If --out is omitted, the script writes <input_basename>_legacy.safetensors next to input.

Behavior:
- Detects if the provided file is a LoRA bundle (lora.safetensors in an internal backup folder),
  a LoRA standalone, a base model (unet/te bundle), or embeddings/VAE candidates.
- For LoRA files it will attempt to locate an appropriate key conversion for the target model
  using the project's omi_model_standards converters if available. If not available, it will
  produce a best-effort legacy conversion (renaming keys heuristically) but will not drop
  tensors.
- Preserves safetensors header fields (ot_* metadata) and appends a note about conversion.

Notes:
- This script depends on "safetensors" and the optional "omi_model_standards" package.
- Run inside the project's Python environment (venv) where requirements are installed.
"""
import argparse
import os
import sys
from pathlib import Path

try:
    from safetensors import safe_open
    from safetensors.torch import save_file, load_file
except Exception as e:
    print(
        "Missing safetensors package. Install requirements or run inside project's venv."
    )
    raise

# omi_model_standards is required for reliable conversion. Fail early with clear message
try:
    from omi_model_standards.convert.lora.convert_lora_util import (
        convert_to_legacy_diffusers,
        convert_to_omi,
    )
except Exception as e:
    print("Required package 'omi_model_standards' not found or failed to import.")
    print(
        "Please install project's global requirements (requirements-global.txt) in your venv."
    )
    raise

# Collect known keyset providers
_KEYSET_PROVIDERS = {}
_known_providers = {
    "sd": "omi_model_standards.convert.lora.convert_sd_lora",
    "sdxl": "omi_model_standards.convert.lora.convert_sdxl_lora",
    "sd3": "omi_model_standards.convert.lora.convert_sd3_lora",
    "flux": "omi_model_standards.convert.lora.convert_flux_lora",
    "chroma": "omi_model_standards.convert.lora.convert_chroma_lora",
    "pixart": "omi_model_standards.convert.lora.convert_pixart_lora",
    "qwen": "omi_model_standards.convert.lora.convert_qwen_lora",
    "sana": "omi_model_standards.convert.lora.convert_sana_lora",
    "hunyuan": "omi_model_standards.convert.lora.convert_hunyuan_video_lora",
    "hidream": "omi_model_standards.convert.lora.convert_hidream_lora",
    "wuerstchen": "omi_model_standards.convert.lora.convert_stable_cascade_lora",
}

for name, modpath in _known_providers.items():
    try:
        mod = __import__(modpath, fromlist=["*"])
        func_name = [n for n in dir(mod) if n.endswith("_key_sets")]
        if func_name:
            func = getattr(mod, func_name[0])
            _KEYSET_PROVIDERS[name] = func
    except Exception:
        pass


def detect_type(state_dict: dict) -> str:
    """Detect artifact type: 'lora', 'model', 'vae', 'embedding', or 'unknown'."""
    keys = list(state_dict.keys())
    low = [k.lower() for k in keys]

    # heuristics
    if any(
        k.startswith("lora") or k.startswith("lora_") or ".lora" in k or "alpha" in k
        for k in low
    ):
        return "lora"
    # common unet/te keys for sd models
    if any("unet" in k or "transformer" in k or "text_encoder" in k for k in low):
        return "model"
    if any("vae" in k or "decoder" in k for k in low):
        return "vae"
    if any("embed" in k or "token" in k or "bundle_emb" in k for k in low):
        return "embedding"
    return "unknown"


def detect_lora_family(keys: list[str]) -> str | None:
    """Detect LoRA family used by the trainer savers (sdxl, sd, sd3, qwen, flux, chroma, pixart, sana, hunyuan, hidream, wuerstchen).

    Returns provider key (e.g. 'sdxl') or None when unknown.
    This mirrors the saver selection used in the trainer pipeline.
    """
    low = [k.lower() for k in keys]

    # SDXL: modern text_model.encoder paths + clip_l/clip_g or bundle_emb
    if any(
        "text_model.encoder" in k or k.startswith("text_model.") for k in low
    ) or any("clip_l" in k or "clip_g" in k or "bundle_emb" in k for k in low):
        return "sdxl"

    # SD classic (v1/v2): older text_encoder or transformer keys and unet
    if any("text_encoder" in k or "transformer" in k for k in low) and any(
        "unet" in k for k in low
    ):
        return "sd"

    if any("sd3" in k for k in low) or any("stable_diffusion3" in k for k in low):
        return "sd3"

    if any("qwen" in k for k in low) or any(
        "transformer" in k and "qwen" in k for k in low
    ):
        return "qwen"

    if any("flux" in k for k in low):
        return "flux"

    if any("chroma" in k for k in low) or any("bundle_emb" in k for k in low):
        return "chroma"

    if any("pixart" in k for k in low):
        return "pixart"

    if any("hunyuan" in k for k in low):
        return "hunyuan"

    if any("hidream" in k for k in low):
        return "hidream"

    if any("wuerst" in k for k in low):
        return "wuerstchen"

    if any("sana" in k for k in low):
        return "sana"

    return None


def _select_key_sets_from_keys(keys: list[str]):
    """Try to pick a keyset provider by simple heuristics on tensor names."""
    low = [k.lower() for k in keys]
    # heuristics mapping
    # SDXL indicators: explicit tags or the modern text encoder path used by SDXL
    if any(
        (
            "sd_xl" in k
            or "sdxl" in k
            or "refiner" in k
            or "text_model.encoder" in k
            or k.startswith("text_model.")
        )
        for k in low
    ):
        return _KEYSET_PROVIDERS.get("sdxl") if _KEYSET_PROVIDERS else None

    # Stable Diffusion v1/v2 style indicators
    if any(
        ("unet" in k and "down" in k) or "text_encoder" in k or "transformer" in k
        for k in low
    ):
        return _KEYSET_PROVIDERS.get("sd") if _KEYSET_PROVIDERS else None
    if any("transformer" in k and "qwen" in k for k in low) or any(
        "qwen" in k for k in low
    ):
        return _KEYSET_PROVIDERS.get("qwen") if _KEYSET_PROVIDERS else None
    if any("flux" in k for k in low):
        return _KEYSET_PROVIDERS.get("flux") if _KEYSET_PROVIDERS else None
    if any("chroma" in k for k in low) or any("bundle_emb" in k for k in low):
        return _KEYSET_PROVIDERS.get("chroma") if _KEYSET_PROVIDERS else None
    if any("pixart" in k for k in low):
        return _KEYSET_PROVIDERS.get("pixart") if _KEYSET_PROVIDERS else None
    if any("hunyuan" in k for k in low):
        return _KEYSET_PROVIDERS.get("hunyuan") if _KEYSET_PROVIDERS else None
    if any("hidream" in k for k in low):
        return _KEYSET_PROVIDERS.get("hidream") if _KEYSET_PROVIDERS else None
    if any("wuerst" in k for k in low):
        return _KEYSET_PROVIDERS.get("wuerstchen") if _KEYSET_PROVIDERS else None
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", help="input safetensors file path")
    p.add_argument("--out", help="output safetensors path (optional)")
    p.add_argument("--force", action="store_true", help="overwrite output if exists")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="do not write output, only show planned actions",
    )
    args = p.parse_args()

    inp = Path(args.input)
    # allow directory input (backup folder)
    if not inp.exists():
        print("Input path not found:", inp)
        sys.exit(2)

    # If directory was passed, try to find lora/lora.safetensors
    if inp.is_dir():
        candidate = inp / "lora" / "lora.safetensors"
        if candidate.exists():
            inp = candidate
            print(f"Using {inp} found in provided directory")
        else:
            # fallback: pick first .safetensors under dir
            found = list(inp.rglob("*.safetensors"))
            if found:
                inp = found[0]
                print(f"Using {inp} (first .safetensors found in directory)")
            else:
                print("No .safetensors file found in directory")
                sys.exit(2)

    out = (
        Path(args.out) if args.out else inp.with_name(inp.stem + "_legacy" + inp.suffix)
    )

    try:
        data = load_file(str(inp))
    except Exception as e:
        print("Error reading safetensors:", e)
        sys.exit(3)

    ttype = detect_type(data)
    print(f"Detected type: {ttype}")

    # Default: if OMI available and this is lora, try convert_to_legacy_diffusers
    converted = None
    header = {}
    try:
        # preserve header
        try:
            with safe_open(str(inp), framework="pt") as f:
                header = f.metadata or {}
                # some safetensors versions expose metadata as a callable
                if callable(header):
                    try:
                        header = header()
                    except Exception:
                        header = {}
        except Exception:
            header = {}

        if ttype != "lora":
            print(
                "Input does not look like a LoRA bundle. Copying tensors as-is into legacy file."
            )
            converted = data
        else:
            print("Converting LoRA to legacy safetensors using omi_model_standards")
            # Prefer exact family detection that mirrors trainer saver selection
            family = detect_lora_family(list(data.keys()))
            keyset_provider = None
            provider_name = None
            if family is not None:
                # try to load explicit provider function from omi_model_standards
                try:
                    if family == "sdxl":
                        from omi_model_standards.convert.lora.convert_sdxl_lora import (
                            convert_sdxl_lora_key_sets as _prov,
                        )
                    elif family == "sd":
                        from omi_model_standards.convert.lora.convert_sd_lora import (
                            convert_sd_lora_key_sets as _prov,
                        )
                    elif family == "sd3":
                        from omi_model_standards.convert.lora.convert_sd3_lora import (
                            convert_sd3_lora_key_sets as _prov,
                        )
                    elif family == "qwen":
                        from omi_model_standards.convert.lora.convert_qwen_lora import (
                            convert_qwen_lora_key_sets as _prov,
                        )
                    elif family == "flux":
                        from omi_model_standards.convert.lora.convert_flux_lora import (
                            convert_flux_lora_key_sets as _prov,
                        )
                    elif family == "chroma":
                        from omi_model_standards.convert.lora.convert_chroma_lora import (
                            convert_chroma_lora_key_sets as _prov,
                        )
                    elif family == "pixart":
                        from omi_model_standards.convert.lora.convert_pixart_lora import (
                            convert_pixart_lora_key_sets as _prov,
                        )
                    elif family == "hunyuan":
                        from omi_model_standards.convert.lora.convert_hunyuan_video_lora import (
                            convert_hunyuan_video_lora_key_sets as _prov,
                        )
                    elif family == "hidream":
                        from omi_model_standards.convert.lora.convert_hidream_lora import (
                            convert_hidream_lora_key_sets as _prov,
                        )
                    elif family == "wuerstchen":
                        from omi_model_standards.convert.lora.convert_stable_cascade_lora import (
                            convert_stable_cascade_lora_key_sets as _prov,
                        )
                    elif family == "sana":
                        from omi_model_standards.convert.lora.convert_sana_lora import (
                            convert_sana_lora_key_sets as _prov,
                        )
                    else:
                        _prov = None

                    if _prov is not None:
                        keyset_provider = _prov
                        provider_name = family
                except Exception:
                    # if explicit import failed, fall back to generic registry
                    keyset_provider = (
                        _KEYSET_PROVIDERS.get(family) if _KEYSET_PROVIDERS else None
                    )
                    provider_name = family if keyset_provider is not None else None

            # fallback: use earlier heuristic-based selection
            if keyset_provider is None:
                keyset_provider = _select_key_sets_from_keys(list(data.keys()))
                if keyset_provider is not None:
                    for _n, _f in _KEYSET_PROVIDERS.items():
                        if _f is keyset_provider:
                            provider_name = _n

            print(f"DEBUG: selected keyset provider: {provider_name}")
            key_sets = None
            if keyset_provider is not None:
                try:
                    key_sets = keyset_provider()
                except Exception:
                    key_sets = None

            # perform conversion (may accept None key_sets)
            try:
                print("DEBUG: key_sets type:", type(key_sets))
                if hasattr(key_sets, "__iter__") and not isinstance(
                    key_sets, (str, bytes)
                ):
                    try:
                        print("DEBUG: key_sets len:", len(key_sets))
                    except Exception:
                        pass
                converted = convert_to_legacy_diffusers(data, key_sets)
                print("DEBUG: conversion returned type:", type(converted))
                if hasattr(converted, "keys"):
                    try:
                        print(
                            "DEBUG: converted keys count:", len(list(converted.keys()))
                        )
                    except Exception:
                        pass
            except Exception as e:
                # provide a more informative message for debugging
                import traceback

                print(
                    "Conversion attempt raised an exception:\n", traceback.format_exc()
                )
                raise

        # For LoRA: follow trainer saver pipeline — use converted (legacy) keys only.
        # Do NOT merge original internal keys into the legacy output because loaders
        # (e.g. InvokeAI) will reject mixed-key bundles. Only fall back to merging
        # if conversion produced an empty result to avoid data loss.
        if ttype == "lora":
            if not converted or len(converted) == 0:
                print(
                    "Warning: conversion produced no keys — falling back to merging to avoid data loss."
                )
                merged = dict(data)
                merged.update(converted)
                converted = merged
            else:
                # use converted as-is (do not keep original internal keys)
                pass
        else:
            # non-lora: keep previous conservative merge behavior
            if set(converted.keys()) != set(data.keys()):
                print(
                    "Warning: keys differ after conversion. Merging to ensure no tensor is lost."
                )
                merged = dict(data)
                merged.update(converted)
                converted = merged

        # update header (ensure it's a dict)
        print("DEBUG: header type before dict():", type(header))
        try:
            if isinstance(header, dict):
                hdr = dict(header)
            else:
                # try to coerce mappings/iterables
                hdr = (
                    dict(header)
                    if hasattr(header, "items") or hasattr(header, "__iter__")
                    else {}
                )
        except Exception:
            hdr = {}
        hdr["ot_converted_to"] = "legacy_safetensors"
        header = hdr

        if args.dry_run:
            print("Dry run enabled - not writing output. Planned output:", out)
        else:
            if out.exists() and not args.force:
                print(f"Output {out} already exists. Use --force to overwrite.")
                sys.exit(5)

            # prepare tensors to ensure no shared memory among entries (safetensors rejects shared memory)
            def _prepare_for_save(state_dict: dict):
                prepared = {}
                try:
                    import torch
                    import numpy as _np
                except Exception:
                    torch = None
                    import numpy as _np

                for k, v in state_dict.items():
                    # torch tensors -> clone & cpu
                    if torch is not None and hasattr(v, "clone") and hasattr(v, "cpu"):
                        try:
                            tv = v.clone().detach().cpu().contiguous()
                            prepared[k] = tv
                            continue
                        except Exception:
                            pass

                    # numpy arrays or array-like -> make copy
                    try:
                        prepared[k] = _np.array(v, copy=True)
                    except Exception:
                        # last resort: coerce via torch
                        if torch is not None:
                            try:
                                prepared[k] = (
                                    torch.as_tensor(v)
                                    .clone()
                                    .detach()
                                    .cpu()
                                    .contiguous()
                                )
                                continue
                            except Exception:
                                pass
                        prepared[k] = v

                return prepared

            prepared = _prepare_for_save(converted)
            save_file(prepared, str(out), header)
            print("Wrote converted file:", out)
    except Exception as e:
        import traceback

        print("Conversion failed. Traceback:\n", traceback.format_exc())
        sys.exit(4)


if __name__ == "__main__":
    main()
