"""promboard.heartmula — HeartMuLa(oss-3B happy-new-year) + HeartCodec 로 가사 · 스타일 태그에서 노래를 만드는 노드.

모델 코드는 HeartMuLa/heartlib(Apache-2.0)의 heartmula · heartcodec 을 nodes/heartlib 로 옮긴 것이다(NOTICE).
생성 순서는 heartlib pipelines/music_generation.py 를 따라 이 파일에 다시 적었다 — 바꾼 점:
  - torch.compile 을 하지 않는다(윈도우에는 triton 이 없고, 한 번 쓰는 노드에서는 컴파일 시간이 더 든다)
  - KV 캐시를 8192 칸 대신 이번 곡 길이만큼만 잡는다(그래픽카드 메모리 절약)
  - HeartMuLa 로 토큰을 다 만든 뒤 내리고 HeartCodec 을 올린다(heartlib lazy_load 와 같은 순서)
  - 프레임마다 진행률 · 취소를 확인하고, 결과는 48kHz 스테레오 WAV 로 쓴다
"""
from __future__ import annotations

import gc
import time
from typing import Any, Dict

MULA_REPO = "HeartMuLa/HeartMuLa-oss-3B-happy-new-year"
CODEC_REPO = "HeartMuLa/HeartCodec-oss-20260123"
GEN_REPO = "HeartMuLa/HeartMuLaGen"
FRAME_MS = 80  # HeartMuLa 한 프레임 = 80ms(12.5Hz)
PARALLEL = 8 + 1  # 오디오 코드북 8 + 글 1
MUQ_DIM = 512
OOM_HELP = "그래픽카드 메모리가 모자랍니다. '길이'를 줄이거나, 다른 프로그램이 그래픽카드를 쓰고 있으면 닫아 주세요."


def _is_oom(exc: BaseException) -> bool:
    import torch

    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _free(device: str) -> None:
    import torch

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def _encode_text(tokenizer, text: str, bos: int, eos: int):
    ids = tokenizer.encode(text).ids
    if not ids or ids[0] != bos:
        ids = [bos] + ids
    if ids[-1] != eos:
        ids = ids + [eos]
    return ids


def _prompt(tokenizer, cfg: Dict[str, int], tags: str, lyrics: str, dtype, guided: bool):
    """heartlib HeartMuLaGenPipeline.preprocess 와 같다(참조 음성은 쓰지 않는다)."""
    import torch

    tags = tags.strip().lower()
    if not tags.startswith("<tag>"):
        tags = f"<tag>{tags}"
    if not tags.endswith("</tag>"):
        tags = f"{tags}</tag>"
    tags_ids = _encode_text(tokenizer, tags, cfg["text_bos_id"], cfg["text_eos_id"])
    lyrics_ids = _encode_text(tokenizer, lyrics.strip().lower(), cfg["text_bos_id"], cfg["text_eos_id"])
    muq_idx = len(tags_ids)
    prompt_len = len(tags_ids) + 1 + len(lyrics_ids)
    tokens = torch.zeros([prompt_len, PARALLEL], dtype=torch.long)
    tokens[: len(tags_ids), -1] = torch.tensor(tags_ids)
    tokens[len(tags_ids) + 1 :, -1] = torch.tensor(lyrics_ids)
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    mask[:, -1] = True
    muq = torch.zeros([MUQ_DIM], dtype=dtype)
    pos = torch.arange(prompt_len, dtype=torch.long)

    def cat(t):
        t = t.unsqueeze(0)
        return torch.cat([t, t], dim=0) if guided else t

    bs = 2 if guided else 1
    return {"tokens": cat(tokens), "mask": cat(mask), "muq": cat(muq), "starts": [muq_idx] * bs, "pos": cat(pos), "len": prompt_len}


def _setup_caches(mula, batch: int, seq_len: int) -> None:
    """HeartMuLa.setup_caches 와 같되 KV 캐시 · 마스크를 seq_len 칸만 잡는다."""
    from .heartlib.heartmula.modeling_heartmula import _create_causal_mask

    dtype = next(mula.parameters()).dtype
    device = next(mula.parameters()).device
    try:
        mula.reset_caches()
    except RuntimeError:
        pass
    with device:
        mula.backbone.setup_caches(batch, dtype, decoder_max_seq_len=seq_len)
        mula.decoder.setup_caches(batch, dtype, decoder_max_seq_len=mula.config.audio_num_codebooks)
    mula.register_buffer("backbone_causal_mask", _create_causal_mask(seq_len, device), persistent=False)
    mula.register_buffer("decoder_causal_mask", _create_causal_mask(mula.config.audio_num_codebooks, device), persistent=False)


def generate_song(ctx, inputs, settings):
    import json
    import os

    import numpy as np
    import soundfile as sf
    import torch
    from tokenizers import Tokenizer

    from .heartlib.heartcodec.modeling_heartcodec import HeartCodec
    from .heartlib.heartmula.modeling_heartmula import HeartMuLa

    lyrics = str(inputs.get("lyrics") or "").strip()
    if not lyrics:
        raise ValueError("가사를 이어 주세요. 연주곡이면 [Instrumental] 처럼 구간 표시만 적어도 됩니다.")
    tags = str(settings["tags"]).strip()
    if not tags:
        raise ValueError("스타일 태그를 적어 주세요(예: pop,piano,female vocal).")
    seconds = int(settings["duration"])
    cfg_scale = float(settings["cfg"])
    temperature = float(settings["temperature"])
    topk = int(settings["topk"])
    seed = int(settings["seed"])
    device = ctx.device
    if device == "cpu":
        raise ValueError("HeartMuLa 는 그래픽카드(CUDA · Apple 실리콘)가 있어야 돕니다.")
    dev = torch.device(device)
    mula_dtype = torch.bfloat16
    guided = cfg_scale != 1.0

    gen_dir = os.path.dirname(ctx.model_path(f"{GEN_REPO}/gen_config.json"))
    with open(os.path.join(gen_dir, "gen_config.json"), encoding="utf-8") as fh:
        gcfg = json.load(fh)
    tokenizer = Tokenizer.from_file(os.path.join(gen_dir, "tokenizer.json"))
    prompt = _prompt(tokenizer, gcfg, tags, lyrics, mula_dtype, guided)
    max_frames = seconds * 1000 // FRAME_MS
    torch.manual_seed(seed)
    started = time.monotonic()

    # ── 1) HeartMuLa: 토큰 만들기 ───────────────────────────────────────
    frames = []
    mula = None
    try:
        ctx.progress(0.02, "load-model")
        mula = HeartMuLa.from_pretrained(os.path.dirname(ctx.model_path(f"{MULA_REPO}/config.json")), device_map=device, dtype=mula_dtype)
        mula.eval()
        ctx.log(f"HeartMuLa loaded in {time.monotonic() - started:.1f}s")
        _setup_caches(mula, 2 if guided else 1, prompt["len"] + max_frames + 8)
        empty_id = int(gcfg.get("empty_id", 0))
        eos = int(gcfg["audio_eos_id"])
        tokens = prompt["tokens"].to(dev)
        tmask = prompt["mask"].to(dev)
        pos = prompt["pos"].to(dev)
        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=mula_dtype):
            cur = mula.generate_frame(tokens=tokens, tokens_mask=tmask, input_pos=pos, temperature=temperature, topk=topk,
                                      cfg_scale=cfg_scale, continuous_segments=prompt["muq"].to(dev), starts=prompt["starts"])
            frames.append(cur[0:1])
            gen_started = time.monotonic()
            for i in range(max_frames):
                padded = torch.full((cur.shape[0], PARALLEL), empty_id, device=dev, dtype=torch.long)
                padded[:, :-1] = cur
                padded = padded.unsqueeze(1)
                pmask = torch.ones_like(padded, dtype=torch.bool)
                pmask[..., -1] = False
                cur = mula.generate_frame(tokens=padded, tokens_mask=pmask, input_pos=pos[..., -1:] + i + 1, temperature=temperature,
                                          topk=topk, cfg_scale=cfg_scale, continuous_segments=None, starts=None)
                if torch.any(cur[0:1, :] >= eos):
                    ctx.log(f"end of song at frame {i}")
                    break
                frames.append(cur[0:1])
                if i % 25 == 0:
                    ctx.check_cancel()
                    ctx.progress(0.05 + 0.7 * i / max_frames, "compose")
        ctx.log(f"{len(frames)} frames in {time.monotonic() - gen_started:.1f}s")
        codes = torch.stack(frames).permute(1, 2, 0).squeeze(0).cpu()
    except Exception as exc:  # noqa: BLE001
        if _is_oom(exc):
            raise RuntimeError(OOM_HELP) from exc
        raise
    finally:
        if mula is not None:
            try:
                mula.reset_caches()
            except Exception:  # noqa: BLE001
                pass
            del mula
        frames = []
        _free(device)
    ctx.check_cancel()

    # ── 2) HeartCodec: 토큰 → 소리 ─────────────────────────────────────
    codec = None
    try:
        ctx.progress(0.78, "load-model")
        codec = HeartCodec.from_pretrained(os.path.dirname(ctx.model_path(f"{CODEC_REPO}/config.json")), device_map=device, dtype=torch.float32)
        codec.eval()
        ctx.progress(0.82, "decode")
        with torch.no_grad():
            wav = codec.detokenize(codes, disable_progress=True)
    except Exception as exc:  # noqa: BLE001
        if _is_oom(exc):
            raise RuntimeError(OOM_HELP) from exc
        raise
    finally:
        if codec is not None:
            del codec
        _free(device)
    ctx.check_cancel()
    audio = wav.to(torch.float32).cpu().numpy()
    if audio.ndim == 2 and audio.shape[0] in (1, 2):
        audio = audio.T
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    dst = ctx.out_path("song.wav")
    sf.write(dst, audio, 48000, subtype="PCM_16")
    ctx.log(f"song {audio.shape[0] / 48000:.1f}s done in {time.monotonic() - started:.1f}s")
    ctx.progress(1.0, "done")
    return {"audio": dst}


NODES = {"song": generate_song}
