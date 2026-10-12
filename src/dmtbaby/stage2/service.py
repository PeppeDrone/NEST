"""Reusable single-load UniAnimate inference service for rendered controls."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import random
import sys
from typing import Any, Callable, Iterable, Mapping

from .controls import CASCADE_FRAME_COUNT, ControlSequence, validate_control_sequence


def _progress(message: str) -> None:
    """Emit a flushed progress line so nohup/cascade logs update immediately."""

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[stage2 {stamp}] {message}", flush=True)


DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, jpeg artifacts, text, watermark, still image, "
    "deformed body, malformed limbs, extra limbs, bad anatomy, bad hands, "
    "bad face, duplicated body, crowded background"
)


@dataclass(frozen=True)
class UniAnimateModelConfig:
    """Filesystem and memory settings shared by an inference batch."""

    unianimate_dir: Path | str = Path("external/UniAnimate-DiT")
    wan_dir: Path | str = Path("external/UniAnimate-DiT/Wan2.1-I2V-14B-720P")
    lora_path: Path | str = Path(
        "external/UniAnimate-DiT/models/UniAnimate-DiT/UniAnimate-Wan2.1-14B-Lora-12000.ckpt"
    )
    official_lora_path: Path | str = Path(
        "external/UniAnimate-DiT/models/UniAnimate-DiT/UniAnimate-Wan2.1-14B-Lora-12000.ckpt"
    )
    lora_alpha: float = 1.0
    persistent_dit_params: int = 6_000_000_000
    device: str = "cuda"

    def __post_init__(self) -> None:
        object.__setattr__(self, "unianimate_dir", Path(self.unianimate_dir).expanduser().resolve())
        object.__setattr__(self, "wan_dir", Path(self.wan_dir).expanduser().resolve())
        object.__setattr__(self, "lora_path", Path(self.lora_path).expanduser().resolve())
        object.__setattr__(
            self,
            "official_lora_path",
            Path(self.official_lora_path).expanduser().resolve(),
        )
        if self.lora_alpha <= 0:
            raise ValueError("lora_alpha must be positive")
        if self.persistent_dit_params < 0:
            raise ValueError("persistent_dit_params cannot be negative")


@dataclass(frozen=True)
class UniAnimateRequest:
    """One strict cascade request in a batch manifest."""

    clip_id: str
    control_dir: Path | str
    reference_image: Path | str
    output: Path | str
    reference_pose: Path | str | None = None
    prompt: str = "a realistic NICU video of a newborn baby lying in a cradle"
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    seed: int = 0
    height: int = 832
    width: int = 480
    num_frames: int = CASCADE_FRAME_COUNT
    num_inference_steps: int = 25
    cfg_scale: float = 1.5
    fps: int = 15
    quality: int = 5
    use_teacache: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "control_dir", Path(self.control_dir).expanduser().resolve())
        object.__setattr__(self, "reference_image", Path(self.reference_image).expanduser().resolve())
        object.__setattr__(self, "output", Path(self.output).expanduser().resolve())
        if self.reference_pose is not None:
            object.__setattr__(self, "reference_pose", Path(self.reference_pose).expanduser().resolve())
        if not self.clip_id.strip():
            raise ValueError("clip_id cannot be empty")
        if self.num_frames != CASCADE_FRAME_COUNT:
            raise ValueError(
                f"Cascade requests require exactly {CASCADE_FRAME_COUNT} frames; got {self.num_frames}."
            )
        if self.num_frames % 4 != 1:
            raise ValueError("num_frames must satisfy Wan's num_frames % 4 == 1 constraint")
        if self.height <= 0 or self.width <= 0:
            raise ValueError("height and width must be positive")
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        if self.fps <= 0:
            raise ValueError("fps must be positive")

    def validate_artifacts(self) -> ControlSequence:
        if not Path(self.reference_image).is_file():
            raise FileNotFoundError(f"Reference image does not exist: {self.reference_image}")
        return validate_control_sequence(self.control_dir, self.reference_pose)

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any], root: str | Path = ".") -> "UniAnimateRequest":
        required = {"clip_id", "control_dir", "reference_image", "output"}
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"Batch request is missing required fields: {', '.join(missing)}")
        unknown = sorted(set(row) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Batch request has unknown fields: {', '.join(unknown)}")
        base = Path(root).expanduser().resolve()
        values = dict(row)
        for field in ("control_dir", "reference_image", "reference_pose", "output"):
            value = values.get(field)
            if value is not None and not Path(value).expanduser().is_absolute():
                values[field] = base / Path(value).expanduser()
        return cls(**values)


@dataclass
class _LoadedRuntime:
    pipeline: Any
    torch: Any
    numpy: Any
    image: Any
    save_video: Callable[..., Any]


PipelineLoader = Callable[[UniAnimateModelConfig], Any]
GenerationFunction = Callable[[Any, UniAnimateRequest, ControlSequence], Iterable[Any]]
VideoSaver = Callable[[Iterable[Any], str, int, int], Any]


class UniAnimateBatchService:
    """Sequential GPU service that loads Wan and the adapter exactly once.

    Optional loader/generator/saver hooks keep orchestration unit-testable
    without importing torch or allocating the 14B model.
    """

    def __init__(
        self,
        config: UniAnimateModelConfig | None = None,
        *,
        pipeline_loader: PipelineLoader | None = None,
        generation_function: GenerationFunction | None = None,
        video_saver: VideoSaver | None = None,
    ) -> None:
        self.config = config or UniAnimateModelConfig()
        self._pipeline_loader = pipeline_loader
        self._generation_function = generation_function
        self._video_saver = video_saver
        self._loaded: Any | None = None
        self._load_count = 0

    @property
    def is_loaded(self) -> bool:
        return self._loaded is not None

    @property
    def load_count(self) -> int:
        return self._load_count

    def load(self) -> Any:
        if self._loaded is None:
            _progress("loading UniAnimate/Wan runtime (one-time)")
            loader = self._pipeline_loader or _load_default_runtime
            self._loaded = loader(self.config)
            self._load_count += 1
            _progress("UniAnimate/Wan runtime ready")
        return self._loaded

    def generate_validated(self, request: UniAnimateRequest, controls: ControlSequence) -> list[Any]:
        loaded = self.load()
        _progress(
            f"generating {request.clip_id}: "
            f"{request.width}x{request.height} frames={request.num_frames} "
            f"steps={request.num_inference_steps} seed={request.seed}"
        )
        if self._generation_function is not None:
            frames = list(self._generation_function(loaded, request, controls))
        else:
            frames = list(_generate_default(loaded, request, controls))
        if not frames:
            raise RuntimeError(f"UniAnimate returned zero generated frames for {request.clip_id}.")
        _progress(f"generated {request.clip_id}: {len(frames)} frames")
        return frames

    def generate(self, request: UniAnimateRequest) -> list[Any]:
        return self.generate_validated(request, request.validate_artifacts())

    def run_batch(self, requests: Iterable[UniAnimateRequest]) -> list[Path]:
        """Validate the complete batch before loading Wan, then run sequentially.

        Resume selection belongs to the fingerprint-aware cascade orchestrator;
        this lower-level service always executes every request it receives.
        """

        request_list = list(requests)
        _progress(f"validating control/reference images for {len(request_list)} request(s)")
        validated = [(request, request.validate_artifacts()) for request in request_list]
        if not validated:
            raise ValueError("The UniAnimate batch is empty.")
        if self._pipeline_loader is None:
            for index, (request, controls) in enumerate(validated, start=1):
                if index == 1 or index == len(validated) or index % 50 == 0:
                    _progress(f"image preflight {index}/{len(validated)}: {request.clip_id}")
                validate_request_images(request, controls)
        outputs = [Path(request.output) for request, _ in validated]
        self.load()
        for index, (request, controls) in enumerate(validated, start=1):
            _progress(f"clip {index}/{len(validated)}: {request.clip_id}")
            frames = self.generate_validated(request, controls)
            output = Path(request.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            self._save(frames, output, request)
            _progress(f"saved {request.clip_id} -> {output}")
        return outputs

    def _save(self, frames: Iterable[Any], output: Path, request: UniAnimateRequest) -> None:
        if self._video_saver is not None:
            self._video_saver(frames, str(output), request.fps, request.quality)
            return
        loaded = self.load()
        if not isinstance(loaded, _LoadedRuntime):
            raise TypeError("A custom pipeline loader must also provide video_saver for run_batch().")
        partial = output.with_name(f".{output.stem}.partial{output.suffix}")
        partial.unlink(missing_ok=True)
        try:
            loaded.save_video(frames, str(partial), fps=request.fps, quality=request.quality)
            if not _is_complete_output(partial):
                raise RuntimeError(f"UniAnimate video writer did not produce a non-empty file: {partial}")
            os.replace(partial, output)
        finally:
            partial.unlink(missing_ok=True)


def _is_complete_output(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def validate_request_images(
    request: UniAnimateRequest,
    controls: ControlSequence | None = None,
) -> None:
    """Decode every image before the 14B model is allocated."""

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Pillow is required for Stage-2 image preflight") from exc

    sequence = controls or request.validate_artifacts()

    def inspect(path: Path, *, pose: bool) -> None:
        try:
            with Image.open(path) as image:
                image_format = str(image.format or "").upper()
                mode = str(image.mode)
                size = tuple(image.size)
                image.verify()
        except Exception as exc:
            raise ValueError(f"Could not decode Stage-2 image {path}: {exc}") from exc
        if pose and (
            image_format != "JPEG" or mode != "RGB" or size != (512, 768)
        ):
            raise ValueError(
                f"DWPose control {path} must be RGB JPEG 512x768; "
                f"got format={image_format}, mode={mode}, size={size}"
            )
        if not pose and (len(size) != 2 or min(size) <= 0):
            raise ValueError(f"Reference image {path} has invalid size {size}")

    inspect(Path(request.reference_image), pose=False)
    inspect(sequence.reference_pose, pose=True)
    for path in sequence.frames:
        inspect(path, pose=True)


def _required_model_paths(config: UniAnimateModelConfig) -> tuple[Path, ...]:
    wan_dir = Path(config.wan_dir)
    tokenizer_dir = wan_dir / "google" / "umt5-xxl"
    paths = (
        *adapter_checkpoint_chain(config),
        wan_dir / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        wan_dir / "models_t5_umt5-xxl-enc-bf16.pth",
        wan_dir / "Wan2.1_VAE.pth",
        tokenizer_dir / "special_tokens_map.json",
        tokenizer_dir / "spiece.model",
        tokenizer_dir / "tokenizer.json",
        tokenizer_dir / "tokenizer_config.json",
        *(wan_dir / f"diffusion_pytorch_model-{index:05d}-of-00007.safetensors" for index in range(1, 8)),
    )
    return tuple(dict.fromkeys(paths))


def adapter_checkpoint_chain(config: UniAnimateModelConfig) -> tuple[Path, ...]:
    """Return the official initialization followed by an optional compact adapter.

    Compact DMTBaby checkpoints intentionally omit the frozen reference-pose
    encoder.  Loading the official checkpoint first supplies that encoder;
    loading the compact checkpoint second overrides only DiT LoRA and the
    trainable dynamic DWPose encoder.  The frozen baseline loads once.
    """

    official = Path(config.official_lora_path).resolve()
    selected = Path(config.lora_path).resolve()
    return (official,) if official == selected else (official, selected)


def validate_model_artifacts(config: UniAnimateModelConfig) -> tuple[Path, ...]:
    """Validate the complete frozen/adapted UniAnimate model inventory."""

    unianimate_dir = Path(config.unianimate_dir)
    if not (unianimate_dir / "diffsynth").is_dir():
        raise FileNotFoundError(
            f"UniAnimate repository is incomplete: {unianimate_dir}"
        )
    required = _required_model_paths(config)
    missing = [path for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        rendered = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing UniAnimate model artifacts:\n{rendered}")
    return required


def _load_default_runtime(config: UniAnimateModelConfig) -> _LoadedRuntime:
    _progress("validating UniAnimate model artifact inventory")
    validate_model_artifacts(config)

    unianimate_dir = Path(config.unianimate_dir)
    if not unianimate_dir.is_dir():
        raise FileNotFoundError(f"UniAnimate repository does not exist: {unianimate_dir}")
    if str(unianimate_dir) not in sys.path:
        sys.path.insert(0, str(unianimate_dir))

    _progress("importing torch/diffsynth/PIL")
    import numpy as np  # type: ignore
    import torch  # type: ignore
    from diffsynth import ModelManager, WanUniAnimateVideoPipeline, save_video  # type: ignore
    from PIL import Image  # type: ignore

    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for UniAnimate, but torch.cuda.is_available() is False.")
    if config.device.startswith("cuda"):
        _progress(
            f"CUDA device={torch.cuda.get_device_name(0)} "
            f"mem_allocated_gb={torch.cuda.memory_allocated(0) / (1024**3):.2f}"
        )

    wan_dir = Path(config.wan_dir)
    manager = ModelManager(device="cpu")
    clip_path = wan_dir / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    _progress(f"loading CLIP image encoder: {clip_path.name}")
    manager.load_models([str(clip_path)], torch_dtype=torch.float32)
    dit_shards = [
        wan_dir / f"diffusion_pytorch_model-{index:05d}-of-00007.safetensors"
        for index in range(1, 8)
    ]
    t5_path = wan_dir / "models_t5_umt5-xxl-enc-bf16.pth"
    vae_path = wan_dir / "Wan2.1_VAE.pth"
    _progress(
        "loading DiT shards + T5 + VAE "
        f"({len(dit_shards)} shards, {t5_path.name}, {vae_path.name})"
    )
    manager.load_models(
        [
            [str(path) for path in dit_shards],
            str(t5_path),
            str(vae_path),
        ],
        torch_dtype=torch.bfloat16,
    )
    adapter_chain = adapter_checkpoint_chain(config)
    for index, adapter_path in enumerate(adapter_chain, start=1):
        _progress(
            f"loading adapter {index}/{len(adapter_chain)}: {adapter_path.name}"
        )
    manager.load_lora_v2(
        str(adapter_chain[0]) if len(adapter_chain) == 1 else [str(path) for path in adapter_chain],
        lora_alpha=config.lora_alpha,
    )
    _progress(f"building WanUniAnimateVideoPipeline on {config.device}")
    pipeline = WanUniAnimateVideoPipeline.from_model_manager(
        manager,
        torch_dtype=torch.bfloat16,
        device=config.device,
    )
    _progress(
        "enabling VRAM management "
        f"(persistent_dit_params={config.persistent_dit_params})"
    )
    pipeline.enable_vram_management(num_persistent_param_in_dit=config.persistent_dit_params)
    if config.device.startswith("cuda") and torch.cuda.is_available():
        _progress(
            f"CUDA mem after load: allocated_gb={torch.cuda.memory_allocated(0) / (1024**3):.2f} "
            f"reserved_gb={torch.cuda.memory_reserved(0) / (1024**3):.2f}"
        )
    return _LoadedRuntime(pipeline=pipeline, torch=torch, numpy=np, image=Image, save_video=save_video)


def _generate_default(
    loaded: Any,
    request: UniAnimateRequest,
    controls: ControlSequence,
) -> list[Any]:
    if not isinstance(loaded, _LoadedRuntime):
        raise TypeError("Default generation requires the default UniAnimate runtime loader.")

    torch = loaded.torch
    np = loaded.numpy
    Image = loaded.image
    random.seed(request.seed)
    torch.manual_seed(request.seed)

    resampling = getattr(Image, "Resampling", Image).BILINEAR

    def as_array(path: Path) -> Any:
        with Image.open(path) as image:
            rgb = image.convert("RGB").resize((request.width, request.height), resampling)
            return torch.from_numpy(np.asarray(rgb).copy())

    with Image.open(request.reference_image) as image:
        reference_image = image.convert("RGB").copy()
    reference_pose = as_array(controls.reference_pose)
    temporal = torch.stack([as_array(path).permute(2, 0, 1) for path in controls.frames], dim=0)
    dwpose_data = temporal.permute(1, 0, 2, 3)

    video = loaded.pipeline(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        input_image=reference_image,
        num_inference_steps=request.num_inference_steps,
        cfg_scale=request.cfg_scale,
        num_frames=CASCADE_FRAME_COUNT,
        seed=request.seed,
        tiled=True,
        dwpose_data=dwpose_data,
        random_ref_dwpose=reference_pose,
        height=request.height,
        width=request.width,
        tea_cache_l1_thresh=0.3 if request.use_teacache else None,
        tea_cache_model_id="Wan2.1-I2V-14B-720P" if request.use_teacache else None,
    )
    return list(video)
