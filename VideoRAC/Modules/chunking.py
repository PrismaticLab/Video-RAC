import logging
import math
import os
import json
from pathlib import Path
from time import time

import cv2
import torch
from scipy.spatial.distance import cosine
from skimage.metrics import structural_similarity as ssim
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor

import VideoRAC.utils.entropy_utils as E
from VideoRAC.utils.logging_utils import get_logger_handler


logger = logging.getLogger(__name__)
logger.propagate = False
if not logger.hasHandlers():
    logger.addHandler(get_logger_handler())
logger.setLevel(logging.INFO)

class HybridChunker:
    """
    Slide-change chunking using a weighted combination of CLIP image-embedding
    similarity and SSIM between consecutive frames.

    Parameters
    ----------
    clip_model : str
        Hugging Face model identifier for CLIP. The default is kept as-is to
        preserve external behavior.
    threshold_embedding : float, optional
        Threshold on the combined similarity used to decide slide changes.
    threshold_ssim : float, optional
        Kept for compatibility; not used directly in the decision.
    interval : int, optional
        Sampling interval (seconds) between analyzed frames.
    alpha : float, optional
        Weight for embedding similarity in the hybrid score (0..1).
    output_dir : str or Path, optional
        Default directory where chunk images will be saved. Can be overridden
        per call in `chunk(...)`. Defaults to "./chunks_out".
    image_format : {"png","jpg","jpeg","webp"}, optional
        File format for saved frames. Defaults to "png".
    """

    def __init__(
        self,
        clip_model: str = 'openai/clip-vit-base-patch32',
        *,
        threshold_embedding: float = 0.8,
        threshold_ssim: float = 0.8,
        interval: int = 1,
        alpha: float = 0.5,
        output_dir: str | os.PathLike = "chunks_out",
        image_format: str = "png",
    ):
        try:
            self._clip_model_id = clip_model
            self._model = CLIPModel.from_pretrained(clip_model)
            self._processor = CLIPProcessor.from_pretrained(clip_model)

            # configurable detection settings
            self._threshold_embedding = float(threshold_embedding)
            self._threshold_ssim = float(threshold_ssim)
            self._interval = int(interval)
            self._alpha = float(alpha)

            # saving configuration
            self._output_dir = Path(output_dir)
            self._image_format = str(image_format).lower().strip(".") or "png"

            # results
            self._chunks = None
            self._exe_time = None
            self._avg_frame_per_chunk = None
            self._mean_entropy = None

            logger.info("✅ HybridChunking initialized with model %s", clip_model)
        except Exception as e:
            logger.exception("💥 Failed to initialize CLIP model/processor: %s", e)
            raise

    # -------------------------------------------------------------------------
    # Properties (read-only where appropriate)
    # -------------------------------------------------------------------------

    @property
    def clip_model_id(self) -> str:
        """Model identifier used for CLIP."""
        return self._clip_model_id

    @property
    def threshold_embedding(self) -> float:
        """Combined-similarity threshold for slide change detection."""
        return self._threshold_embedding

    @property
    def threshold_ssim(self) -> float:
        """SSIM threshold placeholder (kept for compatibility)."""
        return self._threshold_ssim

    @property
    def interval(self) -> int:
        """Sampling interval (seconds) between analyzed frames."""
        return self._interval

    @property
    def alpha(self) -> float:
        """Weight of embedding similarity in the hybrid score (0..1)."""
        return self._alpha

    @property
    def output_dir(self) -> Path:
        """Default directory where chunk images will be saved."""
        return self._output_dir

    @property
    def image_format(self) -> str:
        """Image file format used when saving frames."""
        return self._image_format

    @property
    def chunks(self):
        """List of chunked frame lists (or None before running)."""
        return self._chunks

    @property
    def execution_time(self):
        """Total execution time (seconds) from the last `chunk` call."""
        return self._exe_time

    @property
    def avg_frame_per_chunk(self):
        """Average number of frames per chunk after evaluation."""
        return self._avg_frame_per_chunk

    @property
    def mean_entropy(self):
        """Mean entropy across chunks after evaluation."""
        return self._mean_entropy

    # -------------------------------------------------------------------------
    # Metrics helpers
    # -------------------------------------------------------------------------

    def _get_avg_frame_per_time(self):
        """
        Average number of frames per chunk.

        Returns
        -------
        float or None
            Mean length of chunks if available; otherwise None.
        """
        try:
            avg_frame_per_chunk = sum(len(self._chunks[i]) for i in range(len(self._chunks))) / len(self._chunks)
            logger.info("📈 Average frames per chunk: %.2f", avg_frame_per_chunk)
            return avg_frame_per_chunk
        except Exception as e:
            logger.error("⚠️ Error computing avg_frame_per_time: %s", e)
            return None

    def _get_mean_entropy(self):
        """
        Mean entropy across chunks computed via `entropy_utils`.

        Returns
        -------
        float
            Mean entropy value.
        """
        try:
            mean_entropy = E.chunks_mean_entropy(self._chunks)
            logger.info("🧠 Mean entropy across chunks: %.4f", mean_entropy)
            return mean_entropy
        except Exception as e:
            logger.exception("💥 Error computing mean entropy: %s", e)
            raise

    # -------------------------------------------------------------------------
    # Saving helpers
    # -------------------------------------------------------------------------

    def _ensure_dir(self, path: Path):
        """Create directory if it does not exist."""
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.exception("💥 Failed to create directory %s: %s", path, e)
            raise

    def _save_chunks(
        self,
        *,
        output_dir: str | os.PathLike | None = None,
        image_format: str | None = None,
        prefix: str | None = None,
    ) -> list[list[Path]]:
        """
        Save each frame image from the computed chunks to disk.

        Parameters
        ----------
        output_dir : str or Path, optional
            Root directory where chunks are written. Defaults to the instance
            `output_dir` set at initialization.
        image_format : {"png","jpg","jpeg","webp"}, optional
            File format for saved images. Defaults to the instance `image_format`.
        prefix : str, optional
            Optional prefix to help identify outputs for a particular video.

        Returns
        -------
        list[list[pathlib.Path]]
            Paths to saved images, grouped by chunk. Each inner list contains
            the file paths for images belonging to that chunk.

        Notes
        -----
        - `self._chunks` is a list of lists. Each inner list contains frames
          (images) that will be written individually.
        - Chunk folders are created as: {output_dir}/{prefix_}chunk_0001, ...
        """
        if not self._chunks:
            logger.warning("ℹ️ No chunks to save.")
            return []

        fmt = (image_format or self._image_format).lower().strip(".")
        root = Path(output_dir) if output_dir is not None else self._output_dir
        self._ensure_dir(root)

        saved_paths: list[list[Path]] = []
        for c_idx, frames in enumerate(self._chunks, start=1):
            chunk_dir_name = f"{(prefix + '_') if prefix else ''}chunk_{c_idx:04d}"
            chunk_dir = root / chunk_dir_name
            self._ensure_dir(chunk_dir)

            paths_for_chunk: list[Path] = []
            for f_idx, frame in enumerate(frames, start=1):
                out_path = chunk_dir / f"frame_{f_idx:04d}.{fmt}"
                try:
                    # OpenCV expects BGR; frames are already BGR.
                    ok = cv2.imwrite(str(out_path), frame)
                    if not ok:
                        raise RuntimeError("cv2.imwrite returned False")
                    paths_for_chunk.append(out_path)
                except Exception as e:
                    logger.exception("💥 Failed to write %s: %s", out_path, e)
                    raise
            saved_paths.append(paths_for_chunk)

        logger.info("💾 Saved %d chunks to %s ✅", len(saved_paths), root)
        return saved_paths

    # -------------------------------------------------------------------------
    # NEW: Selected-frames helper (private)
    # -------------------------------------------------------------------------

    def _select_best_frames_per_chunk(
        self,
        *,
        output_dir: str | os.PathLike | None = None,
        image_format: str | None = None,
        prefix: str | None = None,
    ) -> list[list[Path]]:
        """
        For each chunk, compute entropy for all frames and select:
        - first frame
        - frame with maximum entropy
        - last frame

        Saves selected frames under: {output_dir}/selected_frames/{prefix_}chunk_xxxx
        """
        if not self._chunks:
            logger.warning("ℹ️ No chunks available for selection.")
            return []

        fmt = (image_format or self._image_format).lower().strip(".")
        root = Path(output_dir) if output_dir is not None else self._output_dir
        sel_root = root / "selected_frames"
        self._ensure_dir(sel_root)

        all_selected_paths: list[list[Path]] = []

        try:
            for c_idx, frames in enumerate(self._chunks, start=1):
                if not frames:
                    all_selected_paths.append([])
                    continue

                # Compute entropies on grayscale copies
                entropies = []
                for frame in frames:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    entropies.append(E.calculate_entropy(gray))

                # Determine indices to keep: first, argmax, last (deduped, order preserved)
                idx_first = 0
                idx_max = int(max(range(len(entropies)), key=lambda i: entropies[i]))
                idx_last = len(frames) - 1
                ordered_unique = []
                for idx in (idx_first, idx_max, idx_last):
                    if idx not in ordered_unique:
                        ordered_unique.append(idx)

                # Save selected frames
                chunk_dir_name = f"{(prefix + '_') if prefix else ''}chunk_{c_idx:04d}"
                out_dir = sel_root / chunk_dir_name
                self._ensure_dir(out_dir)

                sel_paths: list[Path] = []
                for idx in ordered_unique:
                    out_path = out_dir / f"frame_{(idx+1):04d}.{fmt}"
                    ok = cv2.imwrite(str(out_path), frames[idx])
                    if not ok:
                        raise RuntimeError(f"cv2.imwrite returned False for {out_path}")
                    sel_paths.append(out_path)

                all_selected_paths.append(sel_paths)

            logger.info("🌟 Saved selected frames for %d chunks to %s ✅", len(all_selected_paths), sel_root)
            return all_selected_paths

        except Exception as e:
            logger.exception("💥 Error selecting/saving best frames: %s", e)
            raise

    # -------------------------------------------------------------------------
    # NEW: Metadata helper (private)
    # -------------------------------------------------------------------------

    def _save_metadata(
        self,
        *,
        video_path: str,
        timestamps: list[float],
        output_dir: str | os.PathLike | None = None,
        image_format: str | None = None,
        prefix: str | None = None,
    ) -> Path:
        """
        Save a JSON metadata file describing the run, including:
        - video info, timestamps, counts, config, execution time, and stats.
        """
        try:
            fmt = (image_format or self._image_format).lower().strip(".")
            root = Path(output_dir) if output_dir is not None else self._output_dir
            self._ensure_dir(root)

            video_p = Path(video_path)
            num_chunks = len(self._chunks) if self._chunks else 0
            frames_per_chunk = [len(c) for c in (self._chunks or [])]
            total_frames = int(sum(frames_per_chunk)) if frames_per_chunk else 0

            meta = {
                "video_name": video_p.name,
                "video_path": str(video_p),
                "clip_model": self._clip_model_id,
                "threshold_embedding": self._threshold_embedding,
                "threshold_ssim": self._threshold_ssim,
                "interval_seconds": self._interval,
                "alpha": self._alpha,
                "image_format": fmt,
                "output_dir": str(root),
                "chunks_root": str(root),
                "selected_frames_root": str(root / "selected_frames"),
                "timestamps_seconds": timestamps or [],
                "num_chunks": num_chunks,
                "frames_per_chunk": frames_per_chunk,
                "total_frames": total_frames,
                "execution_time_seconds": self._exe_time,
                "avg_frame_per_chunk": self._avg_frame_per_chunk,
                "mean_entropy": self._mean_entropy,
            }

            meta_name = f"{(prefix + '_') if prefix else ''}{video_p.stem}_metadata.json"
            meta_path = root / meta_name
            with meta_path.open("w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            logger.info("🗂️ Metadata saved to %s ✅", meta_path)
            return meta_path
        except Exception as e:
            logger.exception("💥 Failed to save metadata: %s", e)
            raise

    # -------------------------------------------------------------------------
    # Core internals
    # -------------------------------------------------------------------------

    def _get_frame_embedding(self, frame):
        """
        CLIP image embedding for a single frame.

        Parameters
        ----------
        frame : numpy.ndarray
            BGR frame as produced by OpenCV.

        Returns
        -------
        torch.Tensor
            1-D image-feature tensor.
        """
        try:
            inputs = self._processor(images=frame, return_tensors="pt")
            with torch.no_grad():
                embedding = self._model.get_image_features(**inputs)
            return embedding.squeeze()
        except Exception as e:
            logger.exception("💥 Failed to compute frame embedding: %s", e)
            raise

    def _detect_slide_changes(
        self,
        video_path,
        *,
        threshold_embedding: float | None = None,
        threshold_ssim: float | None = None,  # kept for compatibility
        interval: int | None = None,
        alpha: float | None = None,
    ):
        """
        Detect slide changes using a hybrid similarity score:
        `alpha * cosine(CLIP) + (1 - alpha) * SSIM`.

        Parameters
        ----------
        video_path : str
            Path to a video file.
        threshold_embedding : float, optional
            Threshold on the combined similarity. If None, uses instance setting.
        threshold_ssim : float, optional
            Kept for compatibility; not used directly in the decision.
        interval : int, optional
            Sampling interval in seconds. If None, uses instance setting.
        alpha : float, optional
            Weight for embedding similarity (0..1). If None, uses instance setting.

        Returns
        -------
        tuple[list[list[numpy.ndarray]], list[float]] | list
            (chunks, timestamps) on success; [] if the first frame cannot be read.
        """
        # Resolve effective parameters from instance when not provided
        thr_emb = self._threshold_embedding if threshold_embedding is None else float(threshold_embedding)
        _ = self._threshold_ssim if threshold_ssim is None else float(threshold_ssim)  # unused, compatibility
        step = self._interval if interval is None else int(interval)
        w_alpha = self._alpha if alpha is None else float(alpha)

        try:
            cap = cv2.VideoCapture(video_path)
        except Exception as e:
            logger.exception("💥 Failed to open video: %s", e)
            raise

        try:
            # Estimate total steps for tqdm so it completes at 100%
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
            duration_sec = (frame_count / fps) if (fps > 0 and frame_count > 0) else None
            estimated_steps = math.ceil(duration_sec / step) if duration_sec else None

            success, frame = cap.read()

            frame_lst = []
            hybrid_chunks = []

            if not success:
                logger.error("🚫 Failed to read the first frame from %s", video_path)
                cap.release()
                return []

            prev_embedding = self._get_frame_embedding(frame)
            prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            timestamp = 0
            timestamps = []

            # Progress bar: show total if we could estimate it; ensure it reaches 100%.
            pbar = tqdm(
                total=estimated_steps,
                desc="⏳ Processing frames",
                unit="frame",
                leave=True
            )

            while cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                ret, frame = cap.read()
                if not ret:
                    break

                curr_embedding = self._get_frame_embedding(frame)
                curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                embedding_similarity = 1 - cosine(prev_embedding, curr_embedding)
                frame_similarity = ssim(prev_frame, curr_frame)
                combined_similarity = w_alpha * embedding_similarity + (1 - w_alpha) * frame_similarity

                if combined_similarity < thr_emb:
                    minutes = int(timestamp // 60)
                    seconds = int(timestamp % 60)
                    timestamps.append(timestamp)
                    logger.info("🔀 Slide changed at %02d:%02d ⏱️", minutes, seconds)
                    hybrid_chunks.append(frame_lst)
                    frame_lst = []

                prev_embedding = curr_embedding
                prev_frame = curr_frame
                frame_lst.append(frame)

                timestamp += step
                pbar.update(1)

            # Flush last (possibly open) chunk if it has frames.
            if frame_lst:
                hybrid_chunks.append(frame_lst)

            # Ensure progress bar reaches 100% if total was set.
            if pbar.total is not None and pbar.n < pbar.total:
                pbar.update(pbar.total - pbar.n)
            pbar.close()

            cap.release()
            logger.info("✅ Slide detection complete. Segments: %d 🎬", len(hybrid_chunks))
            return hybrid_chunks, timestamps

        except Exception as e:
            try:
                cap.release()
            except Exception:
                pass
            logger.exception("💥 Error during slide change detection: %s", e)
            raise

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def chunk(
        self,
        video_path: str,
        *,
        save: bool = True,
        output_dir: str | os.PathLike | None = None,
        image_format: str | None = None,
        prefix: str | None = None,
    ):
        """
        Run detection and record execution time using the instance configuration.
        Optionally save the resulting chunk frames to disk.

        Parameters
        ----------
        video_path : str
            Path to a video file.
        save : bool, optional
            If True, saves chunk images at the end using `_save_chunks(...)`.
            Defaults to True.
        output_dir : str or Path, optional
            Root directory for saving results; overrides the instance default.
        image_format : {"png","jpg","jpeg","webp"}, optional
            File format for saved frames; overrides the instance default.
        prefix : str, optional
            Optional prefix for the per-chunk directory names to distinguish
            outputs (e.g., a short video identifier).

        Returns
        -------
        tuple
            (chunks, slide_change_timestamps, execution_time_seconds)
        """
        try:
            start_time = time()
            self._chunks, slide_change_timestamps = self._detect_slide_changes(
                video_path,
                threshold_embedding=self._threshold_embedding,
                threshold_ssim=self._threshold_ssim,
                interval=self._interval,
                alpha=self._alpha,
            )
            end_time = time()
            self._exe_time = end_time - start_time
            logger.info("⏱️ Chunking finished in %.2f s. Chunks: %d 🎉",
                        self._exe_time, len(self._chunks) if self._chunks else 0)

            if save:
                # 1) Save selected frames per chunk (first, max-entropy, last) in a sibling dir
                self._select_best_frames_per_chunk(
                    output_dir=output_dir,
                    image_format=image_format,
                    prefix=prefix,
                )
                # 2) Save all frames per chunk as before (original behavior)
                self._save_chunks(
                    output_dir=output_dir,
                    image_format=image_format,
                    prefix=prefix,
                )
                # 3) Save metadata JSON at the very end
                self._save_metadata(
                    video_path=video_path,
                    timestamps=slide_change_timestamps,
                    output_dir=output_dir,
                    image_format=image_format,
                    prefix=prefix,
                )

            return self._chunks, slide_change_timestamps, self._exe_time
        except Exception as e:
            logger.exception("💥 Error in chunk(): %s", e)
            raise

    def evaluate(self):
        """
        Compute summary statistics for the current chunk set.

        Side Effects
        ------------
        Sets `avg_frame_per_chunk` and `mean_entropy` as read-only properties.
        """
        try:
            self._avg_frame_per_chunk = self._get_avg_frame_per_time()
            self._mean_entropy = self._get_mean_entropy()
            logger.info("📊 Evaluation — AvgFrames: %s | MeanEntropy: %s ✅",
                        f"{self._avg_frame_per_chunk:.2f}" if self._avg_frame_per_chunk is not None else "N/A",
                        f"{self._mean_entropy:.4f}" if self._mean_entropy is not None else "N/A")
        except Exception as e:
            logger.exception("💥 Error in evaluate(): %s", e)
            raise
