import logging
from time import time

import cv2
import torch
from scipy.spatial.distance import cosine
from skimage.metrics import structural_similarity as ssim
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor

import VideoRAC.Modules.entropy_utils as E

logger = logging.getLogger(__name__)


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
    """

    def __init__(
        self,
        clip_model: str = '"openai/clip-vit-base-patch32"',
        *,
        threshold_embedding: float = 0.8,
        threshold_ssim: float = 0.8,
        interval: int = 1,
        alpha: float = 0.5,
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

            # Progress bar without altering loop logic or termination.
            pbar = tqdm(desc="⏳ Processing frames", unit="frame", leave=False)

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

    def chunk(self, video_path):
        """
        Run detection and record execution time using the instance configuration.

        Parameters
        ----------
        video_path : str
            Path to a video file.

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
