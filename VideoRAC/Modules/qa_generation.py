import os
import re
import json
import base64
import pickle
import logging

import cv2
from yt_dlp import YoutubeDL
from youtube_transcript_api import YouTubeTranscriptApi
from tqdm.auto import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logger = logging.getLogger(__name__)


def _sanitize_filename(name: str) -> str:
    """
    Replace characters that are invalid on common filesystems.
    """
    return re.sub(r'[\\/:*?"<>|]+', "_", name).strip()


class VideoQAGenerator:
    """
    End-to-end utilities for:
      - Downloading videos with yt-dlp
      - Fetching YouTube transcripts
      - Describing frames via a user-supplied LLM function
      - Generating Q&A pairs from (frames + transcript)
      - Saving results under a managed output directory

    Only `run()` is public. All other methods/attributes are private by convention.
    """

    def __init__(
        self,
        *,
        video_urls,
        llm_fn,
        chunks_root_path: str = "../chunks/hybrid_clip_ssim_chunking",
        output_dir: str = "./outputs",
        qa_prompt: str | None = None,
        qa_pairs: int = 50,
        image_prompt: str | None = None,
        ydl_opts: dict | None = None,
    ):
        # Validate inputs
        if not llm_fn or not callable(llm_fn):
            logger.error("🚫 llm_fn must be provided and callable.")
            raise ValueError("llm_fn must be provided and callable.")
        if not video_urls or not isinstance(video_urls, (list, tuple)) or not all(isinstance(u, str) for u in video_urls):
            logger.error("🚫 video_urls must be a non-empty list of strings.")
            raise ValueError("video_urls must be a non-empty list of strings.")

        # Private attributes
        self._llm_fn = llm_fn
        self._video_urls = list(video_urls)
        self._chunks_root_path = chunks_root_path

        self._output_dir = os.path.abspath(output_dir)
        self._downloads_dir = os.path.join(self._output_dir, "downloads")
        self._transcripts_dir_root = os.path.join(self._output_dir, "transcripts")

        # Ensure directories exist
        for d in [self._output_dir, self._downloads_dir, self._transcripts_dir_root]:
            os.makedirs(d, exist_ok=True)

        # Prompts
        self._qa_pairs = int(qa_pairs) if qa_pairs and qa_pairs > 0 else 50
        self._qa_prompt = qa_prompt or (
            "*Generate up to {N} unique, relevant question–answer pairs based on the provided description. "
            "Treat them as exam-style questions that benefit from retrieval.*\n\n"
            "- Provide each Q&A in this format:\n\n"
            "1. **question** : [Your question here]\n"
            "**answer** : [Your answer here]\n\n"
            "2. **question** : [Your question here]\n"
            "**answer** : [Your answer here]\n\n"
            "- Avoid redundancy and cover different aspects of the content.\n"
            "- If {N} unique pairs are not possible without repetition, stop earlier."
        )
        self._image_prompt = image_prompt or (
            "Analyze the frame and produce a cohesive description usable for question generation. "
            "Include visible text, salient objects, and overall scene context."
        )

        # yt-dlp options
        default_ydl_opts = {
            "format": "bestvideo+bestaudio/best",
            "merge_output_format": "mp4",
            "outtmpl": os.path.join(self._downloads_dir, "%(id)s.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "restrictfilenames": True,
        }
        self._ydl_opts = {**default_ydl_opts, **(ydl_opts or {})}

        logger.info("✅ Initialized. URLs: %d | Output: %s | QA pairs: %d",
                    len(self._video_urls), self._output_dir, self._qa_pairs)

    # ----------------------
    # Read-only properties
    # ----------------------

    @property
    def video_urls(self) -> list[str]:
        return self._video_urls

    @property
    def output_dir(self) -> str:
        return self._output_dir

    @property
    def downloads_dir(self) -> str:
        return self._downloads_dir

    @property
    def transcripts_dir_root(self) -> str:
        return self._transcripts_dir_root

    @property
    def qa_pairs(self) -> int:
        return self._qa_pairs

    @property
    def qa_prompt(self) -> str:
        return self._qa_prompt

    @property
    def image_prompt(self) -> str:
        return self._image_prompt

    # ----------------------
    # Private helpers (LLM, encode, prompts)
    # ----------------------

    def _encode_np_array_to_base64(self, frame, img_format: str = "jpg") -> str:
        """
        Encode a numpy frame to base64 string.
        """
        success, buffer = cv2.imencode(f".{img_format}", frame)
        if not success:
            logger.error("💥 Could not encode frame as image.")
            raise ValueError("Could not encode frame as image.")
        return base64.b64encode(buffer).decode("utf-8")

    def _image_description(self, frame, image_prompt: str | None = None) -> str:
        """
        Describe a frame using the user-supplied LLM function.
        """
        try:
            prompt = image_prompt or self._image_prompt
            base64_image = self._encode_np_array_to_base64(frame, img_format="jpg")
            messages = [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpg;base64,{base64_image}",
                                "detail": "auto",
                            },
                        },
                    ],
                },
            ]
            return self._llm_fn(messages)
        except Exception as e:
            logger.exception("💥 _image_description failed: %s", e)
            raise

    def _generate_qas(self, description: str, qa_prompt: str | None = None, num_pairs: int | None = None) -> str:
        """
        Generate question–answer pairs using the user-supplied LLM function.
        """
        try:
            N = int(num_pairs) if num_pairs and num_pairs > 0 else self._qa_pairs
            system_prompt = (qa_prompt or self._qa_prompt).replace("{N}", str(N))
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Here is the description of the video:\n\n"
                                f"{description}\n\n"
                                f"Generate up to {N} unique question–answer pairs based on this description."
                            ),
                        }
                    ],
                },
            ]
            return self._llm_fn(messages)
        except Exception as e:
            logger.exception("💥 _generate_qas failed: %s", e)
            raise

    # ----------------------
    # Private helpers (YouTube / transcripts)
    # ----------------------

    def _download_video(self, url: str) -> tuple[str, str]:
        """
        Download a video with yt-dlp; return (file_path, file_name).
        """
        try:
            with YoutubeDL(self._ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                file_path = None
                if isinstance(info, dict):
                    req = info.get("requested_downloads")
                    if req and isinstance(req, list) and "filepath" in req[0]:
                        file_path = req[0]["filepath"]
                    if not file_path:
                        file_path = ydl.prepare_filename(info)
                if not file_path:
                    logger.error("🚫 Could not resolve downloaded file path.")
                    raise RuntimeError("Could not resolve downloaded file path.")
                file_name = os.path.basename(file_path)
                logger.info("⬇️ Downloaded: %s", file_name)
                return file_path, file_name
        except Exception as e:
            logger.exception("💥 _download_video failed for %s: %s", url, e)
            raise

    def _extract_video_id(self, url: str) -> str:
        """
        Extract the YouTube video id from a typical watch URL.
        """
        try:
            video_id = url.split("v=")[1]
            amp = video_id.find("&")
            if amp != -1:
                video_id = video_id[:amp]
            return video_id
        except Exception as e:
            logger.exception("💥 _extract_video_id failed for %s: %s", url, e)
            raise

    def _get_transcript(self, video_id: str) -> str:
        """
        Full transcript string from YouTubeTranscriptApi.
        """
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            return " ".join([entry["text"] for entry in transcript])
        except Exception as e:
            logger.exception("💥 _get_transcript failed for %s: %s", video_id, e)
            raise

    def _get_raw_transcripts(self, video_id: str):
        """
        Raw transcript entries from YouTubeTranscriptApi.
        """
        try:
            return YouTubeTranscriptApi.get_transcript(video_id)
        except Exception as e:
            logger.exception("💥 _get_raw_transcripts failed for %s: %s", video_id, e)
            raise

    # ----------------------
    # Private helpers (frames / combine)
    # ----------------------

    def _analyze_frames(self, frames, image_prompt: str | None = None) -> str:
        """
        Run frame descriptions and join them into a text block.
        """
        try:
            results = []
            for i, frame in enumerate(frames):
                desc = self._image_description(frame, image_prompt=image_prompt)
                results.append(f"Frame {i}: {desc}")
            return "\n".join(results)
        except Exception as e:
            logger.exception("💥 _analyze_frames failed: %s", e)
            raise

    def _combined_results(self, frames, transcript, image_prompt: str | None = None) -> str:
        """
        Combine transcript and frame descriptions into one string.
        """
        try:
            analyzed = self._analyze_frames(frames, image_prompt=image_prompt)
            return f"Transcript:\n{transcript}\n\nAnalyze Frames Results:\n{analyzed}"
        except Exception as e:
            logger.exception("💥 _combined_results failed: %s", e)
            raise

    # ----------------------
    # Private helpers (FS / chunks / saves)
    # ----------------------

    def _video_details(self) -> list[dict]:
        """
        Download videos, fetch transcripts, and return video metadata.
        """
        details: list[dict] = []
        for url in tqdm(self._video_urls, desc="Fetching video details", unit="video"):
            try:
                video_id = self._extract_video_id(url)
                transcript = self._get_transcript(video_id)
                path, file_name = self._download_video(url)
                video_name = os.path.splitext(file_name)[0]
                details.append(
                    {"id": video_id, "url": url, "path": path, "transcript": transcript, "video_name": video_name}
                )
            except Exception:
                # Error already logged
                continue
        logger.info("📦 Prepared %d video(s).", len(details))
        return details

    def _chunk_frames(self, chunks_path: str):
        """
        Load .jpg frames from a directory.
        """
        frames = []
        try:
            if not os.path.isdir(chunks_path):
                logger.warning("⚠️ chunks_path does not exist: %s", chunks_path)
                return frames
            for file in os.listdir(chunks_path):
                full_path = os.path.join(chunks_path, file)
                if os.path.isfile(full_path) and file.lower().endswith(".jpg"):
                    image = cv2.imread(full_path)
                    if image is not None:
                        frames.append(image)
            return frames
        except Exception as e:
            logger.exception("💥 _chunk_frames failed: %s", e)
            raise

    def _save_as_json(self, idx, video_id, video_name, result, video_url, file_path: str | None = None) -> dict:
        """
        Append Q&A entries to a single JSON file under output_dir.
        """
        try:
            dst = file_path or os.path.join(self._output_dir, "Q&A.json")
            os.makedirs(os.path.dirname(dst), exist_ok=True)

            if os.path.exists(dst):
                with open(dst, "r", encoding="utf-8") as fp:
                    metadata = json.load(fp)
            else:
                metadata = []

            entry = {"video_id": video_id, "video_url": video_url, "video_name": video_name, "Q&A": {}}

            blocks = [b for b in result.split("\n\n") if b.strip()]
            for i, QA in enumerate(blocks):
                lines = [l for l in QA.split("\n") if l.strip()]
                if len(lines) < 2:
                    continue
                q_line = lines[0]
                a_line = lines[1]
                q_idx = q_line.find(": ")
                a_idx = a_line.find(": ")
                if q_idx != -1:
                    entry["Q&A"][f"question {i}"] = q_line[q_idx + 2 :].strip()
                if a_idx != -1:
                    entry["Q&A"][f"answer {i}"] = a_line[a_idx + 2 :].strip()

            metadata.append(entry)
            with open(dst, "w", encoding="utf-8") as fp:
                json.dump(metadata, fp, indent=4, ensure_ascii=False)

            logger.info("💾 Saved Q&A for %s to %s", video_name, dst)
            return entry
        except Exception as e:
            logger.exception("💥 _save_as_json failed: %s", e)
            raise

    # ----------------------
    # Public pipeline
    # ----------------------

    def run(self, *, qa_prompt: str | None = None, qa_pairs: int | None = None, image_prompt: str | None = None):
        """
        Execute the full pipeline:
          1) Download video + fetch transcript
          2) Load chunked frames if available
          3) Combine frames + transcript
          4) Generate Q&A pairs
          5) Save results to output_dir/Q&A.json

        Parameters
        ----------
        qa_prompt : str, optional
            Custom system prompt for Q&A generation.
        qa_pairs : int, optional
            Desired number of question–answer pairs.
        image_prompt : str, optional
            Custom system prompt for image/frame descriptions.
        """
        try:
            details = self._video_details()
            for i, v in enumerate(tqdm(details, desc="Processing videos", unit="video")):
                safe_name = _sanitize_filename(v["video_name"])
                frames_dir = os.path.join(self._chunks_root_path, safe_name, "hybrid_clip_ssim_frame_dir")
                frames = self._chunk_frames(frames_dir)

                combined = self._combined_results(frames, v["transcript"], image_prompt=image_prompt)
                qas = self._generate_qas(combined, qa_prompt=qa_prompt, num_pairs=qa_pairs)
                self._save_as_json(i + 1, v["id"], v["video_name"], qas, v["url"])
        except Exception as e:
            logger.exception("💥 run() failed: %s", e)
            raise

    # ----------------------
    # Optional: transcript chunking utilities (private)
    # ----------------------

    def _prepare_concatenation_with_metadata(self, transcripts):
        """
        Concatenate transcript entries and preserve character-span metadata.
        """
        concatenated_text = ""
        transcript_metadata = []
        current_pos = 0

        for item in transcripts:
            text = item["text"]
            start_time = item["start"]
            duration = item["duration"]
            concatenated_text += text + " "
            transcript_metadata.append(
                {
                    "start_pos": current_pos,
                    "end_pos": current_pos + len(text),
                    "start_time": start_time,
                    "end_time": start_time + duration,
                }
            )
            current_pos += len(text) + 1

        return concatenated_text.strip(), transcript_metadata

    def _chunk_transcript(self, transcript):
        """
        Chunk a transcript using a recursive text splitter.
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        documents = [Document(page_content=transcript)]
        return splitter.split_documents(documents)

    def _map_langchain_chunks_to_metadata(self, chunks, concatenated_text, metadata):
        """
        Map each chunk back to approximate time ranges using character spans.
        """
        chunk_metadata = []
        current_pos = 0

        for chunk in chunks:
            chunk_text = chunk.page_content
            start_idx = concatenated_text.find(chunk_text, current_pos)
            if start_idx == -1:
                current_pos = current_pos + len(chunk_text)
                continue
            end_idx = start_idx + len(chunk_text)

            related = [m for m in metadata if not (m["end_pos"] < start_idx or m["start_pos"] > end_idx)]
            if related:
                chunk_metadata.append(
                    {
                        "chunk_text": chunk_text,
                        "start_time": related[0]["start_time"],
                        "end_time": related[-1]["end_time"],
                    }
                )
            current_pos = end_idx

        return chunk_metadata

    def _chunk_videos_transcript(self):
        """
        Create transcript artifacts (full text, spans, chunk metadata) under output_dir.
        """
        details = self._video_details()

        for v in tqdm(details, desc="Chunking transcripts", unit="video"):
            try:
                video_dir = os.path.join(self._transcripts_dir_root, _sanitize_filename(v["video_name"]))
                os.makedirs(video_dir, exist_ok=True)

                transcripts = self._get_raw_transcripts(v["id"])
                full_text, meta_spans = self._prepare_concatenation_with_metadata(transcripts)
                chunks = self._chunk_transcript(full_text)
                meta_chunks = self._map_langchain_chunks_to_metadata(chunks, full_text, meta_spans)

                transcript_txt_path = os.path.join(video_dir, "transcript.txt")
                metadata_json_path = os.path.join(video_dir, "metadata.json")
                metadata_chunks_json_path = os.path.join(video_dir, "metadata_chunks.json")
                chunks_pickle_path = os.path.join(video_dir, "chunks.pkl")

                with open(transcript_txt_path, "w", encoding="utf-8") as f:
                    f.write(full_text)
                with open(metadata_json_path, "w", encoding="utf-8") as f:
                    json.dump(meta_spans, f, indent=4, ensure_ascii=False)
                with open(metadata_chunks_json_path, "w", encoding="utf-8") as f:
                    json.dump(meta_chunks, f, indent=4, ensure_ascii=False)
                with open(chunks_pickle_path, "wb") as f:
                    pickle.dump(chunks, f)

                logger.info("🗂️ Saved transcript artifacts for %s", v["video_name"])
            except Exception as e:
                logger.exception("💥 _chunk_videos_transcript failed for %s: %s", v.get("video_name", "?"), e)
                continue
