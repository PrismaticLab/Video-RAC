"""
VideoRAC - Video Retrieval-Augmented Chunking and Q&A Generation Toolkit
=======================================================================

VideoRAC is designed for building retrieval-augmented video pipelines.
It combines video segmentation, frame analysis, and large-language-model
based question generation in a modular and extensible structure.

Submodules
----------
VideoRAC.Modules.chunking
    Implements hybrid CLIP+SSIM chunking for video segmentation.

VideoRAC.Modules.entropy_utils
    Utility functions for entropy and content diversity metrics.

VideoRAC.Modules.qa_generation
    Full pipeline for transcript-driven and frame-driven Q&A generation.

"""

__name__ = "PrismSSL"
__version__ = "0.2.1"

from VideoRAC.Modules.chunking import HybridChunker
from VideoRAC.Modules.entropy_utils import EntropyUtils
from VideoRAC.Modules.qa_generation import VideoQAGenerator

__all__ = ["HybridChunker", "EntropyUtils", "VideoQAGenerator"]
