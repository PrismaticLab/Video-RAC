from scipy.stats import entropy
import numpy as np

def calculate_entropy(data):
    """Calculate entropy of pixel intensity distribution in a frame."""
    value, counts = np.unique(data, return_counts=True)
    return entropy(counts, base=2)

def chunks_mean_entropy(chunks):
  chunk_entropies = []
  for chunk in chunks:
      frame_entropies = [calculate_entropy(frame) for frame in chunk]
      average_chunk_entropy = sum(frame_entropies) / len(frame_entropies)
      chunk_entropies.append(average_chunk_entropy)

  mean_entropy = sum(chunk_entropies) / len(chunk_entropies)
  return mean_entropy