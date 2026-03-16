"""
Embedding Model Module for Semantic Similarity

This module uses SentenceTransformers for semantic matching between
job requirements and resume skills.

Model: all-MiniLM-L6-v2
- Lightweight and fast
- Good performance on semantic similarity tasks
- 384-dimensional embeddings
"""

from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class EmbeddingModel:
    """
    Class for semantic similarity matching using SentenceTransformers.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the SentenceTransformer model
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("✓ Embedding model loaded successfully!")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            Numpy array of embeddings
        """
        return self.model.encode(texts, convert_to_numpy=True)
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        embeddings = self.model.encode([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(similarity)
    
    def batch_similarity(self, text1: str, texts2: List[str]) -> List[float]:
        """
        Calculate similarity between one text and a list of texts.
        
        Args:
            text1: Single text
            texts2: List of texts to compare against
            
        Returns:
            List of similarity scores
        """
        embedding1 = self.model.encode([text1])[0]
        embeddings2 = self.model.encode(texts2)
        
        similarities = []
        for emb2 in embeddings2:
            sim = np.dot(embedding1, emb2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(emb2)
            )
            similarities.append(float(sim))
        
        return similarities
    
    def find_best_match(self, text: str, candidate_texts: List[str]) -> Tuple[str, float]:
        """
        Find the best matching text from a list of candidates.
        
        Args:
            text: Text to match
            candidate_texts: List of candidate texts
            
        Returns:
            Tuple of (best_match_text, similarity_score)
        """
        similarities = self.batch_similarity(text, candidate_texts)
        best_idx = np.argmax(similarities)
        return candidate_texts[best_idx], similarities[best_idx]
