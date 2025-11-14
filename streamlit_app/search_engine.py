"""
Search engine with text-to-image retrieval and rank fusion
"""
import torch
import numpy as np
from typing import List, Tuple, Dict
import faiss

from config import DEVICE


class SearchEngine:
    """Text-to-image search with support for multiple models and fusion"""
    
    def __init__(self, model_manager, faiss_managers: Dict):
        """
        Initialize search engine
        
        Args:
            model_manager: ModelManager instance
            faiss_managers: dict with keys 'openclip' and 'beit3'
        """
        self.model_manager = model_manager
        self.faiss_managers = faiss_managers
        self.device = DEVICE
    
    def text_to_vector_openclip(self, text: str) -> np.ndarray:
        """
        Encode text with OpenCLIP
        
        Args:
            text: query text
        
        Returns:
            numpy array of shape (1, 512)
        """
        model, _, tokenizer = self.model_manager.get_openclip()
        
        # Tokenize
        text_tokens = tokenizer([text]).to(self.device)
        
        # Encode
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy().astype('float32')
    
    def text_to_vector_beit3(self, text: str) -> np.ndarray:
        """
        Encode text with BEiT3 (same as rank_fusion_demo.py)
        
        Args:
            text: query text
        
        Returns:
            numpy array of shape (1, 768)
        """
        model, tokenizer, _ = self.model_manager.get_beit3()
        
        max_len = 64
        
        # Tokenize text (same as rank_fusion_demo.py)
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        if len(token_ids) > max_len - 2:
            token_ids = token_ids[:max_len - 2]
        
        bos_token_id = tokenizer.bos_token_id
        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id
        
        language_tokens = [bos_token_id] + token_ids + [eos_token_id]
        num_tokens = len(language_tokens)
        padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
        language_tokens = language_tokens + [pad_token_id] * (max_len - num_tokens)
        
        # Convert to tensor
        language_tokens = torch.tensor([language_tokens]).to(self.device)
        padding_mask = torch.tensor([padding_mask]).to(self.device)
        
        # Encode using model
        with torch.no_grad():
            _, language_cls = model(
                text_description=language_tokens,
                padding_mask=padding_mask,
                only_infer=True
            )
        
        return language_cls.cpu().numpy().astype('float32')
    
    def search_openclip(self, text: str, k: int) -> Tuple[np.ndarray, List[str]]:
        """
        Search using OpenCLIP model
        
        Args:
            text: query text
            k: number of results
        
        Returns:
            scores: similarity scores
            uuids: list of image UUIDs
        """
        # Encode text
        query_vector = self.text_to_vector_openclip(text)
        
        # Search in FAISS index
        faiss_manager = self.faiss_managers['openclip']
        scores, uuids = faiss_manager.search(query_vector, k)
        
        return scores, uuids
    
    def search_beit3(self, text: str, k: int) -> Tuple[np.ndarray, List[str]]:
        """
        Search using BEiT3 model
        
        Args:
            text: query text
            k: number of results
        
        Returns:
            scores: similarity scores
            uuids: list of image UUIDs
        """
        # Encode text
        query_vector = self.text_to_vector_beit3(text)
        
        # Search in FAISS index
        faiss_manager = self.faiss_managers['beit3']
        scores, uuids = faiss_manager.search(query_vector, k)
        
        return scores, uuids
    
    def score_based_fusion(
        self,
        results1: List[Tuple[str, float]],
        results2: List[Tuple[str, float]],
        alpha: float = 0.5
    ) -> List[Tuple[str, float]]:
        """
        Score-based Fusion algorithm using cosine similarity scores
        
        Args:
            results1: list of (uuid, similarity_score) tuples from model 1 (OpenCLIP)
            results2: list of (uuid, similarity_score) tuples from model 2 (BEiT3)
            alpha: weight for results1 (1-alpha for results2)
        
        Returns:
            fused results as list of (uuid, fused_score) tuples
        """
        fusion_scores = {}
        
        # Process results1 (OpenCLIP) - use actual similarity scores
        for uuid, score in results1:
            fusion_scores[uuid] = alpha * score
        
        # Process results2 (BEiT3) - use actual similarity scores
        for uuid, score in results2:
            if uuid in fusion_scores:
                fusion_scores[uuid] += (1 - alpha) * score
            else:
                fusion_scores[uuid] = (1 - alpha) * score
        
        # Sort by fused score (higher is better)
        fused_results = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Debug logging
        if len(fused_results) > 0:
            print(f"Fusion scores range: {fused_results[0][1]:.4f} (max) to {fused_results[-1][1]:.4f} (min)")
            print(f"Top 3 fused scores: {[f'{score:.4f}' for _, score in fused_results[:3]]}")
        
        return fused_results
    
    def search_fusion(self, text: str, k: int, alpha: float = 0.5) -> List[Tuple[str, float]]:
        """
        Search using score-based fusion of both models
        
        Args:
            text: query text
            k: number of final results
            alpha: fusion weight (0.0 = all BEiT3, 1.0 = all OpenCLIP)
        
        Returns:
            list of (uuid, fused_score) tuples
        """
        # Get results from both models (retrieve more for better fusion)
        search_k = max(k * 10, 100)
        
        scores1, uuids1 = self.search_openclip(text, search_k)
        scores2, uuids2 = self.search_beit3(text, search_k)
        
        # Create result tuples (uuid, score)
        results1 = list(zip(uuids1, scores1))
        results2 = list(zip(uuids2, scores2))
        
        # Apply score-based fusion
        fused_results = self.score_based_fusion(results1, results2, alpha)
        
        # Return top-k
        return fused_results[:k]
    
    def format_results(
        self,
        uuids: List[str],
        scores: List[float],
        model_name: str
    ) -> List[Dict]:
        """
        Format search results with image paths
        
        Args:
            uuids: list of image UUIDs
            scores: list of similarity scores
            model_name: 'openclip' or 'beit3' (for metadata lookup)
        
        Returns:
            list of dicts with 'uuid', 'path', 'score'
        """
        faiss_manager = self.faiss_managers[model_name]
        
        results = []
        for uuid, score in zip(uuids, scores):
            image_info = faiss_manager.get_image_by_uuid(uuid)
            if image_info:
                results.append({
                    'uuid': uuid,
                    'path': image_info['path'],
                    'score': float(score)
                })
        
        return results
    
    def format_fusion_results(
        self,
        fusion_results: List[Tuple[str, float]]
    ) -> List[Dict]:
        """
        Format fusion results with image paths
        
        Args:
            fusion_results: list of (uuid, fused_score) tuples
        
        Returns:
            list of dicts with 'uuid', 'path', 'score'
        """
        # Try to get path from either index (should be same images in both)
        openclip_manager = self.faiss_managers['openclip']
        beit3_manager = self.faiss_managers['beit3']
        
        results = []
        for uuid, score in fusion_results:
            # Try OpenCLIP first
            image_info = openclip_manager.get_image_by_uuid(uuid)
            
            # If not found, try BEiT3
            if not image_info:
                image_info = beit3_manager.get_image_by_uuid(uuid)
            
            if image_info:
                results.append({
                    'uuid': uuid,
                    'path': image_info['path'],
                    'score': float(score)
                })
        
        return results
