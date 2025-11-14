"""
FAISS Index Management with IndexIDMap2 and UUID support
"""
import faiss
import numpy as np
import json
import os
import uuid
from datetime import datetime
from typing import List, Tuple, Dict, Optional

from config import (
    OPENCLIP_INDEX_FILE,
    BEIT3_INDEX_FILE,
    METADATA_FILE,
    OPENCLIP_DIM,
    BEIT3_DIM
)


class FAISSManager:
    """Manages FAISS IndexIDMap2 with UUID-based identification"""
    
    def __init__(self, model_name: str, dim: int):
        """
        Initialize FAISS manager
        
        Args:
            model_name: 'openclip' or 'beit3'
            dim: Vector dimension (512 for OpenCLIP, 768 for BEiT3)
        """
        self.model_name = model_name
        self.dim = dim
        self.index_file = OPENCLIP_INDEX_FILE if model_name == 'openclip' else BEIT3_INDEX_FILE
        
        # Create or load index
        self.index = self.load_index()
        
        # Load metadata
        self.metadata = self.load_metadata()
        
        # Validate consistency
        self._validate_consistency()
    
    def _validate_consistency(self):
        """Validate that FAISS index and metadata are consistent"""
        faiss_count = self.index.ntotal
        metadata_count = len(self.metadata[self.model_name]['images'])
        
        if faiss_count != metadata_count:
            print(f"⚠️  WARNING: [{self.model_name}] Inconsistency detected!")
            print(f"   FAISS index: {faiss_count} vectors")
            print(f"   Metadata: {metadata_count} images")
            print(f"   This may cause issues. Consider clearing and re-encoding.")
    
    def create_index(self) -> faiss.IndexIDMap2:
        """Create new FAISS IndexIDMap2 with IndexFlatIP base"""
        base_index = faiss.IndexFlatIP(self.dim)
        index = faiss.IndexIDMap2(base_index)
        return index
    
    def load_index(self) -> faiss.IndexIDMap2:
        """Load index from file or create new one"""
        if os.path.exists(self.index_file):
            try:
                index = faiss.read_index(self.index_file)
                print(f"Loaded {self.model_name} index from {self.index_file}")
                return index
            except Exception as e:
                print(f"Error loading index: {e}, creating new index")
                return self.create_index()
        else:
            print(f"Creating new {self.model_name} index")
            return self.create_index()
    
    def save_index(self):
        """Save index and metadata to disk immediately"""
        try:
            # Ensure directory exists before saving
            os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, self.index_file)
            
            # Reload metadata from disk before saving to avoid overwriting other models' data
            self._merge_metadata_before_save()
            
            # Save metadata
            self.save_metadata()
            
            print(f"Saved {self.model_name} index and metadata")
        except Exception as e:
            print(f"Error saving index: {e}")
            raise
    
    def _merge_metadata_before_save(self):
        """Merge current metadata with disk version to preserve other models' data"""
        if os.path.exists(METADATA_FILE):
            try:
                # Load latest metadata from disk
                with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                    disk_metadata = json.load(f)
                
                # Merge: Keep other models' data from disk, use our model's data from memory
                for model_name in disk_metadata.keys():
                    if model_name != self.model_name:
                        # Preserve other model's data from disk
                        self.metadata[model_name] = disk_metadata[model_name]
                
                print(f"[{self.model_name}] Merged metadata before save")
            except Exception as e:
                print(f"[{self.model_name}] Warning: Could not merge metadata: {e}")
                # Continue with save anyway
    
    def load_metadata(self) -> Dict:
        """Load metadata from JSON file and rebuild mappings"""
        if os.path.exists(METADATA_FILE):
            try:
                with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # Ensure structure exists for this model
                if self.model_name not in metadata:
                    metadata[self.model_name] = {
                        'images': [],
                        'uuid_to_index': {},
                        'path_to_uuid': {},
                        'total_images': 0
                    }
                
                # Rebuild mappings from images list to ensure consistency
                model_meta = metadata[self.model_name]
                model_meta['path_to_uuid'] = {}
                model_meta['uuid_to_index'] = {}
                
                for idx, img in enumerate(model_meta['images']):
                    uuid_str = img['uuid']
                    path = img['path']
                    model_meta['path_to_uuid'][path] = uuid_str
                    model_meta['uuid_to_index'][uuid_str] = idx
                
                # Update total_images to match actual count
                model_meta['total_images'] = len(model_meta['images'])
                
                print(f"[{self.model_name}] Loaded metadata: {model_meta['total_images']} images")
                
                return metadata
            except Exception as e:
                print(f"Error loading metadata: {e}, creating new metadata")
                return self._create_empty_metadata()
        else:
            return self._create_empty_metadata()
    
    def _create_empty_metadata(self) -> Dict:
        """Create empty metadata structure"""
        return {
            'openclip': {
                'images': [],
                'uuid_to_index': {},
                'path_to_uuid': {},
                'total_images': 0
            },
            'beit3': {
                'images': [],
                'uuid_to_index': {},
                'path_to_uuid': {},
                'total_images': 0
            }
        }
    
    def save_metadata(self):
        """Save metadata to JSON file"""
        try:
            # Ensure directory exists before saving
            os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
            
            with open(METADATA_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving metadata: {e}")
            raise
    
    def uuid_to_int64(self, uuid_str: str) -> np.int64:
        """Convert UUID string to int64 for FAISS"""
        return np.int64(uuid.UUID(uuid_str).int % (2**63 - 1))
    
    def int64_to_uuid(self, int_val: int) -> Optional[str]:
        """Reverse lookup UUID from int64"""
        model_meta = self.metadata[self.model_name]
        
        # Search through uuid_to_index mapping
        for uuid_str in model_meta['uuid_to_index'].keys():
            if self.uuid_to_int64(uuid_str) == int_val:
                return uuid_str
        
        return None
    
    def add_vectors(self, vectors: np.ndarray, image_paths: List[str]):
        """
        Add vectors to index with UUIDs
        
        Args:
            vectors: numpy array of shape (N, dim)
            image_paths: list of N image paths
        """
        n = len(image_paths)
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(vectors)
        
        # Debug logging
        print(f"[{self.model_name}] Adding {len(image_paths)} images")
        print(f"[{self.model_name}] Vectors shape: {vectors.shape}")
        
        # Generate UUIDs and convert to int64
        uuids = []
        uuid_int64s = []
        
        for path in image_paths:
            # Check if path already exists
            model_meta = self.metadata[self.model_name]
            if path in model_meta['path_to_uuid']:
                print(f"Warning: Image {path} already exists in index, skipping")
                continue
            
            # Generate new UUID
            new_uuid = str(uuid.uuid4())
            uuids.append(new_uuid)
            uuid_int64s.append(self.uuid_to_int64(new_uuid))
        
        # Only add if there are new images
        if len(uuids) == 0:
            print("No new images to add")
            return 0
        
        # Filter vectors to only new images
        new_indices = [i for i, path in enumerate(image_paths) if path not in model_meta['path_to_uuid']]
        vectors_to_add = vectors[new_indices]
        paths_to_add = [image_paths[i] for i in new_indices]
        
        # Add to FAISS index
        uuid_int64_array = np.array(uuid_int64s, dtype=np.int64)
        self.index.add_with_ids(vectors_to_add, uuid_int64_array)
        
        # Update metadata
        for i, (uuid_str, path) in enumerate(zip(uuids, paths_to_add)):
            self.add_image_metadata(uuid_str, path)
        
        # Debug: Check metadata before save
        print(f"[{self.model_name}] Before save: {len(self.metadata[self.model_name]['images'])} images in metadata")
        
        # Save immediately
        self.save_index()
        
        # Debug: Verify save
        print(f"[{self.model_name}] After save: {len(self.metadata[self.model_name]['images'])} images in metadata")
        print(f"Added {len(uuids)} images to {self.model_name} index")
        return len(uuids)
    
    def remove_vectors(self, uuids: List[str]):
        """
        Remove vectors from index by UUIDs
        
        Args:
            uuids: list of UUID strings to remove
        """
        # Convert UUIDs to int64
        uuid_int64s = [self.uuid_to_int64(u) for u in uuids]
        uuid_int64_array = np.array(uuid_int64s, dtype=np.int64)
        
        # Remove from FAISS index
        self.index.remove_ids(uuid_int64_array)
        
        # Remove from metadata
        for uuid_str in uuids:
            self.remove_image_metadata(uuid_str)
        
        # Save immediately
        self.save_index()
        
        print(f"Removed {len(uuids)} images from {self.model_name} index")
    
    def search(self, query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, List[str]]:
        """
        Search for similar vectors
        
        Args:
            query_vector: numpy array of shape (1, dim)
            k: number of results to return
        
        Returns:
            distances: numpy array of similarity scores
            uuids: list of UUID strings
        """
        # Normalize query vector
        faiss.normalize_L2(query_vector)
        
        # Search
        distances, indices = self.index.search(query_vector, k)
        
        # Convert indices (int64) back to UUIDs
        uuids = []
        for idx in indices[0]:
            if idx != -1:  # -1 means no result
                uuid_str = self.int64_to_uuid(idx)
                if uuid_str:
                    uuids.append(uuid_str)
        
        return distances[0][:len(uuids)], uuids
    
    def get_all_images(self) -> List[Dict]:
        """Get all images from metadata"""
        model_meta = self.metadata[self.model_name]
        return model_meta['images']
    
    def add_image_metadata(self, uuid_str: str, path: str):
        """Add image metadata entry"""
        model_meta = self.metadata[self.model_name]
        
        current_index = len(model_meta['images'])
        
        image_entry = {
            'uuid': uuid_str,
            'path': path,
            'added_at': datetime.now().isoformat(),
            'faiss_index': current_index
        }
        
        model_meta['images'].append(image_entry)
        model_meta['uuid_to_index'][uuid_str] = current_index
        model_meta['path_to_uuid'][path] = uuid_str
        model_meta['total_images'] = len(model_meta['images'])
    
    def remove_image_metadata(self, uuid_str: str):
        """Remove image metadata entry"""
        model_meta = self.metadata[self.model_name]
        
        # Find and remove from images list
        model_meta['images'] = [
            img for img in model_meta['images'] if img['uuid'] != uuid_str
        ]
        
        # Get path before removing
        path = None
        for p, u in model_meta['path_to_uuid'].items():
            if u == uuid_str:
                path = p
                break
        
        # Remove from mappings
        if uuid_str in model_meta['uuid_to_index']:
            del model_meta['uuid_to_index'][uuid_str]
        
        if path and path in model_meta['path_to_uuid']:
            del model_meta['path_to_uuid'][path]
        
        # Update total
        model_meta['total_images'] = len(model_meta['images'])
    
    def get_image_by_uuid(self, uuid_str: str) -> Optional[Dict]:
        """Get image info by UUID"""
        model_meta = self.metadata[self.model_name]
        
        for img in model_meta['images']:
            if img['uuid'] == uuid_str:
                return img
        
        return None
    
    def get_uuid_by_path(self, path: str) -> Optional[str]:
        """Get UUID by image path"""
        model_meta = self.metadata[self.model_name]
        return model_meta['path_to_uuid'].get(path, None)
    
    def get_total_images(self) -> int:
        """Get total number of images in index from FAISS (source of truth)"""
        return self.index.ntotal
