"""
Image encoding utilities for OpenCLIP and BEiT3
"""
import torch
import numpy as np
from PIL import Image
from typing import List, Tuple
import os
from tqdm import tqdm

from config import BATCH_SIZE, IMAGE_EXTENSIONS, DEVICE


class ImageEncoder:
    """Encodes images using OpenCLIP or BEiT3 models"""
    
    def __init__(self, model_manager):
        """
        Initialize encoder with model manager
        
        Args:
            model_manager: ModelManager instance
        """
        self.model_manager = model_manager
        self.device = DEVICE
    
    def encode_with_openclip(self, image_paths: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Encode images with OpenCLIP
        
        Args:
            image_paths: list of image file paths
        
        Returns:
            vectors: numpy array of shape (N, 512)
            successfully_encoded_paths: list of successfully encoded image paths
        """
        model, preprocess, _ = self.model_manager.get_openclip()
        
        all_features = []
        successfully_encoded_paths = []
        
        # Process in batches
        for i in range(0, len(image_paths), BATCH_SIZE):
            batch_paths = image_paths[i:i+BATCH_SIZE]
            batch_images = []
            batch_valid_paths = []
            
            for path in batch_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    image_tensor = preprocess(image)
                    batch_images.append(image_tensor)
                    batch_valid_paths.append(path)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    continue
            
            if len(batch_images) == 0:
                continue
            
            # Stack and move to device
            batch_tensor = torch.stack(batch_images).to(self.device)
            
            # Encode
            with torch.no_grad():
                image_features = model.encode_image(batch_tensor)
                # Don't normalize here - will be normalized in faiss_manager
            
            all_features.append(image_features.cpu().numpy())
            successfully_encoded_paths.extend(batch_valid_paths)
        
        # Concatenate all batches
        if len(all_features) == 0:
            return np.array([]), []
        
        vectors = np.vstack(all_features).astype('float32')
        return vectors, successfully_encoded_paths
    
    def encode_with_beit3(self, image_paths: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Encode images with BEiT3
        
        Args:
            image_paths: list of image file paths
        
        Returns:
            vectors: numpy array of shape (N, 768)
            successfully_encoded_paths: list of successfully encoded image paths
        """
        model, tokenizer, image_transform = self.model_manager.get_beit3()
        
        all_features = []
        successfully_encoded_paths = []
        
        # Process in batches
        for i in range(0, len(image_paths), BATCH_SIZE):
            batch_paths = image_paths[i:i+BATCH_SIZE]
            batch_images = []
            batch_valid_paths = []
            
            for path in batch_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    image_tensor = image_transform(image)
                    batch_images.append(image_tensor)
                    batch_valid_paths.append(path)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    continue
            
            if len(batch_images) == 0:
                continue
            
            # Stack and move to device
            batch_tensor = torch.stack(batch_images).to(self.device)
            
            # Encode using model
            with torch.no_grad():
                vision_cls, _ = model(image=batch_tensor, only_infer=True)
            
            all_features.append(vision_cls.cpu().numpy())
            successfully_encoded_paths.extend(batch_valid_paths)
        
        # Concatenate all batches
        if len(all_features) == 0:
            return np.array([]), []
        
        vectors = np.vstack(all_features).astype('float32')
        return vectors, successfully_encoded_paths
    
    def encode_folder(self, folder_path: str, model_name: str, progress_callback=None) -> Tuple[np.ndarray, List[str]]:
        """
        Encode all images in a folder
        
        Args:
            folder_path: path to folder containing images
            model_name: 'openclip' or 'beit3'
            progress_callback: optional callback for progress updates
        
        Returns:
            vectors: numpy array of encoded vectors
            image_paths: list of image paths
        """
        # Scan folder for images
        image_paths = []
        
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                    full_path = os.path.join(root, file)
                    image_paths.append(full_path)
        
        if len(image_paths) == 0:
            print(f"No images found in {folder_path}")
            return np.array([]), []
        
        print(f"Found {len(image_paths)} images in {folder_path}")
        
        # Encode based on model (now returns successfully encoded paths)
        if model_name == 'openclip':
            vectors, successfully_encoded_paths = self.encode_with_openclip(image_paths)
        elif model_name == 'beit3':
            vectors, successfully_encoded_paths = self.encode_with_beit3(image_paths)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        print(f"Successfully encoded {len(successfully_encoded_paths)} out of {len(image_paths)} images")
        return vectors, successfully_encoded_paths
    
    def encode_files(self, file_paths: List[str], model_name: str) -> Tuple[np.ndarray, List[str]]:
        """
        Encode list of image files
        
        Args:
            file_paths: list of image file paths
            model_name: 'openclip' or 'beit3'
        
        Returns:
            vectors: numpy array of encoded vectors
            image_paths: list of image paths
        """
        # Filter valid image files
        valid_paths = [
            path for path in file_paths
            if any(path.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)
        ]
        
        if len(valid_paths) == 0:
            print("No valid image files provided")
            return np.array([]), []
        
        print(f"Encoding {len(valid_paths)} images")
        
        # Encode based on model (now returns successfully encoded paths)
        if model_name == 'openclip':
            vectors, successfully_encoded_paths = self.encode_with_openclip(valid_paths)
        elif model_name == 'beit3':
            vectors, successfully_encoded_paths = self.encode_with_beit3(valid_paths)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        print(f"Successfully encoded {len(successfully_encoded_paths)} out of {len(valid_paths)} images")
        return vectors, successfully_encoded_paths
