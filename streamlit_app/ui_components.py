"""
Reusable UI components for Streamlit app
"""
import streamlit as st
from PIL import Image
import os
from typing import List, Dict


def render_image_grid(results: List[Dict], cols: int = 4):
    """
    Render search results in a grid layout
    
    Args:
        results: list of dicts with 'path' and 'score' keys
        cols: number of columns in grid
    """
    if len(results) == 0:
        st.info("No results found")
        return
    
    # Create grid
    num_rows = (len(results) + cols - 1) // cols
    
    for row in range(num_rows):
        columns = st.columns(cols)
        
        for col_idx in range(cols):
            result_idx = row * cols + col_idx
            
            if result_idx < len(results):
                result = results[result_idx]
                
                with columns[col_idx]:
                    try:
                        # Load and display image
                        if os.path.exists(result['path']):
                            image = Image.open(result['path'])
                            st.image(image, width='stretch')
                            
                            # Display metadata
                            st.caption(f"**Score:** {result['score']:.4f}")
                            st.caption(f"**File:** {os.path.basename(result['path'])}")
                        else:
                            st.error(f"Image not found: {result['path']}")
                    except Exception as e:
                        st.error(f"Error loading image: {e}")


def render_image_selector(images: List[Dict], model_name: str) -> List[str]:
    """
    Render images with checkboxes for selection
    
    Args:
        images: list of dicts with 'uuid', 'path', 'added_at'
        model_name: name of the model (for unique keys)
    
    Returns:
        list of selected UUIDs
    """
    if len(images) == 0:
        st.info("No images in index")
        return []
    
    st.write(f"**Total images:** {len(images)}")
    
    # Search/filter box
    filter_text = st.text_input("Filter by path:", key=f"filter_{model_name}")
    
    # Filter images if search text provided
    if filter_text:
        filtered_images = [
            img for img in images
            if filter_text.lower() in img['path'].lower()
        ]
    else:
        filtered_images = images
    
    if len(filtered_images) == 0:
        st.warning("No images match the filter")
        return []
    
    st.write(f"**Showing:** {len(filtered_images)} images")
    
    # Select all checkbox
    select_all = st.checkbox("Select All", key=f"select_all_{model_name}")
    
    selected_uuids = []
    
    # Display in grid with checkboxes
    cols = 4
    num_rows = (len(filtered_images) + cols - 1) // cols
    
    for row in range(num_rows):
        columns = st.columns(cols)
        
        for col_idx in range(cols):
            img_idx = row * cols + col_idx
            
            if img_idx < len(filtered_images):
                img = filtered_images[img_idx]
                
                with columns[col_idx]:
                    try:
                        if os.path.exists(img['path']):
                            image = Image.open(img['path'])
                            st.image(image, width='stretch')
                            
                            # Checkbox for selection
                            is_selected = st.checkbox(
                                f"Select",
                                value=select_all,
                                key=f"select_{model_name}_{img['uuid']}"
                            )
                            
                            if is_selected:
                                selected_uuids.append(img['uuid'])
                            
                            # Display metadata
                            st.caption(f"**Added:** {img['added_at'][:10]}")
                            st.caption(f"**Path:** {os.path.basename(img['path'])}")
                        else:
                            st.error(f"Not found: {img['path']}")
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    return selected_uuids


def render_model_selector():
    """
    Render model selection in sidebar
    
    Returns:
        tuple of (model_name, fusion_weight)
    """
    st.sidebar.header("Model Selection")
    
    model_name = st.sidebar.radio(
        "Choose Model:",
        options=["OpenCLIP", "BEiT3", "Fusion"],
        index=2,  # Default to Fusion
        key="model_selection"
    )
    
    fusion_weight = 0.5
    
    if model_name == "Fusion":
        fusion_weight = st.sidebar.slider(
            "Fusion Weight (α)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            help="0.0 = all BEiT3, 1.0 = all OpenCLIP",
            key="fusion_weight_slider"
        )
    
    return model_name.lower(), fusion_weight


def render_index_status(faiss_managers: Dict):
    """
    Render index status in sidebar
    
    Args:
        faiss_managers: dict with 'openclip' and 'beit3' FAISSManager instances
    """
    st.sidebar.header("Index Status")
    
    for model_name, manager in faiss_managers.items():
        total = manager.get_total_images()
        st.sidebar.metric(
            label=f"{model_name.upper()} Images",
            value=total
        )
    
    st.sidebar.markdown("---")
