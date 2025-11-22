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
                            st.caption(f"**Score:** {float(result['score'])*100:.2f} %")
                            st.caption(f"**File:** {os.path.basename(result['path'])}")
                        else:
                            st.error(f"Image not found: {result['path']}")
                    except Exception as e:
                        st.error(f"Error loading image: {e}")


def render_image_selector(images: List[Dict], model_name: str, items_per_page: int = 30) -> List[str]:
    """
    Render images with checkboxes for selection (with pagination)
    
    Args:
        images: list of dicts with 'uuid', 'path', 'added_at'
        model_name: name of the model (for unique keys)
        items_per_page: number of images per page
    
    Returns:
        list of selected UUIDs
    """
    if len(images) == 0:
        st.info("No images in index")
        return []
    
    # Initialize session state for pagination
    page_key = f"manage_current_page_{model_name}"
    selected_key = f"manage_selected_set_{model_name}"
    filter_key = f"last_filter_{model_name}"
    
    if page_key not in st.session_state:
        st.session_state[page_key] = 1
    if selected_key not in st.session_state:
        st.session_state[selected_key] = set()
    if filter_key not in st.session_state:
        st.session_state[filter_key] = ""
    
    st.write(f"**Total images:** {len(images)}")
    
    # Search/filter box
    filter_text = st.text_input("Filter by path:", key=f"filter_{model_name}")
    
    # Reset page if filter changed
    if filter_text != st.session_state[filter_key]:
        st.session_state[page_key] = 1
        st.session_state[filter_key] = filter_text
    
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
    
    # Calculate pagination
    import math
    total_pages = math.ceil(len(filtered_images) / items_per_page)
    current_page = st.session_state[page_key]
    
    # Adjust page if out of bounds
    if current_page > total_pages:
        current_page = total_pages
        st.session_state[page_key] = current_page
    
    start_idx = (current_page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(filtered_images))
    page_images = filtered_images[start_idx:end_idx]
    
    st.write(f"**Showing:** {start_idx + 1}-{end_idx} of {len(filtered_images)} images")
    
    # Pagination controls
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("⬅️ Previous", disabled=(current_page == 1), key=f"prev_{model_name}"):
            st.session_state[page_key] -= 1
            st.rerun()
    
    with col2:
        st.write(f"**Page {current_page} / {total_pages}**")
    
    with col3:
        if st.button("Next ➡️", disabled=(current_page == total_pages), key=f"next_{model_name}"):
            st.session_state[page_key] += 1
            st.rerun()
    
    st.markdown("---")
    
    # Callback function for Select All checkbox
    def handle_select_all_change():
        """Callback when Select All checkbox changes"""
        checkbox_state = st.session_state.get(f"select_all_checkbox_{model_name}", False)
        if checkbox_state:
            # Add all page UUIDs
            for img in page_images:
                st.session_state[selected_key].add(img['uuid'])
        else:
            # Remove all page UUIDs
            for img in page_images:
                st.session_state[selected_key].discard(img['uuid'])
    
    # Calculate if all images on current page are selected
    all_selected = all(img['uuid'] in st.session_state[selected_key] for img in page_images)
    
    # Select All checkbox with on_change callback
    st.checkbox(
        "✅ Select All (on this page)",
        value=all_selected,
        key=f"select_all_checkbox_{model_name}",
        on_change=handle_select_all_change
    )
    
    # Display in grid with checkboxes
    cols = 4
    num_rows = (len(page_images) + cols - 1) // cols
    
    for row in range(num_rows):
        columns = st.columns(cols)
        
        for col_idx in range(cols):
            img_idx = row * cols + col_idx
            
            if img_idx < len(page_images):
                img = page_images[img_idx]
                
                with columns[col_idx]:
                    try:
                        if os.path.exists(img['path']):
                            st.image(img['path'],
                                    width='stretch',
                                    #use_container_width=True
)
                            
                            # Callback for individual checkbox
                            def make_checkbox_callback(uuid):
                                def callback():
                                    checkbox_key = f"select_{model_name}_{uuid}"
                                    if st.session_state.get(checkbox_key, False):
                                        st.session_state[selected_key].add(uuid)
                                    else:
                                        st.session_state[selected_key].discard(uuid)
                                return callback
                            
                            # Checkbox for selection (read-only value, update via callback)
                            st.checkbox(
                                f"Select",
                                value=(img['uuid'] in st.session_state[selected_key]),
                                key=f"select_{model_name}_{img['uuid']}",
                                on_change=make_checkbox_callback(img['uuid'])
                            )
                            
                            # Display metadata
                            st.caption(f"**Added:** {img['added_at'][:10]}")
                            st.caption(f"**Path:** {os.path.basename(img['path'])}")
                        else:
                            st.error(f"Not found: {img['path']}")
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    return list(st.session_state[selected_key])


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
            value=0.42,
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
