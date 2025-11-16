"""
Streamlit Image Retrieval Application
Main app with search, encoding, and management features
"""
import streamlit as st
import os

# Import modules
from config import (
    OPENCLIP_DIM,
    BEIT3_DIM,
    DEFAULT_TOP_K,
    GRID_COLUMNS,
    UPLOADS_DIR
)
from models import ModelManager
from faiss_manager import FAISSManager
from image_encoder import ImageEncoder
from search_engine import SearchEngine
from ui_components import (
    render_image_grid,
    render_image_selector,
    render_model_selector,
    render_index_status
)


# Page configuration
st.set_page_config(
    layout="wide",
    page_title="Image Retrieval System",
    page_icon="🔍"
)


def initialize_app():
    """Initialize app components and session state"""
    
    # Initialize model manager
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = ModelManager()
    
    # Initialize FAISS managers
    if 'faiss_managers' not in st.session_state:
        st.session_state.faiss_managers = {
            'openclip': FAISSManager('openclip', OPENCLIP_DIM),
            'beit3': FAISSManager('beit3', BEIT3_DIM)
        }
    
    # Initialize image encoder
    if 'image_encoder' not in st.session_state:
        st.session_state.image_encoder = ImageEncoder(st.session_state.model_manager)
    
    # Initialize search engine
    if 'search_engine' not in st.session_state:
        st.session_state.search_engine = SearchEngine(
            st.session_state.model_manager,
            st.session_state.faiss_managers
        )


def tab_search():
    """Tab 1: Search Interface"""
    st.header("🔍 Image Search")
    
    # Get model selection from session state
    model_name = st.session_state.get('selected_model', 'fusion')
    fusion_weight = st.session_state.get('selected_fusion_weight', 0.5)
    
    # Search input
    col1, col2 = st.columns([4, 1])


    with col1:
        query = st.text_input(
            "Enter search query:",
            placeholder="Ví dụ: Một con chó đang ngậm quả bóng nằm trên cỏ",
            key="search_query"
        )
        query = query.lower().strip()

        #kiểm tra xem câu query có chữ số không (ví dụ: "1", "2", "3") nếu có thì chuyển các chữ số về chữ viết tay ("một", "hai", "ba")
        digit_to_word = {"0": "không", "1": "một", "2": "hai", "3": "ba", "4": "bốn", "5": "năm", "6": "sáu", "7": "bảy", "8": "tám", "9": "chín"}
        for digit, word in digit_to_word.items(): query = query.replace(digit, word)

    with col2:
        top_k = st.number_input(
            "Top K:",
            min_value=1,
            max_value=100,
            value=DEFAULT_TOP_K,
            key="search_top_k"
        )
    
    # Search button
    if st.button("🔍 Search", type="primary"):
        if not query.strip():
            st.warning("Please enter a search query")
            return
        
        with st.spinner("Searching..."):
            try:
                search_engine = st.session_state.search_engine
                
                if model_name == "openclip":
                    scores, uuids = search_engine.search_openclip(query, top_k)
                    results = search_engine.format_results(uuids, scores, 'openclip')
                
                elif model_name == "beit3":
                    scores, uuids = search_engine.search_beit3(query, top_k)
                    results = search_engine.format_results(uuids, scores, 'beit3')
                
                elif model_name == "fusion":
                    fusion_results = search_engine.search_fusion(query, top_k, fusion_weight)
                    results = search_engine.format_fusion_results(fusion_results)
                
                # Display results
                st.success(f"Found {len(results)} results")
                
                if len(results) > 0:
                    st.markdown("---")
                    render_image_grid(results, cols=GRID_COLUMNS)
                else:
                    st.info("No results found. Try encoding some images first!")
            
            except Exception as e:
                st.error(f"Search error: {str(e)}")


def tab_encode():
    """Tab 2: Encode Images"""
    st.header("📂 Encode Images Place")
    
    # Section 1: Encode Folder
    st.subheader("1. Encode Folder")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        folder_path = st.text_input(
            "Folder path:",
            placeholder="D:/images/my_dataset",
            key="encode_folder_path"
        )
    
    with col2:
        folder_model = st.selectbox(
            "Model:",
            options=["Both", "OpenCLIP", "BEiT3"],
            key="encode_folder_model"
        )
    
    if st.button("📂 Encode Folder", type="primary"):
        if not folder_path.strip():
            st.warning("Please enter a folder path")
            return
        
        if not os.path.exists(folder_path):
            st.error(f"Folder not found: {folder_path}")
            return
        
        if not os.path.isdir(folder_path):
            st.error(f"Path is not a folder: {folder_path}")
            return
        
        encoder = st.session_state.image_encoder
        faiss_managers = st.session_state.faiss_managers
        
        models_to_encode = []
        if folder_model == "Both":
            models_to_encode = ["openclip", "beit3"]
        else:
            models_to_encode = [folder_model.lower()]
        
        total_added = 0
        
        for model in models_to_encode:
            with st.spinner(f"Encoding with {model.upper()}..."):
                try:
                    # Encode images
                    vectors, image_paths = encoder.encode_folder(folder_path, model)
                    
                    if len(vectors) == 0:
                        st.warning(f"No images found in folder for {model}")
                        continue
                    
                    # Add to index
                    num_added = faiss_managers[model].add_vectors(vectors, image_paths)
                    total_added += num_added
                    
                    st.success(f"✅ {model.upper()}: Added {num_added} images")
                
                except Exception as e:
                    st.error(f"Error encoding with {model}: {str(e)}")
        
        if total_added > 0:
            st.balloons()
            st.success(f"🎉 Total images added: {total_added}")
            st.rerun()  # Refresh UI to update sidebar
    
    st.markdown("---")
    
    # Section 2: Encode Files
    st.subheader("2. Encode Files")
    
    uploaded_files = st.file_uploader(
        "Upload images:",
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
        accept_multiple_files=True,
        key="encode_files_uploader"
    )
    
    file_model = st.selectbox(
        "Model:",
        options=["Both", "OpenCLIP", "BEiT3"],
        key="encode_files_model"
    )
    
    if st.button("📤 Encode Uploaded Files", type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one image")
            return
        
        # Ensure uploads directory exists
        os.makedirs(UPLOADS_DIR, exist_ok=True)
        
        # Save uploaded files
        saved_paths = []
        
        for uploaded_file in uploaded_files:
            save_path = os.path.join(UPLOADS_DIR, uploaded_file.name)
            
            with open(save_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            saved_paths.append(save_path)
        
        st.info(f"Saved {len(saved_paths)} files")
        
        encoder = st.session_state.image_encoder
        faiss_managers = st.session_state.faiss_managers
        
        models_to_encode = []
        if file_model == "Both":
            models_to_encode = ["openclip", "beit3"]
        else:
            models_to_encode = [file_model.lower()]
        
        total_added = 0
        
        for model in models_to_encode:
            with st.spinner(f"Encoding with {model.upper()}..."):
                try:
                    # Encode files
                    vectors, image_paths = encoder.encode_files(saved_paths, model)
                    
                    if len(vectors) == 0:
                        st.warning(f"No valid images for {model}")
                        continue
                    
                    # Add to index
                    num_added = faiss_managers[model].add_vectors(vectors, image_paths)
                    total_added += num_added
                    
                    st.success(f"✅ {model.upper()}: Added {num_added} images")
                
                except Exception as e:
                    st.error(f"Error encoding with {model}: {str(e)}")
        
        if total_added > 0:
            st.balloons()
            st.success(f"🎉 Total images added: {total_added}")
            st.rerun()  # Refresh UI to update sidebar


def tab_manage():
    """Tab 3: Manage Images"""
    st.header("🗑️ Manage Images")
    
    # Select index to manage
    index_to_manage = st.selectbox(
        "Select Index:",
        options=["OpenCLIP", "BEiT3", "Both"],
        key="manage_index_select"
    )
    
    st.markdown("---")
    
    if index_to_manage == "Both":
        # Handle both indices
        openclip_manager = st.session_state.faiss_managers['openclip']
        beit3_manager = st.session_state.faiss_managers['beit3']
        
        # Get images from both
        openclip_images = openclip_manager.get_all_images()
        beit3_images = beit3_manager.get_all_images()
        
        # Render OpenCLIP section
        st.subheader("📊 OpenCLIP Images")
        selected_openclip = render_image_selector(openclip_images, 'openclip')
        
        st.markdown("---")
        
        # Render BEiT3 section
        st.subheader("📊 BEiT3 Images")
        selected_beit3 = render_image_selector(beit3_images, 'beit3')
        
        st.markdown("---")
        
        # Action Bar for both
        total_selected = len(selected_openclip) + len(selected_beit3)
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            delete_button = st.button(
                "🗑️ Delete Selected",
                type="primary",
                disabled=(total_selected == 0)
            )
        
        with col2:
            if total_selected > 0:
                st.metric("Selected", total_selected)
                st.caption(f"CLIP: {len(selected_openclip)} | BEiT3: {len(selected_beit3)}")
        
        # Delete logic for both
        if delete_button and total_selected > 0:
            with st.spinner("Deleting..."):
                try:
                    deleted_count = 0
                    
                    if len(selected_openclip) > 0:
                        openclip_manager.remove_vectors(selected_openclip)
                        deleted_count += len(selected_openclip)
                        st.session_state['manage_selected_set_openclip'] = set()
                    
                    if len(selected_beit3) > 0:
                        beit3_manager.remove_vectors(selected_beit3)
                        deleted_count += len(selected_beit3)
                        st.session_state['manage_selected_set_beit3'] = set()
                    
                    st.success(f"✅ Deleted {deleted_count} images")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error deleting images: {str(e)}")
    
    else:
        # Handle single index (OpenCLIP or BEiT3)
        model_name = index_to_manage.lower()
        faiss_manager = st.session_state.faiss_managers[model_name]
        
        # Get all images
        images = faiss_manager.get_all_images()
        
        # Render image selector (with pagination)
        selected_uuids = render_image_selector(images, model_name)
        
        st.markdown("---")
        
        # Action Bar with delete button
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            delete_button = st.button(
                "🗑️ Delete Selected",
                type="primary",
                disabled=(len(selected_uuids) == 0)
            )
        
        with col2:
            if len(selected_uuids) > 0:
                st.metric("Selected", len(selected_uuids))
        
        # Delete logic - single button with confirmation
        if delete_button and len(selected_uuids) > 0:
            with st.spinner("Deleting..."):
                try:
                    faiss_manager.remove_vectors(selected_uuids)
                    st.success(f"✅ Deleted {len(selected_uuids)} images")
                    # Clear selection set after successful delete
                    st.session_state[f"manage_selected_set_{model_name}"] = set()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error deleting images: {str(e)}")


def tab_settings():
    """Tab 4: Settings"""
    st.header("⚙️ Settings")
    
    # Model Paths
    st.subheader("Model Paths")
    
    from config import (
        OPENCLIP_MODEL_PATH,
        BEIT3_MODEL_PATH,
        BEIT3_TOKENIZER_PATH,
        BEIT3_CHECKPOINT_PATH
    )
    
    st.code(f"OpenCLIP: {OPENCLIP_MODEL_PATH}")
    st.code(f"BEiT3 Base: {BEIT3_MODEL_PATH}")
    st.code(f"BEiT3 Tokenizer: {BEIT3_TOKENIZER_PATH}")
    st.code(f"BEiT3 Checkpoint: {BEIT3_CHECKPOINT_PATH}")
    
    st.markdown("---")
    
    # Index Status
    st.subheader("Index Status")
    
    faiss_managers = st.session_state.faiss_managers
    
    for model_name in ['openclip', 'beit3']:
        manager = faiss_managers[model_name]
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(f"{model_name.upper()}", "Index Info")
        with col2:
            faiss_count = manager.index.ntotal
            st.metric("FAISS Vectors", faiss_count)
        with col3:
            metadata_count = len(manager.metadata[model_name]['images'])
            st.metric("Metadata Images", metadata_count)
        
        if faiss_count != metadata_count:
            st.error(f"⚠️ Inconsistency detected in {model_name.upper()}!")
    
    st.markdown("---")
    
    # Danger Zone
    st.subheader("⚠️ Danger Zone")
    
    st.warning("Clear indices will delete all encoded images. You will need to re-encode.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear OpenCLIP Index", type="secondary"):
            if st.checkbox("Confirm clear OpenCLIP?", key="confirm_clear_openclip"):
                try:
                    manager = st.session_state.faiss_managers['openclip']
                    manager.index = manager.create_index()
                    manager.metadata['openclip'] = {
                        'images': [],
                        'uuid_to_index': {},
                        'path_to_uuid': {},
                        'total_images': 0
                    }
                    manager.save_index()
                    
                    # Reload metadata to verify
                    manager.metadata = manager.load_metadata()
                    
                    st.success("✅ OpenCLIP index cleared")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with col2:
        if st.button("🗑️ Clear BEiT3 Index", type="secondary"):
            if st.checkbox("Confirm clear BEiT3?", key="confirm_clear_beit3"):
                try:
                    manager = st.session_state.faiss_managers['beit3']
                    manager.index = manager.create_index()
                    manager.metadata['beit3'] = {
                        'images': [],
                        'uuid_to_index': {},
                        'path_to_uuid': {},
                        'total_images': 0
                    }
                    manager.save_index()
                    
                    # Reload metadata to verify
                    manager.metadata = manager.load_metadata()
                    
                    st.success("✅ BEiT3 index cleared")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
    
    if st.button("🗑️ Clear ALL Indices", type="secondary"):
        if st.checkbox("⚠️ Confirm clear EVERYTHING?", key="confirm_clear_all"):
            try:
                for model in ['openclip', 'beit3']:
                    manager = st.session_state.faiss_managers[model]
                    manager.index = manager.create_index()
                    manager.metadata[model] = {
                        'images': [],
                        'uuid_to_index': {},
                        'path_to_uuid': {},
                        'total_images': 0
                    }
                    manager.save_index()
                    
                    # Reload metadata to verify
                    manager.metadata = manager.load_metadata()
                
                st.success("✅ All indices cleared")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")


def main():
    """Main application"""
    
    # Initialize
    initialize_app()
    
    # Title
    st.title("🔍 Image Retrieval System")
    st.markdown("*Search images using OpenCLIP, BEiT3, or Fusion of both models*")
    
    # Sidebar - Render once before tabs
    model_name, fusion_weight = render_model_selector()
    st.session_state.selected_model = model_name
    st.session_state.selected_fusion_weight = fusion_weight
    
    render_index_status(st.session_state.faiss_managers)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Search",
        "📂 Encode Images",
        "🗑️ Manage Images",
        "⚙️ Settings"
    ])
    
    with tab1:
        tab_search()
    
    with tab2:
        tab_encode()
    
    with tab3:
        tab_manage()
    
    with tab4:
        tab_settings()


if __name__ == "__main__":
    main()
