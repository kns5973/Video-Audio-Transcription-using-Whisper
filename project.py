import streamlit as st
import whisper
import traceback
import time
import os
import tempfile

# Load Whisper model (cached by Streamlit for performance)
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("small")

def transcribe_video(model, video_path):
    """Transcribes the video and returns the text."""
    try:
        result = model.transcribe(video_path)
        return True, result.get("text", "")
    except Exception as e:
        tb = traceback.format_exc()
        return False, f"Transcription failed: {e}\n{tb}"

# --- Streamlit UI ---

def main():
    st.set_page_config(page_title="Video Transcriber", layout="wide")
    st.title("Video Transcriber 🎧📝")

    # Instructions
    with st.expander("How it works", expanded=False):
        st.markdown("""
        This tool will:
        1. Transcribe your video using **Whisper AI**.
        2. Display the transcript for you to copy or download.
        
        **Requirement:**
        * A video file (MP4, MKV, AVI, MOV).
        """)

    # File Uploader
    uploaded_file = st.file_uploader(
        "Upload your video file",
        type=["mp4", "mkv", "avi", "mov", "MP4", "MKV", "AVI", "MOV"]
    )

    if uploaded_file is not None:
        st.video(uploaded_file)
        
        if st.button("Transcribe Video", type="primary"):
            # Save uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                video_path = tmp_file.name
            
            try:
                model = load_whisper_model()
                
                # --- Transcription ---
                start_transcribe = time.time()
                with st.spinner("Transcribing video... This may take a while. ⏳"):
                    transcribe_ok, transcript_text = transcribe_video(model, video_path)
                
                if not transcribe_ok:
                    st.error(f"Transcription Failed:\n{transcript_text}")
                    return
                
                elapsed_transcribe = time.time() - start_transcribe
                st.success(f"Transcription complete in {elapsed_transcribe:.1f}s! 🎉")

                # --- Display Results ---
                st.subheader("Raw Transcript")
                st.text_area("Transcript", transcript_text, height=400)
                st.download_button(
                    "Download Transcript",
                    transcript_text,
                    file_name="transcript.txt"
                )

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                st.error(traceback.format_exc())
            finally:
                # Clean up the temporary file
                if os.path.exists(video_path):
                    os.unlink(video_path)

if __name__ == "__main__":
    main()
