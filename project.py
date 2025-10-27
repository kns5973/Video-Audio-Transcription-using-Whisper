import streamlit as st
from faster_whisper import WhisperModel
import traceback
import time
import os
import tempfile

# Load Whisper model (cached by Streamlit for performance)
@st.cache_resource
def load_whisper_model():
    # 1. Using "small" model
    # 2. Optimized for CPU with "int8" compute type
    return WhisperModel("small", device="cpu", compute_type="int8")

def transcribe_video(model, video_path):
    """Transcribes the video and returns the text."""
    try:
        # The transcribe method returns a generator
        # 3. Added vad_filter=True to skip silence
        segments, info = model.transcribe(video_path,
                                          beam_size=5,
                                          vad_filter=True,
                                          vad_parameters=dict(min_silence_duration_ms=500))
        
        # We must iterate over the segments to get the full text
        transcript_text = ""
        for segment in segments:
            transcript_text += segment.text + " "
            
        return True, transcript_text.strip()
        
    except Exception as e:
        tb = traceback.format_exc()
        return False, f"Transcription failed: {e}\n{tb}"

# --- Streamlit UI (No changes needed here) ---

def main():
    st.set_page_config(page_title="Video Transcriber", layout="wide")
    st.title("Fast Video Transcriber ⚡️🎧📝")

    # Instructions
    with st.expander("How it works", expanded=False):
        st.markdown("""
        This tool will:
        1. Transcribe your video using the optimized **faster-whisper** AI.
        2. Use Voice Activity Detection (VAD) to skip silence and speed up processing.
        3. Display the transcript for you to copy or download.
        
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
                with st.spinner("Transcribing video... (skipping silence) ⏳"):
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
