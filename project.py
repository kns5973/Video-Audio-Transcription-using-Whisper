import streamlit as st
from faster_whisper import WhisperModel
import traceback
import time
import os
import tempfile
import yt_dlp

@st.cache_resource
def load_whisper_model():
    return WhisperModel("base", device="cpu", compute_type="int8")

def download_youtube_audio(url):
    try:
        ydl_opts = {
            'format': 'm4a/bestaudio/best',
            'outtmpl': os.path.join(tempfile.gettempdir(), '%(id)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'm4a',
            }],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return True, ydl.prepare_filename(info)
    except Exception as e:
        return False, str(e)

def transcribe_video(model, video_path):
    try:
        segments, info = model.transcribe(video_path,
                                         beam_size=5,
                                         vad_filter=True,
                                         vad_parameters=dict(min_silence_duration_ms=500))
        
        transcript_text = ""
        for segment in segments:
            transcript_text += segment.text + " "
            
        return True, transcript_text.strip()
    except Exception as e:
        tb = traceback.format_exc()
        return False, f"Transcription failed: {e}\n{tb}"

def main():
    st.set_page_config(page_title="Video Transcriber", layout="wide")
    st.title("Fast Video Transcriber ⚡️🎧📝")

    tab1, tab2 = st.tabs(["Upload File", "YouTube Link"])

    with tab1:
        uploaded_file = st.file_uploader(
            "Upload your video file",
            type=["mp4", "mkv", "avi", "mov"]
        )
    
    with tab2:
        yt_url = st.text_input("Paste YouTube URL here")

    if uploaded_file or yt_url:
        if st.button("Transcribe", type="primary"):
            video_path = None
            try:
                model = load_whisper_model()
                
                with st.spinner("Preparing audio... ⏳"):
                    if yt_url:
                        success, result = download_youtube_audio(yt_url)
                        if not success:
                            st.error(f"YouTube Download Error: {result}")
                            return
                        video_path = result
                    else:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            video_path = tmp_file.name

                start_transcribe = time.time()
                with st.spinner("Transcribing... (skipping silence) ⏳"):
                    transcribe_ok, transcript_text = transcribe_video(model, video_path)
                
                if not transcribe_ok:
                    st.error(f"Transcription Failed:\n{transcript_text}")
                    return
                
                elapsed_transcribe = time.time() - start_transcribe
                st.success(f"Transcription complete in {elapsed_transcribe:.1f}s! 🎉")

                st.subheader("Raw Transcript")
                st.text_area("Transcript", transcript_text, height=400)
                st.download_button("Download Transcript", transcript_text, file_name="transcript.txt")

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
            finally:
                if video_path and os.path.exists(video_path):
                    os.unlink(video_path)

if __name__ == "__main__":
    main()
