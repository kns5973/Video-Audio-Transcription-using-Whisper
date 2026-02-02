def download_youtube_audio(url):
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(tempfile.gettempdir(), '%(id)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'm4a',
            }],
            # ADD THESE SETTINGS TO FIX 403 FORBIDDEN
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            },
            'nocheckcertificate': True,
            'ignoreerrors': False,
            'logtostderr': False,
            'quiet': True,
            'no_warnings': True,
            'source_address': '0.0.0.0', # Forces IPv4
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # This helps bypass some of the JS-runtime requirements
            info = ydl.extract_info(url, download=True)
            return True, ydl.prepare_filename(info)
    except Exception as e:
        return False, str(e)
