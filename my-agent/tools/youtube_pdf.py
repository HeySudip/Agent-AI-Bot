import os
import re
import urllib.parse
from langchain.tools import tool
from youtube_transcript_api import YouTubeTranscriptApi
from fpdf import FPDF

@tool
def search_and_extract_youtube_to_pdf(query: str) -> str:
    """
    Searches YouTube for a video based on the query, extracts its transcript, 
    and saves the summarized transcript to a PDF file.
    It returns the file path using the format __FILE_PATH__=/path/to/file.pdf.
    Use this when the user asks to summarize a YouTube video but does NOT provide a URL.
    """
    try:
        from duckduckgo_search import DDGS
        
        # Search DuckDuckGo for a YouTube video matching the query
        with DDGS() as ddgs:
            results = list(ddgs.text(f"site:youtube.com {query}", max_results=3))
            
        if not results:
            return "Could not find any YouTube videos for that topic."
            
        url = results[0].get('href', '')
        title = results[0].get('title', 'video')
        
        # Extract video ID
        video_id = None
        parsed = urllib.parse.urlparse(url)
        if parsed.hostname in ('youtu.be', 'www.youtu.be'):
            video_id = parsed.path[1:]
        elif parsed.hostname in ('youtube.com', 'www.youtube.com'):
            if parsed.path == '/watch':
                query_params = urllib.parse.parse_qs(parsed.query)
                video_id = query_params.get('v', [None])[0]
        
        if not video_id:
            match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
            if match:
                video_id = match.group(1)
                
        if not video_id:
            return f"Found a video but couldn't extract the ID. URL: {url}"

        # Fetch transcript
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        except Exception as e:
            return f"Found video '{title}' ({url}), but it does not have a transcript available. Error: {e}"

        text = " ".join([t['text'] for t in transcript_list])
        
        # Break text into chunks for the PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.set_font("Arial", style='B', size=14)
        pdf.multi_cell(0, 10, title.encode('latin-1', 'replace').decode('latin-1'))
        pdf.ln(5)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, f"URL: {url}")
        pdf.ln(10)
        
        clean_text = text.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 10, clean_text)
        
        # Save to file
        pdf_path = f"/tmp/youtube_transcript_{video_id}.pdf"
        pdf.output(pdf_path)
        
        return f"Found video '{title}'. Generated the transcript PDF. __FILE_PATH__={pdf_path}"
    except Exception as e:
        return f"Error searching and creating YouTube PDF: {e}"

@tool
def extract_youtube_to_pdf(url: str) -> str:
    """
    Extracts the transcript of a YouTube video and saves it to a PDF file.
    It returns the file path of the generated PDF using the format __FILE_PATH__=/path/to/file.pdf.
    Use this when the user asks to summarize a YouTube video AND provides the specific URL.
    """
    try:
        # Extract video ID
        video_id = None
        parsed = urllib.parse.urlparse(url)
        if parsed.hostname in ('youtu.be', 'www.youtu.be'):
            video_id = parsed.path[1:]
        elif parsed.hostname in ('youtube.com', 'www.youtube.com'):
            if parsed.path == '/watch':
                query = urllib.parse.parse_qs(parsed.query)
                video_id = query.get('v', [None])[0]
        
        if not video_id:
            match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
            if match:
                video_id = match.group(1)
                
        if not video_id:
            return "Could not extract a valid YouTube video ID from the URL."

        # Fetch transcript
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        except Exception as e:
            return f"Failed to get transcript for this video: {e}"

        text = " ".join([t['text'] for t in transcript_list])
        
        # We should break text into lines for PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        clean_text = text.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 10, clean_text)
        
        # Save to file
        pdf_path = f"/tmp/youtube_transcript_{video_id}.pdf"
        pdf.output(pdf_path)
        
        return f"Successfully generated the transcript PDF. __FILE_PATH__={pdf_path}"
    except Exception as e:
        return f"Error creating YouTube PDF: {e}"

@tool
def generate_text_to_pdf(text: str, filename: str = "document.pdf") -> str:
    """
    Generates a PDF file containing the provided text and returns its file path.
    It returns the file path using the format __FILE_PATH__=/path/to/file.pdf.
    Use this when the user asks for information in a PDF format (e.g. answer key, notes).
    """
    try:
        pdf = FPDF()
        pdf.add_page()
        
        # Add a Unicode font to support special characters
        # Using a standard font since FPDF doesn't embed true type fonts by default
        pdf.set_font("Helvetica", size=12)
        
        # Ensure we only use latin-1 encodable characters by replacing others
        clean_text = text.encode('latin-1', 'replace').decode('latin-1')
        
        pdf.multi_cell(0, 10, clean_text)
        
        safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', filename)
        if not safe_name.endswith('.pdf'):
            safe_name += '.pdf'
            
        pdf_path = f"/tmp/{safe_name}"
        pdf.output(pdf_path)
        
        return f"Successfully generated the PDF '{filename}'. __FILE_PATH__={pdf_path}"
    except Exception as e:
        return f"Error creating PDF: {e}"
