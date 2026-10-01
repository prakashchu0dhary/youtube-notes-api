import os
import re
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
from google import genai

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
ai_client = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None

class NoteRequest(BaseModel):
    url: str
    exam_type: str = "All Competitive Exams"

def extract_youtube_id(url: str):
    pattern = r"(?:v=|\/|youtu\.be\/)([0-9A-Za-z_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else None

def get_youtube_content(video_id: str) -> str:
    try:
        ytt = YouTubeTranscriptApi()
        transcript = ytt.fetch(video_id, languages=['hi', 'en', 'hi-Latn'])
        return " ".join([item['text'] for item in transcript])
    except Exception:
        raise ValueError("इस वीडियो में सबटाइटल/ट्रांसक्रिप्ट उपलब्ध नहीं है।")

def get_website_content(url: str) -> str:
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        res = requests.get(url, headers=headers, timeout=12)
        soup = BeautifulSoup(res.text, 'html.parser')
        for el in soup(["script", "style", "nav", "footer", "header", "aside"]):
            el.decompose()
        paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'li'])
        text = " ".join([p.get_text(strip=True) for p in paragraphs])
        if len(text) < 100:
            raise ValueError("वेबसाइट से पर्याप्त कंटेंट नहीं मिला।")
        return text
    except Exception as e:
        raise ValueError(f"वेबसाइट लोड करने में त्रुटि: {str(e)}")

@app.get("/")
def home():
    return {"status": "YouTubeToNotes API is running successfully!"}

@app.post("/api/generate-notes")
async def generate_notes(req: NoteRequest):
    if not ai_client:
        raise HTTPException(status_code=500, detail="Gemini API Key सर्वर पर सेट नहीं है।")

    url = req.url.strip()
    yt_id = extract_youtube_id(url)

    try:
        if yt_id:
            raw_text = get_youtube_content(yt_id)
            source_type = "यूट्यूब क्लास वीडियो"
        else:
            raw_text = get_website_content(url)
            source_type = "वेबसाइट स्टडी पोस्ट"
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    clean_text = raw_text[:22000]

    prompt = f"""
    तुम प्रतियोगी परीक्षाओं ({req.exam_type}) के विशेषज्ञ शिक्षक हो।
    नीचे दिए गए स्रोत ({source_type}) के मूल कंटेंट को पढ़कर छात्रों के लिए अत्यंत सुंदर, हाइलाइटेड और परीक्षा-उपयोगी रिवीज़न नोट्स तैयार करो।

    प्रारूप निर्देश:
    1. 📌 **अध्याय / टॉपिक का नाम** और 2 पंक्तियों का क्विक समरी बॉक्स।
    2. 📖 **मुख्य सिद्धांत व परिभाषाएँ:** मुख्य शब्दों और तिथियों को **बोल्ड** करें।
    3. 📊 **परीक्षा-उपयोगी फैक्ट्स टेबल:** मुख्य आँकड़ों और अंतरों के लिए Markdown Table बनाएं।
    4. 💡 **विगत परीक्षा उपयोगी बिंदु (High-Yield Points):** 3-5 सीधे परीक्षा उपयोगी फैक्ट्स।
    5. ⚠️ **Doubt Buster:** जहाँ छात्र अक्सर गलती करते हैं।

    मूल सामग्री:
    {clean_text}
    """

    try:
        response = ai_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return {
            "success": True,
            "source": source_type,
            "notes": response.text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI एरर: {str(e)}")
