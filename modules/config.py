import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration settings for the application"""
    
    # Gemini API Settings
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
    USE_GEMINI = os.getenv('USE_GEMINI', 'true').lower() == 'true'
    
    # Application Settings
    MAX_QUESTIONS_PER_SESSION = 50
    ENABLE_CACHE = True
    DEBUG_MODE = os.getenv('DEBUG', 'false').lower() == 'true'
    
    @classmethod
    def is_gemini_available(cls):
        """Check if Gemini API is configured"""
        return cls.GEMINI_API_KEY != '' and cls.USE_GEMINI