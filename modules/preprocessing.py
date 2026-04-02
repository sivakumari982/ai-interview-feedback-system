import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import string

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class TextPreprocessor:
    """Preprocess and analyze text responses"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Technical terms to preserve
        self.tech_terms = {
            'ml', 'ai', 'api', 'sql', 'nosql', 'json', 'xml', 'html', 'css',
            'javascript', 'python', 'java', 'c++', 'algorithm', 'data structure',
            'oop', 'polymorphism', 'inheritance', 'encapsulation', 'abstraction',
            'regression', 'classification', 'clustering', 'neural network',
            'deep learning', 'machine learning', 'nlp', 'computer vision'
        }
    
    def preprocess(self, text: str) -> str:
        """Complete preprocessing pipeline"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep technical terms
        text = self._preserve_tech_terms(text)
        text = re.sub(r'[^\w\s\.\,\-]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and punctuation
        tokens = [t for t in tokens if t not in self.stop_words 
                 and t not in string.punctuation]
        
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        return ' '.join(tokens)
    
    def _preserve_tech_terms(self, text: str) -> str:
        """Preserve technical terms during preprocessing"""
        for term in self.tech_terms:
            # Replace spaces with placeholder for multi-word terms
            if ' ' in term:
                placeholder = term.replace(' ', '_')
                text = text.replace(term, placeholder)
        return text
    
    def simple_preprocess(self, text: str) -> str:
        """Simple preprocessing for quick analysis"""
        if not text:
            return ""
        
        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_keywords(self, text: str, top_n: int = 10) -> list:
        """Extract important keywords from text"""
        if not text:
            return []
        
        # Preprocess
        processed = self.preprocess(text)
        words = processed.split()
        
        # Count frequency
        word_freq = {}
        for word in words:
            if len(word) > 2:  # Ignore very short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        return [word for word, _ in sorted_words[:top_n]]
    
    def get_sentences(self, text: str) -> list:
        """Split text into sentences"""
        if not text:
            return []
        return sent_tokenize(text)
    
    def get_word_count(self, text: str) -> int:
        """Count words in text"""
        if not text:
            return 0
        return len(text.split())
    
    def check_examples(self, text: str) -> bool:
        """Check if text contains examples"""
        if not text:
            return False
        
        text = text.lower()
        example_indicators = [
            'for example', 'e.g.', 'such as', 'like', 'instance',
            'illustrate', 'demonstrate', 'case', 'sample'
        ]
        
        return any(indicator in text for indicator in example_indicators)