from google import genai
from google.genai import types
from .config import Config

class GeminiFeedbackGenerator:
    """Generate personalized interview feedback using Gemini API (new package)"""
    
    def __init__(self):
        self.available = False
        self.client = None
        self.model_name = None
        
        # Configure Gemini with new package
        if Config.is_gemini_available():
            try:
                # Initialize client with new package
                self.client = genai.Client(api_key=Config.GEMINI_API_KEY)
                
                # Test the connection
                models = self.client.models.list()
                
                # Find available models
                available_models = []
                for model in models:
                    if 'generateContent' in model.supported_actions:
                        available_models.append(model.name.replace('models/', ''))
                
                if available_models:
                    # Prefer newer models
                    preferred_models = ['gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-1.5-pro']
                    
                    for preferred in preferred_models:
                        if any(preferred in model for model in available_models):
                            self.model_name = preferred
                            break
                    
                    if not self.model_name and available_models:
                        self.model_name = available_models[0]
                    
                    self.available = True
                    print(f"✅ Using Gemini model for feedback: {self.model_name}")
                    
            except Exception as e:
                print(f"❌ Gemini feedback initialization failed: {e}")
                self.available = False
    
    def generate_feedback(self, question: str, ideal_answer: str, 
                         user_answer: str, score: int, topic: str) -> str:
        """Generate detailed AI feedback using Gemini"""
        
        if not self.available or not self.client:
            return None
        
        try:
            prompt = self._create_feedback_prompt(
                question, ideal_answer, user_answer, score, topic
            )
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=500,
                )
            )
            
            if response and response.text:
                return response.text.strip()
            
        except Exception as e:
            print(f"Error generating feedback with Gemini: {e}")
            return None
    
    def generate_detailed_analysis(self, question: str, user_answer: str, 
                                   topic: str) -> dict:
        """Generate comprehensive analysis of answer"""
        
        if not self.available or not self.client:
            return None
        
        try:
            prompt = f"""
            Analyze this interview answer thoroughly:
            
            Topic: {topic}
            Question: {question}
            Candidate's Answer: {user_answer}
            
            Provide analysis in this exact format:
            
            STRENGTHS:
            - [strength 1]
            - [strength 2]
            - [strength 3]
            
            WEAKNESSES:
            - [weakness 1]
            - [weakness 2]
            - [weakness 3]
            
            MISSING_CONCEPTS:
            - [concept 1]
            - [concept 2]
            - [concept 3]
            
            IMPROVEMENTS:
            - [suggestion 1]
            - [suggestion 2]
            - [suggestion 3]
            
            OVERALL_ASSESSMENT: [brief summary]
            SCORE: [0-100]
            """
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=600,
                )
            )
            
            if response and response.text:
                return self._parse_analysis_response(response.text)
            
        except Exception as e:
            print(f"Error in detailed analysis: {e}")
            return None
    
    def _create_feedback_prompt(self, question: str, ideal: str, 
                                user: str, score: int, topic: str) -> str:
        """Create prompt for feedback generation"""
        
        return f"""
        You are an expert technical interviewer providing constructive feedback.
        
        Interview Details:
        - Topic: {topic}
        - Question: {question}
        - Ideal Answer: {ideal}
        - Candidate's Answer: {user}
        - Score: {score}%
        
        Provide comprehensive feedback including:
        1. What the candidate did well (specific examples from their answer)
        2. Areas that need improvement (specific)
        3. Missing concepts or key points they should have included
        4. How to improve (actionable steps)
        5. A brief summary of their performance
        
        Be encouraging but honest. Focus on helping them improve.
        Format in clear sections with bullet points.
        """
    
    def _parse_analysis_response(self, response_text: str) -> dict:
        """Parse Gemini analysis into structured format"""
        
        analysis = {
            'strengths': [],
            'weaknesses': [],
            'missing_concepts': [],
            'improvements': [],
            'overall': '',
            'score': 0
        }
        
        current_section = None
        lines = response_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if 'STRENGTHS:' in line:
                current_section = 'strengths'
            elif 'WEAKNESSES:' in line:
                current_section = 'weaknesses'
            elif 'MISSING_CONCEPTS:' in line:
                current_section = 'missing_concepts'
            elif 'IMPROVEMENTS:' in line:
                current_section = 'improvements'
            elif 'OVERALL_ASSESSMENT:' in line:
                analysis['overall'] = line.replace('OVERALL_ASSESSMENT:', '').strip()
            elif 'SCORE:' in line:
                try:
                    score_text = line.replace('SCORE:', '').strip()
                    analysis['score'] = int(score_text.replace('%', ''))
                except:
                    analysis['score'] = 0
            elif line.startswith('-') and current_section:
                analysis[current_section].append(line[1:].strip())
        
        return analysis