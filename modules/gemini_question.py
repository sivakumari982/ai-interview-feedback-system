from google import genai
from google.genai import types
import random
import time
import os
from .config import Config

class GeminiQuestionGenerator:
    """Generate interview questions using Google's Gemini API (new package)"""
    
    def __init__(self):
        self.available = False
        self.client = None
        self.model_name = None
        
        # Configure Gemini with new package
        if Config.is_gemini_available():
            try:
                # Initialize client with new package
                self.client = genai.Client(api_key=Config.GEMINI_API_KEY)
                
                # Test the connection by listing models
                models = self.client.models.list()
                
                # Find available models that support generateContent
                available_models = []
                for model in models:
                    if 'generateContent' in model.supported_actions:
                        available_models.append(model.name.replace('models/', ''))
                        print(f"✅ Available model: {model.name}")
                
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
                    print(f"✅ Using Gemini model: {self.model_name}")
                else:
                    print("❌ No models available for generateContent")
                    
            except Exception as e:
                print(f"❌ Gemini initialization failed: {e}")
                self.available = False
        
        # Cache for generated questions
        self.generated_questions = []
        self.max_cache = 100
        self.used_questions = set()
    
    def generate_question(self, topic: str, difficulty: str = "medium") -> dict:
        """Generate a unique interview question using Gemini"""
        
        if not self.available or not self.client:
            print("Gemini not available, returning None")
            return None
        
        try:
            # Create prompt for Gemini
            prompt = self._create_question_prompt(topic, difficulty)
            
            # Add instruction to avoid repetition
            if self.used_questions:
                prompt += f"\n\nIMPORTANT: Do not generate any of these previously used questions:\n"
                for q in list(self.used_questions)[-5:]:
                    prompt += f"- {q}\n"
            
            # Generate question with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Using new API format
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.8,
                            max_output_tokens=200,
                        )
                    )
                    
                    if response and response.text:
                        question = response.text.strip()
                        
                        # Clean up question
                        if question.lower().startswith("question:"):
                            question = question[9:].strip()
                        
                        # Remove quotes if present
                        question = question.strip('"\'')
                        
                        # Check if it's a new question
                        if question not in self.used_questions:
                            # Track this question
                            self.used_questions.add(question)
                            
                            # Return in expected format
                            return {
                                'question': question,
                                'topic': topic,
                                'difficulty': difficulty,
                                'source': 'gemini_ai',
                                'model': self.model_name,
                                'answer': self._generate_answer_preview(topic, question)
                            }
                        else:
                            print(f"Generated duplicate question, retry {attempt + 1}")
                            
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    time.sleep(1)  # Wait before retry
            
            # If all retries fail, try a different prompt
            return self._generate_fallback_question(topic, difficulty)
            
        except Exception as e:
            print(f"Error generating question with Gemini: {e}")
            return None
    
    def generate_with_answer(self, topic: str, difficulty: str = "medium") -> dict:
        """Generate both question AND ideal answer using Gemini"""
        
        if not self.available or not self.client:
            return None
        
        try:
            prompt = f"""
            Create a technical interview question and comprehensive answer for a CSE student.
            
            Topic: {topic}
            Difficulty Level: {difficulty}
            
            Requirements:
            - Question should test deep understanding
            - Answer should be detailed (150-200 words)
            - Include key concepts and technical terms
            - Make it practical and industry-relevant
            - Include examples where appropriate
            
            Format your response EXACTLY like this:
            
            QUESTION: [The interview question]
            
            ANSWER: [The ideal answer with explanations]
            
            KEY_CONCEPTS: [concept1, concept2, concept3, concept4, concept5]
            """
            
            # Add anti-duplication instruction
            if self.used_questions:
                prompt += f"\n\nIMPORTANT: Generate a NEW question. Avoid these: {list(self.used_questions)[-3:]}"
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=800,
                )
            )
            
            if response and response.text:
                parsed = self._parse_full_response(response.text, topic)
                
                if parsed and parsed.get('question'):
                    # Track this question
                    self.used_questions.add(parsed['question'])
                    
                    parsed['source'] = 'gemini_ai'
                    parsed['difficulty'] = difficulty
                    parsed['model'] = self.model_name
                    return parsed
            
            return None
            
        except Exception as e:
            print(f"Error generating Q&A with Gemini: {e}")
            return None
    
    def _generate_answer_preview(self, topic: str, question: str) -> str:
        """Generate a preview answer for the question"""
        try:
            prompt = f"""
            For this interview question about {topic}:
            "{question}"
            
            Generate a brief answer preview (50 words) that covers the key points.
            """
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=100,
                )
            )
            if response and response.text:
                return response.text.strip()
        except:
            pass
        
        return "Answer will be generated during evaluation."
    
    def _generate_fallback_question(self, topic: str, difficulty: str) -> dict:
        """Generate a fallback question when main generation fails"""
        templates = [
            f"Explain the most important concept in {topic}.",
            f"What are the key principles of {topic}?",
            f"How would you apply {topic} in a real-world scenario?",
            f"Discuss the advantages and disadvantages of using {topic}.",
            f"What are the common challenges when working with {topic}?"
        ]
        
        question = random.choice(templates)
        
        # Ensure uniqueness
        base_question = question
        counter = 1
        while question in self.used_questions and counter < 10:
            question = f"{base_question} (Part {counter})"
            counter += 1
        
        self.used_questions.add(question)
        
        return {
            'question': question,
            'topic': topic,
            'difficulty': difficulty,
            'source': 'gemini_ai_fallback',
            'model': self.model_name,
            'answer': f"This is a {difficulty} level question about {topic}. Provide a comprehensive answer covering key concepts, examples, and applications."
        }
    
    def _create_question_prompt(self, topic: str, difficulty: str) -> str:
        """Create prompt for question generation"""
        
        difficulty_descriptions = {
            "easy": "basic/fundamental concept that tests core understanding",
            "medium": "intermediate concept that requires good understanding and some application",
            "hard": "advanced concept that tests deep understanding and problem-solving",
            "expert": "complex scenario that requires expert knowledge and critical thinking"
        }
        
        desc = difficulty_descriptions.get(difficulty, difficulty_descriptions["medium"])
        
        return f"""
        Generate a UNIQUE technical interview question for a CSE student.
        
        Topic: {topic}
        Difficulty: {difficulty} ({desc})
        
        Requirements:
        - Question must be ORIGINAL and not generic
        - Should test conceptual understanding
        - Be specific and clear
        - Require explanation, not just yes/no
        - Include practical aspects where relevant
        
        Return ONLY the question, no explanations or additional text.
        """
    
    def _parse_full_response(self, response_text: str, topic: str) -> dict:
        """Parse full Q&A response from Gemini"""
        
        lines = response_text.strip().split('\n')
        
        question = ""
        answer = ""
        concepts = []
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith("QUESTION:"):
                question = line[9:].strip()
                current_section = "question"
            elif line.startswith("ANSWER:"):
                answer = line[7:].strip()
                current_section = "answer"
            elif line.startswith("KEY_CONCEPTS:"):
                concepts_text = line[13:].strip()
                concepts = [c.strip() for c in concepts_text.split(',')]
                current_section = "concepts"
            elif current_section == "answer":
                # Continue collecting answer lines
                answer += " " + line
            elif current_section == "question":
                question += " " + line
        
        if not question:
            # Try alternate parsing
            parts = response_text.split("ANSWER:")
            if len(parts) > 1:
                question = parts[0].replace("QUESTION:", "").strip()
                answer = parts[1].strip()
        
        return {
            'question': question,
            'answer': answer,
            'key_concepts': concepts,
            'topic': topic
        }
    
    def generate_follow_up(self, previous_question: str, user_answer: str) -> str:
        """Generate a follow-up question based on previous interaction"""
        
        if not self.available or not self.client:
            return None
        
        try:
            prompt = f"""
            Based on this interview interaction:
            
            Previous Question: {previous_question}
            Candidate's Answer: {user_answer}
            
            Generate a relevant follow-up question that:
            1. Probes deeper into the topic
            2. Challenges the candidate's understanding
            3. Is natural in an interview conversation
            4. Builds upon their previous answer
            
            Return ONLY the follow-up question.
            """
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.8,
                    max_output_tokens=150,
                )
            )
            
            if response and response.text:
                follow_up = response.text.strip()
                if follow_up.lower().startswith("follow-up:"):
                    follow_up = follow_up[10:].strip()
                
                return follow_up
            
        except Exception as e:
            print(f"Error generating follow-up: {e}")
        
        return None