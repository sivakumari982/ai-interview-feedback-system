# Modules package initialization
from .question_generator import QuestionGenerator
from .preprocessing import TextPreprocessor
from .evaluator import AnswerEvaluator
from .feedback import FeedbackGenerator
from .gemini_question import GeminiQuestionGenerator
from .gemini_feedback import GeminiFeedbackGenerator
from .config import Config


__all__ = [
    'QuestionGenerator',
    'TextPreprocessor', 
    'AnswerEvaluator',
    'FeedbackGenerator',
    'GeminiQuestionGenerator',
    'GeminiFeedbackGenerator',
    'Config',
    'PerformanceVisualizer'  # ADD THIS
]