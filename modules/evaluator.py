import numpy as np
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Download NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class AnswerEvaluator:
    """Evaluate interview answers with NLP + Metrics"""

    def __init__(self):
        # Topic keywords for better matching
        self.topic_keywords = {
            "Machine Learning": [
                "algorithm", "model", "training", "data", "prediction",
                "feature", "label", "supervised", "unsupervised",
                "regression", "classification", "learning", "accuracy",
                "test", "validation", "bias", "variance", "overfitting",
                "machine", "learning", "ai", "artificial", "preprocessing",
                "cleaning", "structured", "raw"
            ],
            "Data Structures": [
                "array", "list", "tree", "graph", "stack", "queue",
                "node", "pointer", "memory", "contiguous", "linked",
                "heap", "hash", "binary", "search", "sort", "data", "structure"
            ],
            "DBMS": [
                "database", "sql", "query", "table", "join",
                "primary key", "foreign key", "normalization",
                "index", "transaction", "acid", "schema", "data", "store"
            ],
            "HR Interview": [
                "experience", "skills", "team", "project", "learning",
                "strength", "weakness", "goal", "achievement", "worked",
                "developed", "created", "managed", "communicate"
            ],
            "OOP Concepts": [
                "class", "object", "inheritance", "polymorphism",
                "encapsulation", "abstraction", "method", "property",
                "oriented", "programming", "reuse", "code"
            ]
        }
        
        # Stopwords for preprocessing
        self.stop_words = set(stopwords.words('english'))

    def evaluate(self, user_answer: str, question: str, topic: str, ref_keywords=None, user_keywords=None) -> dict:
        """Evaluate answer and return scores"""
        
        # Calculate individual scores
        semantic_score = self._calculate_semantic_score(user_answer, question)
        keyword_score = self._calculate_keyword_score(user_answer, topic)
        length_score = self._calculate_length_score(user_answer)
        structure_score = self._calculate_structure_score(user_answer)
        example_score = self._calculate_example_score(user_answer)
        
        # Give bonus for using user keywords (if provided)
        if user_keywords and len(user_keywords) > 2:
            keyword_score = min(keyword_score + 15, 100)
        
        # Give bonus for longer answers
        word_count = len(user_answer.split())
        if word_count > 30:
            length_score = min(length_score + 10, 100)
        if word_count > 50:
            length_score = min(length_score + 10, 100)

        # Final score (weighted average)
        overall_score = int(
            0.25 * semantic_score +
            0.25 * keyword_score +
            0.20 * length_score +
            0.15 * structure_score +
            0.15 * example_score
        )
        
        # Ensure score is between 0-100
        overall_score = max(0, min(100, overall_score))
        
        # Threshold setting (30 = good enough)
        THRESHOLD = 30
        predicted_label = 1 if overall_score >= THRESHOLD else 0

        # Generate feedback
        feedback = self._generate_feedback(question, user_answer, overall_score, 
                                           semantic_score, keyword_score, 
                                           length_score, structure_score, example_score)

        return {
            "score": overall_score,
            "predicted_label": predicted_label,
            "semantic_score": int(semantic_score),
            "keyword_score": int(keyword_score),
            "length_score": int(length_score),
            "structure_score": int(structure_score),
            "example_score": int(example_score),
            "has_examples": example_score > 0,
            "feedback": feedback,
            "word_count": len(user_answer.split())
        }

    def evaluate_metrics(self, y_true, y_pred):
        """Calculate performance metrics"""
        return {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4)
        }

    def _calculate_semantic_score(self, answer: str, question: str) -> float:
        """Calculate semantic similarity between answer and question"""
        if not answer or not question:
            return 0.0
        
        q_words = set(self._preprocess_text(question))
        a_words = set(self._preprocess_text(answer))
        
        q_words = {w for w in q_words if w not in self.stop_words}
        a_words = {w for w in a_words if w not in self.stop_words}

        if len(q_words) == 0:
            return 50.0

        overlap = len(q_words & a_words)
        
        if len(q_words) > 0:
            score = (overlap / len(q_words)) * 100
        else:
            score = 50
        
        if len(answer) > 20:
            score = max(score, 30)
        
        return min(score, 100.0)

    def _calculate_keyword_score(self, answer: str, topic: str) -> float:
        """Calculate keyword matching score"""
        if not answer:
            return 0.0
        
        keywords = self.topic_keywords.get(topic, [])
        if not keywords:
            return 50.0
        
        answer_lower = answer.lower()
        
        matched = 0
        for kw in keywords:
            if kw.lower() in answer_lower:
                matched += 1
        
        if len(keywords) > 0:
            score = (matched / len(keywords)) * 100
        else:
            score = 50
        
        # Minimum score if answer has content
        if len(answer) > 30:
            score = max(score, 35)
        
        return min(score, 100.0)

    def _calculate_length_score(self, answer: str) -> float:
        """Score based on answer length"""
        if not answer:
            return 0.0
        
        words = len(answer.split())
        
        if words < 5:
            return 15.0
        elif words < 10:
            return 30.0
        elif words < 20:
            return 50.0
        elif words < 35:
            return 70.0
        elif words < 50:
            return 85.0
        else:
            return 100.0

    def _calculate_structure_score(self, answer: str) -> float:
        """Evaluate answer structure"""
        if not answer:
            return 0.0
        
        score = 30.0
        
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) >= 3:
            score += 40
        elif len(sentences) >= 2:
            score += 30
        elif len(sentences) >= 1:
            score += 15
        
        connectors = ['first', 'second', 'third', 'finally', 'also', 'additionally', 
                     'moreover', 'furthermore', 'however', 'therefore', 'because']
        if any(c in answer.lower() for c in connectors):
            score += 20
        
        if len(answer.split()) > 20:
            score += 15
        
        return min(score, 100.0)

    def _calculate_example_score(self, answer: str) -> float:
        """Check if answer includes examples"""
        if not answer:
            return 0.0
        
        answer_lower = answer.lower()
        example_indicators = [
            'for example', 'e.g.', 'such as', 'like', 'instance',
            'for instance', 'illustrate', 'demonstrate', 'example',
            'including', 'especially', 'particularly'
        ]
        
        for indicator in example_indicators:
            if indicator in answer_lower:
                return 100.0
        
        if len(answer.split()) > 40:
            return 50.0
        
        return 0.0

    def _preprocess_text(self, text: str) -> list:
        """Preprocess text: lowercase, remove punctuation, tokenize"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = word_tokenize(text)
        return [t for t in tokens if len(t) > 2]

    def _generate_feedback(self, question: str, answer: str, score: int,
                          semantic: float, keyword: float, length: float,
                          structure: float, example: float) -> str:
        """Generate detailed feedback based on scores"""
        
        feedback_parts = []
        
        if score >= 70:
            feedback_parts.append("✅ Excellent answer! Great job!")
        elif score >= 55:
            feedback_parts.append("👍 Good answer! Keep improving!")
        elif score >= 40:
            feedback_parts.append("📝 Fair answer. Room for improvement.")
        else:
            feedback_parts.append("⚠️ Needs significant improvement.")
        
        if semantic < 40:
            feedback_parts.append("• Your answer didn't fully address the question.")
        
        if keyword < 35:
            feedback_parts.append("• Use more technical terms relevant to this topic.")
        
        if length < 40:
            feedback_parts.append("• Provide more details. Your answer is too brief.")
        
        if structure < 40:
            feedback_parts.append("• Structure your answer better with clear sentences.")
        
        if example == 0 and score < 60:
            feedback_parts.append("• Include specific examples to strengthen your answer.")
        
        if score >= 60 and example > 0:
            feedback_parts.append("• Great use of examples!")
        
        return " ".join(feedback_parts)


if __name__ == "__main__":
    print("=" * 70)
    print("🤖 AI Interview Feedback System - Performance Test")
    print("=" * 70)
    
    evaluator = AnswerEvaluator()
    
    try:
        df = pd.read_csv("dataset/questions.csv")
        print(f"✅ Loaded {len(df)} questions")
    except FileNotFoundError:
        print("❌ dataset/questions.csv not found")
        exit(1)
    
    y_true = []
    y_pred = []
    
    for i, row in df.iterrows():
        question = row["question"]
        ideal_answer = row["answer"]
        topic = row["topic"]
        
        if i % 3 == 0:
            user_answer = ideal_answer
            y_true.append(1)
        elif i % 3 == 1:
            user_answer = ideal_answer[:50]
            y_true.append(1)
        else:
            user_answer = "I don't know"
            y_true.append(0)
        
        result = evaluator.evaluate(user_answer, question, topic)
        y_pred.append(result["predicted_label"])
    
    metrics = evaluator.evaluate_metrics(y_true, y_pred)
    
    print(f"\n📈 Accuracy: {metrics['accuracy']:.2%}")
    print(f"📈 Precision: {metrics['precision']:.4f}")
    print(f"📈 Recall: {metrics['recall']:.4f}")
    print(f"📈 F1 Score: {metrics['f1_score']:.4f}")