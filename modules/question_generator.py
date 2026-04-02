import random
import json
from typing import Dict, List, Optional

class QuestionGenerator:
    """Generate interview questions for various topics"""
    
    def __init__(self):
        """Initialize with question bank"""
        self.questions = self._load_questions()
        self.used_questions = {}  # Track used questions per topic
    
    def _load_questions(self) -> Dict[str, List[Dict]]:
        """Load question bank"""
        return {
            "HR Interview": [
                {
                    "question": "Tell me about yourself.",
                    "answer": "I am a computer science student with strong interest in software development. I have experience in Python, Java, and web technologies. I've worked on several projects including a machine learning application and a web development project. I'm passionate about solving problems and continuously learning new technologies."
                },
                {
                    "question": "What are your strengths and weaknesses?",
                    "answer": "My strengths include quick learning, problem-solving skills, and teamwork. I adapt well to new technologies. My weakness is public speaking, but I'm actively working on it by participating in presentations and joining toastmasters club."
                },
                {
                    "question": "Where do you see yourself in 5 years?",
                    "answer": "In 5 years, I see myself as a skilled software engineer, having mastered multiple technologies and contributed to significant projects. I aim to take on more responsibility, possibly leading small teams, while continuing to learn and grow in the field."
                }
            ],
            "Machine Learning": [
                {
                    "question": "Explain the bias-variance tradeoff.",
                    "answer": "Bias-variance tradeoff is a fundamental concept in machine learning. Bias is error from wrong assumptions in learning algorithm, leading to underfitting. Variance is error from sensitivity to fluctuations in training set, leading to overfitting. The tradeoff involves finding the right model complexity that minimizes total error."
                },
                {
                    "question": "What is overfitting and how to prevent it?",
                    "answer": "Overfitting occurs when a model learns training data too well, including noise, performing poorly on new data. Prevention methods include cross-validation, regularization (L1/L2), pruning decision trees, using more training data, and ensemble methods like random forests."
                },
                {
                    "question": "Compare supervised and unsupervised learning.",
                    "answer": "Supervised learning uses labeled data to train models for prediction/classification (e.g., regression, classification). Unsupervised learning finds patterns in unlabeled data (e.g., clustering, dimensionality reduction). Semi-supervised learning combines both approaches."
                }
            ],
            "Data Structures": [
                {
                    "question": "Explain the difference between array and linked list.",
                    "answer": "Arrays store elements in contiguous memory locations with constant-time access but fixed size. Linked lists store elements non-contiguously with dynamic size but linear access time. Arrays better for random access, linked lists better for frequent insertions/deletions."
                },
                {
                    "question": "What is a binary search tree?",
                    "answer": "A binary search tree is a tree data structure where each node has at most two children. For each node, left subtree contains smaller values, right subtree contains larger values. This property enables efficient searching, insertion, and deletion operations."
                }
            ],
            "OOP Concepts": [
                {
                    "question": "Explain the four pillars of OOP.",
                    "answer": "The four pillars are: Encapsulation (bundling data and methods), Inheritance (creating hierarchies), Polymorphism (many forms of methods), and Abstraction (hiding complexity). These principles help create modular, reusable, and maintainable code."
                },
                {
                    "question": "What is polymorphism with example?",
                    "answer": "Polymorphism allows objects to take multiple forms. Example: A 'draw()' method can behave differently for Circle, Rectangle, and Triangle classes. Achieved through method overriding (runtime) and overloading (compile-time)."
                }
            ],
            "DBMS": [
                {
                    "question": "Explain SQL joins with examples.",
                    "answer": "SQL joins combine rows from multiple tables. INNER JOIN returns matching rows, LEFT JOIN returns all left table rows, RIGHT JOIN returns all right table rows, FULL OUTER JOIN returns all rows when match exists in either table. Example: SELECT * FROM employees INNER JOIN departments ON employees.dept_id = departments.id;"
                },
                {
                    "question": "What is normalization?",
                    "answer": "Normalization organizes data to reduce redundancy and improve integrity. Normal forms: 1NF (atomic values), 2NF (no partial dependencies), 3NF (no transitive dependencies), BCNF (stronger 3NF). Denormalization may be used for performance optimization."
                }
            ]
        }
    
    def get_question(self, topic: str, avoid_duplicates: bool = True) -> Dict:
        """Get a random question for the specified topic"""
        if topic not in self.questions:
            # Return default question if topic not found
            return {
                "question": f"Explain a key concept in {topic}.",
                "answer": f"This is a sample answer for {topic} question. In a real implementation, you would provide detailed explanation covering key concepts, examples, and applications."
            }
        
        # Get questions for topic
        topic_questions = self.questions[topic]
        
        if avoid_duplicates and topic in self.used_questions:
            # Filter out used questions
            used = self.used_questions[topic]
            available = [q for q in topic_questions if q['question'] not in used]
            
            if available:
                selected = random.choice(available)
            else:
                # All questions used, reset and use any
                selected = random.choice(topic_questions)
                self.used_questions[topic] = []
        else:
            selected = random.choice(topic_questions)
        
        # Track used question
        if topic not in self.used_questions:
            self.used_questions[topic] = []
        self.used_questions[topic].append(selected['question'])
        
        return selected
    
    def add_question(self, topic: str, question: str, answer: str):
        """Add a new question to the bank"""
        if topic not in self.questions:
            self.questions[topic] = []
        
        self.questions[topic].append({
            'question': question,
            'answer': answer
        })