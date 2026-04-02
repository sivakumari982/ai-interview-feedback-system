class FeedbackGenerator:
    """Generate structured feedback for answers"""
    
    def __init__(self):
        self.levels = {
            (0, 30): "Beginner",
            (31, 50): "Developing", 
            (51, 70): "Intermediate",
            (71, 85): "Advanced",
            (86, 100): "Expert"
        }
        
        self.confidence_levels = {
            (0, 30): "Low",
            (31, 60): "Medium",
            (61, 100): "High"
        }
    
    def generate(self, evaluation: dict, answer: str, topic: str, 
                question: str, user_keywords: list = None, 
                ref_keywords: list = None) -> dict:
        """Generate comprehensive feedback"""
        
        score = evaluation['score']
        
        # Determine level
        level = self._get_level(score)
        confidence = self._get_confidence(score)
        
        # Get has_examples from evaluation or calculate it
        has_examples = evaluation.get('has_examples', False)
        if not has_examples:
            # Check if answer has examples
            has_examples = self._check_examples(answer)
        
        # Identify strengths and weaknesses
        used_concepts, missing_concepts = self._analyze_concepts(
            answer, topic, user_keywords, ref_keywords
        )
        
        # Generate summary
        summary = self._generate_summary(
            score, level, used_concepts, missing_concepts, evaluation
        )
        
        # Generate suggestions
        suggestions = self._generate_suggestions(
            missing_concepts, evaluation, answer
        )
        
        return {
            'score': score,
            'level': level,
            'confidence': confidence,
            'word_count': evaluation.get('word_count', len(answer.split())),
            'has_examples': has_examples,
            'used_concepts': used_concepts[:5],
            'missing_concepts': missing_concepts[:5],
            'summary': summary,
            'suggestions': suggestions[:5],
            'detailed_analysis': {
                'semantic_score': evaluation.get('semantic_score', 0),
                'keyword_score': evaluation.get('keyword_score', 0),
                'length_score': evaluation.get('length_score', 0),
                'structure_score': evaluation.get('structure_score', 0)
            }
        }
    
    def _check_examples(self, answer: str) -> bool:
        """Check if answer contains examples"""
        if not answer:
            return False
        
        answer_lower = answer.lower()
        example_indicators = [
            'for example', 'e.g.', 'such as', 'like', 'instance',
            'for instance', 'illustrate', 'demonstrate', 'example'
        ]
        
        return any(indicator in answer_lower for indicator in example_indicators)
    
    def _get_level(self, score: int) -> str:
        """Get proficiency level based on score"""
        for (low, high), level in self.levels.items():
            if low <= score <= high:
                return level
        return "Beginner"
    
    def _get_confidence(self, score: int) -> str:
        """Get confidence level based on score"""
        for (low, high), conf in self.confidence_levels.items():
            if low <= score <= high:
                return conf
        return "Low"
    
    def _analyze_concepts(self, answer: str, topic: str, 
                         user_keywords: list = None, 
                         ref_keywords: list = None) -> tuple:
        """Analyze which concepts are used and missing"""
        
        topic_concepts = {
            "Machine Learning": [
                "supervised", "unsupervised", "regression", "classification",
                "clustering", "neural", "deep", "training", "testing",
                "overfitting", "underfitting", "bias", "variance",
                "accuracy", "precision", "recall", "features", "labels",
                "data", "algorithm", "model", "prediction"
            ],
            "Data Structures": [
                "array", "linked list", "stack", "queue", "tree",
                "graph", "hash table", "heap", "binary search",
                "sorting", "traversal", "recursion", "dynamic"
            ],
            "OOP Concepts": [
                "class", "object", "inheritance", "polymorphism",
                "encapsulation", "abstraction", "interface",
                "method", "property", "constructor"
            ],
            "DBMS": [
                "sql", "database", "table", "query", "join",
                "normalization", "index", "transaction", "acid",
                "primary key", "foreign key", "schema"
            ]
        }
        
        concepts = topic_concepts.get(topic, [])
        
        if ref_keywords:
            concepts.extend(ref_keywords)
        
        concepts = list(set(concepts))
        
        answer_lower = answer.lower()
        used = []
        missing = []
        
        for concept in concepts:
            if concept.lower() in answer_lower:
                used.append(concept)
            else:
                missing.append(concept)
        
        return used, missing
    
    def _generate_summary(self, score: int, level: str, used_concepts: list,
                          missing_concepts: list, evaluation: dict) -> str:
        """Generate summary feedback"""
        
        if score >= 80:
            summary = f"Excellent answer! You're at {level} level. "
            if used_concepts:
                summary += f"You effectively covered concepts like {', '.join(used_concepts[:3])}. "
            if missing_concepts:
                summary += f"Consider adding {', '.join(missing_concepts[:2])} for completeness."
        
        elif score >= 60:
            summary = f"Good answer! You're at {level} level. "
            if used_concepts:
                summary += f"You covered {', '.join(used_concepts[:2])}. "
            summary += f"To improve, add more depth and include {', '.join(missing_concepts[:2]) if missing_concepts else 'more examples'}."
        
        elif score >= 40:
            summary = f"Fair attempt. You're at {level} level. "
            summary += f"Focus on key concepts: {', '.join(missing_concepts[:3]) if missing_concepts else 'the main ideas'}. "
            if not evaluation.get('has_examples', False):
                summary += "Include examples to strengthen your answer."
        
        else:
            summary = f"Your answer needs improvement. Currently at {level} level. "
            summary += f"Review core concepts like {', '.join(missing_concepts[:3]) if missing_concepts else 'the fundamentals'}. "
            summary += "Structure your answer better and provide specific examples."
        
        return summary
    
    def _generate_suggestions(self, missing_concepts: list, 
                             evaluation: dict, answer: str) -> list:
        """Generate specific improvement suggestions"""
        suggestions = []
        
        # Concept-based suggestions
        if missing_concepts:
            if len(missing_concepts) > 2:
                suggestions.append(f"Add these key concepts: {', '.join(missing_concepts[:3])}")
            else:
                suggestions.append(f"Include {', '.join(missing_concepts[:2])} in your answer")
        
        # Example suggestions
        if not evaluation.get('has_examples', False):
            suggestions.append("Add real-world examples to illustrate your points")
        
        # Length suggestions
        word_count = evaluation.get('word_count', len(answer.split()))
        if word_count < 30:
            suggestions.append("Expand your answer with more details and explanations")
        elif word_count < 20:
            suggestions.append("Your answer is too brief. Elaborate on each point")
        
        # Structure suggestions
        structure_score = evaluation.get('structure_score', 0)
        if structure_score < 50:
            suggestions.append("Structure your answer with clear introduction, body, and conclusion")
        
        # Add general suggestions based on score
        semantic_score = evaluation.get('semantic_score', 0)
        if semantic_score < 50:
            suggestions.append("Focus on directly answering the question asked")
        
        keyword_score = evaluation.get('keyword_score', 0)
        if keyword_score < 50:
            suggestions.append("Use more technical terms relevant to the topic")
        
        return suggestions