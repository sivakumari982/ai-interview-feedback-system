import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
from datetime import datetime

class PerformanceMetrics:
    """Calculate performance metrics for the interview system"""
    
    def __init__(self):
        self.predictions = []
        self.actual_scores = []
        self.concept_predictions = []
        self.concept_actuals = []
    
    def add_evaluation(self, system_score, human_score, 
                      system_concepts=None, actual_concepts=None):
        """Add a single evaluation for tracking"""
        self.predictions.append(system_score)
        self.actual_scores.append(human_score)
        
        if system_concepts and actual_concepts:
            # Convert concepts to binary vectors
            all_concepts = list(set(system_concepts + actual_concepts))
            system_vec = [1 if c in system_concepts else 0 for c in all_concepts]
            actual_vec = [1 if c in actual_concepts else 0 for c in all_concepts]
            self.concept_predictions.append(system_vec)
            self.concept_actuals.append(actual_vec)
    
    def calculate_accuracy(self, tolerance=10):
        """
        Calculate accuracy: How often system score is within tolerance of human score
        tolerance: acceptable difference in percentage points
        """
        if len(self.predictions) == 0:
            return 0.0
        
        correct = 0
        for pred, actual in zip(self.predictions, self.actual_scores):
            if abs(pred - actual) <= tolerance:
                correct += 1
        
        return correct / len(self.predictions)
    
    def calculate_precision_recall_f1(self):
        """Calculate precision, recall, and F1 for concept detection"""
        if len(self.concept_predictions) == 0:
            return 0.0, 0.0, 0.0
        
        # Flatten all predictions and actuals
        all_pred = []
        all_actual = []
        
        for pred_vec, actual_vec in zip(self.concept_predictions, self.concept_actuals):
            all_pred.extend(pred_vec)
            all_actual.extend(actual_vec)
        
        precision = precision_score(all_actual, all_pred, average='macro', zero_division=0)
        recall = recall_score(all_actual, all_pred, average='macro', zero_division=0)
        f1 = f1_score(all_actual, all_pred, average='macro', zero_division=0)
        
        return precision, recall, f1
    
    def calculate_r2_score(self):
        """Calculate R² score for score predictions"""
        if len(self.predictions) < 2:
            return 0.0
        
        return r2_score(self.actual_scores, self.predictions)
    
    def calculate_mse(self):
        """Calculate Mean Squared Error"""
        if len(self.predictions) == 0:
            return 0.0
        
        return mean_squared_error(self.actual_scores, self.predictions)
    
    def calculate_rmse(self):
        """Calculate Root Mean Squared Error"""
        mse = self.calculate_mse()
        return np.sqrt(mse)
    
    def calculate_mae(self):
        """Calculate Mean Absolute Error"""
        if len(self.predictions) == 0:
            return 0.0
        
        return np.mean([abs(p - a) for p, a in zip(self.predictions, self.actual_scores)])
    
    def get_all_metrics(self):
        """Get all performance metrics"""
        precision, recall, f1 = self.calculate_precision_recall_f1()
        
        return {
            'accuracy': self.calculate_accuracy(),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'r2_score': self.calculate_r2_score(),
            'mse': self.calculate_mse(),
            'rmse': self.calculate_rmse(),
            'mae': self.calculate_mae(),
            'total_samples': len(self.predictions)
        }
    
    def generate_report(self):
        """Generate a detailed performance report"""
        metrics = self.get_all_metrics()
        
        report = f"""
        📊 **Performance Metrics Report**
        ================================
        
        📈 **Score Prediction Metrics:**
        - R² Score: {metrics['r2_score']:.4f}
        - MSE: {metrics['mse']:.4f}
        - RMSE: {metrics['rmse']:.4f}
        - MAE: {metrics['mae']:.4f}
        - Accuracy (within 10%): {metrics['accuracy']:.2%}
        
        🎯 **Concept Detection Metrics:**
        - Precision: {metrics['precision']:.4f}
        - Recall: {metrics['recall']:.4f}
        - F1-Score: {metrics['f1_score']:.4f}
        
        📊 **Dataset Info:**
        - Total Samples: {metrics['total_samples']}
        """
        
        return report
    
    def reset(self):
        """Reset all stored data"""
        self.predictions = []
        self.actual_scores = []
        self.concept_predictions = []
        self.concept_actuals = []


class HumanEvaluator:
    """Simulate human evaluation for testing"""
    
    @staticmethod
    def evaluate_answer(answer, ideal_answer, key_concepts):
        """Simulate human evaluation (for testing)"""
        # This is a simplified simulation
        # In real scenario, you'd have actual human ratings
        
        # Calculate score based on keyword coverage
        answer_lower = answer.lower()
        covered = sum(1 for concept in key_concepts if concept.lower() in answer_lower)
        score = (covered / len(key_concepts)) * 100 if key_concepts else 70
        
        # Add some random variation to simulate human judgment
        variation = np.random.normal(0, 5)
        final_score = max(0, min(100, score + variation))
        
        return final_score
    
    @staticmethod
    def identify_concepts(answer, all_possible_concepts):
        """Simulate human concept identification"""
        answer_lower = answer.lower()
        return [c for c in all_possible_concepts if c.lower() in answer_lower]