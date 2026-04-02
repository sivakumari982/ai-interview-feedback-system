import pandas as pd
import json
import os
import time
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class BatchTester:
    """Run batch tests to evaluate system performance"""
    
    def __init__(self, modules):
        self.modules = modules
        self.results = []
    
    def load_test_cases(self, file_path="dataset/test_cases.json"):
        """Load test cases from JSON file"""
        try:
            # Try multiple possible paths
            possible_paths = [
                file_path,
                "dataset/test_cases.json",
                "../dataset/test_cases.json",
                os.path.join(os.path.dirname(__file__), "../dataset/test_cases.json")
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        data = json.load(f)
                        print(f"✅ Loaded test cases from: {path}")
                        return data.get('test_cases', [])
            
            print("❌ Test cases file not found")
            return []
            
        except Exception as e:
            print(f"Error loading test cases: {e}")
            return []
    
    def run_test(self, test_cases=None):
        """Run batch test with predefined test cases"""
        
        if test_cases is None:
            test_cases = self.load_test_cases()
        
        if not test_cases:
            print("No test cases to run")
            return pd.DataFrame()
        
        print(f"Running {len(test_cases)} test cases...")
        
        for i, test in enumerate(test_cases):
            print(f"Running test {i+1}/{len(test_cases)}: {test.get('question', 'N/A')[:50]}...")
            
            try:
                # Get answer evaluation
                evaluation = self.modules['evaluator'].evaluate(
                    test['user_answer'],
                    test['question'],
                    test['topic'],
                    None,
                    test.get('keywords', [])
                )
                
                # Calculate concept detection metrics
                user_keywords = self.modules['preprocessor'].extract_keywords(test['user_answer'])
                expected_keywords = test.get('keywords', [])
                
                # Calculate precision, recall for this test
                if expected_keywords:
                    detected = [kw for kw in expected_keywords if kw.lower() in test['user_answer'].lower()]
                    precision = len(detected) / len(user_keywords) if user_keywords else 0
                    recall = len(detected) / len(expected_keywords) if expected_keywords else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                else:
                    precision = recall = f1 = 0
                
                # Store results
                self.results.append({
                    'test_id': test.get('test_id', i + 1),
                    'topic': test['topic'],
                    'question': test['question'][:100],
                    'system_score': evaluation['score'],
                    'human_score': test['human_score'],
                    'semantic_score': evaluation['semantic_score'],
                    'keyword_score': evaluation['keyword_score'],
                    'structure_score': evaluation['structure_score'],
                    'length_score': evaluation['length_score'],
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'word_count': len(test['user_answer'].split())
                })
                
            except Exception as e:
                print(f"Error on test {i+1}: {e}")
                self.results.append({
                    'test_id': test.get('test_id', i + 1),
                    'topic': test['topic'],
                    'question': test['question'][:100],
                    'system_score': 0,
                    'human_score': test['human_score'],
                    'error': str(e)
                })
            
            time.sleep(0.5)  # Delay to avoid rate limits
        
        return pd.DataFrame(self.results)
    
    def get_performance_summary(self, df=None):
        """Calculate performance metrics from results"""
        
        if df is None and self.results:
            df = pd.DataFrame(self.results)
        
        if df is None or df.empty:
            return {}
        
        # Filter out rows with errors
        df_valid = df[df['system_score'] > 0] if 'system_score' in df.columns else df
        
        if len(df_valid) == 0:
            return {}
        
        return {
            'r2_score': r2_score(df_valid['human_score'], df_valid['system_score']),
            'mse': mean_squared_error(df_valid['human_score'], df_valid['system_score']),
            'rmse': np.sqrt(mean_squared_error(df_valid['human_score'], df_valid['system_score'])),
            'mae': mean_absolute_error(df_valid['human_score'], df_valid['system_score']),
            'avg_system_score': df_valid['system_score'].mean(),
            'avg_human_score': df_valid['human_score'].mean(),
            'correlation': df_valid['system_score'].corr(df_valid['human_score']),
            'total_tests': len(df_valid)
        }
    
    def generate_report(self, df=None):
        """Generate a detailed performance report"""
        
        if df is None and self.results:
            df = pd.DataFrame(self.results)
        
        if df is None or df.empty:
            return "No test results available."
        
        metrics = self.get_performance_summary(df)
        
        report = f"""
        📊 **BATCH TEST PERFORMANCE REPORT**
        ===================================
        
        📈 **Score Prediction Metrics:**
        - R² Score: {metrics.get('r2_score', 0):.4f}
        - Correlation: {metrics.get('correlation', 0):.4f}
        - MSE: {metrics.get('mse', 0):.4f}
        - RMSE: {metrics.get('rmse', 0):.4f}
        - MAE: {metrics.get('mae', 0):.4f}
        
        📊 **Score Statistics:**
        - Avg System Score: {metrics.get('avg_system_score', 0):.1f}
        - Avg Human Score: {metrics.get('avg_human_score', 0):.1f}
        - Difference: {abs(metrics.get('avg_system_score', 0) - metrics.get('avg_human_score', 0)):.1f}
        
        📊 **Dataset Info:**
        - Total Tests Run: {metrics.get('total_tests', 0)}
        - Topics Covered: {df['topic'].nunique() if 'topic' in df.columns else 0}
        """
        
        # Add per-topic breakdown
        if 'topic' in df.columns and 'system_score' in df.columns:
            report += "\n\n📂 **Per-Topic Performance:**\n"
            for topic in df['topic'].unique():
                topic_df = df[df['topic'] == topic]
                if len(topic_df) > 0:
                    report += f"\n- {topic}:\n"
                    report += f"  - Samples: {len(topic_df)}\n"
                    report += f"  - Avg Score: {topic_df['system_score'].mean():.1f} (System) vs {topic_df['human_score'].mean():.1f} (Human)\n"
        
        return report
    
    def save_results(self, filename="dataset/test_results.csv"):
        """Save test results to CSV"""
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(filename, index=False)
            print(f"✅ Results saved to {filename}")
            return filename
        return None