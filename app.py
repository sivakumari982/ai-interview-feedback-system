import streamlit as st
import pandas as pd
import time
import random
from datetime import datetime
import nltk
import os
import sys

# ============ DOWNLOAD NLTK DATA (FIX FOR STREAMLIT CLOUD) ============
@st.cache_resource
def download_nltk_data():
    """Download required NLTK data for text processing"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')
        
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
        
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

# Run NLTK download
download_nltk_data()
# =======================================================================

# Import modules
from modules.question_generator import QuestionGenerator
from modules.preprocessing import TextPreprocessor
from modules.evaluator import AnswerEvaluator
from modules.feedback import FeedbackGenerator
from modules.config import Config
from modules.performance import PerformanceMetrics, HumanEvaluator
from modules.batch_test import BatchTester
from modules.visualizations import PerformanceVisualizer

# Try to import Gemini modules (optional - may fail on cloud)
try:
    from modules.gemini_question import GeminiQuestionGenerator
    from modules.gemini_feedback import GeminiFeedbackGenerator
    GEMINI_IMPORT_SUCCESS = True
except ImportError:
    GEMINI_IMPORT_SUCCESS = False
    GeminiQuestionGenerator = None
    GeminiFeedbackGenerator = None
    print("⚠️ Gemini modules not available - using question bank only")

# Initialize performance metrics in session state
if 'performance' not in st.session_state:
    st.session_state.performance = PerformanceMetrics()
if 'collecting_data' not in st.session_state:
    st.session_state.collecting_data = False
if 'human_score' not in st.session_state:
    st.session_state.human_score = 70

# Page config
st.set_page_config(
    page_title="AI Interview Feedback System",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
        font-size: 2.5rem;
    }
    .gemini-badge {
        background: #4285F4;
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 25px;
        font-size: 1rem;
        display: inline-block;
        margin-left: 1rem;
        font-weight: normal;
    }
    .feedback-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f8f9fa;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stProgress > div > div > div > div {
        background-color: #667eea;
    }
    div[data-testid="stButton"] > button {
        width: 100%;
        margin: 5px 0;
        transition: all 0.3s ease;
    }
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .indicator-success {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-weight: bold;
        border-left: 4px solid #28a745;
    }
    .indicator-warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.75rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-weight: bold;
        border-left: 4px solid #ffc107;
    }
    .indicator-info {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 0.75rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-weight: bold;
        border-left: 4px solid #17a2b8;
    }
    .model-badge {
        background: #6c757d;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        display: inline-block;
    }
    .performance-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .score-card-excellent {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        border-radius: 15px;
        margin-bottom: 1rem;
    }
    .score-card-good {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #ffc107 0%, #ff9800 100%);
        border-radius: 15px;
        margin-bottom: 1rem;
    }
    .score-card-poor {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        border-radius: 15px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'current_question': None,
        'current_question_data': None,
        'feedback': None,
        'gemini_feedback': None,
        'gemini_analysis': None,
        'answer_submitted': False,
        'selected_topic': "Machine Learning",
        'last_answer': "",
        'question_generated': False,
        'used_questions': [],
        'total_attempts': 0,
        'session_start': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'history': [],
        'use_gemini': Config.is_gemini_available() if GEMINI_IMPORT_SUCCESS else False,
        'gemini_working': False,
        'gemini_model': None,
        'available_models': [],
        'difficulty': 'medium',
        'ai_enhanced': True,
        'force_reset': False,
        'follow_up_question': None,
        'models_loaded': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Initialize modules with caching
@st.cache_resource
def load_modules():
    """Load all modules with caching"""
    try:
        modules = {
            'question_gen': QuestionGenerator(),
            'preprocessor': TextPreprocessor(),
            'evaluator': AnswerEvaluator(),
            'feedback_gen': FeedbackGenerator()
        }
        
        # Add Gemini modules if available and import succeeded
        if GEMINI_IMPORT_SUCCESS and Config.is_gemini_available():
            try:
                gemini_question = GeminiQuestionGenerator()
                gemini_feedback = GeminiFeedbackGenerator()
                
                modules['gemini_question'] = gemini_question
                modules['gemini_feedback'] = gemini_feedback
                
                # Check if Gemini is actually working
                if gemini_question.available:
                    st.session_state.gemini_working = True
                    st.session_state.gemini_model = gemini_question.model_name
                    print(f"✅ Gemini modules loaded successfully with model: {st.session_state.gemini_model}")
                else:
                    st.session_state.gemini_working = False
                    print("⚠️ Gemini modules loaded but not available")
                    
            except Exception as e:
                st.warning(f"Gemini initialization failed: {str(e)}")
                st.session_state.use_gemini = False
                st.session_state.gemini_working = False
        
        return modules
    except Exception as e:
        st.error(f"Error loading modules: {str(e)}")
        return None

modules = load_modules()

if not modules:
    st.error("Failed to load required modules. Please check your installation.")
    st.stop()

# Header with dynamic badge
if st.session_state.use_gemini and st.session_state.get('gemini_working', False):
    st.markdown(f"""
    <h1 class='main-header'>
        🎯 AI-Powered Interview Feedback System 
        <span class='gemini-badge'>🤖 {st.session_state.gemini_model}</span>
    </h1>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <h1 class='main-header'>
        🎯 AI-Powered Interview Feedback System
    </h1>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📚 Topics")
    
    # Show Gemini status with more details
    if st.session_state.use_gemini:
        if st.session_state.get('gemini_working', False):
            st.success(f"""
            ✅ **Gemini AI Active**
            - Model: `{st.session_state.gemini_model}`
            - Status: Connected
            """)
        else:
            st.warning("""
            ⚠️ **Gemini API Issue**
            - Check your API key in `.env` file
            - Check internet connection
            - Using question bank mode
            """)
    else:
        st.warning("""
        ⚠️ **Using Question Bank Mode**
        
        To enable AI features:
        1. Get API key from [aistudio.google.com](https://aistudio.google.com)
        2. Create `.env` file with GEMINI_API_KEY=your_key
        """)
    
    st.markdown("---")
    
    # All topics
    topics = [
        "HR Interview", "Communication Skills", "Aptitude & Reasoning",
        "Machine Learning", "Deep Learning", "Data Structures",
        "Algorithms", "OOP Concepts", "DBMS", "Operating Systems",
        "Computer Networks", "Cloud Computing", "Cybersecurity", 
        "System Design", "Software Engineering", "Python Programming",
        "Java Programming", "Web Development", "Mobile Development"
    ]
    
    # Topic selection
    selected_topic = st.selectbox(
        "Choose interview topic:", 
        topics,
        index=topics.index(st.session_state.selected_topic) if st.session_state.selected_topic in topics else 0
    )
    
    # Difficulty selector (only when Gemini is active and working)
    if st.session_state.use_gemini and st.session_state.get('gemini_working', False) and st.session_state.get('ai_enhanced', True):
        difficulty = st.select_slider(
            "📊 Question Difficulty",
            options=["easy", "medium", "hard", "expert"],
            value=st.session_state.difficulty
        )
        st.session_state.difficulty = difficulty
    
    # Question source info
    if st.session_state.use_gemini and st.session_state.get('gemini_working', False):
        source_mode = "AI + Question Bank"
    else:
        source_mode = "Question Bank Only"
    st.info(f"📋 Source: **{source_mode}**")
    
    # Reset on topic change
    if selected_topic != st.session_state.selected_topic:
        st.session_state.selected_topic = selected_topic
        st.session_state.current_question = None
        st.session_state.feedback = None
        st.session_state.gemini_feedback = None
        st.session_state.answer_submitted = False
        st.session_state.question_generated = False
        st.session_state.follow_up_question = None
        st.rerun()
    
    st.markdown("---")
    
    # Session stats
    st.markdown("### 📊 Session Statistics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Attempts", st.session_state.total_attempts)
    with col2:
        if st.session_state.feedback:
            st.metric("Last Score", f"{st.session_state.feedback['score']}%")
    
    # Show history count
    if st.session_state.history:
        avg_score = sum([h['score'] for h in st.session_state.history]) / len(st.session_state.history)
        st.metric("Average Score", f"{avg_score:.1f}%")
    
    st.markdown("---")
    
    # ============ PERFORMANCE METRICS SECTION ============
    st.markdown("### 📊 Performance Metrics")
    
    # Show current metrics
    if st.button("📈 Show Performance Dashboard"):
        metrics = st.session_state.performance.get_all_metrics()
        
        if metrics['total_samples'] > 0:
            st.markdown(f"""
            <div class="performance-card">
                <h4>🎯 Score Prediction</h4>
                <p><strong>R² Score:</strong> {metrics['r2_score']:.4f}</p>
                <p><strong>RMSE:</strong> {metrics['rmse']:.2f}</p>
                <p><strong>MAE:</strong> {metrics['mae']:.2f}</p>
                <p><strong>Accuracy (10%):</strong> {metrics['accuracy']:.2%}</p>
                
                <h4>🔍 Concept Detection</h4>
                <p><strong>Precision:</strong> {metrics['precision']:.4f}</p>
                <p><strong>Recall:</strong> {metrics['recall']:.4f}</p>
                <p><strong>F1-Score:</strong> {metrics['f1_score']:.4f}</p>
                
                <h4>📊 Dataset</h4>
                <p><strong>Total Samples:</strong> {metrics['total_samples']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if metrics['r2_score'] > 0.7:
                st.success("✅ Strong correlation with human judgment")
            elif metrics['r2_score'] > 0.5:
                st.warning("⚠️ Moderate correlation - room for improvement")
            else:
                st.error("❌ Weak correlation - needs calibration")
                
            if metrics['f1_score'] > 0.7:
                st.success("✅ Excellent concept detection")
            elif metrics['f1_score'] > 0.5:
                st.warning("⚠️ Good concept detection")
            else:
                st.error("❌ Poor concept detection")
        else:
            st.info("No performance data yet. Submit answers to collect data.")
    
    # Toggle data collection mode
    collecting = st.checkbox("📝 Collect Performance Data", value=st.session_state.collecting_data)
    st.session_state.collecting_data = collecting
    
    if collecting:
        st.info("📊 Data collection ON")
        st.markdown("### 🎯 Rate this answer (0-100)")
        human_score = st.slider("Human Score:", 0, 100, st.session_state.human_score)
        st.session_state.human_score = human_score
    
    # Reset metrics button
    if st.button("🔄 Reset Performance Data"):
        st.session_state.performance.reset()
        st.success("Performance data reset!")
    
    st.markdown("---")
    
    # ============ BATCH TESTING SECTION ============
    st.markdown("### 🧪 Batch Testing")
    
    if st.button("🏃 Run Batch Performance Test"):
        with st.spinner("Running batch tests on sample questions..."):
            try:
                tester = BatchTester(modules)
                results_df = tester.run_test()
                
                if not results_df.empty:
                    st.success(f"✅ Completed {len(results_df)} tests!")
                    batch_metrics = tester.get_performance_summary(results_df)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("R² Score", f"{batch_metrics.get('r2_score', 0):.3f}")
                    with col2:
                        st.metric("RMSE", f"{batch_metrics.get('rmse', 0):.2f}")
                    with col3:
                        st.metric("MAE", f"{batch_metrics.get('mae', 0):.2f}")
                    
                    report = tester.generate_report(results_df)
                    st.text(report)
                    
                    if st.button("💾 Save Batch Results"):
                        filename = tester.save_results()
                        if filename:
                            st.success(f"Results saved to {filename}")
                else:
                    st.error("Test failed. Check if dataset/test_cases.json exists")
            except Exception as e:
                st.error(f"Batch test error: {e}")
    
    st.markdown("---")
    
    # Topic tips
    st.markdown("### 💡 Topic Tips")
    
    tips = {
        "HR Interview": """
        **STAR Method:**
        - **S**ituation: Set the context
        - **T**ask: Describe responsibility
        - **A**ction: Explain what you did
        - **R**esult: Share outcomes
        """,
        "Machine Learning": """
        **Key Areas:**
        - Algorithms & Theory
        - Model Evaluation
        - Feature Engineering
        - Real-world Applications
        """,
        "Data Structures": """
        **Focus On:**
        - Time/Space Complexity
        - Use Cases
        - Trade-offs
        - Implementation Details
        """
    }
    
    st.info(tips.get(selected_topic, "Prepare specific examples and practice explaining concepts clearly."))
    
    st.markdown("---")
    
    # Reset button
    if st.button("🔄 Reset Session", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key not in ['use_gemini', 'gemini_model', 'gemini_working', 'performance', 'collecting_data']:
                del st.session_state[key]
        init_session_state()
        st.rerun()

# Define question generation functions
def generate_new_question(force_new=True):
    """Generate a new question using available sources"""
    question_data = None
    
    # Try Gemini first if available and enabled
    if (st.session_state.use_gemini and 
        st.session_state.get('gemini_working', False) and 
        st.session_state.get('ai_enhanced', True) and
        'gemini_question' in modules):
        
        try:
            with st.spinner("🤖 Generating AI question..."):
                question_data = modules['gemini_question'].generate_with_answer(
                    st.session_state.selected_topic,
                    st.session_state.difficulty
                )
                
                if question_data and question_data.get('question'):
                    st.success("✅ AI-generated question created!")
                    question_data['source'] = 'gemini_ai'
                    question_data['difficulty'] = st.session_state.difficulty
                    return question_data
                
                question = modules['gemini_question'].generate_question(
                    st.session_state.selected_topic,
                    st.session_state.difficulty
                )
                
                if question and question.get('question'):
                    st.info("📝 AI question generated")
                    question['source'] = 'gemini_ai'
                    question['difficulty'] = st.session_state.difficulty
                    return question
                        
        except Exception as e:
            st.warning(f"AI generation temporary issue: {str(e)}")
    
    # Fallback to question bank
    with st.spinner("📚 Loading from question bank..."):
        question_data = modules['question_gen'].get_question(
            st.session_state.selected_topic,
            avoid_duplicates=force_new
        )
        question_data['source'] = 'bank'
        st.info("📖 Using question bank")
    
    return question_data

def generate_different_question():
    """Generate a completely different question"""
    
    if st.session_state.get('force_reset', False):
        if 'gemini_question' in modules:
            modules['gemini_question'].used_questions = set()
        st.session_state.force_reset = False
    
    # Try AI first
    if (st.session_state.use_gemini and 
        st.session_state.get('gemini_working', False) and
        st.session_state.get('ai_enhanced', True) and
        'gemini_question' in modules):
        
        try:
            difficulties = ["easy", "medium", "hard", "expert"]
            start_idx = difficulties.index(st.session_state.difficulty)
            
            for i in range(4):
                diff = difficulties[(start_idx + i) % 4]
                
                question_data = modules['gemini_question'].generate_question(
                    st.session_state.selected_topic,
                    diff
                )
                
                if (question_data and question_data.get('question') and 
                    question_data['question'] != st.session_state.current_question):
                    
                    question_data['difficulty'] = diff
                    question_data['source'] = 'gemini_ai'
                    return question_data
            
            if 'gemini_question' in modules:
                modules['gemini_question'].used_questions = set()
            question_data = modules['gemini_question'].generate_question(
                st.session_state.selected_topic,
                st.session_state.difficulty
            )
            if question_data:
                question_data['source'] = 'gemini_ai'
                return question_data
            
        except Exception as e:
            st.warning(f"AI generation issue: {e}")
    
    # Fallback to bank
    question_data = modules['question_gen'].get_question(
        st.session_state.selected_topic,
        avoid_duplicates=True
    )
    question_data['source'] = 'bank'
    return question_data

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 📝 Interview Practice")
    
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    
    with btn_col1:
        if st.button("🎯 New Question", type="primary", use_container_width=True):
            question_data = generate_new_question()
            
            if question_data:
                st.session_state.current_question_data = question_data
                st.session_state.current_question = question_data['question']
                st.session_state.answer_submitted = False
                st.session_state.feedback = None
                st.session_state.gemini_feedback = None
                st.session_state.gemini_analysis = None
                st.session_state.last_answer = ""
                st.session_state.question_generated = True
                st.session_state.follow_up_question = None
                st.rerun()
    
    with btn_col2:
        if st.session_state.question_generated:
            if st.button("🔄 Practice Another", use_container_width=True):
                question_data = None
                attempts = 0
                
                while attempts < 3:
                    question_data = generate_different_question()
                    if question_data and question_data['question'] != st.session_state.current_question:
                        break
                    attempts += 1
                
                if question_data:
                    st.session_state.current_question_data = question_data
                    st.session_state.current_question = question_data['question']
                    st.session_state.answer_submitted = False
                    st.session_state.feedback = None
                    st.session_state.gemini_feedback = None
                    st.session_state.gemini_analysis = None
                    st.session_state.last_answer = ""
                    st.session_state.follow_up_question = None
                    st.rerun()
    
    with btn_col3:
        if (st.session_state.use_gemini and 
            st.session_state.get('gemini_working', False) and
            st.session_state.question_generated and 
            st.session_state.answer_submitted and
            st.session_state.get('ai_enhanced', True)):
            
            if st.button("🔄 Follow-up Question", use_container_width=True):
                with st.spinner("🤔 Generating follow-up..."):
                    follow_up = modules['gemini_question'].generate_follow_up(
                        st.session_state.current_question,
                        st.session_state.last_answer
                    )
                    
                    if follow_up:
                        st.session_state.follow_up_question = follow_up
                        st.session_state.current_question = follow_up
                        st.session_state.current_question_data = {
                            'question': follow_up,
                            'answer': "Answer will be evaluated.",
                            'source': 'gemini_follow_up',
                            'difficulty': st.session_state.difficulty
                        }
                        st.session_state.answer_submitted = False
                        st.session_state.feedback = None
                        st.session_state.gemini_feedback = None
                        st.rerun()
    
    # Display current question
    if st.session_state.current_question:
        meta_col1, meta_col2, meta_col3 = st.columns(3)
        
        with meta_col1:
            source = st.session_state.current_question_data.get('source', 'bank')
            if source == 'gemini_ai':
                st.markdown("🤖 **AI Generated**")
            elif source == 'gemini_follow_up':
                st.markdown("🔄 **Follow-up**")
            else:
                st.markdown("📚 **Question Bank**")
        
        with meta_col2:
            if 'difficulty' in st.session_state.current_question_data:
                diff = st.session_state.current_question_data['difficulty']
                if diff == 'easy':
                    st.markdown("📗 **Easy**")
                elif diff == 'medium':
                    st.markdown("📘 **Medium**")
                elif diff == 'hard':
                    st.markdown("📙 **Hard**")
                else:
                    st.markdown("🔥 **Expert**")
        
        with meta_col3:
            if 'model' in st.session_state.current_question_data:
                st.markdown(f"🔧 **{st.session_state.current_question_data['model']}**")
            else:
                st.markdown(f"📊 **{st.session_state.selected_topic}**")
        
        st.markdown(f"""
        <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #667eea; margin: 1rem 0;">
            <h4 style="margin: 0; color: #2c3e50;">{st.session_state.current_question}</h4>
        </div>
        """, unsafe_allow_html=True)
        
        answer = st.text_area(
            "✍️ Your Answer:",
            height=200,
            placeholder="Type your detailed answer here... Use examples, explain concepts clearly, and structure your response.",
            value=st.session_state.last_answer,
            key="answer_input"
        )
        st.session_state.last_answer = answer
        
        if answer:
            words = len(answer.split())
            chars = len(answer)
            st.caption(f"📝 {words} words | {chars} characters")
        
        col_submit1, col_submit2 = st.columns([3, 1])
        
        with col_submit1:
            if st.button("📤 Submit Answer for Analysis", type="primary", use_container_width=True):
                if not answer or len(answer.split()) < 5:
                    st.error("Please write a detailed answer (at least 5 words).")
                else:
                    try:
                        with st.spinner("🔍 Analyzing your answer..."):
                            progress = st.progress(0)
                            status = st.empty()
                            
                            status.text("📝 Preprocessing text...")
                            progress.progress(20)
                            time.sleep(0.2)
                            
                            processed_user = modules['preprocessor'].preprocess(answer)
                            
                            status.text("🔑 Extracting keywords...")
                            progress.progress(40)
                            time.sleep(0.2)
                            
                            user_keywords = modules['preprocessor'].extract_keywords(answer)
                            
                            status.text("📊 Evaluating answer quality...")
                            progress.progress(60)
                            time.sleep(0.2)
                            
                            evaluation = modules['evaluator'].evaluate(
                                answer,
                                st.session_state.current_question,
                                st.session_state.selected_topic
                            )
                            
                            status.text("💡 Generating feedback...")
                            progress.progress(80)
                            time.sleep(0.2)
                            
                            feedback = modules['feedback_gen'].generate(
                                evaluation,
                                answer,
                                st.session_state.selected_topic,
                                st.session_state.current_question,
                                user_keywords,
                                None
                            )
                            
                            gemini_feedback = None
                            gemini_analysis = None
                            
                            if (st.session_state.use_gemini and 
                                st.session_state.get('gemini_working', False) and
                                st.session_state.get('ai_enhanced', True) and
                                'gemini_feedback' in modules):
                                
                                status.text("🤖 Getting AI insights...")
                                progress.progress(90)
                                
                                ideal_answer = st.session_state.current_question_data.get('answer', '')
                                
                                gemini_feedback = modules['gemini_feedback'].generate_feedback(
                                    st.session_state.current_question,
                                    ideal_answer,
                                    answer,
                                    evaluation['score'],
                                    st.session_state.selected_topic
                                )
                                
                                gemini_analysis = modules['gemini_feedback'].generate_detailed_analysis(
                                    st.session_state.current_question,
                                    answer,
                                    st.session_state.selected_topic
                                )
                            
                            progress.progress(100)
                            status.text("✅ Analysis complete!")
                            time.sleep(0.3)
                            
                            progress.empty()
                            status.empty()
                            
                            st.session_state.feedback = feedback
                            st.session_state.gemini_feedback = gemini_feedback
                            st.session_state.gemini_analysis = gemini_analysis
                            st.session_state.answer_submitted = True
                            st.session_state.total_attempts += 1
                            
                            if st.session_state.collecting_data:
                                ideal_answer = st.session_state.current_question_data.get('answer', '')
                                actual_concepts = modules['preprocessor'].extract_keywords(ideal_answer)
                                
                                st.session_state.performance.add_evaluation(
                                    system_score=evaluation['score'],
                                    human_score=st.session_state.human_score,
                                    system_concepts=user_keywords,
                                    actual_concepts=actual_concepts
                                )
                            
                            st.session_state.history.append({
                                'timestamp': datetime.now().strftime("%H:%M:%S"),
                                'topic': st.session_state.selected_topic,
                                'question': st.session_state.current_question[:50] + "...",
                                'score': feedback['score'],
                                'source': st.session_state.current_question_data.get('source', 'bank'),
                                'words': len(answer.split()),
                                'model': st.session_state.current_question_data.get('model', 'N/A')
                            })
                            
                            st.rerun()
                            
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
        
        with col_submit2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.last_answer = ""
                st.rerun()
    
    else:
        st.info("""
        👆 **Click 'New Question' to start your interview practice!**
        
        **How it works:**
        1. Select a topic from sidebar
        2. Click 'New Question' for AI-generated questions
        3. Type your answer
        4. Get instant feedback and analysis
        """)
        
        with st.expander("📋 Sample Questions by Topic"):
            samples = {
                "HR Interview": [
                    "Tell me about yourself and your background.",
                    "What are your greatest strengths and weaknesses?",
                    "Where do you see yourself in 5 years?"
                ],
                "Machine Learning": [
                    "Explain the bias-variance tradeoff.",
                    "What is overfitting and how to prevent it?",
                    "Compare supervised and unsupervised learning."
                ],
                "Data Structures": [
                    "Explain the difference between array and linked list.",
                    "What is a binary search tree?",
                    "How does quicksort work?"
                ]
            }
            
            for q in samples.get(selected_topic, ["Select a topic to see sample questions"]):
                st.markdown(f"• {q}")

# ============ ANALYSIS SECTION ============
with col2:
    st.markdown("## 📊 Performance Summary")
    
    if st.session_state.feedback:
        fb = st.session_state.feedback
        
        score = fb['score']
        
        if score >= 70:
            st.markdown(f"""
            <div class="score-card-excellent">
                <h1 style="color: white; margin: 0; font-size: 3rem;">{score}%</h1>
                <p style="color: white; margin: 0; font-size: 1.2rem;">🎉 Excellent Performance!</p>
            </div>
            """, unsafe_allow_html=True)
        elif score >= 50:
            st.markdown(f"""
            <div class="score-card-good">
                <h1 style="color: white; margin: 0; font-size: 3rem;">{score}%</h1>
                <p style="color: white; margin: 0; font-size: 1.2rem;">📈 Good! Keep Improving</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="score-card-poor">
                <h1 style="color: white; margin: 0; font-size: 3rem;">{score}%</h1>
                <p style="color: white; margin: 0; font-size: 1.2rem;">📚 Needs Practice</p>
            </div>
            """, unsafe_allow_html=True)
        
        col_lev, col_conf = st.columns(2)
        with col_lev:
            st.metric("🏆 Level", fb['level'])
        with col_conf:
            st.metric("🎯 Confidence", fb['confidence'])
        
        st.markdown("---")
        st.markdown("### 🔍 Key Insights")
        st.markdown(f"📝 **Word Count:** {fb['word_count']} words")
        
        if fb.get('has_examples', False):
            st.success("✅ **Good:** Includes examples")
        else:
            st.warning("⚠️ **Improvement:** Add examples to strengthen your answer")
        
        st.markdown("### 📊 Component Breakdown")
        
        sem_score = fb['detailed_analysis']['semantic_score']
        st.markdown(f"**Semantic Understanding** - {sem_score}%")
        st.progress(sem_score/100)
        
        key_score = fb['detailed_analysis']['keyword_score']
        st.markdown(f"**Keyword Coverage** - {key_score}%")
        st.progress(key_score/100)
        
        struct_score = fb['detailed_analysis']['structure_score']
        st.markdown(f"**Structure Quality** - {struct_score}%")
        st.progress(struct_score/100)
        
        len_score = fb['detailed_analysis']['length_score']
        st.markdown(f"**Answer Length** - {len_score}%")
        st.progress(len_score/100)
        
        st.markdown("---")
        st.markdown("### 💡 Quick Tip")
        
        scores = {
            'Semantic': sem_score,
            'Keyword': key_score,
            'Structure': struct_score,
            'Length': len_score
        }
        lowest = min(scores, key=scores.get)
        
        tips_dict = {
            'Semantic': "🎯 Focus on directly answering the question. Use keywords from the question in your response.",
            'Keyword': "📚 Use more technical terms related to the topic. Review key concepts before answering.",
            'Structure': "📝 Structure your answer with clear introduction, body, and conclusion. Use transition words.",
            'Length': "✏️ Provide more details and examples. Aim for 50-100 words per answer for better scores."
        }
        
        st.info(tips_dict[lowest])
        
        if st.session_state.gemini_feedback:
            with st.expander("🤖 View AI Detailed Analysis"):
                st.markdown(st.session_state.gemini_feedback)
    
    else:
        st.info("👈 **Submit an answer to see your performance analysis**")
        
        st.markdown("---")
        st.markdown("### 📋 What you'll see here:")
        st.markdown("""
        - 🎯 **Your Score** (0-100%) with color-coded feedback
        - 🏆 **Performance Level** (Beginner to Expert)
        - 📊 **Component Breakdown** with progress bars
        - 💡 **Personalized Tips** to improve your weakest area
        - 🤖 **AI Feedback** (when Gemini is enabled)
        """)
        
        st.markdown("---")
        st.markdown("### 💪 Tips for a Good Answer:")
        st.markdown("""
        1. **Be specific** - Use technical terms correctly
        2. **Provide examples** - Real-world applications strengthen answers
        3. **Structure clearly** - Use introduction, body, conclusion
        4. **Be comprehensive** - Cover key concepts thoroughly
        5. **Stay relevant** - Directly address the question
        """)

# Detailed results section
if st.session_state.answer_submitted and st.session_state.feedback:
    st.markdown("---")
    st.markdown("## 📋 Detailed Feedback")
    
    fb = st.session_state.feedback
    
    st.markdown("### Component Scores")
    col_a, col_b, col_c, col_d = st.columns(4)
    
    with col_a:
        st.metric("Semantic Understanding", f"{fb['detailed_analysis']['semantic_score']}%")
    with col_b:
        st.metric("Keyword Coverage", f"{fb['detailed_analysis']['keyword_score']}%")
    with col_c:
        st.metric("Answer Length", f"{fb['detailed_analysis']['length_score']}%")
    with col_d:
        st.metric("Structure Quality", f"{fb['detailed_analysis']['structure_score']}%")
    
    col_x, col_y = st.columns(2)
    
    with col_x:
        st.markdown("### ✅ Strengths")
        strengths = []
        
        if fb.get('used_concepts'):
            for concept in fb['used_concepts'][:3]:
                strengths.append(f"✅ **{concept}**")
        
        if fb.get('has_examples', False):
            strengths.append("✅ **Good use of examples**")
        
        if fb.get('word_count', 0) > 40:
            strengths.append("✅ **Comprehensive answer**")
        
        if fb['detailed_analysis']['structure_score'] > 70:
            strengths.append("✅ **Well-structured response**")
        
        if strengths:
            for s in strengths:
                st.markdown(s)
        else:
            st.markdown("Good attempt! Keep practicing.")
    
    with col_y:
        st.markdown("### 🔧 Areas to Improve")
        improvements = []
        
        if fb.get('missing_concepts'):
            for concept in fb['missing_concepts'][:3]:
                improvements.append(f"⚠️ Add: **{concept}**")
        
        if not fb.get('has_examples', False):
            improvements.append("⚠️ **Include examples**")
        
        if fb.get('word_count', 0) < 30:
            improvements.append("⚠️ **Provide more detail**")
        
        if fb['detailed_analysis']['structure_score'] < 50:
            improvements.append("⚠️ **Improve structure**")
        
        if improvements:
            for imp in improvements:
                st.markdown(imp)
        else:
            st.markdown("✨ Great job! No major issues.")
    
    st.markdown("### 📝 Summary")
    st.info(fb.get('summary', 'No summary available'))
    
    if st.session_state.gemini_feedback:
        st.markdown("### 🤖 Gemini AI Analysis")
        with st.container():
            st.markdown(f"""
            <div class="feedback-box">
                {st.session_state.gemini_feedback}
            </div>
            """, unsafe_allow_html=True)
    
    if st.session_state.gemini_analysis:
        analysis = st.session_state.gemini_analysis
        
        if analysis.get('strengths'):
            with st.expander("✨ AI-Identified Strengths"):
                for s in analysis['strengths']:
                    st.markdown(f"• {s}")
        
        if analysis.get('weaknesses'):
            with st.expander("🎯 AI-Identified Improvements"):
                for w in analysis['weaknesses']:
                    st.markdown(f"• {w}")
        
        if analysis.get('missing_concepts'):
            with st.expander("📚 Key Concepts to Add"):
                for c in analysis['missing_concepts']:
                    st.markdown(f"• **{c}**")
    
    if fb.get('suggestions'):
        st.markdown("### 💡 Specific Suggestions")
        for i, tip in enumerate(fb['suggestions'][:5], 1):
            st.markdown(f"{i}. {tip}")
    
    if st.session_state.current_question_data and st.session_state.current_question_data.get('answer'):
        with st.expander("📚 View Ideal Answer"):
            ideal = st.session_state.current_question_data['answer']
            st.markdown(ideal)
            
            key_terms = modules['preprocessor'].extract_keywords(ideal)
            if key_terms:
                st.markdown("**Key terms to include:**")
                cols = st.columns(3)
                for i, term in enumerate(key_terms[:6]):
                    cols[i % 3].markdown(f"• `{term}`")

# History section
if st.session_state.history:
    st.markdown("---")
    st.markdown("## 📜 Practice History")
    
    history_df = pd.DataFrame(st.session_state.history[-10:])
    
    def color_score(val):
        if val >= 70:
            return 'background-color: #d4edda'
        elif val >= 50:
            return 'background-color: #fff3cd'
        else:
            return 'background-color: #f8d7da'
    
    styled_df = history_df.style.map(color_score, subset=['score'])
    st.dataframe(styled_df, use_container_width=True)
    
    if len(st.session_state.history) >= 3:
        scores = [h['score'] for h in st.session_state.history[-5:]]
        st.markdown("### 📈 Recent Performance Trend")
        
        trend_data = pd.DataFrame({
            'Attempt': range(1, len(scores) + 1),
            'Score': scores
        })
        
        st.line_chart(trend_data.set_index('Attempt'))

# Footer
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("Made with ❤️ for CSE Students")

with col2:
    st.markdown(f"Session: {st.session_state.session_start}")

with col3:
    if st.session_state.total_attempts > 0:
        avg_score = sum([h['score'] for h in st.session_state.history]) / len(st.session_state.history)
        st.markdown(f"Avg Score: {avg_score:.1f}%")

with col4:
    if st.session_state.use_gemini and st.session_state.get('gemini_working', False):
        st.markdown(f"Powered by {st.session_state.gemini_model}")
    else:
        st.markdown("Using Question Bank")

if st.session_state.get('needs_refresh', False):
    st.session_state.needs_refresh = False
    st.rerun()