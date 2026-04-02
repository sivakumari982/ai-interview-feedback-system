import pandas as pd
import os

print("=" * 50)
print("🤖 AI Interview Question Generator")
print("=" * 50)

# Create dataset folder if it doesn't exist
os.makedirs("dataset", exist_ok=True)

# Check existing questions
existing_file = "dataset/questions.csv"
existing_questions = set()

if os.path.exists(existing_file):
    existing = pd.read_csv(existing_file)
    existing_questions = set(existing['question'])
    print(f"📁 Found {len(existing)} existing questions")
else:
    print("📁 No existing questions file found")
    existing = pd.DataFrame()

# Questions to add (100+)
new_questions = []

# Machine Learning (25 questions)
ml_questions = [
    ("What is linear regression?", "Linear regression models the relationship between dependent and independent variables."),
    ("What is logistic regression?", "Logistic regression is used for binary classification problems."),
    ("What are decision trees?", "Decision trees split data based on feature values for prediction."),
    ("What are random forests?", "Random forests combine multiple decision trees for better accuracy."),
    ("What is SVM?", "SVM finds the optimal hyperplane for classification."),
    ("What is k-means clustering?", "K-means groups similar data points into k clusters."),
    ("What is PCA?", "PCA reduces dimensionality while preserving variance."),
    ("What is gradient descent?", "Gradient descent optimizes model parameters."),
    ("What are neural networks?", "Neural networks mimic biological neurons for learning."),
    ("What is backpropagation?", "Backpropagation computes gradients for neural networks."),
    ("What is overfitting?", "Overfitting occurs when models memorize training data."),
    ("What is underfitting?", "Underfitting happens when models are too simple."),
    ("What is cross-validation?", "Cross-validation evaluates model performance on unseen data."),
    ("What is bias?", "Bias is error from wrong assumptions in learning."),
    ("What is variance?", "Variance is error from sensitivity to training data."),
    ("What is regularization?", "Regularization prevents overfitting by adding penalties."),
    ("What is ensemble learning?", "Ensemble learning combines multiple models."),
    ("What is boosting?", "Boosting sequentially improves weak learners."),
    ("What is bagging?", "Bagging trains multiple models on bootstrap samples."),
    ("What is feature scaling?", "Feature scaling normalizes input features."),
    ("What is a confusion matrix?", "Confusion matrix shows prediction results."),
    ("What is precision?", "Precision measures accuracy of positive predictions."),
    ("What is recall?", "Recall measures ability to find positive samples."),
    ("What is F1 score?", "F1 score is harmonic mean of precision and recall."),
    ("What is transfer learning?", "Transfer learning reuses pre-trained models.")
]

for q, a in ml_questions:
    if q not in existing_questions:
        new_questions.append({"topic": "Machine Learning", "question": q, "answer": a})

# Data Structures (25 questions)
ds_questions = [
    ("What is an array?", "Arrays store elements in contiguous memory locations."),
    ("What is a linked list?", "Linked lists store elements in nodes with pointers."),
    ("What is a stack?", "Stacks follow Last-In-First-Out (LIFO) principle."),
    ("What is a queue?", "Queues follow First-In-First-Out (FIFO) principle."),
    ("What is a tree?", "Trees are hierarchical data structures with nodes."),
    ("What is a binary tree?", "Binary trees have at most two children per node."),
    ("What is a binary search tree?", "BST has left smaller, right larger values."),
    ("What is a heap?", "Heaps are complete binary trees with heap property."),
    ("What is a hash table?", "Hash tables map keys to values using hash functions."),
    ("What is a graph?", "Graphs consist of vertices connected by edges."),
    ("What is BFS?", "BFS explores nodes level by level in graphs."),
    ("What is DFS?", "DFS explores as far as possible along branches."),
    ("What is quicksort?", "Quicksort partitions arrays around a pivot."),
    ("What is merge sort?", "Merge sort divides and merges sorted subarrays."),
    ("What is bubble sort?", "Bubble sort repeatedly swaps adjacent elements."),
    ("What is insertion sort?", "Insertion sort builds sorted array one element at a time."),
    ("What is selection sort?", "Selection sort repeatedly selects minimum element."),
    ("What is dynamic programming?", "DP solves problems with overlapping subproblems."),
    ("What is recursion?", "Recursion solves problems by calling itself."),
    ("What is Big O notation?", "Big O describes algorithm time complexity."),
    ("What is a deque?", "Deque allows insertion/deletion at both ends."),
    ("What is a priority queue?", "Priority queue returns highest priority element."),
    ("What is an AVL tree?", "AVL trees are self-balancing binary search trees."),
    ("What is a red-black tree?", "Red-black trees are balanced binary search trees."),
    ("What is Dijkstra's algorithm?", "Dijkstra finds shortest paths in graphs.")
]

for q, a in ds_questions:
    if q not in existing_questions:
        new_questions.append({"topic": "Data Structures", "question": q, "answer": a})

# DBMS (20 questions)
dbms_questions = [
    ("What is a database?", "A database is an organized collection of data."),
    ("What is SQL?", "SQL manages and manipulates relational databases."),
    ("What is normalization?", "Normalization reduces data redundancy."),
    ("What is a primary key?", "Primary keys uniquely identify rows."),
    ("What is a foreign key?", "Foreign keys reference primary keys in other tables."),
    ("What is an index?", "Indexes speed up data retrieval."),
    ("What is a join?", "Joins combine rows from multiple tables."),
    ("What is a transaction?", "Transactions are units of work with ACID properties."),
    ("What is ACID?", "ACID ensures database reliability."),
    ("What is a view?", "Views are virtual tables based on queries."),
    ("What is a stored procedure?", "Stored procedures are precompiled SQL code."),
    ("What is a trigger?", "Triggers execute automatically on table events."),
    ("What is NoSQL?", "NoSQL databases are non-relational."),
    ("What is denormalization?", "Denormalization adds redundancy for performance."),
    ("What is a composite key?", "Composite keys use multiple columns as primary key."),
    ("What is a candidate key?", "Candidate keys can uniquely identify rows."),
    ("What is a super key?", "Super keys are supersets of candidate keys."),
    ("What is a cluster index?", "Cluster index determines physical order of data."),
    ("What is a non-cluster index?", "Non-cluster index points to data location."),
    ("What is database sharding?", "Sharding partitions data across servers.")
]

for q, a in dbms_questions:
    if q not in existing_questions:
        new_questions.append({"topic": "DBMS", "question": q, "answer": a})

# OOP Concepts (20 questions)
oop_questions = [
    ("What is a class?", "Classes are blueprints for creating objects."),
    ("What is an object?", "Objects are instances of classes."),
    ("What is inheritance?", "Inheritance allows classes to inherit properties."),
    ("What is polymorphism?", "Polymorphism allows objects to take multiple forms."),
    ("What is encapsulation?", "Encapsulation bundles data with methods."),
    ("What is abstraction?", "Abstraction hides complex implementation details."),
    ("What is a constructor?", "Constructors initialize new objects."),
    ("What is a destructor?", "Destructors clean up object resources."),
    ("What is method overloading?", "Overloading has same name, different parameters."),
    ("What is method overriding?", "Overriding redefines parent methods in children."),
    ("What is an interface?", "Interfaces define method signatures without implementation."),
    ("What is an abstract class?", "Abstract classes cannot be instantiated."),
    ("What is composition?", "Composition has parts that don't exist without whole."),
    ("What is aggregation?", "Aggregation has parts that can exist independently."),
    ("What is dependency injection?", "DI provides dependencies to objects."),
    ("What is SOLID?", "SOLID is principles for OOP design."),
    ("What is the single responsibility principle?", "Classes should have one reason to change."),
    ("What is the open-closed principle?", "Open for extension, closed for modification."),
    ("What is Liskov substitution?", "Subtypes must be substitutable for base types."),
    ("What is dependency inversion?", "Depend on abstractions, not concretions.")
]

for q, a in oop_questions:
    if q not in existing_questions:
        new_questions.append({"topic": "OOP Concepts", "question": q, "answer": a})

# HR Interview (15 questions)
hr_questions = [
    ("Tell me about yourself.", "I am a computer science student passionate about technology."),
    ("What are your strengths?", "My strengths include problem-solving and teamwork."),
    ("What are your weaknesses?", "I sometimes focus too much on details."),
    ("Why should we hire you?", "I bring relevant skills and strong work ethic."),
    ("Where do you see yourself in 5 years?", "I see myself in a technical leadership role."),
    ("How do you handle pressure?", "I prioritize tasks and communicate clearly."),
    ("Describe a challenge you overcame.", "I overcame a technical challenge through research."),
    ("What is your greatest achievement?", "Leading a team to complete a major project."),
    ("Why do you want this job?", "I am passionate about this company's mission."),
    ("How do you work in a team?", "I collaborate effectively and respect others."),
    ("What is your work style?", "I am organized and results-oriented."),
    ("How do you handle criticism?", "I view criticism as opportunity to improve."),
    ("What motivates you?", "Solving challenging problems motivates me."),
    ("How do you prioritize work?", "I use time management and task prioritization."),
    ("What are your career goals?", "I want to grow into a technical expert role.")
]

for q, a in hr_questions:
    if q not in existing_questions:
        new_questions.append({"topic": "HR Interview", "question": q, "answer": a})

# Save all questions
if new_questions:
    new_df = pd.DataFrame(new_questions)
    
    if not existing.empty:
        final_df = pd.concat([existing, new_df], ignore_index=True)
    else:
        final_df = new_df
    
    final_df.to_csv("dataset/questions.csv", index=False)
    
    print("\n" + "=" * 50)
    print(f"✅ Added {len(new_questions)} new questions!")
    print(f"✅ Total questions now: {len(final_df)}")
    print("=" * 50)
    
    # Show distribution
    print("\n📊 Topic Distribution:")
    for topic in final_df['topic'].unique():
        count = len(final_df[final_df['topic'] == topic])
        print(f"   {topic}: {count} questions")
else:
    print("\n✅ No new questions to add. You already have a good dataset!")

print("\n🎯 You can now run: python modules/evaluator.py")