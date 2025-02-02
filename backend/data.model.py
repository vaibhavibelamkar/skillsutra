import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample Course Data (Expanded for Better Recommendations)
courses_data = [
    {"_id": "6761546e074c4ffca32f145f", "courseTitle": "Python Basics", "category": "Python", "description": "Learn Python programming from scratch."},
    {"_id": "67615aef258824372f436ac3", "courseTitle": "Advanced Python", "category": "Python", "description": "Deep dive into Python advanced concepts."},
    {"_id": "676193b4d62c0678058ab26d", "courseTitle": "Javascript Bootcamp", "category": "Javascript", "description": "Become an expert in JavaScript programming."},
    {"_id": "67619536d62c0678058ab288", "courseTitle": "Machine Learning with Python", "category": "Python", "description": "Use Python for machine learning and AI."},
    {"_id": "6761a502bee8d16f562ba4bd", "courseTitle": "MERN Stack Bootcamp", "category": "MERN Stack Development", "description": "Master full-stack development using the MERN stack."},
    {"_id": "6762d58c3144abf41284d7da", "courseTitle": "Java for Web Development", "category": "Fullstack Development", "description": "Java mastery for full-stack web development."},
    {"_id": "6763d4347604a73d7136dd71", "courseTitle": "Python for Data Science", "category": "Python", "description": "Learn data science techniques using Python."}
]

# Convert list to DataFrame
courses = pd.DataFrame(courses_data)

# Function to recommend similar courses within the same category
def recommend_courses(course_title, top_n=3):
    if course_title not in courses["courseTitle"].values:
        return ["Course not found."]
    
    # Find the category of the input course
    course_info = courses[courses["courseTitle"] == course_title].iloc[0]
    category = course_info["category"]

    # Filter courses by the same category
    filtered_courses = courses[courses["category"] == category]

    # If no other courses in the same category, return empty
    if len(filtered_courses) <= 1:
        return ["No similar courses found in this category."]
    
    # Convert text into numerical representation using TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(filtered_courses["description"])

    # Compute cosine similarity between courses
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Find the index of the input course in the filtered list
    idx = filtered_courses[filtered_courses["courseTitle"] == course_title].index[0]

    # Get similarity scores and sort
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    # Get course indices and return course titles
    course_indices = [i[0] for i in sim_scores]
    return filtered_courses["courseTitle"].iloc[course_indices].tolist()

# Example: Recommend courses similar to "Python Basics"
print(recommend_courses("Python Basics"))
