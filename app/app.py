

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# =====================================================
# 1. LOAD TRAINED ML MODEL
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "placement_model.pkl"

models = joblib.load(MODEL_PATH)

# Select Logistic Regression model
model = models["Logistic Regression"]


# =====================================================
# 2. SKILL RECOMMENDATION FUNCTION
# =====================================================

def generate_recommendations(student_data):

    skill_recommendations = {

        "programming_score": (
            "Programming",
            "Practice Python fundamentals, loops, functions, "
            "lists, dictionaries and problem-solving."
        ),

        "technical_score": (
            "Technical Knowledge",
            "Revise core computer science concepts and "
            "prepare technical interview questions."
        ),

        "communication_score": (
            "Communication",
            "Practice English speaking, mock interviews "
            "and explaining technical concepts."
        ),

        "aptitude_score": (
            "Aptitude",
            "Practice quantitative aptitude, logical reasoning "
            "and verbal ability."
        ),

        "problem_solving_score": (
            "Problem Solving",
            "Solve logical problems and coding challenges regularly."
        ),

        "coding_score": (
            "Coding",
            "Practice arrays, strings, searching, sorting "
            "and programming challenges."
        ),

        "sql_score": (
            "SQL",
            "Learn joins, subqueries, grouping, aggregate functions "
            "and database queries."
        ),

        "domain_knowledge_score": (
            "Domain Knowledge",
            "Strengthen AI, ML, statistics and data science concepts."
        ),

        "interview_score": (
            "Interview Preparation",
            "Attend mock interviews and practice explaining "
            "your projects and technical decisions."
        ),

        "resume_score": (
            "Resume Building",
            "Improve your resume with relevant projects, "
            "technical skills and internship experience."
        ),

        "python_skill": (
            "Python",
            "Build Python projects and practice NumPy, Pandas "
            "and scikit-learn."
        ),

        "java_skill": (
            "Java",
            "Practice OOP concepts, collections and Java programming."
        ),

        "web_development_skill": (
            "Web Development",
            "Build responsive applications using HTML, CSS, "
            "JavaScript and a frontend framework."
        ),

        "machine_learning_skill": (
            "Machine Learning",
            "Study supervised learning, model evaluation, "
            "feature engineering and practical ML projects."
        ),

        "cloud_skill": (
            "Cloud Computing",
            "Learn cloud fundamentals, deployment and basic "
            "cloud services."
        )
    }

    recommendations = []

    for feature, (skill, advice) in skill_recommendations.items():

        score = student_data[feature]

        if score < 50:

            recommendations.append({
                "skill": skill,
                "score": score,
                "level": "Needs Improvement",
                "priority": "High",
                "advice": advice
            })

        elif score < 70:

            recommendations.append({
                "skill": skill,
                "score": score,
                "level": "Intermediate",
                "priority": "Medium",
                "advice": advice
            })

    # Show lowest-scoring skills first
    recommendations.sort(key=lambda item: item["score"])

    return recommendations


# =====================================================
# 3. STREAMLIT PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Student Placement Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 AI-Powered Student Placement Prediction System")

st.write(
    "Predict student placement outcomes and identify "
    "skills that need improvement."
)

st.divider()


# =====================================================
# 4. STUDENT INPUT FORM
# =====================================================

with st.form("placement_form"):

    st.header("Student Information")

    st.subheader("1. Academic Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        cgpa = st.number_input(
            "CGPA", 0.0, 10.0, 7.0, step=0.1
        )

        tenth = st.number_input(
            "10th Percentage", 0.0, 100.0, 75.0
        )

    with col2:
        twelfth = st.number_input(
            "12th Percentage", 0.0, 100.0, 75.0
        )

        backlogs = st.number_input(
            "Backlog Count", 0, 20, 0
        )

    with col3:
        attendance = st.number_input(
            "Attendance Percentage", 0.0, 100.0, 85.0
        )

        projects = st.number_input(
            "Projects Count", 0, 50, 2
        )

    st.divider()

    # -------------------------------------------------
    # ASSESSMENT SCORES
    # -------------------------------------------------

    st.subheader("2. Assessment Scores")

    col1, col2, col3 = st.columns(3)

    with col1:

        programming = st.slider(
            "Programming Score", 0, 100, 70
        )

        technical = st.slider(
            "Technical Score", 0, 100, 70
        )

        communication = st.slider(
            "Communication Score", 0, 100, 70
        )

        aptitude = st.slider(
            "Aptitude Score", 0, 100, 70
        )

    with col2:

        problem_solving = st.slider(
            "Problem Solving Score", 0, 100, 70
        )

        coding = st.slider(
            "Coding Score", 0, 100, 70
        )

        sql = st.slider(
            "SQL Score", 0, 100, 70
        )

        domain = st.slider(
            "Domain Knowledge Score", 0, 100, 70
        )

    with col3:

        interview = st.slider(
            "Interview Score", 0, 100, 70
        )

        resume = st.slider(
            "Resume Score", 0, 100, 70
        )

        teamwork = st.slider(
            "Teamwork Score", 0, 100, 70
        )

        adaptability = st.slider(
            "Adaptability Score", 0, 100, 70
        )

    leadership = st.slider(
        "Leadership Score", 0, 100, 70
    )

    st.divider()

    # -------------------------------------------------
    # EXPERIENCE AND ACTIVITIES
    # -------------------------------------------------

    st.subheader("3. Experience and Activities")

    col1, col2, col3 = st.columns(3)

    with col1:

        internships = st.number_input(
            "Internship Count", 0, 20, 1
        )

        certifications = st.number_input(
            "Certifications Count", 0, 50, 2
        )

    with col2:

        hackathons = st.number_input(
            "Hackathons Participated", 0, 50, 1
        )

        coding_problems = st.number_input(
            "Coding Problems Solved", 0, 10000, 50
        )

    with col3:

        extracurricular = st.selectbox(
            "Extracurricular Participation",
            ["Low", "Moderate", "High", "Unknown"]
        )

    st.divider()

    # -------------------------------------------------
    # TECHNICAL SKILLS
    # -------------------------------------------------

    st.subheader("4. Technical Skills")

    col1, col2, col3 = st.columns(3)

    with col1:

        python_skill = st.slider(
            "Python Skill", 0, 100, 70
        )

        java_skill = st.slider(
            "Java Skill", 0, 100, 60
        )

    with col2:

        web_skill = st.slider(
            "Web Development Skill", 0, 100, 60
        )

        ml_skill = st.slider(
            "Machine Learning Skill", 0, 100, 50
        )

    with col3:

        cloud_skill = st.slider(
            "Cloud Skill", 0, 100, 40
        )

    st.divider()

    submitted = st.form_submit_button(
        "🚀 Predict Placement",
        use_container_width=True
    )


# =====================================================
# 5. PREDICTION AND RECOMMENDATION
# =====================================================

if submitted:

    # Create input DataFrame with all 29 features

    input_data = pd.DataFrame([{

        "cgpa": cgpa,
        "tenth_percentage": tenth,
        "twelfth_percentage": twelfth,

        "programming_score": programming,
        "technical_score": technical,
        "communication_score": communication,
        "aptitude_score": aptitude,
        "problem_solving_score": problem_solving,
        "coding_score": coding,
        "sql_score": sql,
        "domain_knowledge_score": domain,
        "interview_score": interview,
        "resume_score": resume,

        "projects_count": projects,
        "internship_count": internships,
        "certifications_count": certifications,
        "backlog_count": backlogs,
        "hackathons_participated": hackathons,
        "coding_problems_solved": coding_problems,
        "attendance_percentage": attendance,

        "extracurricular_participation": extracurricular,

        "teamwork_score": teamwork,
        "adaptability_score": adaptability,
        "leadership_score": leadership,

        "python_skill": python_skill,
        "java_skill": java_skill,
        "web_development_skill": web_skill,
        "machine_learning_skill": ml_skill,
        "cloud_skill": cloud_skill

    }])

    # Match the model's expected feature order
    input_data = input_data[model.feature_names_in_]


    # =================================================
    # 6. PLACEMENT PREDICTION
    # =================================================

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    placed_probability = probabilities[1] * 100

    st.divider()

    st.header("📊 Placement Prediction Results")

    col1, col2 = st.columns(2)

    with col1:

        if prediction == 1:

            st.success("Predicted Outcome: Placed")

        else:

            st.warning("Predicted Outcome: Not Placed")

    with col2:

        st.metric(
            "Estimated Placement Probability",
            f"{placed_probability:.2f}%"
        )

    st.progress(float(probabilities[1]))

    st.caption(
        "This is an experimental estimate from a model "
        "trained on synthetic data. It is not a real-world "
        "placement guarantee."
    )


    # =================================================
    # 7. SKILL RECOMMENDATION SYSTEM
    # =================================================

    st.divider()

    st.header("📚 Personalized Skill Recommendations")

    recommendations = generate_recommendations(
        input_data.iloc[0].to_dict()
    )

    if recommendations:

        st.write(
            "The following skills may benefit from additional practice:"
        )

        for item in recommendations:

            with st.expander(
                f"{item['skill']} | "
                f"{item['level']} | "
                f"Score: {item['score']}/100"
            ):

                if item["priority"] == "High":

                    st.error("High Improvement Priority")

                else:

                    st.warning("Medium Improvement Priority")

                st.write("**Recommended Action:**")

                st.write(item["advice"])

    else:

        st.success(
            "All assessed skills are at or above "
            "our current recommendation threshold!"
        )


    # =================================================
    # 8. SKILL SCORE SUMMARY
    # =================================================

    st.divider()

    st.subheader("📈 Skill Performance Summary")

    skill_scores = {
        "Programming": programming,
        "Technical": technical,
        "Communication": communication,
        "Aptitude": aptitude,
        "Problem Solving": problem_solving,
        "Coding": coding,
        "SQL": sql,
        "Domain Knowledge": domain,
        "Interview": interview,
        "Resume": resume,
        "Python": python_skill,
        "Java": java_skill,
        "Web Development": web_skill,
        "Machine Learning": ml_skill,
        "Cloud": cloud_skill
    }

    skill_df = pd.DataFrame(
        list(skill_scores.items()),
        columns=["Skill", "Score"]
    )

    st.bar_chart(
        skill_df.set_index("Skill")
    )

    st.caption(
        "Recommendation thresholds are initial project-defined "
        "rules and should be validated before real-world use."
    )
