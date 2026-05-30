# Student Performance Predictor 🎓

Predicts exam scores based on study habits,
attendance, sleep and lifestyle factors.

## Live Demo
[Click here](https://student-performance-predictor-7ljm9r4rgredl9kolr5pbk.streamlit.app)

## Features
- Predict score with grade and improvement tips
- Animated gauge chart for score visualization
- Score distribution with pass/distinction marks
- Study hours vs score correlation
- Factor analysis: motivation, sleep, stress
- 3 model comparison: GB, RF, Linear Regression
- Actual vs predicted scatter plot

## Key Findings
- Study hours is the strongest predictor
- Attendance above 75% significantly boosts scores
- 7-8 hours of sleep is optimal
- High stress reduces scores by ~8 points

## Tools Used
- Python, Streamlit, Plotly, Scikit-learn

## How to Run Locally
pip install streamlit pandas numpy plotly scikit-learn
streamlit run app.py
