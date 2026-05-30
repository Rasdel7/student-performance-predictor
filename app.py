import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
    train_test_split, cross_val_score)
from sklearn.metrics import (
    mean_absolute_error, r2_score)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Predictor")
st.markdown("Predict exam scores based on study "
            "habits, attendance and lifestyle factors.")
st.markdown("---")

# Generate dataset
@st.cache_data
def generate_data():
    np.random.seed(42)
    n = 1000

    study_hours    = np.random.uniform(0, 10, n)
    attendance     = np.random.uniform(40, 100, n)
    sleep_hours    = np.random.uniform(4, 10, n)
    prev_scores    = np.random.uniform(30, 100, n)
    parent_edu     = np.random.choice(
        ['No Education', 'High School',
         'Graduate', 'Post Graduate'],
        n, p=[0.1, 0.3, 0.4, 0.2])
    internet       = np.random.choice(
        ['Yes', 'No'], n,
        p=[0.75, 0.25])
    extra_classes  = np.random.choice(
        ['Yes', 'No'], n,
        p=[0.45, 0.55])
    motivation     = np.random.choice(
        ['Low', 'Medium', 'High'], n,
        p=[0.25, 0.45, 0.30])
    gender         = np.random.choice(
        ['Male', 'Female'], n)
    part_time_job  = np.random.choice(
        ['Yes', 'No'], n,
        p=[0.25, 0.75])
    extracurricular = np.random.choice(
        ['Yes', 'No'], n,
        p=[0.40, 0.60])
    stress_level   = np.random.choice(
        ['Low', 'Medium', 'High'], n,
        p=[0.25, 0.45, 0.30])

    # Score calculation
    score = (
        study_hours * 4.5 +
        attendance * 0.25 +
        prev_scores * 0.35 +
        (sleep_hours - 4) * 1.5
    )

    # Adjustments
    edu_map = {
        'No Education': -5,
        'High School':  0,
        'Graduate':     5,
        'Post Graduate':8
    }
    score += np.array([
        edu_map[e] for e in parent_edu])
    score += np.where(
        internet == 'Yes', 3, -3)
    score += np.where(
        extra_classes == 'Yes', 5, 0)
    mot_map = {
        'Low': -8, 'Medium': 0, 'High': 8}
    score += np.array([
        mot_map[m] for m in motivation])
    score += np.where(
        part_time_job == 'Yes', -4, 0)
    str_map = {
        'Low': 3, 'Medium': 0, 'High': -5}
    score += np.array([
        str_map[s] for s in stress_level])
    score += np.random.normal(0, 5, n)
    score  = np.clip(score, 0, 100)

    return pd.DataFrame({
        'study_hours':     study_hours.round(1),
        'attendance':      attendance.round(1),
        'sleep_hours':     sleep_hours.round(1),
        'prev_scores':     prev_scores.round(1),
        'parent_edu':      parent_edu,
        'internet':        internet,
        'extra_classes':   extra_classes,
        'motivation':      motivation,
        'gender':          gender,
        'part_time_job':   part_time_job,
        'extracurricular': extracurricular,
        'stress_level':    stress_level,
        'score':           score.round(1)
    })

df = generate_data()

# Train models
@st.cache_resource
def train_models(df):
    df_ml = df.copy()
    les   = {}
    cat_cols = ['parent_edu', 'internet',
                'extra_classes', 'motivation',
                'gender', 'part_time_job',
                'extracurricular', 'stress_level']
    for col in cat_cols:
        le          = LabelEncoder()
        df_ml[col]  = le.fit_transform(df_ml[col])
        les[col]    = le

    features = [
        'study_hours', 'attendance',
        'sleep_hours', 'prev_scores',
        'parent_edu', 'internet',
        'extra_classes', 'motivation',
        'gender', 'part_time_job',
        'extracurricular', 'stress_level'
    ]
    X = df_ml[features]
    y = df_ml['score']

    X_train, X_test, y_train, y_test = \
        train_test_split(
            X, y, test_size=0.2,
            random_state=42)

    models = {
        'Gradient Boosting':
            GradientBoostingRegressor(
                n_estimators=150,
                random_state=42),
        'Random Forest':
            RandomForestRegressor(
                n_estimators=100,
                random_state=42),
        'Linear Regression':
            LinearRegression()
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae   = mean_absolute_error(
            y_test, preds)
        r2    = r2_score(y_test, preds)
        results[name] = {
            'model': model,
            'mae':   round(mae, 2),
            'r2':    round(r2, 3)
        }

    best_name = max(
        results,
        key=lambda x: results[x]['r2'])
    return results, les, features, \
           best_name, X_test, y_test

results, les, features, best_name, \
    X_test, y_test = train_models(df)
best_model = results[best_name]['model']

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Predict Score",
    "📊 Data Insights",
    "🏆 Factor Analysis",
    "📈 Model Performance"
])

# Tab 1 — Predict
with tab1:
    st.markdown("### Enter Student Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 📚 Academic")
        study_hours   = st.slider(
            "Daily study hours:",
            0.0, 10.0, 4.0, 0.5)
        attendance    = st.slider(
            "Attendance (%):",
            40.0, 100.0, 75.0, 1.0)
        prev_scores   = st.slider(
            "Previous exam score:",
            30.0, 100.0, 65.0, 1.0)
        extra_classes = st.selectbox(
            "Takes extra classes:",
            ["Yes", "No"])

    with col2:
        st.markdown("#### 🌙 Lifestyle")
        sleep_hours  = st.slider(
            "Sleep hours/night:",
            4.0, 10.0, 7.0, 0.5)
        motivation   = st.selectbox(
            "Motivation level:",
            ["High", "Medium", "Low"])
        stress_level = st.selectbox(
            "Stress level:",
            ["Low", "Medium", "High"])
        part_time    = st.selectbox(
            "Part-time job:",
            ["No", "Yes"])

    with col3:
        st.markdown("#### 👤 Personal")
        gender          = st.selectbox(
            "Gender:",
            ["Male", "Female"])
        parent_edu      = st.selectbox(
            "Parent education:",
            ["Post Graduate", "Graduate",
             "High School", "No Education"])
        internet        = st.selectbox(
            "Internet access:",
            ["Yes", "No"])
        extracurricular = st.selectbox(
            "Extracurricular:",
            ["Yes", "No"])

    if st.button("🔮 Predict Score",
                 type="primary"):
        input_dict = {
            'study_hours':     study_hours,
            'attendance':      attendance,
            'sleep_hours':     sleep_hours,
            'prev_scores':     prev_scores,
            'parent_edu':      les['parent_edu']
                               .transform(
                [parent_edu])[0],
            'internet':        les['internet']
                               .transform(
                [internet])[0],
            'extra_classes':   les['extra_classes']
                               .transform(
                [extra_classes])[0],
            'motivation':      les['motivation']
                               .transform(
                [motivation])[0],
            'gender':          les['gender']
                               .transform(
                [gender])[0],
            'part_time_job':   les['part_time_job']
                               .transform(
                [part_time])[0],
            'extracurricular': les['extracurricular']
                               .transform(
                [extracurricular])[0],
            'stress_level':    les['stress_level']
                               .transform(
                [stress_level])[0]
        }

        input_df = pd.DataFrame(
            [input_dict])[features]
        predicted = best_model.predict(
            input_df)[0]
        predicted = np.clip(predicted, 0, 100)

        st.markdown("---")

        if predicted >= 90:
            grade = "A+ 🌟"
            color = "#27ae60"
            msg   = "Outstanding performance!"
        elif predicted >= 80:
            grade = "A 🎉"
            color = "#2ecc71"
            msg   = "Excellent work!"
        elif predicted >= 70:
            grade = "B+ 👍"
            color = "#f39c12"
            msg   = "Good performance!"
        elif predicted >= 60:
            grade = "B 📚"
            color = "#e67e22"
            msg   = "Keep studying!"
        elif predicted >= 50:
            grade = "C ⚠️"
            color = "#e74c3c"
            msg   = "Needs improvement!"
        else:
            grade = "D 🔴"
            color = "#c0392b"
            msg   = "Critical — study more!"

        st.markdown(
            f"<h2 style='text-align:center;"
            f"color:{color}'>"
            f"Predicted Score: "
            f"{predicted:.1f}/100 — {grade}"
            f"</h2>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<p style='text-align:center;"
            f"color:gray'>{msg}</p>",
            unsafe_allow_html=True
        )

        # Score gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=predicted,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Predicted Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar':  {'color': color},
                'steps': [
                    {'range': [0, 50],
                     'color': "#fadbd8"},
                    {'range': [50, 70],
                     'color': "#fdebd0"},
                    {'range': [70, 85],
                     'color': "#d5f5e3"},
                    {'range': [85, 100],
                     'color': "#a9dfbf"}
                ],
                'threshold': {
                    'line': {'color': "red",
                             'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig,
                        use_container_width=True)

        # Improvement tips
        st.markdown("### 💡 Improvement Tips")
        tips = []
        if study_hours < 4:
            tips.append(
                "📚 Increase study hours to "
                "at least 4-6 hours daily")
        if attendance < 75:
            tips.append(
                "🏫 Improve attendance to "
                "above 75% — it's mandatory")
        if sleep_hours < 7:
            tips.append(
                "😴 Get 7-8 hours of sleep — "
                "it improves memory retention")
        if motivation == "Low":
            tips.append(
                "🔥 Work on motivation — "
                "set small daily goals")
        if stress_level == "High":
            tips.append(
                "🧘 Manage stress — "
                "try meditation or exercise")
        if extra_classes == "No":
            tips.append(
                "📖 Consider extra classes "
                "for difficult subjects")

        if tips:
            for tip in tips:
                st.info(tip)
        else:
            st.success(
                "✅ Great habits! "
                "Keep it up!")

# Tab 2 — Insights
with tab2:
    st.markdown("### 📊 Dataset Insights")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students",   len(df))
    c2.metric("Avg Score",
              f"{df['score'].mean():.1f}")
    c3.metric("Pass Rate",
              f"{(df['score']>=50).mean():.1%}")
    c4.metric("Distinction Rate",
              f"{(df['score']>=75).mean():.1%}")

    col1, col2 = st.columns(2)

    with col1:
        fig2 = px.histogram(
            df, x='score',
            nbins=30,
            title='Score Distribution',
            color_discrete_sequence=['#3498db']
        )
        fig2.add_vline(
            x=50, line_dash="dash",
            line_color="red",
            annotation_text="Pass Mark")
        fig2.add_vline(
            x=75, line_dash="dash",
            line_color="green",
            annotation_text="Distinction")
        fig2.update_layout(
            height=350,
            template='plotly_white'
        )
        st.plotly_chart(fig2,
                        use_container_width=True)

    with col2:
        study_score = df.copy()
        study_score['study_group'] = pd.cut(
            study_score['study_hours'],
            bins=[0, 2, 4, 6, 8, 10],
            labels=['0-2h', '2-4h',
                    '4-6h', '6-8h', '8-10h']
        )
        sg = study_score.groupby(
            'study_group',
            observed=True
        )['score'].mean()

        fig3 = px.bar(
            x=sg.index.astype(str),
            y=sg.values,
            title='Avg Score by Study Hours',
            color=sg.values,
            color_continuous_scale='Greens'
        )
        fig3.update_layout(
            height=350,
            template='plotly_white',
            yaxis_title='Average Score',
            xaxis_title='Daily Study Hours'
        )
        st.plotly_chart(fig3,
                        use_container_width=True)

    # Correlation
    col3, col4 = st.columns(2)

    with col3:
        fig4 = px.scatter(
            df.sample(300),
            x='study_hours',
            y='score',
            color='motivation',
            title='Study Hours vs Score',
            trendline='ols',
            labels={
                'study_hours': 'Study Hours',
                'score': 'Score'
            }
        )
        fig4.update_layout(
            height=350,
            template='plotly_white'
        )
        st.plotly_chart(fig4,
                        use_container_width=True)

    with col4:
        fig5 = px.scatter(
            df.sample(300),
            x='attendance',
            y='score',
            color='stress_level',
            title='Attendance vs Score',
            trendline='ols',
            labels={
                'attendance': 'Attendance %',
                'score': 'Score'
            }
        )
        fig5.update_layout(
            height=350,
            template='plotly_white'
        )
        st.plotly_chart(fig5,
                        use_container_width=True)

# Tab 3 — Factor Analysis
with tab3:
    st.markdown(
        "### 🏆 What Affects Performance?")

    col1, col2 = st.columns(2)

    with col1:
        # Motivation impact
        mot_score = df.groupby(
            'motivation')['score'].mean()
        fig6 = px.bar(
            x=mot_score.index,
            y=mot_score.values,
            title='Avg Score by Motivation',
            color=mot_score.values,
            color_continuous_scale='RdYlGn'
        )
        fig6.update_layout(
            height=300,
            template='plotly_white',
            yaxis_title='Average Score'
        )
        st.plotly_chart(fig6,
                        use_container_width=True)

    with col2:
        # Parent education impact
        par_score = df.groupby(
            'parent_edu')['score'].mean()\
            .sort_values()
        fig7 = px.bar(
            x=par_score.values,
            y=par_score.index,
            orientation='h',
            title='Avg Score by Parent Education',
            color=par_score.values,
            color_continuous_scale='Blues'
        )
        fig7.update_layout(
            height=300,
            template='plotly_white',
            xaxis_title='Average Score'
        )
        st.plotly_chart(fig7,
                        use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Sleep impact
        df['sleep_group'] = pd.cut(
            df['sleep_hours'],
            bins=[3, 5, 6, 7, 8, 10],
            labels=['<5h', '5-6h',
                    '6-7h', '7-8h', '8+h']
        )
        sleep_score = df.groupby(
            'sleep_group',
            observed=True
        )['score'].mean()

        fig8 = px.line(
            x=sleep_score.index.astype(str),
            y=sleep_score.values,
            title='Avg Score by Sleep Hours',
            markers=True
        )
        fig8.update_layout(
            height=300,
            template='plotly_white',
            yaxis_title='Average Score',
            xaxis_title='Sleep Hours'
        )
        st.plotly_chart(fig8,
                        use_container_width=True)

    with col4:
        # Stress level
        str_score = df.groupby(
            'stress_level')['score'].mean()
        fig9 = px.bar(
            x=str_score.index,
            y=str_score.values,
            title='Avg Score by Stress Level',
            color=str_score.values,
            color_continuous_scale='RdYlGn_r'
        )
        fig9.update_layout(
            height=300,
            template='plotly_white',
            yaxis_title='Average Score'
        )
        st.plotly_chart(fig9,
                        use_container_width=True)

    # Feature importance
    st.markdown("#### 🎯 Feature Importance")
    if hasattr(best_model, 'feature_importances_'):
        feat_imp = pd.Series(
            best_model.feature_importances_,
            index=features
        ).sort_values(ascending=True)

        fig10 = px.bar(
            x=feat_imp.values,
            y=feat_imp.index,
            orientation='h',
            title='What Predicts Score Most?',
            color=feat_imp.values,
            color_continuous_scale='Viridis'
        )
        fig10.update_layout(
            height=400,
            template='plotly_white',
            xaxis_title='Importance Score'
        )
        st.plotly_chart(fig10,
                        use_container_width=True)

# Tab 4 — Model Performance
with tab4:
    st.markdown("### 📈 Model Comparison")

    model_data = pd.DataFrame([{
        'Model': name,
        'R² Score': res['r2'],
        'MAE':      res['mae']
    } for name, res in results.items()])

    col1, col2 = st.columns(2)

    with col1:
        fig11 = px.bar(
            model_data,
            x='Model', y='R² Score',
            title='R² Score Comparison',
            color='R² Score',
            color_continuous_scale='Greens'
        )
        fig11.update_layout(
            height=350,
            template='plotly_white',
            yaxis_range=[0, 1]
        )
        st.plotly_chart(fig11,
                        use_container_width=True)

    with col2:
        fig12 = px.bar(
            model_data,
            x='Model', y='MAE',
            title='Mean Absolute Error',
            color='MAE',
            color_continuous_scale='Reds_r'
        )
        fig12.update_layout(
            height=350,
            template='plotly_white'
        )
        st.plotly_chart(fig12,
                        use_container_width=True)

    st.dataframe(model_data,
                 use_container_width=True,
                 hide_index=True)

    # Actual vs predicted
    best_preds = best_model.predict(X_test)
    fig13 = px.scatter(
        x=y_test,
        y=best_preds,
        title=f'Actual vs Predicted — '
              f'{best_name}',
        labels={
            'x': 'Actual Score',
            'y': 'Predicted Score'
        },
        opacity=0.5
    )
    fig13.add_shape(
        type='line',
        x0=0, y0=0, x1=100, y1=100,
        line=dict(color='red', dash='dash')
    )
    fig13.update_layout(
        height=400,
        template='plotly_white'
    )
    st.plotly_chart(fig13,
                    use_container_width=True)

st.markdown("---")
st.markdown(
    "Built by **Jyotiraditya** | "
    "Student Performance Predictor | "
    "1000 students analyzed"
)