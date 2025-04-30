from flask import Flask, render_template, request, url_for
import pickle
import numpy as np
import json

app = Flask(__name__)

# Load model + scaler once at startup
with open('../rf_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('../scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

LABELS = {
    'fruits_veggies': 'Fruits & Vegetables Intake',
    'daily_stress': 'Daily Stress Level',
    'places_visited': 'Monthly Hangouts',
    'core_circle': 'Friends Count',
    'supporting_others': 'Support You Provide',
    'social_network': 'Social Network Strength',
    'achievement': 'Achievements/Goals Reached',
    'donation': 'Causes You Donate To',
    'bmi_range': 'BMI Category',
    'todo_completed': 'Tasks Completed Per Week',
    'flow': 'Focus Level',
    'daily_steps': 'Daily Steps',
    'live_vision': 'Clarity of Life Goals',
    'sleep_hours': 'Average Sleep Hours',
    'lost_vacation': 'Unused Vacation Days',
    'daily_shouting': 'Anger Frequency',
    'sufficient_income': 'Comfort with Income',
    'personal_awards': 'Awards Received',
    'time_for_passion': 'Time for Passion/Hobby',
    'weekly_meditation': 'Meditation Sessions per Week',
    'age': 'Age Group',
    'gender': 'Gender',
}

CHOICES = {
    'fruits_veggies': [(0, "<1 serving/day"), (1, "1–2"), (2, "3–4"), (3, "5–6"), (4, "7–8"), (5, "9+")],
    'daily_stress': [(i, f"Level {i}/10") for i in range(1, 11)],
    'places_visited': [
        (0, "No places visited"),
        (1, "1-2 places/month"),
        (2, "3-5 places/month"),
        (3, "6-10 places/month"),
        (4, "10+ places/month")
    ],
    'core_circle': [(i, f"{i} friends") for i in range(1, 11)],
    'supporting_others': [
        (0, "I don’t support anyone actively"),
        (1, "I support a few people"),
        (2, "I support several people"),
        (3, "I support a good number of people"),
        (4, "I actively support a large number of people")
    ],
    'social_network': [(i, f"{i}/10 network") for i in range(0, 11)],
    'achievement': [(i, f"{i} goals") for i in range(0, 11)],
    'donation': [
        (0, "I don’t donate to any causes"),
        (1, "I donate to one cause"),
        (2, "I donate to a few causes"),
        (3, "I donate to several causes"),
        (4, "I donate to many causes")
    ],
    'bmi_range': [('Less than 20', "Underweight"), ('21 to 35', "Normal"), ('36 to 50', "Overweight"),
                  ('51 or more', "Obese")],
    'todo_completed': [(i, f"{i} tasks") for i in range(0, 11)],
    'flow': [
        (0, "I rarely experience deep focus"),
        (1, "I experience deep focus occasionally"),
        (2, "I experience deep focus regularly"),
        (3, "I experience deep focus often"),
        (4, "I’m frequently in a state of deep focus")
    ],
    'daily_steps': [('< 1000', "Very low"), ('1000–4999', "Low"), ('5000–9999', "Medium"), ('10k+', "High")],
    'live_vision': [
        (0, "I don’t have specific life goals"),
        (1, "I have a few goals"),
        (2, "I have several goals"),
        (3, "I have many goals"),
        (4, "I have a large number of life goals")
    ],
    'sleep_hours': [('<5', "Very poor"), ('5–6', "Below avg"), ('6–7', "Avg"), ('7–8', "Good"), ('8+', "Excellent")],
    'lost_vacation': [
        (0, "I used all my vacation days"),
        (1, "I have a few unused vacation days"),
        (2, "I have several unused vacation days"),
        (3, "I have a lot of unused vacation days"),
        (4, "I have a significant number of unused vacation days")
    ],
    'daily_shouting': [('Never', "Never"), ('Rarely', "Rarely"), ('Sometimes', "Sometimes"), ('Often', "Often"),
                       ('Always', "Always")],
    'sufficient_income': [(i, f"{i}/10 comfort") for i in range(0, 11)],
    'personal_awards': [(i, f"{i} awards") for i in range(0, 11)],
    'time_for_passion': [(i, f"{i} hrs/wk") for i in range(0, 11)],
    'weekly_meditation': [(i, f"{i} sessions") for i in range(0, 11)],
    'age': [('<20', "Under 20"), ('20–30', "20-30"), ('30–40', "30-40"), ('40–50', "40-50"), ('50+', "50+")],
    'gender': [('Male', "Male"), ('Female', "Female")],
}

# Interpret continuous score into category + message
MIN_SCORE = 480
MAX_SCORE = 860
STEP = (MAX_SCORE - MIN_SCORE) / 4  # = 95


def interpret(score):
    """
    Categorize a raw score (480–860) into four bands.
    """
    if score < MIN_SCORE + STEP:
        return "Poor", (
            "Your work–life balance is in the lowest quartile. "
            "Consider setting firm boundaries, taking breaks, "
            "and seeking social support."
        )
    if score < MIN_SCORE + 2 * STEP:
        return "Fair", (
            "Your balance is below average. Try scheduling leisure "
            "activities, mindfulness practices, and time-blocking."
        )
    if score < MIN_SCORE + 3 * STEP:
        return "Good", (
            "You’re above average! Maintain healthy habits: "
            "regular exercise, quality sleep, and clear work hours."
        )
    return "Excellent", (
        "You’re in the top quartile! Keep up these great habits "
        "and share your strategies with others."
    )


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/benefits')
def benefits():
    return render_template('benefits.html')


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        vals = []
        for field in CHOICES.keys():
            sel = request.form[field]
            # map back to numeric
            if field == 'daily_steps':
                m = {'< 1000': 0, '1000–4999': 1, '5000–9999': 2, '10k+': 3}
                v = m[sel]
            elif field == 'daily_shouting':
                m = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3, 'Always': 4}
                v = m[sel]
            elif field == 'sleep_hours':
                m = {'<5': 0, '5–6': 1, '6–7': 2, '7–8': 3, '8+': 4}
                v = m[sel]
            elif field == 'age':
                m = {'<20': 0, '20–30': 1, '30–40': 2, '40–50': 3, '50+': 4}
                v = m[sel]
            elif field == 'bmi_range':
                order = [val for val, _ in CHOICES['bmi_range']]
                v = order.index(sel)
            elif field == 'gender':
                v = 0 if sel == 'Male' else 1
            else:
                # numeric scales
                try:
                    v = float(sel)
                except:
                    # fallback: match by description
                    for val, desc in CHOICES[field]:
                        if desc == sel:
                            v = val
                            break
            vals.append(v)

        X = np.array(vals).reshape(1, -1)
        Xs = scaler.transform(X)
        raw = model.predict(Xs)[0]
        cat, msg = interpret(raw)

        # Ensure that the factors are passed as JSON serializable data
        factors = json.dumps(LABELS)

        return render_template('result.html', raw=round(raw, 2), category=cat, message=msg, factors=factors)

    return render_template('prediction.html', choices=CHOICES, labels= LABELS)


if __name__ == '__main__':
    app.run(debug=True)
