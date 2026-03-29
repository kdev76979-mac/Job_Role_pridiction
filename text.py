import joblib
import pandas as pd

# 1. Model load karein
model = joblib.load('Job_role.pkl')

# 2. Input data taiyar karein (Niche wale numbers ko aap badal sakte hain)
# Dhyan rahe: 1 matlab Yes, 0 matlab No
data = {
    "cgpa": 8.5,
    "python": 1,
    "sql": 1,
    "ml": 1,
    "web": 0,
    "cloud": 0,
    "kali_linux": 0,
    "communication": 1
}

# Data ko DataFrame mein badlein (kyunki model ne training mein DataFrame dekha tha)
new_student = pd.DataFrame([data])

# 3. Predict karein
prediction = model.predict(new_student)

# 4. Result print karein
print("Predicted Domain:", prediction[0])