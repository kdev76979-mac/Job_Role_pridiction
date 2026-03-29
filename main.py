import joblib
import pandas as pd
import warnings

# Warnings ko chhupane ke liye
warnings.filterwarnings("ignore")

def predict_job_role():
    try:
        # 1. Model load karein
        model = joblib.load('Job_role.pkl')
        
        print("--- Job Role Suitability Predictor ---")
        print("Please enter the following details:\n")

        # 2. User se Input lena
        cgpa = float(input("Enter your CGPA (e.g., 8.5): "))
        
        print("\nBaki sawalon ke liye: 1 (Yes) ya 0 (No) likhein")
        python = int(input("Do you know Python? (1/0): "))
        sql = int(input("Do you know SQL? (1/0): "))
        ml = int(input("Do you know Machine Learning? (1/0): "))
        web = int(input("Do you know Web Development? (1/0): "))
        cloud = int(input("Do you know Cloud Computing? (1/0): "))
        kali_linux = int(input("Do you know Kali Linux/Cybersecurity? (1/0): "))
        communication = int(input("Are your Communication Skills good? (1/0): "))

        # 3. Data ko sahi format mein taiyar karna
        user_data = {
            "cgpa": cgpa,
            "python": python,
            "sql": sql,
            "ml": ml,
            "web": web,
            "cloud": cloud,
            "kali_linux": kali_linux,
            "communication": communication
        }

        # DataFrame banana (Kyunki model ko columns ke naam chahiye)
        df = pd.DataFrame([user_data])

        # 4. Prediction karna
        prediction = model.predict(df)

        print("\n" + "="*30)
        print(f"RESULT: Based on your skills, the suggested domain is: {prediction[0]}")
        print("="*30)

    except FileNotFoundError:
        print("Error: 'Job_role.pkl' file nahi mili. Check karein ki file sahi folder mein hai.")
    except Exception as e:
        print(f"Kuch galat hua: {e}")

if __name__ == "__main__":
    predict_job_role()