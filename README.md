🔹 CIRCUIT-TIMING-ANALYSIS USING XGBOOST

An AI-powered solution to predict combinational depth in logic circuits using machine learning.

📌 Overview
This project utilizes XGBoost Regression to accurately predict the Logic Depth of digital circuits based on circuit parameters such as:

✔ Fan-In 

✔ Fan-Out

✔ Number of Gates

✔ Path Length


By training on datasets that include Logic Circuit and RTL data, the model helps in optimizing circuit performance and reducing design bottlenecks.

📂 Dataset Information
We use multiple datasets stored in the datasets/ folder:

📌 logic_circuit_dataset.csv (Base dataset)

📌 rtl_dataset1.csv → rtl_dataset7.csv (RTL module datasets)


The combined dataset is used to train the model for improved generalization and accuracy.

⚙️ Model Training & Architecture
Uses XGBoost Regressor with:

✅ n_estimators=200

✅ max_depth=6

✅ learning_rate=0.05

Dataset is split 80% Train / 20% Test.

Model performance is evaluated using:

✔ Mean Absolute Error (MAE)

✔ Mean Squared Error (MSE)

✔ R-squared (R² Score)

Once trained, the model is saved as optimized_timing_model.pkl for real-time predictions.


🚀 How to Run
1️⃣ Install Dependencies

pip install pandas numpy joblib xgboost matplotlib scikit-learn

2️⃣ Run the Prediction Script

python predict_logic_depth.py

➡️ Enter Fan-In, Fan-Out, Number of Gates, and Path Length to get the Predicted Logic Depth.

📊 Model Performance

✅ Achieves high prediction accuracy using real-world circuit data.

✅ Provides insights into circuit complexity for optimization.

✅ Scales efficiently with large datasets.

📈 Visualization
The model generates a scatter plot comparing Actual vs Predicted Logic Depth to validate performance.

![image](https://github.com/user-attachments/assets/20f9a98e-924e-4a2b-a7a9-72155f98e97b)


🛠 Future Enhancements

🔹 Integration with EDA tools for automated optimization.

🔹 Support for more complex circuit structures.

🔹 Hybrid AI models for improved accuracy.

👨‍💻 Contributors

🚀 Developed by Dharshini

