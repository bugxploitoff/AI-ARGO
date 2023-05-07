from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the dataset
df = pd.read_csv("./Crop_recommendation.csv.xls")
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["label"])
x = df.drop(["label"], axis=1)
y = df["label"]

# Train the model
rf_clf = RandomForestClassifier()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
rf_clf.fit(x_train, y_train)
print("training done")

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = []
    for i in range(7):
        feature_val = int(request.form[f'param{i+1}'])
        features.append(feature_val)
    prediction = label_encoder.inverse_transform(rf_clf.predict([features]))
    return render_template('predict.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
