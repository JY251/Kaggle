import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# file_path = "data/test.csv"
# df = pd.read_csv(file_path)
# print(df) # debug

train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")
# print(test_data) # debug

combined_data = pd.concat([train_data, test_data], axis=0, ignore_index=True)

numerical_features = combined_data.select_dtypes(include=['int64', 'float64']).columns
categorical_features = combined_data.select_dtypes(include=['object']).columns

# remove "Survived" from numerical_features in order to prevent NaN in test_data to be filled
numerical_features = numerical_features.drop("Survived")

# Imputer: fill missing values
imputer_num = SimpleImputer(strategy="median")
imputer_cat = SimpleImputer(strategy="most_frequent")

# print("combined_data (before imputer):")
# print(combined_data)
# print("end combined_data (before imputer)")

combined_data[numerical_features] = imputer_num.fit_transform(combined_data[numerical_features])
combined_data[categorical_features] = imputer_cat.fit_transform(combined_data[categorical_features])

# print("combined_data(after imputer):")
# print(combined_data)
# print("end combined_data (after imputer)")


label_encoder = LabelEncoder()
combined_data["Sex"] = label_encoder.fit_transform(combined_data["Sex"])
combined_data["Embarked"] = label_encoder.fit_transform(combined_data["Embarked"])

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
target = "Survived"

train_set = combined_data[combined_data["Survived"].notnull()]
test_set = combined_data[combined_data["Survived"].isnull()]

# print("train_set:")
# print(train_set)
# print("end train_set")

# print("test_set:")
# print(test_set)
# print("end test_set")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(train_set[features], train_set[target])

test_set["Survived"] = model.predict(test_set[features])

submission = test_set[["PassengerId", "Survived"]]
submission["PassengerId"] = submission["PassengerId"].astype("int32")
submission["Survived"] = submission["Survived"].astype("int32")
submission.to_csv("data/submission.csv", index=False)