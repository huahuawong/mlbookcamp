# 1. What's the version of pipenv you installed?
import pipenv

print(pipenv.__version__)
# 2020.11.15


# 2. Use Pipenv to install Scikit-Learn version 1.0
# What's the first hash for scikit-learn you get in Pipfile.lock?
# "sha256:121f78d6564000dc5e968394f45aac87981fcaaf2be40cfcd8f07b2baa1e1829"

# 3.Unpack the model and make predictions
import pickle

model_file = 'model1.bin'
with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

dict_file = 'dv.bin'
with open(dict_file, 'rb') as f_in:
    dv = pickle.load(f_in)

customer = {"contract": "two_year",
            "tenure": 12,
            "monthlycharges": 19.7}


x = dv.transform([customer])
model.predict_proba(x)[0, 1]
# Churn probability: 0.11549580587832914

