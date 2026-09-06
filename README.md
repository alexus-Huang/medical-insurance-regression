# Medical Insurance Cost Prediction

A multiple linear regression project predicting medical insurance charges using scikit-learn, 
with a focus on investigating prediction errors and discovering a meaningful feature interaction.

## Dataset
[Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance) 
(1,338 records) — age, sex, BMI, children, smoker status, and region, predicting insurance charges.

## What's covered
- Exploratory data analysis (scatter plots, box plots, distribution analysis)
- One-hot encoding of categorical variables (sex, smoker, region)
- Multiple linear regression using scikit-learn
- Model evaluation (MSE, MAE, RMSE) on a held-out validation set
- Diagnosing a systematic pattern in prediction errors using a colored actual-vs-predicted plot
- Engineering a bmi × smoker interaction term to capture a real, discovered relationship
- Testing additional candidate interactions (age × smoker, bmi × age) and rejecting them based 
  on evidence
- Final predictions on new, hypothetical inputs, with a visual sweep showing the interaction effect

## Key finding
Smoking status is by far the strongest predictor of insurance charges, but its effect isn't 
independent of BMI. For smokers, each additional BMI point adds ~$1,507 to predicted charges — 
compared to only ~$17 for non-smokers. Adding this interaction term improved validation RMSE 
from $5,861 to $4,639 (~21% improvement).

## Setup
\`\`\`bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
\`\`\`

Open `notebooks/insurance_regression.ipynb` in Jupyter or VS Code.

## What I learned
- A model's coefficients can reveal relationships (like the effect of `children`) that weren't 
  visible in raw scatter plots, since those relationships only emerge once other factors are 
  held constant.
- Plotting actual vs. predicted values, colored by a suspected factor, is a strong way to 
  diagnose *why* a model is making systematic errors, not just *that* it is.
- Interaction terms let a linear model represent one feature's effect changing based on another 
  feature — something a plain additive model can't do on its own.
- Not every interaction is worth keeping — testing and rejecting two additional interaction 
  terms was as valuable as finding the one that worked, and made the final model simpler and 
  more interpretable.