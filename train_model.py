import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import psycopg2
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')
sns.set_palette("husl")

df = pd.read_csv('raw_milk_data.csv')

features = ['days_since_start', 'temperature', 'ph_20c', 'density_20c']
target = 'titratable_acidity'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
print(f"Training set rows: {len(X_train)}")
print(f"Test set rows: {len(X_test)}")

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (Test): {mse:.4f}")

y_full_pred = model.predict(X_test)
full_pred_df = pd.DataFrame({
    "rforest_pred": y_full_pred,
    "titratable_acidity": y_test
})
print("\nFull Dataset Predictions vs. Real Values (all 607 rows):")
print(full_pred_df[["rforest_pred", "titratable_acidity"]].to_string(index=False))

joblib.dump(model, 'acidity_model.pkl')

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Real Titratable Acidity')
plt.ylabel('Predicted Titratable Acidity')
plt.title('Real vs. Predicted Titratable Acidity')
plt.tight_layout()
plt.savefig('real_vs_predicted.png')
plt.close()

feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
})
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance in Titratable Acidity Prediction')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

print("\nFeature Importance Values:")
for index, row in feature_importance.iterrows():
    print(f"{row['feature']}: {row['importance']:.4f} (Most important: {row['importance'] == feature_importance['importance'].max()})")
print(f"Most important feature: {feature_importance.loc[feature_importance['importance'].idxmax(), 'feature']}")

print("Graph has been saved as 'feature_importance.png'.")

plt.figure(figsize=(12, 6))
plt.scatter(X_train['days_since_start'], y_train, color='blue', alpha=0.5, label='Training Data (Real)', s=50)

plt.scatter(X_test['days_since_start'], y_test, color='green', alpha=0.5, label='Test Data (Real)', s=50)

plt.scatter(X_test['days_since_start'], y_pred, color='red', alpha=0.5, label='Test Data (Predicted)', marker='^', s=50)

plt.xlabel('Days Since Start')
plt.ylabel('Titratable Acidity')
plt.title('Titratable Acidity Over Time: Training vs Test')
plt.legend()
plt.tight_layout()
plt.savefig('titratable_acidity_over_time.png')
plt.close()
