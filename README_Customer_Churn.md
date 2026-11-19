# Customer Churn Prediction

**Notebook:** `No_Churn_FirstPerson.ipynb` · **Stack:** Python, pandas, scikit-learn, matplotlib, seaborn

## Goal
Predict which telecom customers will churn (leave) so retention effort can be targeted.

## Data
1,000 customers with demographics, usage (calls, SMS, data) and a churn flag. **17.9% churned** (179 customers), so the classes are imbalanced.

## What I did
1. **Checked and cleaned the data:** about 3% of rows per usage column had negative values (invalid), so I fixed them. I did not trust the `state`/`city` columns (cities appear under many states) and dropped `state`.
2. **EDA:** no category or group showed a churn rate significantly different from the overall 17.9%. Only calls and SMS showed a faint U-shape.
3. **Leak-free modelling:** stratified 80/20 split, scaling inside a pipeline, stratified 5-fold CV.
4. **Compared** a dummy baseline, Logistic Regression, Random Forest and Gradient Boosting, using recall, precision, F1 and ROC-AUC (not accuracy, since a dummy "always stay" gets 82%).
5. **Handled imbalance two ways:** class weights and SMOTE (applied only inside CV folds, written in scikit-learn).
6. **Stress-tested the result:** shuffled-label runs, permutation importance on the test set, and a random-noise column as a feature.

## Results (test set: 200 customers, 36 churners)
| Model | ROC-AUC | Recall | Precision |
|---|---|---|---|
| Dummy (always stay) | 0.500 | 0.00 | 0.00 |
| Logistic Regression (balanced) | 0.536 | 0.64 | 0.22 |
| Random Forest | 0.513 | 0.03 | 0.50 |
| Gradient Boosting | 0.549 | 0.00 | 0.00 |

## Conclusion
ROC-AUC stays at 0.50–0.55 for every model and SMOTE did not help. A random-noise column ranked above most real features, and no column was reliably important. **These attributes carry little usable signal for churn**, so I would not deploy a model on this data. If one had to be used, Logistic Regression with class weights is the simplest and most explainable, accepting many false alarms.

## Limitations
Only 36 churners in the test set, so every test metric is noisy. The `date` and location columns are unreliable.
