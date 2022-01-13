# Drift Detection for Categorical data

Data drift is one of the top reasons model accuracy degrades over time. For machine learning models, data drift is the change in model input data that leads to model performance degradation. Monitoring data drift helps detect these model performance issues.

Causes of data drift include:

Upstream process changes, such as a sensor being replaced that changes the units of measurement from inches to centimeters.
Data quality issues, such as a broken sensor always reading 0.
Natural drift in the data, such as mean temperature changing with the seasons.
Change in relation between features, or covariate shift.

In this project, I have used "Chebyshev distance" as a measure of drift between a reference dataset and novel dataset.

![image](https://user-images.githubusercontent.com/68592826/149387737-cc339e0f-ef90-463a-b12c-cd5e52d2f2ac.png)

All the libraries utilized are mentioned in requirements.
