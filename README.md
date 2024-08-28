# Auto Analytics: Advanced Estimation & Deployment 🛠️

This project focuses on using machine learning algorithms to estimate car prices. Various regression algorithms were implemented, including:

- Linear Regression
- Lasso Regression
- Ridge Regression
- Decision Tree
- Random Forest
- XGBoost

Model evaluation, grid-search, and cross-validation were performed, resulting in the following scores:

| Model             | R2    | MAE     | RMSE    | MAPE   |
|-------------------|-------|---------|---------|--------|
| XGBoost           | 0.917 | 1738.874| 2611.473| 0.131  |
| Random Forest     | 0.904 | 1919.038| 2814.804| 0.153  |
| Lasso             | 0.836 | 2474.291| 3670.117| 0.216  | 
| Linear Regression | 0.838 | 2479.831| 3651.842| 0.219  |
| ElasticNet        | 0.836 | 2460.996| 3667.654| 0.212  |
| Decision Tree     | 0.813 | 2720.283| 3918.682| 0.221  |

The final models chosen were Random Forest and XGBoost. Feature importance was determined separately for each model to reduce feature counts. The models were saved using Pickle and converted into a Streamlit file for deployment outside the notebook environment. The Streamlit file was published on both AWS EC2 instances and the Streamlit website, enabling users to make predictions interactively.

## Access Links
- [Streamlit Live](https://autoscout.streamlit.app)
- [AWS EC2]()
- [GitHub Notebook Link](https://github.com/sahinasli/Deployment_Project_Solution)
- [LinkedIn](https://www.linkedin.com/in/sahin-asli/)
