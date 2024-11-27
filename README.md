# Predicting Airbnb Prices

This was a machine learning project aimed at predicting the price of Airbnb listings based on their attributes. The project explores a variety of machine learning techniques, ranging from baseline models to advanced ensembling and boosting approaches. 

## Dataset

The dataset used for this project was sourced from the [Kaggle competition](https://www.kaggle.com/competitions/knn-sp24-sec21-airbnb-prices) and included attributes of Airbnb listings such as:
- **Host characteristics** (e.g., superhost status, host response rate)
- **Property features** (e.g., number of bedrooms, location)
- **Listing details** (e.g., nightly price, availability)

Data was split into training and testing sets (`train_regression.csv` and `test_regression.csv`), with the target variable being the nightly price of a listing.

## Libraries Used

- **Core Libraries**: `pandas`, `numpy`
- **Data Visualization**: `matplotlib`, `seaborn`
- **Machine Learning**: `scikit-learn`, `xgboost`, `lightgbm`, `catboost`

## Modeling Process

### 1. **Data Preprocessing**
   - Addressed missing values and encoded categorical variables.
   - Conducted exploratory data analysis (EDA) to identify key features influencing listing prices.
   - Applied feature scaling and log-transformation to normalize predictors and address skewness.

### 2. **Baseline Models**
   - **K-Nearest Neighbors (KNN)**: Implemented as a baseline model, experimenting with different values of `k`.
   - **Random Forests**: Used for capturing feature interactions and handling high-dimensional data.

### 3. **Boosting Models**
   - **Gradient Boosting**: Established a foundational boosting baseline.
   - **XGBoost**: Used for its speed, scalability, and advanced regularization.
   - **LightGBM**: Optimized for efficiency, especially with large datasets and high-cardinality features.
   - **CatBoost**: Utilized to handle categorical variables directly without extensive preprocessing.

   ### Reflections on Boosting
   - Initial attempts involved hyperparameter tuning for XGBoost, LightGBM, and CatBoost using `RandomizedSearchCV` for its computational efficiency. However, these tuned models failed to reach the desired Kaggle score threshold of 105.
   - A critical improvement came from revisiting the data preprocessing stage. By applying log-transformation to skewed predictors, an untuned CatBoost model outperformed all previous attempts, achieving the best performance.

### 4. **Ensemble Models**
   - Models were ensembled using different approaches, including:
     - **StackingRegressor**: Combined predictions from boosting models.
     - **Variable-Specific Ensembles**: Created models based on different predictor sets (e.g., raw predictors vs. polynomial features with Lasso selection).
   - The ensemble strategy that achieved the best result involved untuned models with carefully log-transformed predictors.

   ### Reflections on Ensembling
   - Over 30 attempts were made to tune hyperparameters and ensemble models. Despite extensive tuning efforts, an untuned CatBoost model, combined with thoughtful data preprocessing, ultimately achieved the best result.
   - Hyperparameter tuning required significant computational resources, with grids running overnight and during long idle periods. The final ensemble, however, required minimal tuning and performed optimally with default settings.

### 5. **Hyperparameter Tuning**
   - **Methods Used**:
     - **GridSearchCV**: Used for smaller grids where exhaustive search was feasible.
     - **RandomizedSearchCV**: Applied for larger parameter spaces due to its efficiency.
   - **Challenges**:
     - Uncertainty about the effectiveness of tuning actions, particularly when performance worsened after adjustments.
     - Long tuning times for boosting models (up to 5 hours per run).
   - **Solution**:
     - Used smaller grids for daytime tuning to iterate quickly and larger grids overnight.
     - Revisited preprocessing to address predictor skewness, improving model performance without requiring extensive tuning.

### 6. **Evaluation Metrics**
   - Metrics such as **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)** were used to compare model performance.

## Key Findings

- **Baseline Models**: Random Forests performed better than KNN due to their ability to model complex interactions.
- **Boosting Insights**:
  - LightGBM trained the fastest and performed competitively.
  - CatBoost excelled in handling categorical variables, achieving the best results when paired with log-transformed predictors.
  - XGBoost provided robust results and useful feature importance insights.
- **Ensembling Success**:
  - The most effective ensemble used untuned boosting models (ironic!) on thoughtfully preprocessed data, outperforming extensively tuned models.

## Next Steps

- Explore additional features influencing Airbnb prices, such as user reviews and seasonal factors.
- Conduct hyperparameter optimization using advanced methods like Bayesian Optimization.
- Extend the analysis to include geographic clustering for location-based insights.

## Acknowledgements

- [Kaggle Competition](https://www.kaggle.com/competitions/knn-sp24-sec21-airbnb-prices)
- [NUStat Course Notes](https://nustat.github.io/STAT303-3-class-notes/)
- Professor Arvind Krishna
