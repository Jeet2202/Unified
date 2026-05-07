# Unified Health & Development Analytics

A data analysis and visualization project that explores life expectancy, mortality rates, and socio-economic indicators across South and East Asian countries from 1990 to 2019. The project includes a Jupyter Notebook for backend data processing and machine learning, and a Streamlit dashboard for interactive exploration.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Technologies](#technologies)
- [License](#license)

---

## Overview

This project analyzes a unified health dataset covering 150+ indicators for countries including India, Pakistan, Bangladesh, Nepal, Bhutan, Sri Lanka, China, Afghanistan, and Myanmar. The analysis covers:

- Life expectancy trends and distributions across countries
- Infant and under-5 mortality rates
- GDP per capita and its correlation with health outcomes
- Diet composition and cereal consumption patterns
- Risk categorization using life expectancy thresholds
- Predictive modeling with Linear Regression, Decision Tree, and Random Forest

The Streamlit frontend provides an interactive dashboard for filtering, comparing, and visualizing these insights without writing code.

---

## Dataset

**File:** `UnifiedDataset.csv`

- **Rows:** 22,050
- **Columns:** 150
- **Time Range:** 1990 - 2019
- **Granularity:** Country, Year, Gender

Key columns include:

| Category | Examples |
|---|---|
| Demographics | Life Expectancy, Birth Rate, Death Rate, Population |
| Mortality | Infant Mortality Rate, Under 5 Mortality Rate, Maternal Mortality Ratio |
| Air Pollution | Death Rate by Stroke, Ischaemic Heart Disease, COPD, Lung Cancers |
| Nutrition | Diet Calories (Fat, Carbohydrates, Protein), Cereal Consumption, Fruit Consumption |
| Socio-economic | GDP per Capita, GNI per Capita, Income per Capita, Government Expenditure |
| Health Services | Doctors, Nurses, Basic Drinking Water, Sanitation, Universal Health Coverage |

---

## Project Structure

```
Unified/
  UnifiedDataset.csv      # Source dataset
  unified.ipynb            # Backend analysis and ML models (Jupyter Notebook)
  app.py                   # Streamlit frontend dashboard
  README.md                # Project documentation
```

---

## Features

### Backend (unified.ipynb)

- Data loading, cleaning, and preprocessing (missing value imputation, duplicate removal)
- Filtering for selected Asian countries
- Exploratory data analysis with matplotlib and seaborn
- Correlation heatmaps
- Life expectancy trend analysis
- Scatter plots and box plots for mortality indicators
- Machine learning models for life expectancy prediction
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
- Model evaluation using MAE, MSE, R-squared
- Confusion matrices for risk categorization (High / Medium / Low)

### Frontend (app.py)

- **Sidebar Filters** — Country selection, year range slider, gender filter
- **KPI Metrics** — Average life expectancy, infant mortality, under-5 mortality, GDP per capita
- **Overview Tab** — Line charts, box plots, scatter plots, bar charts for life expectancy analysis
- **Country Comparison Tab** — Side-by-side metric comparison of two countries with trend lines
- **Radar Chart Tab** — Normalized radar plot comparing two countries across multiple indicators with distinct overlapping color fills
- **ML Predictions Tab** — Train and evaluate three regression models, actual vs predicted scatter, risk categorization histogram

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Steps

1. Clone the repository:

```bash
git clone https://github.com/Jeet2202/Unified.git
cd Unified
```

2. Install dependencies:

```bash
pip install streamlit plotly scikit-learn pandas numpy matplotlib seaborn
```

3. Ensure `UnifiedDataset.csv` is in the project root directory.

---

## Usage

### Run the Streamlit Dashboard

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### Run the Jupyter Notebook

```bash
jupyter notebook unified.ipynb
```

---

## Screenshots

> Screenshots can be added here after running the dashboard.

---

## Technologies

| Tool | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computation |
| Matplotlib | Static visualizations (notebook) |
| Seaborn | Statistical plots (notebook) |
| Plotly | Interactive charts (dashboard) |
| Scikit-learn | Machine learning models |
| Streamlit | Web-based interactive dashboard |
| Jupyter Notebook | Exploratory analysis environment |

---

## License

This project is for educational and academic purposes.
