# Project Details
- **Authors:** Yuqing Liu(QF), Filipe Alexandre De Sousa Correia(QF),  Niyetali Kaliyev(QF),  Ilya Shnip(QF)
- **University:** University of Warsaw
- **Project Title:**: Market Risk Modelling for S&P 500, Bitcoin, and 1-Year U.S. Treasury Bond using VaR metrics

# Overview
This project evaluates the market risk of three key financial assets with different risk profiles: the S&P 500, Bitcoin, and the 1-Year U.S. Treasury Bond. The evaluation uses various Value at Risk (VaR) metrics over a decade-long period from May 1, 2014, to May 1, 2024.

# Main Components:
- Historical VaR
- Parametric VaR
- Monte Carlo VaR
- Expected Shortfall (ES)/Conditional VaR

# Dataset Description
The dataset includes 52 columns with financial and macroeconomic variables, focusing primarily on the S&P 500, Bitcoin, and the 1-Year U.S. Treasury Bond. The dataset starts in May 1, 2014 to May 1, 2024.

# Project Setup
The project was created using Anaconda, therefore it is advised to use Anaconda to run this project. 

1. Clone the Repository to your local machine.

`git clone https://github.com/IlyaSshnip/RR_project.git
cd RR_project`

2. Create a virtual environment to manage your dependencies:

`python3 -m venv env`
source env/bin/activate  # On Windows use `env\Scripts\activate`

3. Install Dependencies

`pip install pandas numpy matplotlib seaborn statsmodels pandas_datareader scipy`

## Run the Project:
First download the scripts. To execute the code separately, run the following scripts:

cd "Downloads/Users/path.."

4. Data Preparation
`python data_preparation.py`

5. EDA
`python EDA.py`

6. Feature Engineering
`python feature_engineering.py`

7. VaR Models - run this script:
`python var_models.py`

**To execute the full code**, run the following script:
`python full_code.py`

# Additional Information

### Folders:
- **scripts**: contains the scripts you will execute.
- **dataset**: contains the historical data (10y) for S&P500, Bitcoin and 1Y US Treasury Bond.
- **history**: contains papers and attempts using garch models (garch, egarch, tgarch, aparch), which were excluded from the main code.

### Pre-Link
https://www.canva.com/design/DAGDKXHzg_g/3t_lt50fitsJuiLsjyIewQ/edit

### Results until the meeting held on 24.05.2024
1) Loading dependencies, and data preparation.
2) Combined Literature Review;
   
### Results of the meeting held on 24.05.2024
2) Adding Bitcoin and 1-Year U.S. Treasury Bond as alternative assets (3 assets with different risk profiles);
3) Upload new data to the repository;
4) **Discuss the next steps:** exploratory data analysis (EDA); feature engineering.
5) https://colab.research.google.com/drive/1BjmCZfZejh0ytwZlnS-_1jRJpCbbx4lp?usp=sharing

### Results of the meeting held on 30.05.2024
1) **Review:** Previous steps, including Data Preparation, EDA, and feature engineering;
2) Add the some points in Statistical Analysis (part of EDA);
4) **Discuss modeling:** Methods used for volatility modeling - GARCH, EGARCH, apARCH, TGARCH;
5) **Discuss VaR metrics and validation:** MC VaR, HVaR, Parametric, CVaR.

### Results of the meeting held on 09.06.2024
**Discuss final steps:**
1) Final file, with some adjustments :)
2) Uploading the presentation.
