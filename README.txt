NEDBANK TRANSACTION VOLUME FORECASTING CHALLENGE
=================================================

OBJECTIVE
---------
Predict the total number of bank transactions each customer will make
in the next 3-month window (November 2015 - January 2016).

TARGET VARIABLE: next_3m_txn_count (non-negative integer)
EVALUATION METRIC: RMSLE (Root Mean Squared Logarithmic Error)


IMPORTANT: REGISTRATION REQUIRED
---------------------------------
In addition to entering on Zindi, you MUST register at:

    http://register.data.challenge.nedbank.co.za/

This ensures Nedbank can contact you about the in-person finale,
employment opportunities, and event logistics. You will not be
eligible for prizes or the finale without Nedbank registration.


EMPLOYMENT OPPORTUNITY
-----------------------
This is more than a competition. Nedbank values the skills
demonstrated in this challenge, and outstanding performers will
be considered for career opportunities within Nedbank's data
and analytics teams.


FILES IN THIS PACKAGE
---------------------

  Participant files:
    Train.csv                         Training labels (8,360 customers)
    Test.csv                          Test customer IDs (3,584 customers)
    SampleSubmission.csv              Submission template (fill in predictions)
    transactions_features.parquet     18M transaction rows (Dec 2012 - Oct 2015)
    financials_features.parquet       372K financial snapshot rows (Dec 2013 - Oct 2015)
    demographics_clean.parquet        11,944 customer profiles (1 row per customer)
    VariableDefinitions.csv           Data dictionary for all columns
    StarterNotebook.ipynb             Baseline model notebook
    evaluate.py                       Local scoring script
    README.txt                        This file

  Platform-only (NOT for participants):
    PublicReference.csv               Public leaderboard ground truth
    PrivateReference.csv              Private leaderboard ground truth


QUICK START
-----------

  1. Load Train.csv to see training labels (UniqueID + next_3m_txn_count)
  2. Load Test.csv to see which customers you need to predict
  3. Join transactions_features.parquet on UniqueID to get transaction history
  4. Optionally join financials_features.parquet and demographics_clean.parquet
  5. Engineer features, train a model, predict for Test customers
  6. Save predictions as CSV with columns: UniqueID, next_3m_txn_count
  7. Submit to Zindi

  Quick load example (Python):
    import pandas as pd
    train = pd.read_csv('Train.csv')
    test = pd.read_csv('Test.csv')
    txn = pd.read_parquet('transactions_features.parquet')


JOIN KEYS
---------
  - UniqueID: Customer-level join key (present in ALL files)
  - AccountID: Account-level key (in transactions + financials only)
  - A customer may have multiple accounts


DATA NOTES
----------
  - All transaction dates are BEFORE the prediction window (max: Oct 2015)
  - 567 of 11,944 customers have no financials data - handle with left joins
  - BirthDate contains data quality challenges — inspect before using as a feature
  - AnnualGrossIncome is null for ~6% of customers
  - financials AccountID is 100% null for Mortgages rows (join via UniqueID)
  - TransactionAmount is signed: negative = debit, positive = credit
  - The prediction window (Nov-Jan) includes holiday seasonality


LOCAL SCORING
-------------
  python evaluate.py my_submission.csv PublicReference.csv

  (PublicReference.csv is only available to you if you are testing locally
   before platform upload. On the Zindi platform, scoring is automatic.)


LEADERBOARD
-----------
  Public leaderboard:  30% of test set (1,075 customers)
  Private leaderboard: 70% of test set (2,509 customers) - determines final ranking
  Select your best 2 submissions before the deadline.


LICENSE
-------
  CC-BY SA 4.0 (Creative Commons Attribution-ShareAlike 4.0 International)
