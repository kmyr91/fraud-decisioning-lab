# Fraud Decisioning Lab

## Objective
This project simulates a production-style fraud decisioning workflow for online transactions.  
It uses transaction and identity data to train fraud risk models, convert risk scores into approve/review/decline decisions, and evaluate business tradeoffs such as fraud capture, customer friction, review workload, and estimated policy cost.

## Project Summary
Rather than stopping at binary fraud prediction, this project focuses on how fraud models are used in practice. It compares multiple operating policies, backtests their outcomes, and evaluates champion-challenger models in a shadow deployment setting to support promotion decisions.

## Workflow
The project follows this workflow:

1. Load and merge transaction and identity data  
2. Build a baseline fraud modeling dataset with selected features  
3. Train baseline fraud risk models  
4. Convert fraud scores into approve / review / decline decisions  
5. Backtest multiple operating policies  
6. Compare champion and challenger models  
7. Simulate shadow deployment and promotion decision logic  

## Dataset
This project uses transaction and identity data from the IEEE-CIS fraud detection dataset.

Main input files:
- `train_transaction.csv`
- `train_identity.csv`

The transaction table contains the fraud label (`isFraud`) and core transaction features.  
The identity table provides additional device and identity-related signals, which are merged using `TransactionID`.

## Features Used in the Baseline Dataset
The baseline dataset includes a smaller subset of features to create a manageable first modeling table.

Examples of selected features:
- Transaction amount and time
- Product code
- Card-related features
- Address and distance features
- Email domain features
- Device type and device info
- Selected identity fields such as `id_30`, `id_31`, `id_32`, and `id_33`

Missing values were handled explicitly:
- categorical fields were filled with `"missing"`
- numerical fields were filled with `-999`

## Models
Two models were tested:

- **Champion:** Random Forest
- **Challenger:** XGBoost

The project compares both models using:
- ROC AUC
- policy-level fraud capture
- false decline rate
- estimated business cost

## Decision Policies
Fraud scores are translated into three decision buckets:

- **approve**
- **review**
- **decline**

Three operating policies were tested:

- **Conservative**
- **Balanced**
- **Aggressive**

These policies represent different tradeoffs between:
- fraud prevention
- customer friction
- review workload

## Key Results
The Random Forest champion with the **balanced** policy produced the best overall tradeoff.

Main findings:
- approve rate: **93.1%**
- review rate: **5.2%**
- decline rate: **1.7%**
- fraud capture in review + decline: **84.0%**
- false decline rate among good users: **0.07%**
- estimated total cost: **98,400**

The decline bucket had a fraud rate above **96%**, showing that the model concentrated the riskiest transactions effectively.

## Champion-Challenger Outcome
The XGBoost challenger was evaluated against the Random Forest champion using the same policies and cost framework.

Result:
- the challenger underperformed the champion on both ROC AUC and estimated business cost
- the challenger did not meet promotion criteria
- final decision: **keep champion / shadow only**

## Shadow Deployment Logic
A shadow evaluation step was used to simulate promotion criteria for the challenger.

Promotion required:
1. lower estimated total cost than the champion
2. comparable fraud capture in review + decline
3. no material worsening in false decline rate

The challenger failed these criteria, so it remained in shadow mode only.

## Repository Structure
```text
fraud_decisioning_lab/
├── data/
├── outputs/
├── src/
├── README.md
└── requirements.txt