
# ML vs Chain Ladder: Claims Forecasting

This is a fun private project exploring how simple machine learning models compare to classical Chain Ladder methods for forecasting ultimate insurance claims.

The project generates synthetic claims data, trains a multi-layer perceptron (MLP) model on partial development data, and compares its predictions to Chain Ladder estimates.

> **Note**: This is a work in progress and has not yet undergone a thorough theoretical audit.

## 📁 Structure

- `src/generate_data.py`: Creates the synthetic claims dataset.
- `src/prepare_data.py`: Prepares training data for the ML model.
- `src/train_model.py`: Trains an MLP to predict residual-to-ultimate claims.
- `src/predict_model.py`: Evaluates model performance on a test set.
- `src/chain_ladder.py`: Computes Chain Ladder forecasts.
- `src/plot_results.py`: Visualizes ML vs. Chain Ladder predictions.

## 🚀 Deployment

Deployment to AWS (e.g., for visualization or inference API) is currently in progress.

## 🔮 Future Scope

- Creation of a relational database for IFRS accounting and SST/Solvency II balance sheets
- Generation of payment patterns for liquidity, capital, and financial planning
- Automation of solvency calculations and reporting
- AI-driven analytics across actuarial and finance use cases

