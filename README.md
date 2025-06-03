Machine Learning vs Chain Ladder: Claims Forecasting
This is a hands-on project exploring how simple machine learning models compare to classical Chain Ladder methods for forecasting ultimate insurance claims.

The project generates synthetic claims triangle data, trains a multi-layer perceptron (MLP) on partial development patterns, and compares ML predictions with traditional Chain Ladder forecasts. The system is fully deployed on AWS, supporting both interactive visualization and real-time prediction via API.

Note: This is a proof of concept using synthetic data. It is not a production-ready actuarial model and has not undergone theoretical audit.

📁 Project Structure
src/generate_data.py: Creates synthetic individual contract-level claims data.

src/prepare_data.py: Prepares train/test splits and model input formats.

src/train_model.py: Trains an MLP model to predict residuals to ultimate.

src/predict_model.py: Evaluates model predictions against true ultimates.

src/chain_ladder.py: Computes classic Chain Ladder development estimates.

src/plot_results.py: Visualizes ML vs. Chain Ladder results.

app.py: Streamlit dashboard for local or EC2 deployment.

☁️ AWS Cloud Deployment
The project includes a full AWS deployment for both visualization and inference:

Architecture:
EC2 – hosts the Streamlit web app (live demo)

S3 – stores model artifacts and synthetic data

Lambda + Docker + ECR – containerized model inference endpoint

API Gateway – provides public HTTP access to the model

IAM + VPC – secure networking and access control

CloudWatch – logs and monitors Lambda + EC2 activity

Live Resources:
🔗 GitHub: github.com/ihanprasetyo/ml-vs-chainladder

🌐 Web App: Streamlit (EC2)

🧠 API Endpoint: Public prediction API via AWS API Gateway

🔮 Future Scope
Integrate a PostgreSQL database for IFRS/Solvency II reporting

Generate realistic payment and reporting patterns

Build dashboards for solvency monitoring and planning

Add support for more complex models (e.g., attention-based)

Extend to real-world datasets (when available)

Fine-tune cost and security architecture
