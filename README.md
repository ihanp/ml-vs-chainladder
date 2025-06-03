
## Machine Learning (Multi-Layer Perceptron) vs Chain Ladder: Claims Forecasting

This is a private proof-of-concept project exploring how simple machine learning models compare to classical Chain Ladder methods for forecasting ultimate insurance claims.

The pipeline generates synthetic claims triangle data, trains a multi-layer perceptron (MLP) model on partial development data, and benchmarks its predictions against Chain Ladder estimates.

The project also includes a full AWS deployment for inference and visualization, integrating:

- 🔁 **Lambda** with a custom Docker image
- 🧠 **Model and scaler loading** from S3
- 🔐 **Secure infrastructure** using IAM and VPC
- 🔍 **Real-time monitoring** via CloudWatch
- 🌐 **Public inference endpoint** via API Gateway
- 🖥️ **Interactive web app** served from EC2 with Streamlit
- 📦 **Containerized model code** with ECR

> **Note**: This project is for educational/demo purposes only and has not been reviewed for actuarial or regulatory rigor.


## 📁 Structure

- `src/generate_data.py`: Creates synthetic insurance claims triangle data.
- `src/prepare_data.py`: Prepares input features and targets for ML training.
- `src/train_model.py`: Trains a multi-layer perceptron (MLP) to predict residual-to-ultimate claims.
- `src/predict_model.py`: Generates ML predictions on the test set for evaluation.
- `src/chain_ladder.py`: Computes classical Chain Ladder forecasts for comparison.
- `src/plot_results.py`: Visualizes ML vs. Chain Ladder performance.
- `app.py`: Streamlit app for interactive forecasting UI hosted on EC2.
- `lambda/lambda_predict.py`: AWS Lambda function for real-time ML predictions via API Gateway.
- `Dockerfile`: Containerizes the Lambda function with required dependencies.
- `requirements.txt`: Python dependencies for both local and EC2 environments.
- `cloudwatch-config.json`: Optional config for streaming EC2 metrics to Amazon CloudWatch.

## 🔮 Future Scope

- Creation of a relational database for IFRS accounting and SST/Solvency II balance sheets  
- Generation of payment patterns for liquidity, capital, and financial planning  
- Automation of solvency calculations and reporting  
- AI-driven analytics across actuarial and finance use cases  
- Integration with industry-standard reserving tools such as ResQ or Prophet  
- Incorporation of advanced ML models (e.g., LSTMs, transformers) for sequential development forecasting  
- Application of Bayesian methods to estimate uncertainty and prediction intervals  
- Expansion to model multiple lines of business and hierarchical reserving structures  
- Implementation of Infrastructure-as-Code (IaC) using Terraform or AWS CDK  
- Deployment of real-time inference using AWS EventBridge or Kinesis  
- Development of explainability tools (e.g., SHAP, LIME) to increase model transparency  
- Storage of triangle history using a data lake with versioning and audit trails  
- Setup of automated model retraining and deployment pipelines (e.g., SageMaker, Step Functions)  
- Extension of the front-end dashboard with user roles and access control  
- Support for interactive scenario testing and stress testing in the web interface  
- Visualization of reserve risk distribution using simulation-based techniques (e.g., bootstrapping, copulas)  
- Alignment with ESG reporting and long-term capital planning frameworks 
