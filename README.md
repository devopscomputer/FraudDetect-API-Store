# FraudDetect-API

## About the Dataset
This dataset contains credit card transactions made by European cardholders in the year 2023. It comprises over 550,000 records, and the data has been anonymized to protect the cardholders' identities. The primary objective of this dataset is to facilitate the development of fraud detection algorithms and models to identify potentially fraudulent transactions.

[Credit Card Fraud Detection Dataset 2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023/code?datasetId=3752264&sortBy=voteCount)

### Key Features

- **id**: Unique identifier for each transaction
- **V1-V28**: Anonymized features representing various transaction attributes (e.g., time, location, etc.)
- **Amount**: The transaction amount
- **Class**: Binary label indicating whether the transaction is fraudulent (1) or not (0)

---

## Fraud Detection API

This is a FastAPI-based API for fraud detection using a pre-trained machine learning model. It provides endpoints for making predictions based on input data.

### Getting Started

To get started with this API, follow these steps:

1. **Clone this repository to your local machine**:

    ```bash
    git clone git@github.com:ManolisTr/FraudDetect-API.git
    ```

2. **Install the required dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the API server**:

    ```bash
    uvicorn main:app --reload
    ```

4. **Access the API documentation**: Once the server is running, you can access the API documentation at [http://localhost:8000/docs](http://localhost:8000/docs) to test the endpoints using Swagger UI.

---

### Usage

#### Making Predictions

To make predictions, send a POST request to the `/v1/prediction` endpoint with input data in the request body. The input data should be a JSON object containing a list of floats representing the features for prediction.

**Example Request**:

```json
{
    "user_id": 123,
    "amount": 150.75,
    "product_id": 456,
    "transaction_type": "purchase",
    "timestamp": "2023-03-18T12:00:00"
}