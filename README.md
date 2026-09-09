# Customer Churn Predictor with AI Business Advisor

Customer churn prediction uses historical data and machine learning to forecast which customers are likely to stop using a service. By identifying at-risk accounts early through signals like short tenure, month-to-month contracts, or high monthly charges, businesses can proactively target retention efforts before cancellations occur.

This project goes one step further: alongside the churn prediction, it uses an LLM (Google Gemini) to explain *why* a customer is at risk and suggest concrete retention actions — combining a traditional ML classifier for the prediction with an LLM for the business-facing explanation.

## Problem Statement

Develop a web application that predicts whether a customer is likely to churn based on attributes such as tenure, monthly charges, contract type, and services used, then use an LLM to explain the prediction and suggest retention actions.

## Tech Stack

- **Data & ML:** Python, pandas, scikit-learn (Logistic Regression, Decision Tree, Random Forest)
- **UI:** Streamlit
- **AI Business Advisor:** Google Gemini API (`gemini-2.0-flash`)
- **Dataset:** [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Project Structure

```
customer-churn-predictor/
├── data/                    # raw dataset
├── churn_predictor.ipynb    # exploration, preprocessing, training, evaluation (mirrors the Colab notebook)
├── src/
│   ├── preprocess.py        # converts raw form input into the model's exact feature vector
│   └── llm_advisor.py       # Gemini API call for churn explanation + retention suggestions
├── app/
│   ├── app.py                # Streamlit application
│   └── model/                 # saved model, scaler, and feature schema (.pkl files)
├── .env                      # GEMINI_API_KEY (not committed - see .gitignore)
└── requirements.txt
```

## Model Performance

Three models were trained and compared, with **recall on the churn class prioritized** — a missed churner (false negative) costs the business a lost customer with no retention attempt, which is more expensive than a false positive.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression | 0.739 | 0.505 | 0.783 | 0.614 |
| **Decision Tree (selected)** | 0.738 | 0.504 | **0.818** | 0.624 |
| Random Forest | 0.764 | 0.538 | 0.783 | 0.638 |

**Decision Tree** was selected for its higher recall, catching the most actual churners at an acceptable cost to precision.

## How to Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-predictor.git
cd customer-churn-predictor
pip install -r requirements.txt
```

Create a `.env` file in the repo root with your Gemini API key:
```
GEMINI_API_KEY=your-key-here
```

Then run the app:
```bash
streamlit run app/app.py
```

## Deliverables

- ✅ This GitHub repository (source code + commit history by task)
- ✅ [Google Colab notebook](PASTE_YOUR_COLAB_SHARE_LINK_HERE) — exploration through model export, with outputs
- ✅ Project report — problem statement, methodology, results, screenshots
