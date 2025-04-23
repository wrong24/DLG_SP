# NSE Stock Analysis Project

This project aims to analyze stock data from the National Stock Exchange (NSE) of India using advanced machine learning techniques, including Temporal Graph Attention Networks (TGAT) and Hypergraph Neural Networks (HGNN). The analysis focuses on identifying the best-performing sectors and stocks based on historical data and temporal relationships.

## Project Structure

```
nse-stock-analysis
├── src
│   ├── data
│   │   ├── read_nse_data.py      # Script to read and store NSE stock data
│   │   └── preprocess.py          # Script for data preprocessing and feature engineering
│   ├── models
│   │   ├── tgat.py                # Implementation of the Temporal Graph Attention Network
│   │   └── hgnn.py                # Implementation of the Hypergraph Neural Network
│   ├── ensemble
│   │   └── predict.py             # Script to combine predictions from TGAT and HGNN
│   └── utils
│       └── __init__.py            # Utility functions and classes
├── requirements.txt                # List of project dependencies
└── README.md                       # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd nse-stock-analysis
   ```

2. **Install the required dependencies:**
   Create a virtual environment and install the dependencies listed in `requirements.txt`:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. **Reading NSE Data:**
   Run the `read_nse_data.py` script to fetch and store the NSE stock data in a CSV file:
   ```
   python src/data/read_nse_data.py
   ```

2. **Preprocessing Data:**
   After fetching the data, preprocess it using the `preprocess.py` script:
   ```
   python src/data/preprocess.py
   ```

3. **Training Models:**
   Train the TGAT and HGNN models by executing their respective scripts:
   ```
   python src/models/tgat.py
   python src/models/hgnn.py
   ```

4. **Making Predictions:**
   Finally, use the `predict.py` script to generate predictions for the best-performing sectors and stocks:
   ```
   python src/ensemble/predict.py
   ```

## Overview

This project leverages historical stock data to build predictive models that can identify trends and potential growth in various sectors. By utilizing TGAT and HGNN, the project aims to provide a comprehensive analysis of the stock market, helping investors make informed decisions.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.