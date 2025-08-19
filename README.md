# Payment Transaction Management Scripts

This repository contains scripts for managing payment transactions in the Paidy system.

## Scripts

1. `get-transactions-that-can-be-completed.sh`: Fetches transactions that can be completed.
2. `complete-transactions.sh`: Completes a transaction based on the provided payment ID.

## Usage

To run the scripts, ensure you have the necessary environment variables set for API endpoints.

Example:
```bash
export ELASTICSEARCH_API_ROOT=http://your-elastic-url
export KYC_API_ROOT=http://your-kyc-url
bash get-transactions-that-can-be-completed.sh
```