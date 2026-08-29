New-Item -ItemType Directory -Force -Path "deploy\space" | Out-Null
---
title: IEEE-CIS Fraud Detection API
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Fraud Detection API

Scores card transactions for fraud risk. Built on the IEEE-CIS dataset.

**Open [`/docs`](./docs) to try it.** Fill in the example transaction, tick
`explain`, and press Execute.

## Endpoints

| Endpoint | What it does |
|----------|--------------|
| `GET /health` | Is the service up and the model loaded |
| `GET /model` | What model is being served and how it scored |
| `POST /predict` | Score one transaction, optionally with an explanation |
| `POST /predict/batch` | Score up to 500 at once |

## About the model

LightGBM, 284 engineered features, validation PR-AUC 0.607 against a 0.034
baseline. Kaggle private leaderboard 0.914.

The threshold is 0.4222, not 0.5. It was chosen by a cost model at a 2%
manual review capacity, which is what a real fraud team can handle.

Most transaction fields are optional. Anything you leave out is treated as
unknown, which the model handles natively at every split.

Full project, including the data pipeline, drift monitoring, and CI:
https://github.com/Dee-ui/ieee-cis-fraud-detection
