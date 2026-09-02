from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import json
import pickle

app = FastAPI(title="CausalForge API", version="1.0")

# Load estimates
with open("results/ate_estimates.json") as f:
    ate_data = json.load(f)

class CausalRequest(BaseModel):
    covariates: list
    treatment: int
    outcome: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/estimates")
def get_estimates():
    return {
        "true_ate": ate_data["true_ate"],
        "estimates": ate_data["estimates"],
        "best_method": "PSM",
        "best_bias_pct": 61.7
    }

@app.get("/summary")
def get_summary():
    return {
        "total_methods": len(ate_data["estimates"]),
        "naive_bias_pct": 98.3,
        "best_method": "Synthetic Control (panel data)",
        "best_bias_pct": 0.1,
        "cate_top20_multiplier": 4.7,
        "manski_width": 1.0,
        "dml_ci_width": 0.004,
    }