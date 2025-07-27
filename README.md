# Open Polymer Prediction 2025

Predicting polymer performance from chemical structure using machine learning  
Accelerating sustainable materials discovery with open data

---

## Project Overview

Polymers are the backbone of innovation in medicine, electronics, and sustainability. Our goal is to predict a polymer's **real-world performance** directly from its **chemical structure**.

This project supports the **Open Polymer Prediction 2025** initiative by leveraging machine learning on a large-scale, open-source dataset — ten times larger than any existing resource.

We predict **five key physical properties** from SMILES strings:
-  **Density**
-  **Thermal conductivity (Tc)**
-  **Glass transition temperature (Tg)**
-  **Radius of gyration (Rg)**
-  **Fractional free volume (FFV)**

---

## 📁 Project Structure

```bash
.
├── data/                   # Input datasets (SMILES, labels)
├── src/
│   ├── preprocessing.py    # Molecular parsing and feature extraction
│   ├── models/             # ML/DL model definitions
│   ├── train.py            # Training pipeline
│   ├── evaluate.py         # Evaluation scripts
├── notebooks/              # Jupyter notebooks for EDA and prototyping
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
