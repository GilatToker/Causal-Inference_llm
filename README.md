# 12.2023 

# Evaluation Language Models’ Ability To Capture Causal Effects Through Task-Based Learning

## Project Overview

The goal of this project is to evaluate the causal learning capabilities of large language models (LLMs) in understanding and capturing the relationships between various factors that influence outcomes. Specifically, we want to determine if models trained on a specific task can implicitly learn the causal effects of these factors. To achieve this, we developed a controlled synthetic dataset focused on resume screening, aiming to explore how models perceive the influence of demographic and professional variables in hiring decisions, wich quantifies how much a change in a specific concept (e.g., gender) affects a model's predictions, helping us assess a model's causal understanding.

The evaluation is based on the Individual Causal Concept Effect (ICaCE), wich quantifies how much a change in a specific concept (e.g., gender) affects a model's predictions, helping us assess a model's causal understanding.

## Data Generation Approach
To analyze these causal relationships effectively, we designed a synthetic dataset. This dataset was constructed using a meticulously planned five-step approach, which includes defining the relevant aspects and generating textual representations based on the relationships between these aspects.

- ### 1. Causal Graph Construction
- ### 2. Aspect Value Sampling
- ### 3. Human-like Template Direction
- ### 4. Persona Direction
- ### 5. Text and Counterfactual (CF) Generation

## Evaluation Pipeline
Once the dataset was created, we trained three different models—OV, CB, and CW—on the task of resume screening. These models were then evaluated based on their ability to capture causal relationships using ICaCE. The average treatment effect (ATE) was calculated for each aspect change, and bootstrap methods were used to assess the reliability of our ATE estimates.

## References
For a detailed report on the project and a complete bibliography, please refer to Project_Report.pdf.

This project was completed as part of the 097400 - Causal Inference course and originated from Gilat's master's thesis. The dataset and research concepts were adapted and expanded specifically for this project, enabling us to explore new applications and gain further insights.
