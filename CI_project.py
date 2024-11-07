import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import DebertaTokenizer, DebertaForSequenceClassification, pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import resample


# Function to calculate average effect of aspect change
def calculate_average_aspect_effect(df):
    grouped = df.groupby('Candidate_id')  # Assuming 'group_id' identifies org and its related CFs
    effect_diffs_dict = {}

    for id, group in grouped:
        org_row = group[group['CF_on'] == 'ORG'].iloc[0] if not group[group['CF_on'] == 'ORG'].empty else None
        if org_row is not None:
            cf_df = group[group['CF_on'] != 'ORG']
            for _, cf_row in cf_df.iterrows():
                on_aspect = cf_row['CF_on']
                from_value = cf_row['CF_from']
                to_value = cf_row['CF_to']
                from_label = org_row['Good_Employee']
                to_label = cf_row['Good_Employee']

                if on_aspect not in effect_diffs_dict:
                    effect_diffs_dict[on_aspect] = {}
                if (from_value, to_value) not in effect_diffs_dict[on_aspect]:
                    effect_diffs_dict[on_aspect][(from_value, to_value)] = []

                # Measure the difference
                effect_diff = to_label - from_label
                effect_diffs_dict[on_aspect][(from_value, to_value)].append(effect_diff)

    # Calculate average effect for each aspect change and bootstrap confidence intervals
    results_dict = {}

    for aspect in effect_diffs_dict:
        for change in effect_diffs_dict[aspect]:
            effect_diffs_dict[aspect][change] = np.mean(effect_diffs_dict[aspect][change])



    # Visualize the average effect in a table
    changes = []
    scores = []
    for aspect in effect_diffs_dict:
        for change, score in effect_diffs_dict[aspect].items():
            changes.append(f"{aspect} ({change[0]} -> {change[1]})")
            scores.append(score)
    df_visual = pd.DataFrame({'Aspect Change': changes, 'Average Effect Score': scores})

    return effect_diffs_dict, df_visual


def calculate_ate_bootstrap(df, n_bootstrap=1000, confidence_level=0.95):
    grouped = df.groupby('Candidate_id')  # Assuming 'group_id' identifies org and its related CFs
    effect_diffs_dict = {}

    for id, group in grouped:
        org_row = group[group['CF_on'] == 'ORG'].iloc[0] if not group[group['CF_on'] == 'ORG'].empty else None
        if org_row is not None:
            cf_df = group[group['CF_on'] != 'ORG']
            for _, cf_row in cf_df.iterrows():
                on_aspect = cf_row['CF_on']
                from_value = cf_row['CF_from']
                to_value = cf_row['CF_to']
                from_label = org_row['Good_Employee']
                to_label = cf_row['Good_Employee']

                if on_aspect not in effect_diffs_dict:
                    effect_diffs_dict[on_aspect] = {}
                if (from_value, to_value) not in effect_diffs_dict[on_aspect]:
                    effect_diffs_dict[on_aspect][(from_value, to_value)] = []

                # Measure the difference
                effect_diff = to_label - from_label
                effect_diffs_dict[on_aspect][(from_value, to_value)].append(effect_diff)

    # Calculate average effect for each aspect change and bootstrap confidence intervals
    results_dict = {}
    for aspect in effect_diffs_dict:
        results_dict[aspect] = {}
        for change in effect_diffs_dict[aspect]:
            # Get all effect differences
            effect_diffs = np.array(effect_diffs_dict[aspect][change])

            # Calculate ATE as the mean of the effect differences
            ate = np.mean(effect_diffs)

            # Bootstrap to calculate confidence intervals
            bootstrap_ates = []
            for _ in range(n_bootstrap):
                sample = resample(effect_diffs)
                bootstrap_ates.append(np.mean(sample))
            bootstrap_ates = np.array(bootstrap_ates)

            # Calculate confidence interval
            lower_bound = np.percentile(bootstrap_ates, ((1 - confidence_level) / 2) * 100)
            upper_bound = np.percentile(bootstrap_ates, (1 - (1 - confidence_level) / 2) * 100)

            # Store results
            results_dict[aspect][change] = {
                'ATE': ate,
                'CI_lower': lower_bound,
                'CI_upper': upper_bound
            }

    # Create a DataFrame to visualize the results
    changes = []
    ates = []
    lower_bounds = []
    upper_bounds = []
    for aspect in results_dict:
        for change, stats in results_dict[aspect].items():
            changes.append(f"{aspect} ({change[0]} -> {change[1]})")
            ates.append(stats['ATE'])
            lower_bounds.append(stats['CI_lower'])
            upper_bounds.append(stats['CI_upper'])

    df_visual = pd.DataFrame({
        'Aspect Change': changes,
        'Average Effect Score (ATE)': ates,
        'CI Lower Bound': lower_bounds,
        'CI Upper Bound': upper_bounds
    })

    print(df_visual)

    return results_dict, df_visual


def mean_aspect_affect(aspect_effect_dict, cf_row):
    on_aspect = cf_row['CF_on']
    from_value = cf_row['CF_from']
    to_value = cf_row['CF_to']
    return aspect_effect_dict.get(on_aspect, {}).get((from_value, to_value), None)
def models_pred(model, tokenizer, row):
    text = "CV_statement"
    # conditions = ['No_violence', 'Verbal_violence', 'Physical_violence']
    # id = "Nurse_id"

    # Create a pipeline for sequence classification
    classification_pipeline = pipeline(
        "text-classification", model=model, tokenizer=tokenizer, top_k=None,
        device=0 if torch.cuda.is_available() else -1
    )

    # # Check the length of the tokenized input
    # tokens = tokenizer(row[text], return_tensors='pt')

    predictions = classification_pipeline(row[text])
    max_label = max(predictions[0], key=lambda x: x['score'])['label']
    label_int = int(max_label.split('_')[1])

    return label_int
def calculate_sample_diff(model_path, org_row, cf_row):
    model = DebertaForSequenceClassification.from_pretrained(model_path)
    tokenizer = DebertaTokenizer.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Get predictions for the sample row
    pred_org = models_pred(model, tokenizer, org_row)
    pred_CF = models_pred(model, tokenizer, cf_row)

    model_diff = pred_CF - pred_org
    return model_diff


def count_model_wins(df):
    # Extract columns related to models
    model_columns = [col for col in df.columns if col.startswith('Average Effect Score (')]

    # Add a column to track which model wins each row
    df['Winning Model'] = df.apply(
        lambda row: min(model_columns, key=lambda col: abs(row['Average Effect Score'] - row[col])), axis=1)

    # Count the number of wins for each model
    model_win_counts = df['Winning Model'].value_counts()

    # Convert the result to a dictionary for easier readability
    model_win_counts_dict = model_win_counts.to_dict()

    # Print the result in a readable format
    for model, count in model_win_counts_dict.items():
        print(f"Model '{model}' wins: {count} times")

    return model_win_counts

if __name__ == "__main__":
    # df = pd.read_csv('combined_average_effects.csv')
    # model_win_counts = count_model_wins(df)

    df_w = pd.read_csv("Cv_dataset/Final_cv_w_cf.csv", encoding='utf-8-sig')
    df_w['Good_Employee'] = df_w['Good_Employee'].astype(int)
    print("test_dataset_wo_n size:", df_w.shape)

    # aspect_effect_dict, df_visual_org = calculate_average_aspect_effect(df_w)
    aspect_effect_dict, df_visual_org = calculate_ate_bootstrap(df_w)
    df_visual_org.to_csv('ate_bootstrap.csv', index=False)

    models_paths = ["Trained_models/cv/CW_cv_model_lr1e-05_epochs5_bs8",
                    "Trained_models/cv/CB_cv_model_lr3e-05_epochs7_bs8",
                    "Trained_models/cv/OV_cv_model_lr3e-05_epochs15_bs8"]

    results = {model_path: {} for model_path in models_paths}

    grouped = df_w.groupby('Candidate_id')
    for model_path in models_paths:
        print("Processing model:", model_path)
        for id, group in grouped:
            org_row = group[group['CF_on'] == 'ORG'].iloc[0] if not group[group['CF_on'] == 'ORG'].empty else None
            if org_row is not None:
                cf_df = group[group['CF_on'] != 'ORG']
                for index, row in cf_df.iterrows():
                    cf_row = row
                    sample_diff = calculate_sample_diff(model_path, org_row, cf_row)
                    on_aspect = cf_row['CF_on']
                    from_value = cf_row['CF_from']
                    to_value = cf_row['CF_to']

                    if on_aspect not in results[model_path]:
                        results[model_path][on_aspect] = {}
                    if (from_value, to_value) not in results[model_path][on_aspect]:
                        results[model_path][on_aspect][(from_value, to_value)] = []
                    # Measure the difference
                    results[model_path][on_aspect][(from_value, to_value)].append(sample_diff)
            else:
                print(f"No 'ORG' row found for ID: {id}")

        # Calculate average effect for each aspect change
        for aspect in results[model_path]:
            for change in results[model_path][aspect]:
                results[model_path][aspect][change] = np.mean(results[model_path][aspect][change])

        # Visualize the average effect in a table
        changes = []
        scores = []
        for aspect in results[model_path]:
            for change, score in results[model_path][aspect].items():
                changes.append(f"{aspect} ({change[0]} -> {change[1]})")
                scores.append(score)
        df_visual_model = pd.DataFrame({'Aspect Change': changes, f'Average Effect Score ({model_path})': scores})
        df_visual_org = df_visual_org.merge(df_visual_model, on='Aspect Change', how='outer')

    # Save the DataFrame to a CSV file
    df_visual_org.to_csv('ATE_Bootstrap_models.csv', index=False)


