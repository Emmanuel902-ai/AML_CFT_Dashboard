import os
import base64
import io
import datetime
import time
import dash
from dash import html, dcc, dash_table, Output, Input, State, callback, ctx
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import plotly.express as px
import joblib

# Initialize app and set server as WSGI application
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server

# Define constants
MODEL_PATH = "models"
TOP_FEATURES = joblib.load("data/processed/top_features.pkl")
models = {
    "Random Forest": joblib.load(os.path.join(MODEL_PATH, "random_forest.pkl")),
    "Logistic Regression": joblib.load(os.path.join(MODEL_PATH, "logistic_regression.pkl")),
    "HDBSCAN": joblib.load(os.path.join(MODEL_PATH, "hdbscan_model.pkl")),
    "Isolation Forest": joblib.load(os.path.join(MODEL_PATH, "isolation_forest.pkl"))
}
numerical_cols = ['Amount', 'Recipient_diversity', 'Sender_diversity', 'Daily_frequency',
                  'Avg_velocity', 'Total_inflow', 'Total_outflow', 'Inflow_Outflow_Ratio',
                  'Txn_sequence', 'Rolling_avg_amt', 'Weekday', 'Day', 'Month']
categorical_cols = ['Payment_type', 'Received_currency', 'Receiver_bank_location']

# Training metrics based on your evaluation (replace Logistic Regression with actual values when available)
TRAINING_METRICS = {
    "Random Forest": {"precision": 1.00, "recall": 0.91, "f1_score": 0.95},
    "Logistic Regression": {"precision": 0.80, "recall": 0.82, "f1_score": 0.81},  # Placeholder, update with actual values
    "HDBSCAN": {"precision": 0.45, "recall": 0.16, "f1_score": 0.24},
    "Isolation Forest": {"precision": 0.71, "recall": 0.39, "f1_score": 0.51}
}

# Layout
app.layout = html.Div([
    # Header with logo and title
    html.Div([
        html.Img(src=app.get_asset_url('bank_logo.png'), style={'height': '50px', 'float': 'left'}),
        html.H1("Bank AML/CFT Dashboard", style={'textAlign': 'center', 'margin-left': '60px', 'color': '#2c3e50'})
    ], style={'backgroundColor': '#f8f9fa', 'padding': '15px', 'borderBottom': '2px solid #3498db'}),
    
    dbc.Tabs([
        dbc.Tab(label='Dashboard', tab_id='tab-dashboard', children=[
            dcc.Upload(
                id='upload_data',
                children=html.Button('Upload Live Data Feed', id='upload-button', style={'margin': '10px', 'backgroundColor': '#3498db', 'color': 'white', 'border': 'none', 'padding': '10px'}),
                multiple=False
            ),
            dcc.Dropdown(
                id='model_selector',
                options=[{'label': k, 'value': k} for k in models.keys()],
                value='Random Forest',
                clearable=False,
                style={'width': '200px', 'margin': '10px', 'borderColor': '#3498db'}
            ),
            dcc.Dropdown(id='filter-sender', options=[], placeholder="Filter by Sender Account", style={'width': '200px', 'margin': '10px'}),
            dcc.Dropdown(id='filter-prediction', options=[{'label': 'All', 'value': 'all'}, {'label': 'Laundering', 'value': 1}, {'label': 'Not Laundering', 'value': 0}], value='all', style={'width': '200px', 'margin': '10px'}),
            html.Button("Submit", id="submit-button", n_clicks=0, style={'margin': '10px', 'backgroundColor': '#2ecc71', 'color': 'white', 'border': 'none', 'padding': '10px'}),
            html.Button("Download Report", id="download-button", n_clicks=0, style={'margin': '10px', 'backgroundColor': '#2ecc71', 'color': 'white', 'border': 'none', 'padding': '10px'}),
            dcc.Download(id="download-data"),
            dcc.Loading(
                id="loading",
                type="circle",
                children=[
                    html.Div(id='output_metrics', style={'margin': '10px', 'backgroundColor': '#ecf0f1', 'padding': '10px', 'borderRadius': '5px'}),
                    html.Div(id='prediction_table', style={'margin': '10px', 'backgroundColor': '#ecf0f1', 'padding': '10px', 'borderRadius': '5px'}),
                    dcc.Graph(id='pie_chart', style={'margin': '10px', 'border': '1px solid #3498db', 'borderRadius': '5px'}),
                    dcc.Graph(id='metrics-chart', style={'margin': '10px', 'border': '1px solid #3498db', 'borderRadius': '5px'}),
                    html.Div(id='alert-popup', style={'margin': '10px'}),
                    html.Div(id='alert-history-log', style={'margin': '10px', 'backgroundColor': '#ecf0f1', 'padding': '10px', 'borderRadius': '5px'}),
                    html.Div(id='upload-feedback', style={'margin': '10px', 'backgroundColor': '#ecf0f1', 'padding': '10px', 'borderRadius': '5px'}),
                ]
            ),
            dcc.Store(id='alert-history', storage_type='memory'),
            dbc.Modal([
                dbc.ModalHeader("Transaction Details"),
                dbc.ModalBody(id='transaction-details'),
                dbc.ModalFooter(dbc.Button("Close", id='close-modal', className="ml-auto"))
            ], id='modal', is_open=False)
        ]),
        dbc.Tab(label='Performances', tab_id='tab-modeling', children=[
            html.Div([
                html.H3("Model Performance Overview", style={'textAlign': 'center', 'color': '#2c3e50'}),
                html.P("This page displays a summary of model performance based on current data.", style={'textAlign': 'center', 'color': '#7f8c8d'}),
                dash_table.DataTable(
                    id='model-performance-table',
                    columns=[
                        {"name": "Model", "id": "model"},
                        {"name": "Precision", "id": "precision"},
                        {"name": "Recall", "id": "recall"},
                        {"name": "F1 Score", "id": "f1_score"}
                    ],
                    data=[],
                    style_table={'margin': '20px', 'backgroundColor': '#ecf0f1', 'padding': '10px', 'borderRadius': '5px'},
                    style_cell={'textAlign': 'center'},
                    style_header={'backgroundColor': '#3498db', 'color': 'white'}
                ),
                html.P(id='modeling-note', style={'textAlign': 'center', 'color': '#7f8c8d', 'margin': '20px'}),
                dcc.Store(id='model-performance-data', storage_type='memory', data={})
            ])
        ])
    ], active_tab='tab-dashboard', style={'margin': '20px'}),
    
    html.Div("© Bank 2025 - Contact: emmynahimana1999@gmail.com", style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderTop': '2px solid #3498db', 'color': '#7f8c8d'})
], style={'fontFamily': 'Arial, sans-serif', 'maxWidth': '1200px', 'margin': 'auto', 'boxShadow': '0 4px 8px rgba(0,0,0,0.1)'})

@callback(
    [Output('output_metrics', 'children'),
     Output('prediction_table', 'children'),
     Output('pie_chart', 'figure'),
     Output('alert-popup', 'children'),
     Output('alert-history-log', 'children'),
     Output('metrics-chart', 'figure'),
     Output('upload-feedback', 'children'),
     Output('filter-sender', 'options'),
     Output('upload-button', 'disabled'),
     Output('model_selector', 'disabled'),
     Output('filter-sender', 'disabled'),
     Output('filter-prediction', 'disabled'),
     Output('download-button', 'disabled'),
     Output('model-performance-data', 'data', allow_duplicate=True),
     Output('modeling-note', 'children', allow_duplicate=True)],
    [Input('submit-button', 'n_clicks'),
     Input('upload_data', 'contents'),
     Input('model_selector', 'value'),
     Input('filter-sender', 'value'),
     Input('filter-prediction', 'value')],
    [State('upload_data', 'filename'),
     State('alert-history', 'data'),
     State('model-performance-data', 'data')],
    prevent_initial_call=True
)
def update_output(submit_n_clicks, contents, model_name, sender_filter, pred_filter, filename, alert_history, model_performance_data):
    print(f"Processing file: {filename} at {datetime.datetime.now()}")
    if contents is None or not models or not submit_n_clicks:
        return ["Please upload a file, select a model, and click Submit to load data.", html.P("Interpretation: Upload data and submit to see predicted laundering status. Values of 1 indicate predicted laundering, while 0 indicates no laundering.")], None, {}, None, alert_history, {}, html.P("No file uploaded yet or submit not clicked.", style={'color': 'gray'}), [], False, False, False, False, False, model_performance_data, "Note: Upload data with 'Is_laundering' column to view model performance metrics."

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8-sig')))
        feedback = html.P("Data uploaded successfully!", style={'color': 'green'})
    except Exception as e:
        return [f"❌ Failed to parse CSV: {e}", html.P("Interpretation: Ensure the uploaded file is a valid CSV with required columns. Values of 1 indicate predicted laundering, while 0 indicates no laundering.")], None, {}, None, alert_history, {}, html.P(f"Upload failed: {e}", style={'color': 'red'}), [], False, False, False, False, False, model_performance_data, "Note: Upload data with 'Is_laundering' column to view model performance metrics."

    df_original = df.copy()

    if set(TOP_FEATURES).issubset(df.columns):
        X = df[TOP_FEATURES].copy()
        y_true = df['Is_laundering'] if 'Is_laundering' in df.columns else None
    else:
        try:
            if X.shape[0] > 10000:
                X = X.sample(n=10000, random_state=43)
                print("Dataset reduced to 10,000 rows for performance.")
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])
            df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce')
            df = df.sort_values(by=['Sender_account', 'Date', 'Time'])
            df['Total_inflow'] = df.groupby('Receiver_account')['Amount'].cumsum()
            df['Total_outflow'] = df.groupby('Sender_account')['Amount'].cumsum()
            df['Inflow_Outflow_Ratio'] = df['Total_inflow'] / (df['Total_outflow'] + 1e-6)
            df['Recipient_diversity'] = df.groupby('Sender_account')['Receiver_account'].apply(
                lambda x: x.expanding(min_periods=1).apply(lambda y: y.nunique())
            ).reset_index(level=0, drop=True)
            df['Sender_diversity'] = df.groupby('Receiver_account')['Sender_account'].apply(
                lambda x: x.expanding(min_periods=1).apply(lambda y: y.nunique())
            ).reset_index(level=0, drop=True)
            df['Daily_frequency'] = df.groupby(['Sender_account', 'Date']).transform('size')
            df['Avg_velocity'] = df.groupby('Sender_account')['Daily_frequency'].transform(
                lambda x: x.rolling(window=7, min_periods=1).mean()
            )
            df['Txn_sequence'] = df.groupby('Sender_account').cumcount() + 1
            df['Rolling_avg_amt'] = df.groupby('Sender_account')['Amount'].rolling(
                window=3, min_periods=1).mean().reset_index(0, drop=True)
            df['Hour'] = df['Time'].dt.hour
            df['Minute'] = df['Time'].dt.minute
            df['Weekday'] = df['Date'].dt.weekday
            df['Day'] = df['Date'].dt.day
            df['Month'] = df['Date'].dt.month
            df = df.drop(columns=['Time', 'Laundering_type'] if 'Laundering_type' in df.columns else ['Time'])
            df['Sender_account'] = pd.to_numeric(df['Sender_account'], errors='coerce').fillna(0).abs().astype('int32')
            df['Receiver_account'] = pd.to_numeric(df['Receiver_account'], errors='coerce').fillna(0).abs().astype('int32')
            df['Weekday'] = df['Weekday'].apply(lambda x: x if pd.notna(x) and 0 <= x <= 6 else 0)
            df['Day'] = df['Day'].apply(lambda x: x if pd.notna(x) and 1 <= x <= 31 else 1)
            df['Month'] = df['Month'].apply(lambda x: x if pd.notna(x) and 1 <= x <= 12 else 1)
            df['Amount'] = df['Amount'].astype('float32')
            if 'Is_laundering' in df.columns:
                df['Is_laundering'] = df['Is_laundering'].astype('int8')
            for col in ['Recipient_diversity', 'Sender_diversity', 'Daily_frequency', 'Avg_velocity',
                        'Total_inflow', 'Total_outflow', 'Inflow_Outflow_Ratio', 'Txn_sequence', 'Rolling_avg_amt']:
                df[col] = df[col].astype('float32')
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
            df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median(numeric_only=True))
            y_true = df['Is_laundering'] if 'Is_laundering' in df.columns else None
            X = df.reindex(columns=TOP_FEATURES, fill_value=0)
        except Exception as e:
            return [f"❌ Preprocessing failed: {e}", html.P("Interpretation: Check data integrity or column names. Values of 1 indicate predicted laundering, while 0 indicates no laundering.")], None, {}, None, alert_history, {}, feedback, [], False, False, False, False, False, model_performance_data, "Note: Upload data with 'Is_laundering' column to view model performance metrics."

    if X.shape[1] != len(TOP_FEATURES):
        return [f"❌ Feature mismatch: Expected {len(TOP_FEATURES)} features, got {X.shape[1]}", html.P("Interpretation: Ensure all required features are present in the uploaded data. Values of 1 indicate predicted laundering, while 0 indicates no laundering.")], None, {}, None, alert_history, {}, feedback, [], False, False, False, False, False, model_performance_data, "Note: Upload data with 'Is_laundering' column to view model performance metrics."

    print(f"X shape: {X.shape}")
    print(f"X columns: {X.columns.tolist()}")
    if not X.empty:
        print(f"Sample X head: {X.head()}")
    print(f"Model: {model_name}")

    model = models.get(model_name)
    if model is None:
        return [f"❌ Model {model_name} not loaded.", html.P("Interpretation: Model loading failed; verify model files. Values of 1 indicate predicted laundering, while 0 indicates no laundering.")], None, {}, None, alert_history, {}, feedback, [], False, False, False, False, False, model_performance_data, "Note: Upload data with 'Is_laundering' column to view model performance metrics."

    try:
        if model_name == "HDBSCAN":
            labels = model.fit_predict(X)
            y_pred = (labels == -1).astype(int)
        else:
            y_pred = model.predict(X)
            if model_name == "Isolation Forest":
                y_pred = np.where(y_pred == -1, 1, 0)
            if model_name in ["Random Forest", "Logistic Regression"]:
                try:
                    prob = model.predict_proba(X)[:, 1]
                    df_original['Confidence'] = prob
                except:
                    pass
    except Exception as e:
        return [f"❌ Model prediction failed: {e}", html.P("Interpretation: Prediction error; check model compatibility with data. Values of 1 indicate predicted laundering, while 0 indicates no laundering.")], None, {}, None, alert_history, {}, feedback, [], False, False, False, False, False, model_performance_data, "Note: Upload data with 'Is_laundering' column to view model performance metrics."

    print(f"y_pred distribution: {pd.Series(y_pred).value_counts()}")
    print(f"Unique predictions: {np.unique(y_pred)}")

    df_original['Prediction'] = y_pred
    risk_score = (sum(y_pred) / len(y_pred)) * 100 if len(y_pred) > 0 else 0
    df_original['Risk_Score'] = y_pred * 100

    alert = html.Div([html.H5("⚠️ High Risk Alert")], style={'color': 'red'}) if risk_score > 50 else ""

    if y_true is not None:
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        if sum(y_true) / len(y_true) < 0.1:
            metrics = html.Div([
                html.P("⚠️ Warning: Model may be overfitting due to imbalanced data (less than 10% laundering cases).", style={'color': 'orange'}),
                html.H4("Model Performance Metrics"),
                html.P(f"Precision: {report['1']['precision']:.2f} - This measures how accurate the model is when predicting laundering (1), i.e., the proportion of true laundering cases among predicted laundering cases."),
                html.P(f"Recall: {report['1']['recall']:.2f} - This indicates how well the model identifies actual laundering cases (1), i.e., the proportion of true laundering cases captured by the model."),
                html.P(f"F1 Score: {report['1']['f1-score']:.2f} - This is the harmonic mean of Precision and Recall, providing a balanced measure of the model's performance on laundering detection."),
                html.P(f"Risk Score: {risk_score:.1f}% - This represents the percentage of transactions flagged as potential laundering (1), with higher values indicating greater risk."),
                alert,
            ])
        else:
            metrics = html.Div([
                html.H4("Model Performance Metrics"),
                html.P(f"Precision: {report['1']['precision']:.2f} - This measures how accurate the model is when predicting laundering (1), i.e., the proportion of true laundering cases among predicted laundering cases."),
                html.P(f"Recall: {report['1']['recall']:.2f} - This indicates how well the model identifies actual laundering cases (1), i.e., the proportion of true laundering cases captured by the model."),
                html.P(f"F1 Score: {report['1']['f1-score']:.2f} - This is the harmonic mean of Precision and Recall, providing a balanced measure of the model's performance on laundering detection."),
                html.P(f"Risk Score: {risk_score:.1f}% - This represents the percentage of transactions flagged as potential laundering (1), with higher values indicating greater risk."),
                alert,
            ])
        # Compute metrics for all models
        performance_data = {}
        for model_name in models.keys():
            model = models[model_name]
            try:
                if model_name == "HDBSCAN":
                    labels = model.fit_predict(X)
                    y_pred_model = (labels == -1).astype(int)
                else:
                    y_pred_model = model.predict(X)
                    if model_name == "Isolation Forest":
                        y_pred_model = np.where(y_pred_model == -1, 1, 0)
                report_model = classification_report(y_true, y_pred_model, output_dict=True, zero_division=0)
                performance_data[model_name] = {
                    'precision': report_model['1']['precision'],
                    'recall': report_model['1']['recall'],
                    'f1_score': report_model['1']['f1-score']
                }
                print(f"Model: {model_name}, Precision: {report_model['1']['precision']:.2f}, Recall: {report_model['1']['recall']:.2f}, F1: {report_model['1']['f1-score']:.2f}")
            except Exception as e:
                print(f"Error calculating metrics for {model_name}: {e}")
                performance_data[model_name] = {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
        model_performance_data = performance_data
    else:
        metrics = html.Div([
            html.H4("Prediction Summary"),
            html.P(f"{sum(y_pred)} transactions predicted as laundering (1)."),
            html.P(f"Risk Score: {risk_score:.1f}% - This is the percentage of transactions flagged as potential laundering (1), based solely on model predictions since no ground truth is available."),
            alert,
            html.P("Interpretation: Without actual laundering data (Is_laundering), the model assigns 1 to suspected laundering and 0 to normal transactions. Review predictions with caution.")
        ])
        model_performance_data = TRAINING_METRICS  # Use pre-trained metrics as fallback
        print("No ground truth available; using pre-trained metrics.")

    key_fields = ['Transaction_ID', 'Sender_account', 'Receiver_account', 'Date', 'Time', 'Amount']
    available_key_fields = [col for col in key_fields if col in df_original.columns]
    columns_to_use = available_key_fields + [col for col in df_original.columns if col in TOP_FEATURES or col in ['Risk_Score', 'Prediction', 'Confidence']]
    columns_to_use = [col for col in columns_to_use if col not in ['Weekday', 'Day', 'Month']]
    columns_to_use = list(dict.fromkeys(columns_to_use))
    table_columns = [{"name": i, "id": i} for i in columns_to_use]
    filtered_data = df_original[columns_to_use].head(50)
    if sender_filter:
        filtered_data = filtered_data[filtered_data['Sender_account'] == int(sender_filter)]
    if pred_filter != 'all':
        filtered_data = filtered_data[filtered_data['Prediction'] == int(pred_filter)]
    table_data = filtered_data.to_dict('records')
    table = dash_table.DataTable(
        columns=table_columns,
        data=table_data,
        style_table={'overflowX': 'auto'},
        style_data_conditional=[
            {'if': {'filter_query': '{Prediction} eq 1'}, 'backgroundColor': '#ffcccc', 'color': 'black'}
        ],
        page_size=10,
        id='transaction-table'
    )

    fig = px.pie(df_original, names='Prediction', title=f'Prediction Distribution (Risk Score: {risk_score:.1f}%) - 1 indicates laundering, 0 indicates no laundering')

    alert_content = dbc.Alert("High Risk Transaction Detected! Action Required.", color="danger", duration=4000, is_open=True) if risk_score > 50 else None

    total_transactions = len(df_original)
    flagged_transactions = sum(y_pred)
    detailed_metrics = html.Div([
        metrics,
        html.P(f"Total Transactions: {total_transactions} - The total number of transactions analyzed.", style={'margin-top': '10px'}),
        html.P(f"Flagged Transactions: {flagged_transactions} ({risk_score:.1f}%) - The number and percentage of transactions predicted as laundering (1).")
    ])

    if risk_score > 50:
        alert_history = [html.P(f"Alert at {datetime.datetime.now()}: Risk Score {risk_score:.1f}%", style={'color': 'red'})]
        if alert_history:
            alert_history = alert_history + (alert_history if not isinstance(alert_history, list) else [])
        return detailed_metrics, table, fig, alert_content, alert_history, {}, feedback, df_original['Sender_account'].dropna().unique(), False, False, False, False, False, model_performance_data, "Note: Upload data with 'Is_laundering' column to view model performance metrics."

    metrics_fig = {}
    if y_true is not None:
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        metrics_fig = {
            'data': [{
                'x': ['Precision', 'Recall', 'F1 Score'],
                'y': [report['1']['precision'], report['1']['recall'], report['1']['f1-score']],
                'type': 'bar',
                'marker': {'color': '#3498db'}
            }],
            'layout': {'title': 'Model Performance Metrics'}
        }

    modeling_note = "Note: No 'Is_laundering' column detected in the uploaded data. Displaying pre-trained performance metrics. Upload data with ground truth for updated metrics." if y_true is None else "Note: Model performance metrics are based on the uploaded data with ground truth."

    return detailed_metrics, table, fig, alert_content, alert_history, metrics_fig, feedback, df_original['Sender_account'].dropna().unique(), False, False, False, False, False, model_performance_data, modeling_note

@callback(
    Output('model-performance-table', 'data'),
    [Input('model-performance-data', 'data')],
    prevent_initial_call=True
)
def update_model_performance_table(performance_data):
    if not performance_data:
        return []
    return [
        {"model": model_name, "precision": round(data['precision'], 2), "recall": round(data['recall'], 2), "f1_score": round(data['f1_score'], 2)}
        for model_name, data in performance_data.items()
    ]

@callback(
    [Output('modal', 'is_open', allow_duplicate=True),
     Output('transaction-details', 'children')],
    [Input('transaction-table', 'active_cell')],
    [State('modal', 'is_open'),
     State('transaction-table', 'derived_virtual_data')],
    prevent_initial_call='initial_duplicate'
)
def toggle_modal(active_cell, is_open, data):
    if active_cell and data:
        row = data[active_cell['row']]
        details = [html.P(f"{k}: {v}") for k, v in row.items()]
        return not is_open, details
    return is_open, []

@callback(
    [Output('modal', 'is_open', allow_duplicate=True)],
    [Input('close-modal', 'n_clicks')],
    [State('modal', 'is_open')],
    prevent_initial_call='initial_duplicate'
)
def close_modal(n, is_open):
    return [not is_open]

@callback(
    Output("download-data", "data"),
    [Input("download-button", "n_clicks")],
    [State("model_selector", "value"),
     State("upload_data", 'contents')],
    prevent_initial_call=True
)
def generate_report(n_clicks, model_name, contents):
    if contents is None or model_name not in models:
        return None
    content_type, content_string = contents.split(',')
    df = pd.read_csv(io.StringIO(base64.b64decode(content_string).decode('utf-8-sig')))
    df_original = df.copy()
    X = df[TOP_FEATURES].copy() if set(TOP_FEATURES).issubset(df.columns) else df.reindex(columns=TOP_FEATURES, fill_value=0)
    y_pred = models[model_name].predict(X)
    df_original['Prediction'] = y_pred
    total_transactions = len(df_original)
    flagged_transactions = sum(y_pred)
    risk_score = (sum(y_pred) / len(y_pred)) * 100 if len(y_pred) > 0 else 0
    summary_data = pd.DataFrame({
        'Metric': ['Total Transactions', 'Flagged Transactions', 'Risk Score (%)'],
        'Value': [total_transactions, flagged_transactions, risk_score]
    })
    combined_df = pd.concat([df_original, summary_data], ignore_index=True)
    return dcc.send_data_frame(combined_df.to_csv, filename=f"aml_report_{model_name}_{time.strftime('%Y%m%d')}.csv")

if __name__ == '__main__':
    app.run(debug=True, port=8051)