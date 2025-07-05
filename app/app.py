# Dashboard for AML/CFT monitoring using simulated data and top features; planned for live data and full features post-hire
import os
import base64
import io
import datetime
import time
import dash
from dash import html, dcc, dash_table, Output, Input, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import plotly.express as px
import joblib

# TODO: Add basic authentication (e.g., Dash login) for user access to show future security intent without code changes.
# TODO: Enhance interval to simulate real-time data updates to highlight potential for live monitoring.
# TODO: Add user-configurable alert threshold (e.g., slider) to suggest dynamic alerts without modifying the current 50% trigger.

# Initialize app and set server as WSGI application
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # Explicitly define server as the Flask WSGI instance

# Define constants (updated with correct path)
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

# Layout with enhanced design and modeling page
app.layout = html.Div([
    # Header with logo and title
    html.Div([
        html.Img(src=app.get_asset_url('bank_logo.png'), style={'height': '50px', 'float': 'left'}),
        html.H1("Bank AML/CFT Dashboard", style={'textAlign': 'center', 'margin-left': '60px', 'color': '#2c3e50'})
    ], style={'backgroundColor': '#f8f9fa', 'padding': '15px', 'borderBottom': '2px solid #3498db'}),
    
    # Tab system for enhanced design and modeling page
    dbc.Tabs([
        dbc.Tab(label='Dashboard', tab_id='tab-dashboard', children=[
            dcc.Interval(id='interval-component', interval=1*60*1000, n_intervals=0),
            dcc.Upload(
                id='upload_data',
                children=html.Button('Upload Live Data Feed', style={'margin': '10px', 'backgroundColor': '#3498db', 'color': 'white', 'border': 'none', 'padding': '10px'}),
                multiple=False
            ),
            dcc.Dropdown(
                id='model_selector',
                options=[{'label': k, 'value': k} for k in models.keys()],
                value='Random Forest',
                clearable=False,
                style={'width': '200px', 'margin': '10px', 'borderColor': '#3498db'}
            ),
            html.Button("Download Report", id="download-button", n_clicks=0, style={'margin': '10px', 'backgroundColor': '#2ecc71', 'color': 'white', 'border': 'none', 'padding': '10px'}),
            dcc.Download(id="download-data"),
            html.Div(id='output_metrics', style={'margin': '10px', 'backgroundColor': '#ecf0f1', 'padding': '10px', 'borderRadius': '5px'}),
            html.Div(id='prediction_table', style={'margin': '10px', 'backgroundColor': '#ecf0f1', 'padding': '10px', 'borderRadius': '5px'}),
            dcc.Graph(id='pie_chart', style={'margin': '10px', 'border': '1px solid #3498db', 'borderRadius': '5px'}),
            html.Div(id='alert-popup', style={'margin': '10px'})
        ]),
        dbc.Tab(label='Modeling', tab_id='tab-modeling', children=[
            html.Div([
                html.H3("Model Performance Overview", style={'textAlign': 'center', 'color': '#2c3e50'}),
                html.P("This page will display detailed model statistics and comparisons once live data is integrated.", style={'textAlign': 'center', 'color': '#7f8c8d'}),
                # Placeholder for future modeling content
                html.Div(id='modeling-content', style={'margin': '20px', 'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '5px'})
            ])
        ])
    ], active_tab='tab-dashboard', style={'margin': '20px'}),
    
    # Footer branding
    html.Div("© Bank 2025 - Contact: emmynahimana1999@gmail.com", style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderTop': '2px solid #3498db', 'color': '#7f8c8d'})
], style={'fontFamily': 'Arial, sans-serif', 'maxWidth': '1200px', 'margin': 'auto', 'boxShadow': '0 4px 8px rgba(0,0,0,0.1)'})

@callback(
    [Output('output_metrics', 'children'),
     Output('prediction_table', 'children'),
     Output('pie_chart', 'figure'),
     Output('alert-popup', 'children')],
    [Input('upload_data', 'contents'), Input('interval-component', 'n_intervals')],
    [State('upload_data', 'filename'),
     State('model_selector', 'value')]
)
def update_output(contents, n, filename, model_name):
    print(f"Processing file: {filename} at {datetime.datetime.now()}")
    if contents is None or not models:
        return ["Please upload a file or ensure models are available."], None, {}, None

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8-sig')))
    except Exception as e:
        return [f"❌ Failed to parse CSV: {e}"], None, {}, None

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
            df['Sender_account'] = df['Sender_account'].astype('int32')
            df['Receiver_account'] = df['Receiver_account'].astype('int32')
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
            return [f"❌ Preprocessing failed: {e}"], None, {}, None

    if X.shape[1] != len(TOP_FEATURES):
        return [f"❌ Feature mismatch: Expected {len(TOP_FEATURES)} features, got {X.shape[1]}"], None, {}, None

    print(f"X shape: {X.shape}")
    print(f"X columns: {X.columns.tolist()}")
    if not X.empty:
        print(f"Sample X head: {X.head()}")
    print(f"Model: {model_name}")

    model = models.get(model_name)
    if model is None:
        return [f"❌ Model {model_name} not loaded."], None, {}, None

    try:
        if model_name == "HDBSCAN":
            labels = model.fit_predict(X)
            y_pred = (labels == -1).astype(int)
        else:
            y_pred = model.predict(X)
            if model_name == "Isolation Forest":
                y_pred = np.where(y_pred == -1, 1, 0)
    except Exception as e:
        return [f"❌ Model prediction failed: {e}"], None, {}, None

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
                html.P("⚠️ Warning: Model may be overfitting due to imbalanced data.", style={'color': 'orange'}),
                html.H4("Model Performance Metrics"),
                html.P(f"Precision: {report['1']['precision']:.2f}"),
                html.P(f"Recall: {report['1']['recall']:.2f}"),
                html.P(f"F1 Score: {report['1']['f1-score']:.2f}"),
                html.P(f"Risk Score: {risk_score:.1f}%"),
                alert
            ])
        else:
            metrics = html.Div([
                html.H4("Model Performance Metrics"),
                html.P(f"Precision: {report['1']['precision']:.2f}"),
                html.P(f"Recall: {report['1']['recall']:.2f}"),
                html.P(f"F1 Score: {report['1']['f1-score']:.2f}"),
                html.P(f"Risk Score: {risk_score:.1f}%"),
                alert
            ])
    else:
        metrics = html.Div([
            html.H4("Prediction Summary"),
            html.P(f"{sum(y_pred)} transactions predicted as laundering."),
            html.P(f"Risk Score: {risk_score:.1f}%"),
            alert
        ])

    # Enhanced DataTable to include only available key transaction fields
    key_fields = ['Transaction_ID', 'Sender_account', 'Receiver_account', 'Date', 'Time', 'Amount']
    available_key_fields = [col for col in key_fields if col in df_original.columns]
    # Remove duplicate 'Date' if it exists from preprocessing
    columns_to_use = available_key_fields + [col for col in df_original.columns if col in TOP_FEATURES or col in ['Risk_Score', 'Is_laundering', 'Prediction']]
    columns_to_use = list(dict.fromkeys(columns_to_use))  # Deduplicate while preserving order
    table_columns = [{"name": i, "id": i} for i in columns_to_use]
    table_data = df_original[columns_to_use].head(50).to_dict('records')
    table = dash_table.DataTable(
        columns=table_columns,
        data=table_data,
        style_table={'overflowX': 'auto'},
        style_data_conditional=[
            {'if': {'filter_query': '{Prediction} eq 1'}, 'backgroundColor': '#ffcccc', 'color': 'black'}
        ],
        page_size=10
    )

    fig = px.pie(df_original, names='Prediction', title=f'Prediction Distribution (Risk Score: {risk_score:.1f}%)')

    # Alert System: Dynamic in-app alert for high-risk cases
    alert_content = dbc.Alert("High Risk Transaction Detected! Action Required.", color="danger", duration=4000, is_open=True) if risk_score > 50 else None

    # Detailed Reporting: Enhanced metrics with total flagged transactions
    total_transactions = len(df_original)
    flagged_transactions = sum(y_pred)
    detailed_metrics = html.Div([
        metrics,
        html.P(f"Total Transactions: {total_transactions}", style={'margin-top': '10px'}),
        html.P(f"Flagged Transactions: {flagged_transactions} ({risk_score:.1f}%)")
    ])

    return detailed_metrics, table, fig, alert_content

@callback(
    Output("download-data", "data"),
    Input("download-button", "n_clicks"),
    State("model_selector", "value"),
    State("upload_data", "contents"),
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
    # Add summary statistics as additional rows to the CSV
    total_transactions = len(df_original)
    flagged_transactions = sum(y_pred)
    risk_score = (sum(y_pred) / len(y_pred)) * 100 if len(y_pred) > 0 else 0
    false_positives = 0
    if 'Is_laundering' in df_original.columns:
        false_positives = sum((y_pred == 1) & (df_original['Is_laundering'] == 0))
    summary_data = pd.DataFrame({
        'Metric': ['Total Transactions', 'Flagged Transactions', 'Risk Score (%)', 'False Positives'],
        'Value': [total_transactions, flagged_transactions, risk_score, false_positives]
    })
    combined_df = pd.concat([df_original, summary_data], ignore_index=True)
    return dcc.send_data_frame(combined_df.to_csv, filename=f"aml_report_{model_name}_{time.strftime('%Y%m%d')}.csv")

if __name__ == '__main__':
    app.run(debug=True, port=8051)