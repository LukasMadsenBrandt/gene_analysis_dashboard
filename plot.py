from gene_analysis_kutsche.data_preprocessing import load_and_preprocess_data as load_and_preprocess_kutsche
from gene_analysis_kutsche.data_filtering import filter_data_proximity_based_weights as filter_proximity_kutsche
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.io as pio
import pandas as pd
import os
from statsmodels.tsa.api import VAR

# Load and filter data
df = load_and_preprocess_kutsche(os.path.join('Data', 'Kutsche', 'genes_all.txt'))
df_human, raw_data, day_map = filter_proximity_kutsche(df)

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div(style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'width': '100vw', 'height': '100vh'}, children=[
    html.H1("Gene Expression Plotter", style={'textAlign': 'center', 'marginTop': '20px'}),

    # Dropdown for gene selection
    dcc.Dropdown(
        id='gene-selector',
        options=[{'label': gene, 'value': gene} for gene in df_human.index],
        multi=True,  # Allows multiple selection
        placeholder="Select genes to plot",
        style={'width': '50%', 'marginBottom': '20px'}
    ),

    # Toggle normalization
    html.Button("Normalize (VAR Model)", id="normalize-btn", n_clicks=0, style={'marginBottom': '20px'}),
    html.Div(id='normalization-status', style={'marginBottom': '20px'}),

    # Graph display
    dcc.Graph(id='gene-plot', style={'flexGrow': 1, 'width': '90vw'}),

    # Buttons for downloading the plot
    html.Div(style={'display': 'flex', 'justifyContent': 'center', 'marginTop': '20px'}, children=[
        html.Button("Download as HTML", id="download-html-btn", style={'marginRight': '10px'}),
        dcc.Download(id="download-html"),
        
        html.Button("Download as SVG", id="download-svg-btn", style={'marginLeft': '10px'}),
        dcc.Download(id="download-svg")
    ])
])

def normalize_data(df_selected):
    model = VAR(df_selected.T)  # Transpose because VAR expects time-series data in columns
    results = model.fit(maxlags=1)  # Fit with lag order 1 for simplicity
    normalized_data = results.fittedvalues.T  # Transpose back after fitting
    return normalized_data

@app.callback(
    [Output('gene-plot', 'figure'),
     Output('normalization-status', 'children')],
    [Input('gene-selector', 'value'),
     Input('normalize-btn', 'n_clicks')],
    [State('normalize-btn', 'n_clicks_timestamp')]
)
def update_plot(selected_genes, n_clicks, n_clicks_timestamp):
    if not selected_genes:
        return go.Figure(), ""  # Empty figure

    fig = go.Figure()
    x_values = df_human.columns.astype(int)
    is_normalized = n_clicks % 2 == 1  # Normalize if the button has been clicked an odd number of times

    # Extract selected gene data
    df_selected = df_human.loc[selected_genes]

    if is_normalized:
        normalized_data = normalize_data(df_selected)
        normalization_status = "Data normalized using VAR model"
    else:
        normalized_data = df_selected
        normalization_status = "Data is not normalized"

    # Plot the (normalized or raw) data
    for gene, y_values in normalized_data.iterrows():
        fig.add_trace(go.Scatter(x=x_values, y=y_values.values, mode='lines+markers', name=gene))

    # Apply a consistent layout every time the figure is updated
    fig.update_layout(
        title="Gene Expression Over Time",
        xaxis_title="Time (days)",
        yaxis_title="Expression Level",
        template="plotly_white",
        margin={'l': 40, 'r': 40, 't': 40, 'b': 40},  # Adjust margins for a cleaner look
        height=600,  # Set a fixed height for better appearance
        showlegend=True  # Ensure legend is always shown
    )

    return fig, normalization_status

@app.callback(
    Output("download-html", "data"),
    [Input("download-html-btn", "n_clicks"),
     Input('gene-plot', 'figure')],
    prevent_initial_call=True
)
def download_html(n_clicks, figure):
    if n_clicks:
        return dcc.send_string(go.Figure(figure).to_html(full_html=True), "gene_expression_plot.html")

@app.callback(
    Output("download-svg", "data"),
    [Input("download-svg-btn", "n_clicks"),
     Input('gene-plot', 'figure')],
    prevent_initial_call=True
)
def download_svg(n_clicks, figure):
    if n_clicks:
        # Convert the Plotly figure to an SVG string using the kaleido engine
        fig = go.Figure(figure)
        svg_str = pio.to_image(fig, format="svg", engine="kaleido")

        # Send the SVG string as a downloadable file
        return dcc.send_bytes(svg_str, "gene_expression_plot.svg")

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
