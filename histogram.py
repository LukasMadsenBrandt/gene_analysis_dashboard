import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Load the CSV data into a DataFrame
data = pd.read_csv("granger_causality_results.csv")

# Extract the p-values column
p_values = data['p-value']

# Define bin edges that strictly cover the range [0, 1]
bin_edges = np.linspace(0, 1, 21)  # Creates 20 bins between 0 and 1

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the app layout
app.layout = html.Div([
    html.H1("Interactive P-value Distribution Analysis", style={'font-size': '60px'}),
    html.P("Click on a bar in the histogram to zoom into that range. The next plot will show a more detailed view. \"0 - 0.05\" includes values from 0 up to but not including 0.05",
           style={'font-size': '32px'}),
    html.Div([
        dcc.Graph(id='histogram', style={'height': '80vh', 'width': '30vw'}),
        dcc.Graph(id='expanded_histogram', style={'height': '80vh', 'width': '30vw'}),
        dcc.Graph(id='further_expanded_histogram', style={'height': '80vh', 'width': '30vw'}),
    ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'space-around'}),
])

# Callback to update the expanded view based on the clicked bin
@app.callback(
    Output('expanded_histogram', 'figure'),
    Input('histogram', 'clickData')
)
def update_expanded_histogram(clickData):
    if clickData is None:
        return go.Figure()

    x_center = clickData['points'][0]['x']
    bin_size = bin_edges[1] - bin_edges[0]
    bin_start = np.floor(x_center / bin_size) * bin_size
    bin_end = bin_start + bin_size

    expanded_p_values = p_values[(p_values >= bin_start) & (p_values < bin_end)]

    hist_expanded = go.Histogram(
        x=expanded_p_values,
        xbins=dict(start=bin_start, end=bin_end, size=(bin_end - bin_start) / 20),
        marker_color='orange'
    )

    fig = go.Figure(data=[hist_expanded])
    fig.update_layout(
        title=f"Expanded View: {bin_start:.2f} to {bin_end:.2f}",
        title_font=dict(size=24),
        xaxis_title="P-value",
        xaxis_title_font=dict(size=28),
        yaxis_title="Count",
        yaxis_title_font=dict(size=28),
        xaxis=dict(
            range=[bin_start, bin_end],
            tickfont=dict(size=24),
        ),
        yaxis=dict(
            tickfont=dict(size=24),
        ),
        hoverlabel=dict(
            font_size=24  # Increase the font size for tooltips
        ),
        bargap=0.1,
    )

    return fig

# Callback to update the further expanded view based on the clicked bin in the expanded histogram
@app.callback(
    Output('further_expanded_histogram', 'figure'),
    Input('expanded_histogram', 'clickData')
)
def update_further_expanded_histogram(clickData):
    if clickData is None:
        return go.Figure()

    x_center = clickData['points'][0]['x']
    expanded_bin_size = (bin_edges[1] - bin_edges[0]) / 20
    bin_start = np.floor(x_center / expanded_bin_size) * expanded_bin_size
    bin_end = bin_start + expanded_bin_size

    finer_bin_size = (bin_end - bin_start) / 20
    further_expanded_p_values = p_values[(p_values >= bin_start) & (p_values < bin_end)]

    hist_further_expanded = go.Histogram(
        x=further_expanded_p_values,
        xbins=dict(start=bin_start, end=bin_end, size=finer_bin_size),
        marker_color='green'
    )

    fig = go.Figure(data=[hist_further_expanded])
    fig.update_layout(
        title=f"Further Expanded View: {bin_start:.6f} to {bin_end:.6f}",
        title_font=dict(size=24),
        xaxis_title="P-value",
        xaxis_title_font=dict(size=28),
        yaxis_title="Count",
        yaxis_title_font=dict(size=28),
        xaxis=dict(
            range=[bin_start, bin_end],
            tickfont=dict(size=24),
            tickformat=".6f"
        ),
        yaxis=dict(
            tickfont=dict(size=24),
        ),
        hoverlabel=dict(
            font_size=24  # Increase the font size for tooltips
        ),
        bargap=0.1,
    )

    return fig

# Callback to display the main histogram
@app.callback(
    Output('histogram', 'figure'),
    Input('expanded_histogram', 'id')
)
def display_histogram(_):
    hist_original = go.Histogram(
        x=p_values,
        xbins=dict(start=0, end=1, size=(bin_edges[1] - bin_edges[0])),
        marker_color='blue'
    )

    fig = go.Figure(data=[hist_original])
    fig.update_layout(
        title="P-value Distribution",
        title_font=dict(size=24),
        xaxis_title="P-value",
        xaxis_title_font=dict(size=28),
        yaxis_title="Count",
        yaxis_title_font=dict(size=28),
        xaxis=dict(
            range=[0, 1],
            tickfont=dict(size=24),
        ),
        yaxis=dict(
            tickfont=dict(size=24),
        ),
        hoverlabel=dict(
            font_size=24  # Increase the font size for tooltips
        ),
        bargap=0.1,
    )

    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
