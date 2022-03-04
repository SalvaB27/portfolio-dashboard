import pandas as pd
import numpy as np
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.express as px
import turms
import plotly.graph_objects as go
from datetime import date
from scipy.stats import norm


# Read the crypto data into pandas dataframe
crypto_df = pd.read_csv("ticker_returns_wo_shiba.csv", index_col='Date').drop(columns='Unnamed: 0')


# Create a dash application
app = dash.Dash(__name__)

# Create an app layout

coins=crypto_df.columns



app.layout = html.Div(children=[html.H1('My Portfolio Dashboard (in process of making)',
                                        style={'textAlign': 'center', 'color': '#503D36',
                                               'font-size': 40}),
                                # TASK 1: Add a dropdown list to enable Launch Site selection
                                # The default select value is for ALL sites
                                dcc.Dropdown(id='dropdown', 
                                        options=[x for x in coins] + ['All'], 
                                        value='BTC-USD', 
                                        multi=True),

                                html.Br(),

                                dcc.RadioItems(id='radio', 
                                options=['Market Value', 'Returns %', 'Drawdown'], 
                                value='Market Value'),


                                dcc.DatePickerRange(id='date-picker-range',
                                                    min_date_allowed=date(2020, 11, 25),
                                                    max_date_allowed=date(2022,3,1),
                                                    start_date=date(2020, 11, 25),
                                                    end_date=date(2022,3,1)
                                                    ),

                                html.Br(),


                                # TASK 2: Add a pie chart to show the total successful launches count for all sites
                                # If a specific launch site was selected, show the Success vs. Failed counts for the site
                                html.Div(dcc.Graph(id='crypto-returns-graph')),
                                html.Br(),

                                html.H2('Markowitz Portfolio',
                                        style={'textAlign': 'center', 'color': '#503D36',
                                               'font-size': 30}),

                                dcc.Checklist(
                                        id="checklist2",
                                        options=['Bias', 'GMP', 'CML', 'Show Individual Assets'],
                                        value=[],
                                        labelStyle={'display': 'inline-block', 'float':'left'}
                                    ),
                            
                                html.Br(),

                                html.Div(dcc.Graph(id='efficient-frontier')),

                                html.Br(),

                                html.H2('Value at Risk',
                                        style={'textAlign': 'center', 'color': '#503D36',
                                               'font-size': 30}),

                                html.Div(children=[dcc.Graph(id='normal-histogram', 
                                        style={'width':'50%', 'float':'left'}), 

                                        html.Div(html.H2('ANNOTATIONS...',
                                                        style={'textAlign': 'center', 'color': '#503D36',
                                                        'font-size': 30}), 
                                                style={'width':'48%', 'float':'left'}
                                                        )
                                    ])
                                ])

# TASK 2:
# Add a callback function for `site-dropdown` as input, `success-pie-chart` as output
@app.callback(
Output(component_id='crypto-returns-graph', component_property='figure'),
[Input(component_id='dropdown', component_property='value'),
Input(component_id='radio', component_property='value'),
Input(component_id='date-picker-range', component_property='start_date'),
Input(component_id='date-picker-range', component_property='end_date'),])

def get_line_chart(value, value2, start_date, end_date):
    start = start_date
    end = end_date

    if value2 == 'Returns %':
        dataframe = crypto_df.pct_change().fillna(0)

    elif value2 == 'Drawdown':
        pct = crypto_df.pct_change().fillna(0)
        wealth_index = 1000*(1 + pct).cumprod()
        previous_peak = wealth_index.cummax()
        dataframe = (wealth_index - previous_peak) / previous_peak

    else:
        dataframe = crypto_df

    dataframe = dataframe[start:end]
    
    if value == ['All']:
        fig = px.line(dataframe, x=dataframe.index, y=dataframe.columns, title='Price for All Cryptos')
        return fig

    else:
        filtered_df = dataframe[value]
        title = f"Market Price for for site {value}"
        fig = px.line(filtered_df, x=filtered_df.index, y=value, title=title)
        return fig

# Add a callback function for `site-dropdown` and `payload-slider` as inputs, `success-payload-scatter-chart` as output
@app.callback(
Output(component_id='efficient-frontier', component_property='figure'),
[Input(component_id='checklist2', component_property='value')]
)

def get_markowitz(value):
    pct = crypto_df.pct_change().fillna(0)
    er = turms.annualize_rets(pct, 365)
    cov = crypto_df.cov()

    n_points = 50
    weights = turms.optimal_weights(n_points, er, cov)
    rets = [turms.portfolio_return(w, er) for w in weights]
    vols = [turms.portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        'Returns': rets, 
        'Volatility': vols
        })
    fig = px.line(ef, x='Volatility', y='Returns', title='Efficient Frontier of Possible Portfolios')

    if 'Bias' in value:
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = turms.portfolio_return(w_ew, er)
        vol_ew = turms.portfolio_vol(w_ew, cov)
        fig.add_trace(go.Scatter(x=[vol_ew], y=[r_ew], mode = 'markers',
                         marker_symbol = 'star',
                         marker_size = 15))

    if 'GMP' in value:
        w_gmv = turms.gmv(cov)
        r_gmv = turms.portfolio_return(w_gmv, er)
        vol_gmv = turms.portfolio_vol(w_gmv, cov)
        # Display GMV
        fig.add_trace(go.Scatter(x=[vol_gmv], y=[r_gmv], mode = 'markers',
                         marker_symbol = 'star',
                         marker_size = 15))

    if 'CML' in value:
        w_msr = turms.msr(0.3, er, cov)
        r_msr = turms.portfolio_return(w_msr, er)
        vol_msr = turms.portfolio_vol(w_msr, cov)
        # Add CML
        cml_x = [0, vol_msr]
        cml_y = [0.3, r_msr]
        fig.add_trace(go.Scatter(x =cml_x, y =cml_y,
                            mode = 'lines+markers',
                            name = 'lines+markers'
                        ))

    if 'Show Individual Assets' in value:
        ann_vol = turms.annualize_vol(pct, 365)
        fig.add_trace(go.Scatter(x =ann_vol, y =er,
                            mode = 'markers',
                            text = er.index
                        ))
    return fig


@app.callback(
Output(component_id='normal-histogram', component_property='figure'),
[Input(component_id='dropdown', component_property='value')]
)

def ret_norm(value):
    pct = crypto_df.pct_change().fillna(0)
    mu, std = norm.fit(pct[value])
    fig = px.histogram(pct, x=value, 
                        title = "(1st) Fit results: mu = %.2f,  std = %.2f" % (mu, std))
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server()