# Import required libraries
import pandas as pd
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import plotly.express as px

# Read the airline data into pandas dataframe
spacex_df = pd.read_csv("dash_data.csv")
max_payload = spacex_df['Payload Mass (kg)'].max()
min_payload = spacex_df['Payload Mass (kg)'].min()

# Create a dash application
app = dash.Dash(__name__)

# Create an app layout
options_list = [
    {'label': 'All sites', 'value': 'ALL'}] + [
    {'label':site, 'value': site} for site in spacex_df['Launch Site'].unique()
        ]
app.layout = html.Div(children=[html.H1('SpaceX Launch Records Dashboard',
                                        style={'textAlign': 'center', 'color': '#503D36',
                                               'font-size': 40}),
                                # TASK 1: Add a dropdown list to enable Launch Site selection
                                # The default select value is for ALL sites
                                # dcc.Dropdown(id='site-dropdown',...)
                                html.Br(),
                                html.Div([dcc.Dropdown(
                                    id='site-dropdown', options=options_list, 
                                    value = 'ALL', searchable=True, 
                                    placeholder='Choose a specific launch site to filter the data'
                                    )]),
                                # TASK 2: Add a pie chart to show the total successful launches count for all sites
                                # If a specific launch site was selected, show the Success vs. Failed counts for the site
                                html.Div(dcc.Graph(id='success-pie-chart')),
                                html.Br(),
                                html.P("Payload range (Kg):"),
                                # TASK 3: Add a slider to select payload range
                                # dcc.RangeSlider(...)
                                html.Div(dcc.RangeSlider(id='payload-slider',
                                    min=0, max=10000, step=1000,
                                    value=[min_payload, max_payload],
                                    tooltip={
                                        "always_visible": True,
                                        "color": "LightSteelBlue", "fontSize": "20px"})),
                                # TASK 4: Add a scatter chart to show the correlation between payload and launch success
                                html.Div(dcc.Graph(id='success-payload-scatter-chart')),
                                ])

# TASK 2:
# Add a callback function for `site-dropdown` as input, `success-pie-chart` as output
@app.callback(
    Output(component_id='success-pie-chart', component_property='figure'),
    Input(component_id='site-dropdown', component_property='value'))
def update_pie_chart(chosen_site):
    if chosen_site == 'ALL':
        title = 'Successful launches per site'
        data = spacex_df.loc[spacex_df['class'] == 1, ['Launch Site']].value_counts().reset_index()
        names = 'Launch Site'
        ft = 'value'
    else:
        title = f'Successful vs Failed Counts at {chosen_site}'
        data = spacex_df.loc[spacex_df['Launch Site'] == chosen_site, ['Launch Site', 'class']].value_counts()
        data = data.reset_index()
        data['Outcome'] = ['Success' if c else 'Failure' for c in data['class']]
        names = 'Outcome'
        ft = 'percent'

    fig = px.pie(data, values='count', names=names, title=title)
    fig.update_traces(textinfo=ft)
    return fig

# TASK 4:
# Add a callback function for `site-dropdown` and `payload-slider` as inputs, `success-payload-scatter-chart` as output
@app.callback(
    Output(component_id='success-payload-scatter-chart', component_property='figure'),
    [Input(component_id='payload-slider', component_property='value'),
     Input(component_id='site-dropdown', component_property='value')])
def update_scatter_plot(payload: list, chosen_site: str):
    if chosen_site != 'ALL':
        data = spacex_df.loc[spacex_df['Launch Site'] == chosen_site]
        sites = chosen_site
    else:
        sites = 'all sites'
        data = spacex_df
    data['outcome'] = ['Success' if c else 'Failure' for c in data['class']]
    data = data.loc[spacex_df['Payload Mass (kg)'] >= payload[0]]
    data = data.loc[spacex_df['Payload Mass (kg)'] <= payload[1]]
    fig = px.scatter(data, x='Payload Mass (kg)', y='class', labels='outcome', title=f'Success Rate vs. Payload Weight ({sites})')
    return fig


# Run the app
if __name__ == '__main__':
    # pie_data = spacex_df.loc[spacex_df['Launch Site'] == spacex_df['Launch Site'].unique()[0], 'class'].groupby(level=0).count()
    # print(pie_data)
    app.run()
