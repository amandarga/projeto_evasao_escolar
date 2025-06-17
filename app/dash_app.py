import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
import dash_bootstrap_components as dbc

# Load data
df = pd.read_csv("../dados/AbandonoEscolar_RendaMedia_2013_2023.csv")

# Preprocess
df['Evasao_Alta'] = df['Taxa_Abandono'].apply(lambda x: 1 if x > 0.20 else 0)
X = df[['Renda_Media']]
y = df['Evasao_Alta']

# Balance with SMOTE
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)

# Train model
model = GradientBoostingClassifier()
model.fit(X_resampled, y_resampled)

# DASH App
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

# Layout
app.layout = html.Div([
    html.H2("Dashboard de Risco de Evasão Escolar", style={'textAlign': 'center'}),
    html.Div([
        html.Label("Ano:"),
        dcc.Slider(
            min=df['Ano'].min(),
            max=df['Ano'].max(),
            step=1,
            value=df['Ano'].max(),
            marks={int(year): str(year) for year in df['Ano'].unique()},
            id='ano-slider'
        ),
        html.Br(),
        html.Label("Renda Média (R$):"),
        dcc.Slider(
            min=int(df['Renda_Media'].min()),
            max=int(df['Renda_Media'].max()),
            step=50,
            value=int(df['Renda_Media'].mean()),
            marks={int(r): f"{int(r/1000)}k" for r in range(int(df['Renda_Media'].min()), int(df['Renda_Media'].max())+1, 250)},
            id='renda-slider'
        ),
    ], style={'padding': '0px 20px 20px 20px'}),
    
    html.Div(id='saida-modelo', style={'fontSize': '18px', 'textAlign': 'center', 'padding': '10px'}),

    dcc.Graph(id='grafico-distribuicao')
])

@app.callback(
    Output('grafico-distribuicao', 'figure'),
    Output('saida-modelo', 'children'),
    Input('ano-slider', 'value'),
    Input('renda-slider', 'value')
)
def atualizar_graficos(ano_selecionado, renda_valor):
    df_filtrado = df[df['Ano'] == ano_selecionado]
    fig = px.histogram(df_filtrado, x='Renda_Media', color='Evasao_Alta',
                       barmode='overlay', nbins=20, opacity=0.6,
                       title=f'Distribuição de Renda e Evasão no Ano {ano_selecionado}')
    fig.update_layout(
        plot_bgcolor='#F9FAFB',
        paper_bgcolor='#F9FAFB',
        font=dict(size=14)
    )

    proba = model.predict_proba([[renda_valor]])[0][1]
    risco = "🔥 RISCO ALTO" if proba > 0.20 else "🟢 RISCO BAIXO"
    texto = f"Probabilidade de evasão: {proba:.2%} → {risco}"

    return fig, texto

if __name__ == '__main__':
    app.run(debug=True)
