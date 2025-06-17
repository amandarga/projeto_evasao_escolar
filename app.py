import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import plotly.io as pio
import datetime

# Carregar dados
caminho_dados = "dados/AbandonoEscolar_RendaMedia_2013_2023.csv"
df = pd.read_csv(caminho_dados)
df['Evasao_Alta'] = (df['Taxa_Abandono'] > 20).astype(int)

X = df[['Renda_Media']]
y = df['Evasao_Alta']
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
model = GradientBoostingClassifier()
model.fit(X_res, y_res)

importancias = pd.DataFrame({
    'Feature': ['Renda_Media'],
    'Importancia': model.feature_importances_
})

df_2025_simulado = df[df['Ano'] == df['Ano'].max()].copy()
df_2025_simulado['Ano'] = 2025
df_2025_simulado['Prob_Evasao'] = model.predict_proba(df_2025_simulado[['Renda_Media']])[:, 1]
df_2025_simulado['Risco_Critico'] = df_2025_simulado['Prob_Evasao'] > 0.20
n_risco_2025 = df_2025_simulado['Risco_Critico'].sum()

fig_risco_2025 = px.histogram(
    df_2025_simulado,
    x="Prob_Evasao",
    nbins=30,
    title="Distribuição da Probabilidade de Evasão – Previsão 2025",
    color="Risco_Critico",
    color_discrete_map={True: "#FF0000", False: "#00AA00"},
    labels={"Risco_Critico": "Risco Crítico"}
)

app = Dash(__name__, external_stylesheets=[dbc.themes.SOLAR])
server = app.server

# Layout do login
login_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("🔐 Acesso ao Dashboard de Evasão Escolar", className="text-center text-light")),
                dbc.CardBody([
                    dbc.Input(id='usuario', placeholder='Usuário', type='text', className="mb-3"),
                    dbc.Input(id='senha', placeholder='Senha', type='password', className="mb-3"),
                    dbc.Input(id='token', placeholder='Token MFA (Ex: 654321)', type='text', className="mb-3"),
                    dbc.Button('Entrar', id='botao-login', color="success", className="w-100"),
                    html.Div(id='login-status', className="mt-3")
                ])
            ], className="shadow-lg")
        ], width=6)
    ], justify="center", className="vh-100 align-items-center")
])

# Layout do dashboard

def layout_dashboard():
    return dbc.Container([
        html.H2("📊 Dashboard de Risco de Evasão Escolar", className="text-center my-4 text-primary"),

        dbc.Row([
            dbc.Col([
                html.Label("Ano:"),
                dcc.Slider(
                    id='ano-slider',
                    min=df['Ano'].min(),
                    max=df['Ano'].max(),
                    value=df['Ano'].max(),
                    marks={int(ano): str(ano) for ano in df['Ano'].unique()},
                    step=1
                ),
            ], md=6),

            dbc.Col([
                html.Label("Renda Média (R$):"),
                dcc.Slider(
                    id='renda-slider',
                    min=int(df['Renda_Media'].min()),
                    max=int(df['Renda_Media'].max()),
                    value=int(df['Renda_Media'].mean()),
                    marks={int(r): f"R${int(r)}" for r in range(int(df['Renda_Media'].min()), int(df['Renda_Media'].max())+1, 500)},
                    step=100
                )
            ], md=6)
        ], className="mb-4"),

        html.Div(id='saida-modelo', className="text-center fs-5 mb-4 text-info"),

        dbc.Row([
            dbc.Col([
                html.H5("Distribuição de evasão por renda"),
                dcc.Graph(id='grafico-distribuicao')
            ], md=6),

            dbc.Col([
                html.H5("Comparativo por dependência administrativa"),
                dcc.Graph(id='grafico-dependencia')
            ], md=6),
        ], className="mb-4"),

        html.Hr(),
        html.H5("🔎 Ranking de Fatores Mais Influentes"),
        dcc.Graph(figure=px.bar(importancias, x='Feature', y='Importancia', title="Importância das Variáveis")),

        html.Hr(),
        html.H5("📈 Distribuição da Probabilidade de Evasão – Previsão 2025"),
        dcc.Graph(figure=fig_risco_2025, id='grafico-previsao-2025'),

        html.Div([
            html.Button("📥 Baixar gráfico (PNG)", id="download-btn", className="btn btn-primary", style={"marginTop": "20px"}),
            dcc.Download(id="download-graph")
        ]),

        dbc.Alert(
            f"📌 Predição de Evasão Escolar: {n_risco_2025} escolas identificadas com risco crítico para 2025",
            color="danger",
            className="mt-4 text-center"
        )
    ])

app.layout = html.Div([
    dcc.Location(id='url'),
    html.Div(id='page-content')
])

@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname'),
    prevent_initial_call='initial_duplicate'
)
def display_page(pathname):
    if pathname == '/dashboard':
        return layout_dashboard()
    return login_layout

@app.callback(
    Output('url', 'pathname'),
    Output('login-status', 'children'),
    Input('botao-login', 'n_clicks'),
    State('usuario', 'value'),
    State('senha', 'value'),
    State('token', 'value'),
    prevent_initial_call=True
)
def validar_login(n_clicks, usuario, senha, token):
    if usuario == 'admin' and senha == '123' and token == '654321':
        with open("app.log", "a") as f:
            f.write(f"[{datetime.datetime.now()}] Login OK - Usuário: {usuario}\n")
        return '/dashboard', None
    else:
        with open("app.log", "a") as f:
            f.write(f"[{datetime.datetime.now()}] Falha Login - Usuário: {usuario}\n")
        return '/', dbc.Alert("Erro na autenticação ❌", color="danger")

@app.callback(
    Output('grafico-distribuicao', 'figure'),
    Output('grafico-dependencia', 'figure'),
    Output('saida-modelo', 'children'),
    Input('ano-slider', 'value'),
    Input('renda-slider', 'value')
)
def atualizar(ano, renda):
    df_ano = df[df['Ano'] == ano]
    fig1 = px.histogram(df_ano, x='Renda_Media', color='Evasao_Alta',
                        nbins=30, title=f"Evasão no Ano {ano} por Renda Média",
                        color_discrete_map={0: "#4CAF50", 1: "#FF5733"})

    if 'Dependencia_Administrativa' in df.columns:
        fig2 = px.box(df_ano, x='Dependencia_Administrativa', y='Taxa_Abandono',
                      color='Dependencia_Administrativa', title="Evasão por Tipo de Rede")
    else:
        fig2 = px.box(title="(Coluna 'Dependencia_Administrativa' não encontrada)")

    prob = model.predict_proba([[renda]])[0][1]
    risco = "🔴 RISCO CRÍTICO" if prob > 0.20 else "🟢 RISCO BAIXO"
    texto = f"Probabilidade de evasão: {prob:.2%} → {risco}"
    return fig1, fig2, texto

@app.callback(
    Output("download-graph", "data"),
    Input("download-btn", "n_clicks"),
    prevent_initial_call=True
)
def download_graph(n_clicks):
    img_bytes = pio.to_image(fig_risco_2025, format="png", width=800, height=600, scale=2)
    return dcc.send_bytes(img_bytes, filename="previsao_evasao_2025.png")

if __name__ == '__main__':
    app.run_server(host="0.0.0.0", port=8080, debug=True)
