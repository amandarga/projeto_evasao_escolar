import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import plotly.io as pio
import datetime

# Preparar dados
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

# App e MFA
app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server

usuario_autenticado = False

def layout_dashboard():
    return dbc.Container([
        html.Div([
            dbc.Button("🚪 Sair", id="logout-btn", color="danger", className="mb-3"),

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

            html.Label("Escolha o gráfico para exibir:"),
            dcc.Dropdown(
                id='grafico-dropdown',
                options=[
                    {'label': 'Distribuição de Evasão por Renda', 'value': 'grafico1'},
                    {'label': 'Evasão por Tipo de Rede', 'value': 'grafico2'},
                    {'label': 'Ranking de Fatores Influentes', 'value': 'grafico3'},
                    {'label': 'Previsão de Evasão para 2025', 'value': 'grafico4'},
                ],
                value='grafico1',
                clearable=False,
                className="mb-4"
            ),

            html.Div(id="conteudo-grafico"),

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
    ])

app.layout = dbc.Container([
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("🔐 Login com MFA")),
        dbc.ModalBody([
            dbc.Input(id='usuario', placeholder='Usuário', type='text', className="mb-2"),
            dbc.Input(id='senha', placeholder='Senha', type='password', className="mb-2"),
            dbc.Input(id='token', placeholder='Token MFA (Ex: 654321)', type='text', className="mb-2"),
            dbc.Button('Entrar', id='botao-login', color="primary", className="me-2"),
            html.Div(id='login-status', className="mt-2")
        ]),
    ], id="modal-login", is_open=True, centered=True),

    html.Div(id='conteudo-dashboard', style={'display': 'none'})
])

@app.callback(
    Output('login-status', 'children'),
    Output('conteudo-dashboard', 'children'),
    Output('conteudo-dashboard', 'style'),
    Output('modal-login', 'is_open'),
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
        return dbc.Alert("Login OK ✅", color="success"), layout_dashboard(), {'display': 'block'}, False
    else:
        with open("app.log", "a") as f:
            f.write(f"[{datetime.datetime.now()}] Falha Login - Usuário: {usuario}\n")
        return dbc.Alert("Erro na autenticação ❌", color="danger"), "", {'display': 'none'}, True

@app.callback(
    Output("conteudo-dashboard", "style"),
    Output("conteudo-dashboard", "children"),
    Output("modal-login", "is_open"),
    Input("logout-btn", "n_clicks"),
    prevent_initial_call=True
)
def realizar_logout(n):
    return {'display': 'none'}, "", True

@app.callback(
    Output('conteudo-grafico', 'children'),
    Input('grafico-dropdown', 'value'),
    Input('ano-slider', 'value'),
    Input('renda-slider', 'value')
)
def renderizar_grafico(grafico, ano, renda):
    df_ano = df[df['Ano'] == ano]
    prob = model.predict_proba([[renda]])[0][1]
    risco = "🔴 RISCO CRÍTICO" if prob > 0.20 else "🟢 RISCO BAIXO"
    texto = html.P(f"Probabilidade de evasão: {prob:.2%} → {risco}", className="text-info")

    if grafico == 'grafico1':
        fig = px.histogram(df_ano, x='Renda_Media', color='Evasao_Alta', nbins=30,
                           title=f"Evasão no Ano {ano} por Renda Média",
                           color_discrete_map={0: "#4CAF50", 1: "#FF5733"})
    elif grafico == 'grafico2':
        if 'Dependencia_Administrativa' in df.columns:
            fig = px.box(df_ano, x='Dependencia_Administrativa', y='Taxa_Abandono',
                         color='Dependencia_Administrativa', title="Evasão por Tipo de Rede")
        else:
            return html.Div([texto, html.P("(Coluna 'Dependencia_Administrativa' não encontrada)")])
    elif grafico == 'grafico3':
        fig = px.bar(importancias, x='Feature', y='Importancia', title="Importância das Variáveis")
    elif grafico == 'grafico4':
        fig = fig_risco_2025
    else:
        return html.Div("Gráfico não encontrado")

    return html.Div([texto, dcc.Graph(figure=fig)])

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
