# 📊 Projeto: Predição de Evasão Escolar com Machine Learning

Este projeto tem como objetivo prever escolas com risco crítico de evasão escolar no Brasil utilizando Machine Learning, Dash, dados socioeconômicos e educacionais.

## 🚀 Funcionalidades

- Dashboard interativo com filtros por ano e renda
- Simulação de risco de evasão para 2025
- Gráficos de distribuição e comparação entre redes administrativas
- Ranking de variáveis mais influentes
- Sistema de login com autenticação multifatorial (MFA Simulado)
- Exportação de gráfico em PNG

## 🔐 Login de Acesso

- Usuário: `admin`
- Senha: `123`
- Token MFA: `654321`

## 🌐 Acesso Online

Acesse o dashboard hospedado via Render:  
🔗 https://projeto-evasao-escolar.onrender.com

## 🛠️ Tecnologias Utilizadas

- Python
- Dash e Dash Bootstrap Components
- Pandas e Plotly Express
- Scikit-learn
- SMOTE (Imbalanced-learn)
- Render (Cloud Hosting)

## 📁 Estrutura de Pastas

```
├── app.py                   # Arquivo principal com o dashboard e login
├── dados/
│   └── AbandonoEscolar_RendaMedia_2013_2023.csv
├── requirements.txt         # Dependências
├── render.yaml              # Configuração para deploy Render
├── Procfile                 # Executável para Render
├── app.log                  # Logs de acesso (autenticação)
```

## 📦 Instalação Local

1. Clone o repositório
```bash
git clone https://github.com/seu-usuario/nome-do-repositorio.git
cd nome-do-repositorio
```

2. Crie e ative um ambiente virtual (opcional)
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Instale as dependências
```bash
pip install -r requirements.txt
```

4. Execute o app
```bash
python app.py
```

Acesse via http://localhost:8080

## 📈 Resultados
- Modelo Gradient Boosting treinado com SMOTE
- Identificadas 342 escolas com risco crítico para 2025

## 📋 Requisitos da Atividade Atendidos

- [x] Autenticação MFA simulada
- [x] Logs auditáveis
- [x] Implantado em cloud (Render)
- [x] Visualização interativa de dados
- [x] Simulador e exportação de gráfico

## 📄 Licença
Este projeto é de uso educacional e experimental.

