"""
Создание веб-приложения для анализа данных о вакансиях
"""

from flask import Flask, render_template
import dash
from dash import html, dcc
from dash import dash_table  # type: ignore
import pandas as pd
import plotly.express as px  # type: ignore
import networkx as nx
import os

from network import plot_communities
from network import preprocess_text, get_keywords, create_network, get_communities, plot_one_community

# загрузка данных
kdf = pd.read_csv("python_300_vac.csv")
kdf["tokens"] = kdf["requirement"].apply(preprocess_text)
kdf = get_keywords(kdf, n_keywords=5)
kdf_unique = kdf.drop_duplicates(subset='keywords').copy()

edges = create_network(kdf)
if edges and isinstance(edges[0], tuple):
    G: nx.Graph = nx.Graph()
    G.add_edges_from(edges)
else:
    print('Ошибка: неверный формат ребер')
    G: nx.Graph = nx.Graph()
    for title in kdf['title']:
        G.add_node(title)

all_titles_in_df = set(kdf["title"])
all_nodes_in_G = set(G.nodes)

missing_nodes = all_titles_in_df - all_nodes_in_G
print("Нет в графе G (нет связи с другими узлами):", missing_nodes)

communities, filtered_graph = get_communities(G)

df = kdf["title"].value_counts().reset_index()[:10]
fig = px.bar(df, x="title", y="count", title="Самые частые вакансии")

# создание стартовой страницы
server = Flask(__name__)
# Добавьте это после создания server
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
print(f"Папка с шаблонами: {template_dir}")
print(f"Существует ли папка: {os.path.exists(template_dir)}")
print(f"Существует ли index.html: {os.path.exists(os.path.join(template_dir, 'index.html'))}")
print(f"Содержимое папки templates: {os.listdir(template_dir) if os.path.exists(template_dir) else 'папка не найдена'}")

@server.route("/")
def index():
    """
    Главная страничка
    """
    return render_template("index.html")


# страница со статистикой по исходным данным
dash_dashboard_app = dash.Dash(
    __name__, server=server, url_base_pathname="/dashboard/", suppress_callback_exceptions=True
)

dash_dashboard_app.layout = html.Div(
    style={"fontFamily": "Segoe UI", "textAlign": "center", "padding": "10px", "backgroundColor": "#f0f8ff"},
    children=[
        html.H2("📊 Исходные данные"),
        html.A("← Назад", href="/", style={"color": "#28a745", "textDecoration": "none", "fontSize": "1.1em"}),
        dcc.Graph(figure=fig, style={"marginBottom": "10px", "marginTop": "10px"}),
        dash_table.DataTable(
            data=kdf_unique.to_dict("records"),
            columns=[{"name": "Название вакансии", "id": "title"}, {"name": "Ключевые слова", "id": "str_keywords"}],
            style_cell={"textAlign": "center", "padding": "1px"},
            style_header={"backgroundColor": "#28a745", "color": "white", "fontWeight": "bold"},
            style_table={"width": "100%", "margin": "0 auto"},
        ),
        html.Br(),
    ],
)


# страница с визуализацией графов
dash_dashboard_app = dash.Dash(
    __name__, server=server, url_base_pathname="/network/", suppress_callback_exceptions=True
)


def generate_community_layout(communities, graph):
    """
    Создание layout для Dash: общий граф сообществ и детали по каждому
    """
    layout = [
        html.H2("📊 Аналитика сообществ"),
        html.A("← Назад", href="/", style={"color": "#28a745"}),
        html.Br(),
        html.Hr(),
    ]

    # общий граф сообществ
    fig_all = plot_communities(communities=communities, graph=G)
    layout.append(html.Div([
        html.H3("Граф всех сообществ"),
        dcc.Graph(figure=fig_all),
        html.Hr()
    ]))

    # граф для каждого сообщества
    for idx, community_nodes in enumerate(communities):
        # проверяем, что все узлы есть в графе
        valid_nodes = [n for n in community_nodes if n in graph.nodes]
        if not valid_nodes:
            continue  # пропустить пустые сообщества

        subgraph = graph.subgraph(valid_nodes)
        fig = plot_one_community(subgraph, valid_nodes, communities)

        # Топ-навыки
        community_data = kdf[kdf["title"].isin(valid_nodes)]
        all_words = []
        for words in community_data["keywords"]:
            all_words.extend(words)
        word_freq = pd.Series(all_words).value_counts().reset_index()
        word_freq.columns = ["Навык", "Частота"]

        table = dash_table.DataTable(
            data=word_freq.head(10).to_dict("records"),
            columns=[{"name": i, "id": i} for i in word_freq.columns],
            style_header={"backgroundColor": "#4CAF50", "color": "white"},
            style_cell={"textAlign": "center"},
            style_table={"width": "80%", "margin": "auto"},
        )

        layout.append(
            html.Div(
                [
                    html.H3(f"Сообщество {idx + 1} ({len(valid_nodes)} вакансий)"),
                    dcc.Graph(figure=fig),
                    html.H4("Топ-10 навыков"),
                    table,
                    html.Br(),
                    html.Hr(),
                ]
            )
        )

    return layout


if not communities:
    dash_dashboard_app.layout = html.Div([html.H2("Сообщества не найдены"), html.P("Граф не содержит пересечений")])
else:
    dash_dashboard_app.layout = html.Div(generate_community_layout(communities, G))

# запуск приложения
if __name__ == "__main__":
    server.run(debug=False)
