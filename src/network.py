"""
Создание сообществ вакансий
"""

from copy import deepcopy
import math
from itertools import chain, combinations
import re

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import plotly.graph_objects as go  # type: ignore
import networkx as nx


def preprocess_text(text: str) -> str:
    """
    Принимает на вход текст с требуемыми навыками.
    Выполняет очистку, лемматизацию и фильтрацию по части речи.
    Возвращает отфильтрованные леммы через пробел.
    """
    if not isinstance(text, str):
        return ""

    # Удаление HTML-тегов и highlighttext из HH.ru
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"</?highlighttext>", "", text)

    # Очистка лишних символов, оставляем только кириллицу, латиницу и дефисы
    text = re.sub(r"[^a-zA-Zа-яА-ЯёЁ\- ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Лемматизация
    nlp = spacy.load("ru_core_news_sm")
    doc = nlp(text)

    # Фильтруем по части речи и убираем короткие слова (например, 1 буква)
    filtered_tokens = [token.lemma_.lower() for token in doc if token.pos_ in {"NOUN", "X"} and len(token.lemma_) > 1]

    return " ".join(filtered_tokens)


def get_keywords(df, n_keywords=5):
    """
    Принимает на вход датафрейм с вакансиями и полем обработанных навыков.
    Возвращает датафрейм, состоящий из двух столбцов:
    название вакансии и столбец с ключевыми словами.
    df: входной датафрейм
    n_keywords: число ключевых слов, которое надо извлечь
    """
    df = deepcopy(df)
    vectorizer = TfidfVectorizer()
    df["list of requirements"] = df["tokens"].apply(lambda x: x.split())
    vectors = vectorizer.fit_transform(df["tokens"]).toarray()
    reversed_dict = {v: k for k, v in vectorizer.vocabulary_.items()}

    keywords_list = []
    for vec in vectors:
        top_indices = sorted(range(len(vec)), key=lambda i: vec[i], reverse=True)[  # pylint: disable=cell-var-from-loop
            :n_keywords
        ]
        keywords_list.append([reversed_dict[i] for i in top_indices])

    df["keywords"] = keywords_list
    df["str_keywords"] = df["keywords"].apply(lambda x: ", ".join(x))
    return df[["title", "keywords", "str_keywords"]]


def create_network(df):
    """
    Принимает на вход датафрейм с вакансиями и ключевыми словами.
    Возвращает список кортежей из пар вакансий и количества их общих ключевых слов.
    Вид кортежа внутри списка ожидается такой: (ребро1, ребро2, {'weight': вес_ребра})
    """
    network_edges = []
    job = df.columns[0]
    for (tit1, key1), (tit2, key2) in combinations(df[[job, "keywords"]].values, 2):
        if tit1 != tit2:
            common = set(key1) & set(key2)
            if common:
                network_edges.append((min(tit1, tit2), max(tit2, tit1), {"weight": len(common)}))
    return network_edges


def plot_network(vac_edges):
    """
    Строит визуализацию графа с помощью matplotlib.
    """
    graph = nx.Graph()
    graph.add_edges_from(vac_edges)
    nx.draw(graph, with_labels=False, font_weight="bold", node_size=30)
    plt.show()


def get_communities(graph):
    """
    Возвращает список сообществ (каждое — список названий вакансий),
    а также подграф этих узлов.
    """
    raw_communities = nx.community.louvain_communities(graph, resolution=0.8)

    # Преобразуем set → list и фильтруем по длине
    communities = [sorted(list(comm)) for comm in raw_communities if len(comm) > 5]

    if not communities:
        return [], nx.Graph()

    all_nodes = list(chain.from_iterable(communities))

    # Убедимся, что все узлы существуют в графе
    existing_nodes = [node for node in all_nodes if node in graph.nodes]

    if not existing_nodes:
        return [], nx.Graph()

    final_subgraph = graph.subgraph(existing_nodes).copy()
    return communities, final_subgraph


def create_community_node_colors(graph, communities):
    """
    Создание цветов узлов сообществ
    """
    colors = list(set(mcolors.TABLEAU_COLORS.values()))
    node_colors = []
    for node in graph:
        current_community_index = 0
        found = False
        for community in communities:
            if node in community:
                node_colors.append(colors[current_community_index])
                found = True
                break
            current_community_index += 1
        if not found:
            node_colors.append('#808080')
    return node_colors


def plot_communities(graph, communities):
    """
    Строит интерактивный график всех сообществ.
    При наведении отображает подпись: название вакансии + степень.
    """
    pos = nx.spring_layout(graph, iterations=500, seed=42, k=3 / math.sqrt(len(graph)), scale=10.0)


    edge_x, edge_y = [], []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])


    # Подписи для ховера
    node_degrees = [graph.degree(node) for node in graph.nodes()]
    hover_text = [f"Вакансия: {node}<br>Степень: {degree}" for node, degree in zip(graph.nodes(), node_degrees)]


    # Раскраска узлов по сообществам
    community_colors = list(mcolors.TABLEAU_COLORS.values())
    node_colors = []


    for node in graph.nodes():
        color_idx = 0
        found = False
        for comm in communities:
            if node in comm:
                found = True
                break
            color_idx += 1
        node_colors.append(community_colors[color_idx % len(community_colors)] if found else "#808080")


    fig = go.Figure()


    # Рёбра
    fig.add_trace(go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.5, color="gray"),
        hoverinfo="none"
    ))


    # Узлы с ховером
    fig.add_trace(go.Scatter(
        x=[pos[node][0] for node in graph.nodes()],
        y=[pos[node][1] for node in graph.nodes()],
        mode="markers",
        marker=dict(size=[d * 1.2 + 10 for d in node_degrees], color=node_colors),
        hoverinfo="text",
        text=hover_text
    ))


    fig.update_layout(
        title=f"Граф всех сообществ ({len(communities)} кластеров)",
        showlegend=False,
        plot_bgcolor="white",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )


    return fig


def plot_one_community(graph, community, all_communities=None):
    """
    Строит график одного сообщества.

    :param graph: полный граф (networkx.Graph)
    :param community: список вакансий, принадлежащих одному сообществу
    :param all_communities: список всех сообществ (для раскраски)
    :return: go.Figure
    """
    subgraph = graph.subgraph(community)

    # Расположение узлов
    pos = nx.spring_layout(subgraph, seed=30, k=3 / math.sqrt(len(subgraph)), scale=10.0)

    edge_x, edge_y = [], []
    for edge in subgraph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x = [pos[node][0] for node in subgraph.nodes()]
    node_y = [pos[node][1] for node in subgraph.nodes()]
    node_text = list(subgraph.nodes())

    # Используем all_communities для определения цвета
    colors = list(mcolors.TABLEAU_COLORS.values())
    community_index = 0

    if all_communities is not None:
        # Найдём индекс текущего сообщества среди всех сообществ
        found = False
        for idx, comm in enumerate(all_communities):
            if set(community) <= set(comm):
                community_index = idx
                found = True
                break
        color = colors[community_index % len(colors)]
    else:
        color = "skyblue"

    fig = go.Figure()

    # Рёбра
    fig.add_trace(go.Scatter(
        x=edge_x,
        y=edge_y,
        mode='lines',
        line=dict(width=0.5, color='gray'),
        hoverinfo='none'
    ))

    # Узлы с подписями и ховером
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        marker=dict(size=10, color=color),
        hoverinfo='text',
        hovertext=[f"Вакансия: {node}<br>Степень: {subgraph.degree(node)}" for node in node_text],
    ))

    fig.update_layout(
        title=f"Сообщество из {len(community)} вакансий",
        showlegend=False,
        plot_bgcolor="white",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=5, r=5, t=40, b=20)
    )

    return fig




