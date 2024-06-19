import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import networkx as nx


def get_umst_graph_figure(
    initial_graph: nx.Graph,
    umst_graph: nx.Graph,
) -> go.Figure:
    figure = go.Figure()
    get_graph_figure(
        initial_graph, figure=figure
    )
    get_graph_figure(umst_graph, edges_color="#e81095", nodes_color="#e81095", figure=figure)
    return figure


def get_umst_graph_abstract_figure(
    initial_graph: nx.Graph, umst_graph: nx.Graph
) -> go.Figure:
    figure = go.Figure()
    pos = nx.spring_layout(initial_graph, seed=7)
    get_graph_figure_abstract(
        initial_graph,
        pos=pos,
        figure=figure,
    )
    get_graph_figure_abstract(umst_graph, pos=pos, edges_color="#e81095", nodes_color="#e81095", figure=figure)
    return figure


def get_graph_figure_abstract(
    graph: nx.Graph,
    pos: dict = None,
    edges_color: str = "blue",
    nodes_color: str = "red",
    figure: go.Figure = None,
) -> go.Figure:
    if not figure:
        figure = go.Figure()
    if not pos:
        pos = nx.spring_layout(graph, seed=7)

    edge_x = []
    edge_y = []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x = []
    node_y = []
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    figure.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=2, color=edges_color),
            hoverinfo="none",
        )
    )

    figure.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            marker=dict(size=10, color=nodes_color),
            text=[data["capital"] for idx, data in graph.nodes(data=True)],
            textposition="top right",
            hoverinfo="text",
        )
    )

    center = get_graph_center_coords(graph)
    center_lat = center[0]
    center_lon = center[1]

    figure.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
        hovermode="closest",
        margin=dict(l=0, r=0, t=0, b=0),
    )

    return figure


def get_graph_figure(
    graph: nx.Graph,
    edges_color: str = "blue",
    nodes_color: str = "red",
    figure: go.Figure = None,
) -> go.Figure:
    edge_x = []
    edge_y = []
    for edge in graph.edges():
        x0, y0 = graph.nodes[edge[0]]["pos"]
        x1, y1 = graph.nodes[edge[1]]["pos"]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x = []
    node_y = []
    for node in graph.nodes():
        x, y = graph.nodes[node]["pos"]
        node_x.append(x)
        node_y.append(y)

    if figure:
        fig = figure
    else:
        fig = go.Figure()

    fig.add_trace(
        go.Scattermapbox(
            lon=edge_x,
            lat=edge_y,
            mode="lines",
            line=dict(width=2, color=edges_color),
            hoverinfo="none",
        )
    )

    fig.add_trace(
        go.Scattermapbox(
            lon=node_x,
            lat=node_y,
            mode="markers+text",
            marker=dict(size=10, color=nodes_color),
            text=[data["capital"] for idx, data in graph.nodes(data=True)],
            textposition="top right",
            hoverinfo="text",
        )
    )

    center = get_graph_center_coords(graph)
    center_lat = center[0]
    center_lon = center[1]
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(
            center=go.layout.mapbox.Center(
                lat=center_lat,
                lon=center_lon,
            ),
            zoom=3,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )

    return fig


def get_graph_center_coords(
    initial_graph: nx.Graph,
) -> tuple[float, float]:
    p = initial_graph.nodes(data="pos")
    lons = [pos[1][0] for pos in p]
    lats = [pos[1][1] for pos in p]
    min_lon = min(lons)
    max_lon = max(lons)
    min_lat = min(lats)
    max_lat = max(lats)

    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2

    return center_lat, center_lon


def get_graph(adjanecy: np.ndarray, capitals: pd.DataFrame) -> nx.Graph:
    graph_matrix = adjanecy.copy()
    graph = nx.Graph()
    n = len(capitals.index)
    for idx, row in capitals.iterrows():
        graph.add_node(
            idx,
            pos=(row["lon"], row["lat"]),
            capital=f'{row["capital"]}({idx})',
        )

    vweight = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i >= j:
                graph_matrix[i][j] = 0
            node_a, node_b = capitals.loc[i, "capacity"], capitals.loc[j, "capacity"]
            min_capacity = np.minimum(node_a, node_b)
            vweight[i][j] = min_capacity * graph_matrix[i][j]
            if graph_matrix[i][j] != 0:
                graph.add_edge(i, j, weight=vweight[i][j])
    return graph
