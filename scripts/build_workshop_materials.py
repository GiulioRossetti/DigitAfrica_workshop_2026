from __future__ import annotations

import sqlite3
import textwrap
from pathlib import Path

import nbformat as nbf
import networkx as nx
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
MODULES_DIR = NOTEBOOKS_DIR / "modules"
EXERCISES_DIR = NOTEBOOKS_DIR / "exercises"
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip() + "\n")


def write_notebook(path: Path, cells: list) -> None:
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(nbf.writes(notebook), encoding="utf-8")


def build_demo_graph(seed: int = 42) -> nx.Graph:
    rng = np.random.default_rng(seed)
    communities = [
        {"label": "left-mainstream", "camp": "left", "size": 36, "mean_opinion": -0.55, "enclave": 0},
        {"label": "left-enclave", "camp": "left", "size": 18, "mean_opinion": -0.88, "enclave": 1},
        {"label": "right-mainstream", "camp": "right", "size": 36, "mean_opinion": 0.55, "enclave": 0},
        {"label": "right-enclave", "camp": "right", "size": 18, "mean_opinion": 0.88, "enclave": 1},
    ]
    sizes = [c["size"] for c in communities]
    probs = [
        [0.18, 0.08, 0.02, 0.004],
        [0.08, 0.25, 0.006, 0.001],
        [0.02, 0.006, 0.18, 0.08],
        [0.004, 0.001, 0.08, 0.25],
    ]
    graph = nx.stochastic_block_model(sizes, probs, seed=seed)
    graph.graph.clear()

    node_id = 0
    for community in communities:
        for _ in range(community["size"]):
            graph.nodes[node_id].update(
                {
                    "label": f"user_{node_id:03d}",
                    "community_label": community["label"],
                    "camp": community["camp"],
                    "opinion": float(np.clip(rng.normal(community["mean_opinion"], 0.09), -1.0, 1.0)),
                    "enclave": int(community["enclave"]),
                    "activity": int(rng.poisson(9 if community["enclave"] else 6) + 1),
                }
            )
            node_id += 1
    return graph


def write_demo_graph_files(seed: int = 42) -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    graph = build_demo_graph(seed=seed)

    nodes_df = pd.DataFrame(
        [
            {"node_id": node, **attrs}
            for node, attrs in graph.nodes(data=True)
        ]
    )
    edges_df = pd.DataFrame([{"source": u, "target": v} for u, v in graph.edges()])

    nx.write_graphml(graph, DATA_RAW / "workshop_network.graphml")
    nx.write_gexf(graph, DATA_RAW / "workshop_network.gexf")
    nx.write_edgelist(graph, DATA_RAW / "workshop_network.edgelist", data=False)
    nodes_df.to_csv(DATA_RAW / "workshop_nodes.csv", index=False)
    edges_df.to_csv(DATA_PROCESSED / "workshop_edges.csv", index=False)


def write_ysocial_demo_db(path: Path | None = None, seed: int = 42) -> None:
    db_path = path or DATA_RAW / "ysocial_demo.sqlite"
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        return

    rng = np.random.default_rng(seed)
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    cursor.executescript(
        """
        CREATE TABLE rounds (id INTEGER PRIMARY KEY, day INTEGER, hour INTEGER);
        CREATE TABLE user_mgmt (
            id INTEGER PRIMARY KEY,
            username TEXT,
            avatar TEXT,
            bio TEXT,
            role TEXT,
            leaning TEXT,
            age INTEGER,
            oe REAL,
            co REAL,
            ex REAL,
            ag REAL,
            ne REAL,
            content_recsys REAL,
            language TEXT,
            region TEXT,
            education TEXT,
            joined INTEGER,
            social_recsys REAL,
            gender TEXT,
            nationality TEXT,
            toxicity REAL,
            is_page INTEGER,
            left_on INTEGER,
            daily_activity_level REAL,
            profession TEXT
        );
        CREATE TABLE follow (user_id INTEGER, follower_id INTEGER, action TEXT, round INTEGER);
        CREATE TABLE post (
            id INTEGER PRIMARY KEY,
            text TEXT,
            media TEXT,
            user_id INTEGER,
            comment_to INTEGER,
            thread_id INTEGER,
            round INTEGER,
            news_id INTEGER,
            shared_from INTEGER,
            image_id INTEGER
        );
        CREATE TABLE mentions (post_id INTEGER, user_id INTEGER);
        CREATE TABLE hashtags (id INTEGER PRIMARY KEY, hashtag TEXT);
        CREATE TABLE post_hashtags (post_id INTEGER, hashtag_id INTEGER);
        CREATE TABLE interests (iid INTEGER PRIMARY KEY, interest TEXT);
        CREATE TABLE user_interest (user_id INTEGER, interest_id INTEGER, round INTEGER);
        CREATE TABLE post_topics (post_id INTEGER, topic_id INTEGER);
        CREATE TABLE emotions (id INTEGER PRIMARY KEY, emotion TEXT);
        CREATE TABLE post_emotions (post_id INTEGER, emotion_id INTEGER);
        CREATE TABLE post_toxicity (
            post_id INTEGER PRIMARY KEY,
            model TEXT,
            toxicity REAL,
            severe_toxicity REAL,
            identity_attack REAL,
            insult REAL,
            profanity REAL,
            threat REAL,
            sexual_explicit REAL,
            flirtation REAL
        );
        """
    )

    for round_id in range(24):
        cursor.execute("INSERT INTO rounds (id, day, hour) VALUES (?, ?, ?)", (round_id, 1 + round_id // 12, round_id % 12))

    for hashtag_id, hashtag in {1: "#civicdata", 2: "#community", 3: "#publichealth", 4: "#marketfreedom", 5: "#innovation", 6: "#trustednews"}.items():
        cursor.execute("INSERT INTO hashtags (id, hashtag) VALUES (?, ?)", (hashtag_id, hashtag))

    for interest_id, interest in {1: "health", 2: "governance", 3: "innovation", 4: "markets", 5: "media"}.items():
        cursor.execute("INSERT INTO interests (iid, interest) VALUES (?, ?)", (interest_id, interest))

    for emotion_id, emotion in {1: "joy", 2: "anger", 3: "fear", 4: "hope"}.items():
        cursor.execute("INSERT INTO emotions (id, emotion) VALUES (?, ?)", (emotion_id, emotion))

    agents = []
    for agent_id in range(1, 21):
        leaning = "left" if agent_id <= 10 else "right"
        enclave = agent_id in {3, 4, 5, 13, 14, 15}
        profession = "journalist" if agent_id % 5 == 0 else "researcher"
        agents.append(
            (
                agent_id, f"agent_{agent_id:02d}", None, None, "user", leaning, int(rng.integers(22, 60)),
                round(float(rng.uniform(0.3, 0.9)), 3), round(float(rng.uniform(0.3, 0.9)), 3),
                round(float(rng.uniform(0.3, 0.9)), 3), round(float(rng.uniform(0.3, 0.9)), 3),
                round(float(rng.uniform(0.3, 0.9)), 3), 0.85 if enclave else 0.35, "en", "EU", "graduate",
                0, 0.8 if enclave else 0.4, "female" if agent_id % 2 == 0 else "male",
                "Kenya" if agent_id % 3 == 0 else "Italy", 0.45 if enclave else 0.18, 0, None,
                2.5 if enclave else 1.7, profession,
            )
        )

    cursor.executemany(
        '''
        INSERT INTO user_mgmt (
            id, username, avatar, bio, role, leaning, age, oe, co, ex, ag, ne,
            content_recsys, language, region, education, joined, social_recsys,
            gender, nationality, toxicity, is_page, left_on, daily_activity_level, profession
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''',
        agents,
    )

    follow_edges = [
        (1, 2), (1, 3), (1, 6), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5), (4, 7), (5, 8),
        (6, 1), (6, 2), (7, 1), (7, 3), (8, 4), (8, 5), (9, 2), (9, 10), (10, 1), (10, 6),
        (11, 12), (11, 13), (11, 16), (12, 13), (12, 14), (13, 14), (13, 15), (14, 15), (14, 17), (15, 18),
        (16, 11), (16, 12), (17, 11), (17, 13), (18, 14), (18, 15), (19, 12), (19, 20), (20, 11), (20, 16),
        (6, 11), (10, 12), (16, 1), (17, 2),
    ]
    for round_id, (follower_id, user_id) in enumerate(follow_edges, start=1):
        cursor.execute(
            "INSERT INTO follow (user_id, follower_id, action, round) VALUES (?, ?, ?, ?)",
            (user_id, follower_id, "follow", round_id % 24),
        )

    post_id = 1
    for round_id in range(24):
        for user_id in rng.choice(np.arange(1, 21), size=4, replace=False):
            leaning = "left" if user_id <= 10 else "right"
            enclave = user_id in {3, 4, 5, 13, 14, 15}
            hashtags = [1, 6] if leaning == "left" else [4, 5]
            hashtags.append(2 if not enclave else 3)
            topic_id = 2 if leaning == "left" else 4
            if enclave:
                topic_id = 5
            emotion_id = 2 if enclave else 4
            mention_target = int(rng.choice(np.arange(1, 11) if leaning == "left" else np.arange(11, 21)))
            if round_id % 7 == 0:
                mention_target = 12 if leaning == "left" else 2

            cursor.execute(
                '''
                INSERT INTO post (id, text, media, user_id, comment_to, thread_id, round, news_id, shared_from, image_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    post_id,
                    f"Round {round_id}: {'solidarity' if leaning == 'left' else 'competition'} and platform curation shape what we see.",
                    None,
                    int(user_id),
                    None,
                    round_id,
                    round_id,
                    None,
                    None,
                    None,
                ),
            )
            cursor.execute("INSERT INTO mentions (post_id, user_id) VALUES (?, ?)", (post_id, mention_target))
            for hashtag_id in hashtags:
                cursor.execute("INSERT INTO post_hashtags (post_id, hashtag_id) VALUES (?, ?)", (post_id, hashtag_id))
            cursor.execute("INSERT INTO post_topics (post_id, topic_id) VALUES (?, ?)", (post_id, topic_id))
            cursor.execute("INSERT INTO post_emotions (post_id, emotion_id) VALUES (?, ?)", (post_id, emotion_id))
            cursor.execute(
                '''
                INSERT INTO post_toxicity (
                    post_id, model, toxicity, severe_toxicity, identity_attack,
                    insult, profanity, threat, sexual_explicit, flirtation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (post_id, "demo", 0.42 if enclave else 0.17, 0.1 if enclave else 0.03, 0.05 if enclave else 0.02, 0.18 if enclave else 0.05, 0.12 if enclave else 0.04, 0.02, 0.0, 0.0),
            )
            post_id += 1

    for agent_id in range(1, 21):
        leaning = "left" if agent_id <= 10 else "right"
        for round_id, interest_id in enumerate(([1, 2, 5] if leaning == "left" else [3, 4, 5]), start=1):
            cursor.execute("INSERT INTO user_interest (user_id, interest_id, round) VALUES (?, ?, ?)", (agent_id, interest_id, round_id + agent_id))

    connection.commit()
    connection.close()


COMMON_SETUP = """
from pathlib import Path
import os

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "data").exists():
    PROJECT_ROOT = PROJECT_ROOT.parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
CACHE_DIR = PROJECT_ROOT / ".cache"
(CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
"""


DEMO_GRAPH_CODE = """
def build_demo_graph(seed=42):
    rng = np.random.default_rng(seed)
    communities = [
        {"label": "left-mainstream", "camp": "left", "size": 36, "mean_opinion": -0.55, "enclave": 0},
        {"label": "left-enclave", "camp": "left", "size": 18, "mean_opinion": -0.88, "enclave": 1},
        {"label": "right-mainstream", "camp": "right", "size": 36, "mean_opinion": 0.55, "enclave": 0},
        {"label": "right-enclave", "camp": "right", "size": 18, "mean_opinion": 0.88, "enclave": 1},
    ]
    sizes = [community["size"] for community in communities]
    probability_matrix = [
        [0.18, 0.08, 0.02, 0.004],
        [0.08, 0.25, 0.006, 0.001],
        [0.02, 0.006, 0.18, 0.08],
        [0.004, 0.001, 0.08, 0.25],
    ]

    graph = nx.stochastic_block_model(sizes, probability_matrix, seed=seed)
    graph.graph.clear()

    node_id = 0
    for community in communities:
        for _ in range(community["size"]):
            graph.nodes[node_id]["label"] = f"user_{node_id:03d}"
            graph.nodes[node_id]["community_label"] = community["label"]
            graph.nodes[node_id]["camp"] = community["camp"]
            graph.nodes[node_id]["opinion"] = float(np.clip(rng.normal(community["mean_opinion"], 0.09), -1.0, 1.0))
            graph.nodes[node_id]["enclave"] = int(community["enclave"])
            graph.nodes[node_id]["activity"] = int(rng.poisson(9 if community["enclave"] else 6) + 1)
            node_id += 1

    return graph


def write_demo_graph_files(seed=42):
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    graph = build_demo_graph(seed=seed)

    node_frame = pd.DataFrame(
        [{"node_id": node, **attrs} for node, attrs in graph.nodes(data=True)]
    )
    edge_frame = pd.DataFrame(
        [{"source": source, "target": target} for source, target in graph.edges()]
    )

    nx.write_graphml(graph, DATA_RAW / "workshop_network.graphml")
    nx.write_gexf(graph, DATA_RAW / "workshop_network.gexf")
    nx.write_edgelist(graph, DATA_RAW / "workshop_network.edgelist", data=False)
    node_frame.to_csv(DATA_RAW / "workshop_nodes.csv", index=False)
    edge_frame.to_csv(DATA_PROCESSED / "workshop_edges.csv", index=False)


def load_graph(path):
    path = Path(path)
    if path.suffix == ".graphml":
        graph = nx.read_graphml(path)
    elif path.suffix == ".gexf":
        graph = nx.read_gexf(path)
    else:
        graph = nx.read_edgelist(path, nodetype=int)
        node_frame = pd.read_csv(DATA_RAW / "workshop_nodes.csv")
        attributes = node_frame.set_index("node_id").to_dict(orient="index")
        nx.set_node_attributes(graph, attributes)

    graph = nx.convert_node_labels_to_integers(graph, label_attribute="original_id")
    for _, attrs in graph.nodes(data=True):
        attrs["opinion"] = float(attrs["opinion"])
        attrs["activity"] = int(attrs["activity"])
        attrs["enclave"] = int(attrs["enclave"])
    return graph
"""


YSOCIAL_DB_CODE = """
def ensure_ysocial_demo_db(db_path=DATA_RAW / "ysocial_demo.sqlite", seed=42):
    if Path(db_path).exists():
        return Path(db_path)

    rng = np.random.default_rng(seed)
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    cursor.executescript(
        '''
        CREATE TABLE rounds (id INTEGER PRIMARY KEY, day INTEGER, hour INTEGER);
        CREATE TABLE user_mgmt (
            id INTEGER PRIMARY KEY,
            username TEXT,
            avatar TEXT,
            bio TEXT,
            role TEXT,
            leaning TEXT,
            age INTEGER,
            oe REAL,
            co REAL,
            ex REAL,
            ag REAL,
            ne REAL,
            content_recsys REAL,
            language TEXT,
            region TEXT,
            education TEXT,
            joined INTEGER,
            social_recsys REAL,
            gender TEXT,
            nationality TEXT,
            toxicity REAL,
            is_page INTEGER,
            left_on INTEGER,
            daily_activity_level REAL,
            profession TEXT
        );
        CREATE TABLE follow (user_id INTEGER, follower_id INTEGER, action TEXT, round INTEGER);
        CREATE TABLE post (
            id INTEGER PRIMARY KEY,
            text TEXT,
            media TEXT,
            user_id INTEGER,
            comment_to INTEGER,
            thread_id INTEGER,
            round INTEGER,
            news_id INTEGER,
            shared_from INTEGER,
            image_id INTEGER
        );
        CREATE TABLE mentions (post_id INTEGER, user_id INTEGER);
        CREATE TABLE hashtags (id INTEGER PRIMARY KEY, hashtag TEXT);
        CREATE TABLE post_hashtags (post_id INTEGER, hashtag_id INTEGER);
        CREATE TABLE interests (iid INTEGER PRIMARY KEY, interest TEXT);
        CREATE TABLE user_interest (user_id INTEGER, interest_id INTEGER, round INTEGER);
        CREATE TABLE post_topics (post_id INTEGER, topic_id INTEGER);
        CREATE TABLE emotions (id INTEGER PRIMARY KEY, emotion TEXT);
        CREATE TABLE post_emotions (post_id INTEGER, emotion_id INTEGER);
        CREATE TABLE post_toxicity (
            post_id INTEGER PRIMARY KEY,
            model TEXT,
            toxicity REAL,
            severe_toxicity REAL,
            identity_attack REAL,
            insult REAL,
            profanity REAL,
            threat REAL,
            sexual_explicit REAL,
            flirtation REAL
        );
        '''
    )

    for round_id in range(24):
        cursor.execute("INSERT INTO rounds (id, day, hour) VALUES (?, ?, ?)", (round_id, 1 + round_id // 12, round_id % 12))

    for hashtag_id, hashtag in {1: "#civicdata", 2: "#community", 3: "#publichealth", 4: "#marketfreedom", 5: "#innovation", 6: "#trustednews"}.items():
        cursor.execute("INSERT INTO hashtags (id, hashtag) VALUES (?, ?)", (hashtag_id, hashtag))
    for interest_id, interest in {1: "health", 2: "governance", 3: "innovation", 4: "markets", 5: "media"}.items():
        cursor.execute("INSERT INTO interests (iid, interest) VALUES (?, ?)", (interest_id, interest))
    for emotion_id, emotion in {1: "joy", 2: "anger", 3: "fear", 4: "hope"}.items():
        cursor.execute("INSERT INTO emotions (id, emotion) VALUES (?, ?)", (emotion_id, emotion))

    agents = []
    for agent_id in range(1, 21):
        leaning = "left" if agent_id <= 10 else "right"
        enclave = agent_id in {3, 4, 5, 13, 14, 15}
        profession = "journalist" if agent_id % 5 == 0 else "researcher"
        agents.append(
            (
                agent_id, f"agent_{agent_id:02d}", None, None, "user", leaning, int(rng.integers(22, 60)),
                round(float(rng.uniform(0.3, 0.9)), 3), round(float(rng.uniform(0.3, 0.9)), 3),
                round(float(rng.uniform(0.3, 0.9)), 3), round(float(rng.uniform(0.3, 0.9)), 3),
                round(float(rng.uniform(0.3, 0.9)), 3), 0.85 if enclave else 0.35, "en", "EU", "graduate",
                0, 0.8 if enclave else 0.4, "female" if agent_id % 2 == 0 else "male",
                "Kenya" if agent_id % 3 == 0 else "Italy", 0.45 if enclave else 0.18, 0, None,
                2.5 if enclave else 1.7, profession,
            )
        )

    cursor.executemany(
        '''
        INSERT INTO user_mgmt (
            id, username, avatar, bio, role, leaning, age, oe, co, ex, ag, ne,
            content_recsys, language, region, education, joined, social_recsys,
            gender, nationality, toxicity, is_page, left_on, daily_activity_level, profession
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''',
        agents,
    )

    follow_edges = [
        (1, 2), (1, 3), (1, 6), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5), (4, 7), (5, 8),
        (6, 1), (6, 2), (7, 1), (7, 3), (8, 4), (8, 5), (9, 2), (9, 10), (10, 1), (10, 6),
        (11, 12), (11, 13), (11, 16), (12, 13), (12, 14), (13, 14), (13, 15), (14, 15), (14, 17), (15, 18),
        (16, 11), (16, 12), (17, 11), (17, 13), (18, 14), (18, 15), (19, 12), (19, 20), (20, 11), (20, 16),
        (6, 11), (10, 12), (16, 1), (17, 2),
    ]
    for round_id, (follower_id, user_id) in enumerate(follow_edges, start=1):
        cursor.execute(
            "INSERT INTO follow (user_id, follower_id, action, round) VALUES (?, ?, ?, ?)",
            (user_id, follower_id, "follow", round_id % 24),
        )

    post_id = 1
    for round_id in range(24):
        for user_id in rng.choice(np.arange(1, 21), size=4, replace=False):
            leaning = "left" if user_id <= 10 else "right"
            enclave = user_id in {3, 4, 5, 13, 14, 15}
            hashtags = [1, 6] if leaning == "left" else [4, 5]
            hashtags.append(2 if not enclave else 3)
            topic_id = 2 if leaning == "left" else 4
            if enclave:
                topic_id = 5
            emotion_id = 2 if enclave else 4
            mention_target = int(rng.choice(np.arange(1, 11) if leaning == "left" else np.arange(11, 21)))
            if round_id % 7 == 0:
                mention_target = 12 if leaning == "left" else 2

            cursor.execute(
                '''
                INSERT INTO post (id, text, media, user_id, comment_to, thread_id, round, news_id, shared_from, image_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    post_id,
                    f"Round {round_id}: {'solidarity' if leaning == 'left' else 'competition'} and platform curation shape what we see.",
                    None,
                    int(user_id),
                    None,
                    round_id,
                    round_id,
                    None,
                    None,
                    None,
                ),
            )
            cursor.execute("INSERT INTO mentions (post_id, user_id) VALUES (?, ?)", (post_id, mention_target))
            for hashtag_id in hashtags:
                cursor.execute("INSERT INTO post_hashtags (post_id, hashtag_id) VALUES (?, ?)", (post_id, hashtag_id))
            cursor.execute("INSERT INTO post_topics (post_id, topic_id) VALUES (?, ?)", (post_id, topic_id))
            cursor.execute("INSERT INTO post_emotions (post_id, emotion_id) VALUES (?, ?)", (post_id, emotion_id))
            cursor.execute(
                '''
                INSERT INTO post_toxicity (
                    post_id, model, toxicity, severe_toxicity, identity_attack,
                    insult, profanity, threat, sexual_explicit, flirtation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (post_id, "demo", 0.42 if enclave else 0.17, 0.1 if enclave else 0.03, 0.05 if enclave else 0.02, 0.18 if enclave else 0.05, 0.12 if enclave else 0.04, 0.02, 0.0, 0.0),
            )
            post_id += 1

    for agent_id in range(1, 21):
        leaning = "left" if agent_id <= 10 else "right"
        interest_ids = [1, 2, 5] if leaning == "left" else [3, 4, 5]
        for round_id, interest_id in enumerate(interest_ids, start=1):
            cursor.execute("INSERT INTO user_interest (user_id, interest_id, round) VALUES (?, ?, ?)", (agent_id, interest_id, round_id + agent_id))

    connection.commit()
    connection.close()
    return Path(db_path)
"""


def build_environment_check() -> None:
    cells = [
        md(
            """
            # Notebook 00 - Environment Check

            This pre-flight notebook verifies the workshop stack and creates the demo graph files plus the YSocial-style SQLite database used later in the day.
            """
        ),
        code(
            COMMON_SETUP
            + """
import importlib
import sqlite3

import networkx as nx
import numpy as np
import pandas as pd
"""
        ),
        code(DEMO_GRAPH_CODE + "\n" + YSOCIAL_DB_CODE),
        code(
            """
required = ["numpy", "pandas", "matplotlib", "networkx", "cdlib", "ndlib", "sklearn", "ysights"]
for package in required:
    module = importlib.import_module(package)
    print(f"{package:10s} OK  {getattr(module, '__file__', 'built-in')}")
"""
        ),
        code(
            """
write_demo_graph_files()
db_path = ensure_ysocial_demo_db()

print("Created demo assets:")
for path in sorted((DATA_RAW).glob("*")):
    print(" ", path.relative_to(PROJECT_ROOT))
print(" ", (DATA_PROCESSED / "workshop_edges.csv").relative_to(PROJECT_ROOT))
print("SQLite DB:", db_path.relative_to(PROJECT_ROOT))
"""
        ),
    ]
    write_notebook(NOTEBOOKS_DIR / "00_environment_check.ipynb", cells)


def build_module_1() -> None:
    cells = [
        md(
            """
            # Module 1 - Foundations: Mapping the Terrain

            **Time:** 09:30-11:00  
            **Tool focus:** NetworkX

            This notebook follows the first workshop block from the schedule PDF: loading standard graph formats, measuring baseline connectivity, extracting the network core, and visually inspecting segregation.
            """
        ),
        code(
            COMMON_SETUP
            + """
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(context="talk", style="whitegrid")
"""
        ),
        code(DEMO_GRAPH_CODE),
        md(
            """
            ## 1. Create or refresh the demo files

            The schedule explicitly mentions GraphML, GEXF, and edge lists. We create all three so participants can see how the same network is represented in different formats.
            """
        ),
        code(
            """
write_demo_graph_files()

paths = {
    "graphml": DATA_RAW / "workshop_network.graphml",
    "gexf": DATA_RAW / "workshop_network.gexf",
    "edgelist": DATA_RAW / "workshop_network.edgelist",
}
paths
"""
        ),
        code(
            """
graphml_graph = load_graph(paths["graphml"])
gexf_graph = load_graph(paths["gexf"])
edgelist_graph = load_graph(paths["edgelist"])

pd.DataFrame(
    [
        {"format": "GraphML", "nodes": graphml_graph.number_of_nodes(), "edges": graphml_graph.number_of_edges()},
        {"format": "GEXF", "nodes": gexf_graph.number_of_nodes(), "edges": gexf_graph.number_of_edges()},
        {"format": "Edge list", "nodes": edgelist_graph.number_of_nodes(), "edges": edgelist_graph.number_of_edges()},
    ]
)
"""
        ),
        md("## 2. Establish a baseline for connectivity"),
        code(
            """
G = graphml_graph.copy()
largest_component = G.subgraph(max(nx.connected_components(G), key=len)).copy()

baseline = pd.Series(
    {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "average_degree": np.mean([degree for _, degree in G.degree()]),
        "average_clustering": nx.average_clustering(G),
        "average_path_length_lcc": nx.average_shortest_path_length(largest_component),
        "transitivity": nx.transitivity(G),
    }
).round(4)
baseline
"""
        ),
        code(
            """
degree_stats = pd.DataFrame(
    [
        {
            "node_id": node,
            "degree": G.degree(node),
            "clustering": nx.clustering(G, node),
            "camp": G.nodes[node]["camp"],
            "community_label": G.nodes[node]["community_label"],
            "opinion": G.nodes[node]["opinion"],
        }
        for node in G.nodes()
    ]
).sort_values(["degree", "node_id"], ascending=[False, True])

degree_stats.head(10)
"""
        ),
        code(
            """
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(degree_stats, x="degree", hue="camp", bins=15, ax=axes[0])
axes[0].set_title("Degree distribution")

sns.boxplot(data=degree_stats, x="camp", y="clustering", ax=axes[1])
axes[1].set_title("Local clustering by camp")

plt.tight_layout()
"""
        ),
        md(
            """
            ## 3. Separate the core from the periphery

            We first inspect the degree distribution, then derive a `k` value from it. This keeps the filtering logic visible instead of treating the core extraction as a black box.
            """
        ),
        code(
            """
degree_values = np.array([degree for _, degree in G.degree()])
k_threshold = max(2, int(np.quantile(degree_values, 0.7)))
core = nx.k_core(G, k=k_threshold)

pd.Series(
    {
        "k_threshold": k_threshold,
        "core_nodes": core.number_of_nodes(),
        "core_edges": core.number_of_edges(),
        "core_density": round(nx.density(core), 4),
    }
)
"""
        ),
        code(
            """
pos = nx.spring_layout(G, seed=7)
camp_colors = ["#33658A" if G.nodes[node]["camp"] == "left" else "#D1495B" for node in G.nodes()]
core_nodes = set(core.nodes())

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
nx.draw_networkx(G, pos=pos, node_color=camp_colors, node_size=50, edge_color="#BBBBBB", alpha=0.7, with_labels=False, ax=axes[0])
axes[0].set_title("Full network")

nx.draw_networkx(
    G,
    pos=pos,
    node_color=["#F4D35E" if node in core_nodes else "#DDDDDD" for node in G.nodes()],
    node_size=60,
    edge_color="#BBBBBB",
    alpha=0.7,
    with_labels=False,
    ax=axes[1],
)
axes[1].set_title("Core participants highlighted")
plt.tight_layout()
"""
        ),
    ]
    write_notebook(MODULES_DIR / "01_foundations_mapping_the_terrain.ipynb", cells)


def build_module_2() -> None:
    cells = [
        md(
            """
            # Module 2 - Detecting Epistemic Enclaves

            **Time:** 11:15-12:45  
            **Tool focus:** CDlib

            This notebook compares several community-detection algorithms and then evaluates the resulting clusters as potential epistemic enclaves.
            """
        ),
        code(
            COMMON_SETUP
            + """
import networkx as nx
import numpy as np
import pandas as pd
from cdlib import algorithms
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
"""
        ),
        code(DEMO_GRAPH_CODE),
        code(
            """
write_demo_graph_files()
G = load_graph(DATA_RAW / "workshop_network.graphml")
"""
        ),
        md("## 1. Run several community-detection algorithms"),
        code(
            """
louvain = algorithms.louvain(G)
leiden = algorithms.leiden(G)
infomap = algorithms.infomap(G)
label_prop = algorithms.label_propagation(G)

pd.DataFrame(
    [
        {"algorithm": "Louvain", "communities": len(louvain.communities)},
        {"algorithm": "Leiden", "communities": len(leiden.communities)},
        {"algorithm": "Infomap", "communities": len(infomap.communities)},
        {"algorithm": "Label propagation", "communities": len(label_prop.communities)},
    ]
)
"""
        ),
        code(
            """
def partition_frame(partition, column_name):
    rows = []
    for cluster_id, community in enumerate(partition.communities):
        for node in community:
            rows.append({"node_id": int(node), column_name: cluster_id})
    frame = pd.DataFrame(rows)
    node_attributes = pd.DataFrame(
        [
            {
                "node_id": node,
                "camp": G.nodes[node]["camp"],
                "community_label": G.nodes[node]["community_label"],
                "enclave": G.nodes[node]["enclave"],
            }
            for node in G.nodes()
        ]
    )
    return node_attributes.merge(frame, on="node_id", how="left")
"""
        ),
        code(
            """
louvain_frame = partition_frame(louvain, "cluster")
leiden_frame = partition_frame(leiden, "cluster")
infomap_frame = partition_frame(infomap, "cluster")
label_prop_frame = partition_frame(label_prop, "cluster")

pd.DataFrame(
    [
        {
            "algorithm": "Louvain vs Louvain",
            "ARI": 1.0,
            "NMI": 1.0,
        },
        {
            "algorithm": "Leiden vs Louvain",
            "ARI": round(adjusted_rand_score(louvain_frame["cluster"], leiden_frame["cluster"]), 4),
            "NMI": round(normalized_mutual_info_score(louvain_frame["cluster"], leiden_frame["cluster"]), 4),
        },
        {
            "algorithm": "Infomap vs Louvain",
            "ARI": round(adjusted_rand_score(louvain_frame["cluster"], infomap_frame["cluster"]), 4),
            "NMI": round(normalized_mutual_info_score(louvain_frame["cluster"], infomap_frame["cluster"]), 4),
        },
        {
            "algorithm": "Label propagation vs Louvain",
            "ARI": round(adjusted_rand_score(louvain_frame["cluster"], label_prop_frame["cluster"]), 4),
            "NMI": round(normalized_mutual_info_score(louvain_frame["cluster"], label_prop_frame["cluster"]), 4),
        },
    ]
)
"""
        ),
        md("## 2. Global scale: assortativity"),
        code(
            """
camp_assortativity = nx.attribute_assortativity_coefficient(G, "camp")
enclave_assortativity = nx.attribute_assortativity_coefficient(
    nx.relabel_nodes(G, {node: node for node in G.nodes()}), "enclave"
)

pd.Series(
    {
        "camp_assortativity": round(camp_assortativity, 4),
        "enclave_assortativity": round(enclave_assortativity, 4),
    }
)
"""
        ),
        md("## 3. Meso scale: purity and separation"),
        code(
            """
community_rows = []
for cluster_id, chunk in louvain_frame.groupby("cluster"):
    nodes = set(chunk["node_id"])
    subgraph = G.subgraph(nodes)
    internal_edges = subgraph.number_of_edges()
    external_edges = nx.cut_size(G, nodes)
    dominant_camp = chunk["camp"].mode().iat[0]
    purity = (chunk["camp"] == dominant_camp).mean()
    separation = internal_edges / max(internal_edges + external_edges, 1)
    community_rows.append(
        {
            "cluster": cluster_id,
            "size": len(nodes),
            "purity": round(float(purity), 4),
            "internal_edges": internal_edges,
            "external_edges": int(external_edges),
            "separation": round(float(separation), 4),
        }
    )

pd.DataFrame(community_rows).sort_values("cluster").reset_index(drop=True)
"""
        ),
        md("## 4. Local scale: node conformity"),
        code(
            """
conformity_rows = []
for node in G.nodes():
    neighbors = list(G.neighbors(node))
    same_neighbor_share = np.nan
    if neighbors:
        same_neighbor_share = np.mean(
            [G.nodes[neighbor]["camp"] == G.nodes[node]["camp"] for neighbor in neighbors]
        )
    conformity_rows.append(
        {
            "node_id": node,
            "camp": G.nodes[node]["camp"],
            "community_label": G.nodes[node]["community_label"],
            "same_neighbor_share": same_neighbor_share,
        }
    )

conformity = pd.DataFrame(conformity_rows)
conformity.groupby("camp")["same_neighbor_share"].describe().round(3)
"""
        ),
    ]
    write_notebook(MODULES_DIR / "02_detecting_epistemic_enclaves.ipynb", cells)


def build_module_3() -> None:
    cells = [
        md(
            """
            # Module 3 - Simulating Polarisation Dynamics

            **Time:** 13:45-15:15  
            **Tool focus:** NDlib plus explicit bounded-confidence simulation code

            The notebook keeps the simulation logic visible so participants can see exactly how confidence thresholds and biased exposure alter the opinion trajectories.
            """
        ),
        code(
            COMMON_SETUP
            + """
import matplotlib.pyplot as plt
import ndlib
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(context="talk", style="whitegrid")
print("NDlib version:", getattr(ndlib, "__version__", "installed"))
"""
        ),
        code(DEMO_GRAPH_CODE),
        code(
            """
write_demo_graph_files()
G = load_graph(DATA_RAW / "workshop_network.graphml")
"""
        ),
        code(
            """
def initialize_opinions(graph):
    return {node: float(graph.nodes[node]["opinion"]) for node in graph.nodes()}


def record_history(step, opinions, sample_nodes, records):
    for node in sample_nodes:
        records.append({"step": step, "node_id": node, "opinion": opinions[node]})


def count_clusters(opinions, tolerance=0.08):
    values = sorted(opinions.values())
    if not values:
        return 0
    clusters = 1
    for previous, current in zip(values, values[1:]):
        if abs(current - previous) > tolerance:
            clusters += 1
    return clusters


def summarise_state(label, opinions):
    values = np.array(list(opinions.values()))
    return {
        "scenario": label,
        "mean": round(float(values.mean()), 4),
        "std": round(float(values.std()), 4),
        "clusters": count_clusters(opinions),
        "range": round(float(values.max() - values.min()), 4),
    }
"""
        ),
        code(
            """
def run_deffuant(graph, epsilon, mu=0.35, steps=2500, seed=42, sample_every=25):
    rng = np.random.default_rng(seed)
    edges = list(graph.edges())
    opinions = initialize_opinions(graph)
    sample_nodes = sorted(rng.choice(list(graph.nodes()), size=min(12, graph.number_of_nodes()), replace=False).tolist())
    records = []
    record_history(0, opinions, sample_nodes, records)

    for step in range(1, steps + 1):
        source, target = edges[rng.integers(len(edges))]
        if abs(opinions[source] - opinions[target]) <= epsilon:
            delta = mu * (opinions[target] - opinions[source])
            opinions[source] += delta
            opinions[target] -= delta
        if step % sample_every == 0:
            record_history(step, opinions, sample_nodes, records)

    return pd.DataFrame(records), opinions


def run_biased_exposure(graph, epsilon, bias_strength=6.0, mu=0.35, steps=2500, seed=42, sample_every=25):
    rng = np.random.default_rng(seed)
    opinions = initialize_opinions(graph)
    sample_nodes = sorted(rng.choice(list(graph.nodes()), size=min(12, graph.number_of_nodes()), replace=False).tolist())
    records = []
    record_history(0, opinions, sample_nodes, records)

    nodes = list(graph.nodes())
    for step in range(1, steps + 1):
        source = int(nodes[rng.integers(len(nodes))])
        neighbors = list(graph.neighbors(source))
        if neighbors:
            distances = np.array([abs(opinions[source] - opinions[target]) for target in neighbors])
            weights = np.exp(-bias_strength * distances)
            weights = weights / weights.sum()
            target = int(rng.choice(neighbors, p=weights))
            if abs(opinions[source] - opinions[target]) <= epsilon:
                delta = mu * (opinions[target] - opinions[source])
                opinions[source] += delta
                opinions[target] -= delta
        if step % sample_every == 0:
            record_history(step, opinions, sample_nodes, records)

    return pd.DataFrame(records), opinions
"""
        ),
        md("## 1. Phase transition under different confidence thresholds"),
        code(
            """
epsilons = [0.45, 0.25, 0.10]
dw_summaries = []
dw_histories = []

for epsilon in epsilons:
    history, final_state = run_deffuant(G, epsilon=epsilon, seed=42 + int(epsilon * 100))
    dw_histories.append(history.assign(model=f"DW epsilon={epsilon}"))
    dw_summaries.append(summarise_state(f"DW epsilon={epsilon}", final_state))

pd.DataFrame(dw_summaries)
"""
        ),
        code(
            """
dw_history = pd.concat(dw_histories, ignore_index=True)
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=dw_history, x="step", y="opinion", hue="model", units="node_id", estimator=None, alpha=0.35, ax=ax)
ax.set_title("Opinion trajectories under different confidence thresholds")
plt.tight_layout()
"""
        ),
        md("## 2. Add algorithmic bias to the interaction rule"),
        code(
            """
unbiased_history, unbiased_state = run_deffuant(G, epsilon=0.25, seed=123)
biased_history, biased_state = run_biased_exposure(G, epsilon=0.25, bias_strength=7.5, seed=123)

pd.DataFrame(
    [
        summarise_state("Unbiased DW (epsilon=0.25)", unbiased_state),
        summarise_state("Biased neighbor selection", biased_state),
    ]
)
"""
        ),
        code(
            """
trajectory_frame = pd.concat(
    [
        unbiased_history.assign(model="Unbiased"),
        biased_history.assign(model="Biased"),
    ],
    ignore_index=True,
)

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
for axis, model_name in zip(axes, ["Unbiased", "Biased"]):
    chunk = trajectory_frame[trajectory_frame["model"] == model_name]
    sns.lineplot(data=chunk, x="step", y="opinion", units="node_id", estimator=None, alpha=0.4, ax=axis)
    axis.set_title(model_name)
plt.tight_layout()
"""
        ),
    ]
    write_notebook(MODULES_DIR / "03_simulating_polarisation_dynamics.ipynb", cells)


def build_module_4() -> None:
    cells = [
        md(
            """
            # Module 4 - The Controlled Sandbox

            **Time:** 15:30-17:00  
            **Tool focus:** YSocial plus `ysights`

            This notebook assumes a YSocial simulation database is available and keeps the `ysights` analysis explicit, so the follow network, mention network, and content signals can be tied back to enclaves and polarisation.
            """
        ),
        code(
            COMMON_SETUP
            + """
import sqlite3

import networkx as nx
import numpy as np
import pandas as pd
from cdlib import algorithms
from ysights import YDataHandler
"""
        ),
        code(YSOCIAL_DB_CODE),
        code(
            """
DB_PATH = ensure_ysocial_demo_db()
handler = YDataHandler(str(DB_PATH))

pd.Series(
    {
        "db_path": str(DB_PATH.relative_to(PROJECT_ROOT)),
        "agents": handler.number_of_agents(),
        "time_range": handler.time_range(),
    }
)
"""
        ),
        md("## 1. Read the agents and attach their attributes"),
        code(
            """
agents = handler.agents().get_agents()
agents_df = pd.DataFrame(
    [
        {
            "agent_id": agent.id,
            "username": agent.username,
            "leaning": agent.leaning,
            "content_recsys": agent.recsys["content"],
            "social_recsys": agent.recsys["social"],
            "toxicity": agent.toxicity,
            "profession": agent.profession,
        }
        for agent in agents
    ]
)
agents_df.head()
"""
        ),
        md("## 2. Recover the follow and mention networks with `ysights`"),
        code(
            """
social_graph = handler.social_network()
mention_graph = handler.mention_network()

leaning_lookup = dict(zip(agents_df["agent_id"], agents_df["leaning"]))
nx.set_node_attributes(social_graph, leaning_lookup, "leaning")
nx.set_node_attributes(mention_graph, leaning_lookup, "leaning")

pd.DataFrame(
    [
        {"network": "follow", "nodes": social_graph.number_of_nodes(), "edges": social_graph.number_of_edges()},
        {"network": "mention", "nodes": mention_graph.number_of_nodes(), "edges": mention_graph.number_of_edges()},
    ]
)
"""
        ),
        code(
            """
pd.Series(
    {
        "follow_assortativity": round(nx.attribute_assortativity_coefficient(social_graph, "leaning"), 4),
        "mention_assortativity": round(nx.attribute_assortativity_coefficient(mention_graph, "leaning"), 4),
    }
)
"""
        ),
        md("## 3. Detect enclave-like clusters inside the simulation output"),
        code(
            """
follow_partition = algorithms.louvain(social_graph.to_undirected())

cluster_rows = []
for cluster_id, community in enumerate(follow_partition.communities):
    for node in community:
        cluster_rows.append({"node_id": node, "cluster": cluster_id})

cluster_frame = pd.DataFrame(cluster_rows).merge(
    agents_df[["agent_id", "leaning"]],
    left_on="node_id",
    right_on="agent_id",
    how="left",
)
cluster_frame.head()
"""
        ),
        code(
            """
community_rows = []
for cluster_id, chunk in cluster_frame.groupby("cluster"):
    nodes = set(chunk["node_id"])
    subgraph = social_graph.to_undirected().subgraph(nodes)
    internal_edges = subgraph.number_of_edges()
    external_edges = nx.cut_size(social_graph.to_undirected(), nodes)
    dominant_leaning = chunk["leaning"].mode().iat[0]
    purity = (chunk["leaning"] == dominant_leaning).mean()
    community_rows.append(
        {
            "cluster": cluster_id,
            "size": len(nodes),
            "purity": round(float(purity), 4),
            "internal_edges": internal_edges,
            "external_edges": int(external_edges),
            "separation": round(internal_edges / max(internal_edges + external_edges, 1), 4),
        }
    )

pd.DataFrame(community_rows).sort_values("cluster").reset_index(drop=True)
"""
        ),
        code(
            """
bridge_scores = pd.DataFrame(
    [{"node_id": node, "betweenness": score} for node, score in nx.betweenness_centrality(social_graph).items()]
).sort_values("betweenness", ascending=False)

bridge_scores.merge(
    agents_df[["agent_id", "username", "leaning"]],
    left_on="node_id",
    right_on="agent_id",
    how="left",
).head(10)
"""
        ),
        md("## 4. Inspect content signals with `ysights`"),
        code(
            """
hashtag_rows = []
for agent_id in agents_df["agent_id"]:
    for hashtag, count in handler.agent_hashtags(agent_id).items():
        hashtag_rows.append({"agent_id": agent_id, "hashtag": hashtag, "count": count})

hashtag_frame = pd.DataFrame(hashtag_rows).merge(
    agents_df[["agent_id", "leaning"]],
    on="agent_id",
    how="left",
)

hashtag_frame.groupby(["leaning", "hashtag"])["count"].sum().reset_index().sort_values(["leaning", "count"], ascending=[True, False])
"""
        ),
        code(
            """
toxicity_by_leaning = pd.DataFrame(
    handler.custom_query(
        '''
        SELECT u.leaning, AVG(pt.toxicity) AS avg_toxicity, COUNT(*) AS n_posts
        FROM post_toxicity pt
        JOIN post p ON pt.post_id = p.id
        JOIN user_mgmt u ON p.user_id = u.id
        GROUP BY u.leaning
        ORDER BY u.leaning
        '''
    ),
    columns=["leaning", "avg_toxicity", "n_posts"],
)
toxicity_by_leaning
"""
        ),
    ]
    write_notebook(MODULES_DIR / "04_ysocial_sandbox.ipynb", cells)


def build_exercises() -> None:
    exercises = [
        (
            EXERCISES_DIR / "01_foundations_exercises.ipynb",
            [
                md("# Exercise 1 - Foundations\n\nRepeat the loading and filtering steps manually rather than calling a helper pipeline."),
                code(
                    COMMON_SETUP
                    + """
import networkx as nx
import numpy as np
import pandas as pd
"""
                ),
                code(DEMO_GRAPH_CODE),
                code(
                    """
write_demo_graph_files()
G = load_graph(DATA_RAW / "workshop_network.gexf")

baseline = pd.Series(
    {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
    }
)
baseline
"""
                ),
                md("1. Reload the graph from the edge-list file instead of GEXF.\n2. Change the quantile value below.\n3. Compare how the selected core changes."),
                code(
                    """
degree_values = np.array([degree for _, degree in G.degree()])
k_threshold = max(2, int(np.quantile(degree_values, 0.8)))
core = nx.k_core(G, k=k_threshold)
pd.Series({"k_threshold": k_threshold, "core_nodes": core.number_of_nodes(), "core_edges": core.number_of_edges()})
"""
                ),
            ],
        ),
        (
            EXERCISES_DIR / "02_enclaves_exercises.ipynb",
            [
                md("# Exercise 2 - Enclave Detection\n\nCompare one extra partition and decide whether it identifies the same enclaves as the lecture solution."),
                code(
                    COMMON_SETUP
                    + """
import networkx as nx
import numpy as np
import pandas as pd
from cdlib import algorithms
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
"""
                ),
                code(DEMO_GRAPH_CODE),
                code(
                    """
write_demo_graph_files()
G = load_graph(DATA_RAW / "workshop_network.graphml")
louvain = algorithms.louvain(G)
infomap = algorithms.infomap(G)

def partition_labels(partition):
    labels = {}
    for cluster_id, community in enumerate(partition.communities):
        for node in community:
            labels[int(node)] = cluster_id
    return pd.Series(labels).sort_index()
"""
                ),
                code(
                    """
louvain_labels = partition_labels(louvain)
infomap_labels = partition_labels(infomap)

pd.Series(
    {
        "ARI": round(adjusted_rand_score(louvain_labels, infomap_labels), 4),
        "NMI": round(normalized_mutual_info_score(louvain_labels, infomap_labels), 4),
    }
)
"""
                ),
            ],
        ),
        (
            EXERCISES_DIR / "03_dynamics_exercises.ipynb",
            [
                md("# Exercise 3 - Polarisation Dynamics\n\nChange the tolerance and bias settings, then explain the effect on the number of surviving opinion clusters."),
                code(
                    COMMON_SETUP
                    + """
import networkx as nx
import numpy as np
import pandas as pd
"""
                ),
                code(DEMO_GRAPH_CODE),
                code(
                    """
write_demo_graph_files()
G = load_graph(DATA_RAW / "workshop_network.graphml")

def initialize_opinions(graph):
    return {node: float(graph.nodes[node]["opinion"]) for node in graph.nodes()}

def count_clusters(opinions, tolerance=0.08):
    values = sorted(opinions.values())
    clusters = 1
    for previous, current in zip(values, values[1:]):
        if abs(current - previous) > tolerance:
            clusters += 1
    return clusters

def run_biased_exposure(graph, epsilon, bias_strength=6.0, mu=0.35, steps=2500, seed=42):
    rng = np.random.default_rng(seed)
    opinions = initialize_opinions(graph)
    nodes = list(graph.nodes())
    for _ in range(steps):
        source = int(nodes[rng.integers(len(nodes))])
        neighbors = list(graph.neighbors(source))
        if neighbors:
            distances = np.array([abs(opinions[source] - opinions[target]) for target in neighbors])
            weights = np.exp(-bias_strength * distances)
            weights = weights / weights.sum()
            target = int(rng.choice(neighbors, p=weights))
            if abs(opinions[source] - opinions[target]) <= epsilon:
                delta = mu * (opinions[target] - opinions[source])
                opinions[source] += delta
                opinions[target] -= delta
    return opinions
"""
                ),
                code(
                    """
epsilon = 0.2
bias_strength = 8.0
biased_state = run_biased_exposure(G, epsilon=epsilon, bias_strength=bias_strength, seed=101)

pd.Series(
    {
        "epsilon": epsilon,
        "bias_strength": bias_strength,
        "clusters": count_clusters(biased_state),
        "range": round(max(biased_state.values()) - min(biased_state.values()), 4),
    }
)
"""
                ),
            ],
        ),
        (
            EXERCISES_DIR / "04_ysocial_sandbox_exercises.ipynb",
            [
                md("# Exercise 4 - YSocial Sandbox\n\nPick a focal agent and inspect their content and exposure with explicit `ysights` calls."),
                code(
                    COMMON_SETUP
                    + """
import sqlite3

import pandas as pd
from ysights import YDataHandler
"""
                ),
                code(YSOCIAL_DB_CODE),
                code(
                    """
DB_PATH = ensure_ysocial_demo_db()
handler = YDataHandler(str(DB_PATH))

focal_agent = 3
pd.Series(
    {
        "hashtags": handler.agent_hashtags(focal_agent),
        "interests": handler.agent_interests(focal_agent),
        "posts": len(handler.posts_by_agent(focal_agent, enrich_dimensions=[]).get_posts()),
    }
)
"""
                ),
            ],
        ),
    ]

    for path, cells in exercises:
        write_notebook(path, cells)


def main() -> None:
    MODULES_DIR.mkdir(parents=True, exist_ok=True)
    EXERCISES_DIR.mkdir(parents=True, exist_ok=True)

    write_demo_graph_files()
    write_ysocial_demo_db()

    build_environment_check()
    build_module_1()
    build_module_2()
    build_module_3()
    build_module_4()
    build_exercises()


if __name__ == "__main__":
    main()
