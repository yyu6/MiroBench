import json, os, sqlite3, tempfile
from product_reddit_sim.exporter import export_discussion, _render_markdown

PROFILES = [
    {"user_id": 1, "username": "AudiophileMax", "karma": 18000,
     "name": "Max", "bio": "", "persona": ""},
    {"user_id": 2, "username": "BudgetHunter99", "karma": 500,
     "name": "Dave", "bio": "", "persona": ""},
]

META = {
    "product_category": "headphones",
    "hint": "commuters",
    "agent_count": 2,
    "seed": 42,
    "simulated_hours": 48,
    "rounds": 10,
    "run_id": "test_run_001",
}


def _make_db(tmp: str) -> str:
    """Create a minimal OASIS-style trace DB."""
    db_path = os.path.join(tmp, "reddit_simulation.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE trace (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            action TEXT,
            info TEXT,
            created_at TEXT
        )
    """)
    rows = [
        (1, "create_post",
         json.dumps({"content": "What headphones for commuting?", "post_id": 101}),
         "2026-04-11T19:00:00"),
        (2, "create_comment",
         json.dumps({"content": "Sony XM5 is great!", "post_id": 101}),
         "2026-04-11T19:30:00"),
        (1, "like_post",
         json.dumps({"post_id": 101}),
         "2026-04-11T19:05:00"),
        (2, "CREATE_POST",  # test uppercase handling
         json.dumps({"content": "Budget earbuds under $50?", "post_id": 102}),
         "2026-04-11T20:00:00"),
    ]
    conn.executemany(
        "INSERT INTO trace (user_id, action, info, created_at) VALUES (?,?,?,?)", rows
    )
    conn.commit()
    conn.close()
    return db_path


def _make_profiles_file(tmp: str) -> str:
    path = os.path.join(tmp, "reddit_profiles.json")
    with open(path, "w") as f:
        json.dump(PROFILES, f)
    return path


def _make_live_style_db(tmp: str) -> str:
    """Create a minimal DB using MiroFish's post/comment tables."""
    db_path = os.path.join(tmp, "reddit_simulation.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE post (
            post_id INTEGER PRIMARY KEY,
            user_id INTEGER,
            content TEXT,
            created_at TEXT,
            num_likes INTEGER DEFAULT 0,
            num_dislikes INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE comment (
            comment_id INTEGER PRIMARY KEY,
            post_id INTEGER,
            user_id INTEGER,
            content TEXT,
            created_at TEXT,
            num_likes INTEGER DEFAULT 0,
            num_dislikes INTEGER DEFAULT 0,
            parent_comment_id INTEGER,
            depth INTEGER DEFAULT 0
        )
    """)
    conn.executemany(
        "INSERT INTO post (post_id, user_id, content, created_at, num_likes, num_dislikes) VALUES (?,?,?,?,?,?)",
        [
            (101, 1, "What headphones for commuting?", "2026-04-11T19:00:00", 3, 0),
            (102, 2, "Budget earbuds under $50?", "2026-04-11T20:00:00", 1, 0),
        ],
    )
    conn.executemany(
        "INSERT INTO comment (comment_id, post_id, user_id, content, created_at, num_likes, num_dislikes, parent_comment_id, depth) VALUES (?,?,?,?,?,?,?,?,?)",
        [
            (201, 101, 2, "Sony XM5 is great!", "2026-04-11T19:30:00", 2, 0, None, 0),
        ],
    )
    conn.commit()
    conn.close()
    return db_path


def test_creates_discussion_json():
    with tempfile.TemporaryDirectory() as tmp:
        db = _make_db(tmp)
        pf = _make_profiles_file(tmp)
        json_path, _ = export_discussion(db, pf, tmp, META)
        assert os.path.exists(json_path)


def test_creates_discussion_md():
    with tempfile.TemporaryDirectory() as tmp:
        db = _make_db(tmp)
        pf = _make_profiles_file(tmp)
        _, md_path = export_discussion(db, pf, tmp, META)
        assert os.path.exists(md_path)


def test_json_contains_posts():
    with tempfile.TemporaryDirectory() as tmp:
        db = _make_db(tmp)
        pf = _make_profiles_file(tmp)
        json_path, _ = export_discussion(db, pf, tmp, META)
        with open(json_path) as f:
            data = json.load(f)
    assert len(data["posts"]) == 2


def test_author_names_resolved():
    with tempfile.TemporaryDirectory() as tmp:
        db = _make_db(tmp)
        pf = _make_profiles_file(tmp)
        json_path, _ = export_discussion(db, pf, tmp, META)
        with open(json_path) as f:
            data = json.load(f)
    authors = {p["author"] for p in data["posts"]}
    assert "AudiophileMax" in authors and "BudgetHunter99" in authors


def test_handles_uppercase_action_types():
    """OASIS may store CREATE_POST or create_post — both must be handled."""
    with tempfile.TemporaryDirectory() as tmp:
        db = _make_db(tmp)
        pf = _make_profiles_file(tmp)
        json_path, _ = export_discussion(db, pf, tmp, META)
        with open(json_path) as f:
            data = json.load(f)
    # Should have 2 posts (one lowercase, one uppercase action)
    assert len(data["posts"]) == 2


def test_meta_included_in_json():
    with tempfile.TemporaryDirectory() as tmp:
        db = _make_db(tmp)
        pf = _make_profiles_file(tmp)
        json_path, _ = export_discussion(db, pf, tmp, META)
        with open(json_path) as f:
            data = json.load(f)
    assert data["meta"]["product_category"] == "headphones"
    assert data["meta"]["agent_count"] == 2


def test_markdown_contains_usernames():
    with tempfile.TemporaryDirectory() as tmp:
        db = _make_db(tmp)
        pf = _make_profiles_file(tmp)
        _, md_path = export_discussion(db, pf, tmp, META)
        with open(md_path) as f:
            md = f.read()
    assert "AudiophileMax" in md and "BudgetHunter99" in md


def test_markdown_has_header():
    with tempfile.TemporaryDirectory() as tmp:
        db = _make_db(tmp)
        pf = _make_profiles_file(tmp)
        _, md_path = export_discussion(db, pf, tmp, META)
        with open(md_path) as f:
            md = f.read()
    assert "headphones" in md
    assert "test_run_001" in md


def test_prefers_live_post_comment_tables():
    with tempfile.TemporaryDirectory() as tmp:
        db = _make_live_style_db(tmp)
        pf = _make_profiles_file(tmp)
        json_path, _ = export_discussion(db, pf, tmp, META)
        with open(json_path) as f:
            data = json.load(f)
    assert len(data["posts"]) == 2
    assert data["posts"][0]["likes"] == 3
    assert len(data["posts"][0]["comments"]) == 1
    assert data["posts"][0]["comments"][0]["content"] == "Sony XM5 is great!"


def test_exports_nested_comment_replies():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "reddit_simulation.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE post (
                post_id INTEGER PRIMARY KEY,
                user_id INTEGER,
                content TEXT,
                created_at TEXT,
                num_likes INTEGER DEFAULT 0,
                num_dislikes INTEGER DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE comment (
                comment_id INTEGER PRIMARY KEY,
                post_id INTEGER,
                user_id INTEGER,
                content TEXT,
                created_at TEXT,
                num_likes INTEGER DEFAULT 0,
                num_dislikes INTEGER DEFAULT 0,
                parent_comment_id INTEGER,
                depth INTEGER DEFAULT 0
            )
        """)
        conn.execute(
            "INSERT INTO post (post_id, user_id, content, created_at, num_likes, num_dislikes) VALUES (101,1,'What card?','2026-04-11T19:00:00',3,0)"
        )
        conn.executemany(
            "INSERT INTO comment (comment_id, post_id, user_id, content, created_at, num_likes, num_dislikes, parent_comment_id, depth) VALUES (?,?,?,?,?,?,?,?,?)",
            [
                (201, 101, 2, "Top level", "2026-04-11T19:30:00", 2, 0, None, 0),
                (202, 101, 1, "Reply", "2026-04-11T19:31:00", 1, 0, 201, 1),
            ],
        )
        conn.commit()
        conn.close()
        pf = _make_profiles_file(tmp)
        json_path, md_path = export_discussion(db_path, pf, tmp, META)
        with open(json_path) as f:
            data = json.load(f)
        with open(md_path) as f:
            md = f.read()

    assert data["posts"][0]["comments"][0]["comment_id"] == 201
    assert data["posts"][0]["comments"][0]["replies"][0]["comment_id"] == 202
    assert "Reply" in md
