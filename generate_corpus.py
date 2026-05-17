"""
Synthetic Reddit corpus generator for r/sportscards, r/basketballcards,
r/baseballcards. Schema matches a PRAW comments pull. Outputs posts.csv,
comments.csv, and a per-player weekly price index in prices.csv.
"""
import json, random, numpy as np, pandas as pd
from datetime import datetime, timedelta

random.seed(42); np.random.seed(42)

SUBS = ["sportscards", "basketballcards", "baseballcards"]

# Player roster with weighting. Multipliers drive mention frequency.
PLAYERS = {
    "Cooper Flagg":       {"sport": "nba", "weight": 9.0, "aliases": ["Flagg", "Cooper"]},
    "Dylan Harper":       {"sport": "nba", "weight": 7.0, "aliases": ["Harper", "Dylan H"]},
    "Victor Wembanyama":  {"sport": "nba", "weight": 8.0, "aliases": ["Wemby", "Wembanyama", "VW"]},
    "Luka Doncic":        {"sport": "nba", "weight": 6.0, "aliases": ["Luka", "Doncic", "Lukers"]},
    "Jayson Tatum":       {"sport": "nba", "weight": 4.0, "aliases": ["Tatum"]},
    "LeBron James":       {"sport": "nba", "weight": 3.5, "aliases": ["LeBron", "Bron", "LBJ"]},
    "Stephen Curry":      {"sport": "nba", "weight": 3.0, "aliases": ["Curry", "Steph"]},
    "Paul Skenes":        {"sport": "mlb", "weight": 6.5, "aliases": ["Skenes"]},
    "Roman Anthony":      {"sport": "mlb", "weight": 5.0, "aliases": ["Roman A", "R Anthony"]},
    "Ronald Acuna":       {"sport": "mlb", "weight": 4.0, "aliases": ["Acuna"]},
    "Shohei Ohtani":      {"sport": "mlb", "weight": 5.5, "aliases": ["Ohtani", "Shohei"]},
    "Mike Trout":         {"sport": "mlb", "weight": 3.0, "aliases": ["Trout"]},
    "Bobby Witt":         {"sport": "mlb", "weight": 3.5, "aliases": ["Witt", "Bobby W"]},
    "Caitlin Clark":      {"sport": "wnba","weight": 5.5, "aliases": ["Clark", "Caitlin"]},
    "Paige Bueckers":     {"sport": "wnba","weight": 4.0, "aliases": ["Bueckers", "Paige"]},
}

SETS = {
    "Bowman Chrome":   {"weight": 7.0, "aliases": ["Bowman", "Bowman Chrome"]},
    "Topps Chrome":    {"weight": 6.0, "aliases": ["Topps Chrome", "TC"]},
    "Panini Prizm":    {"weight": 6.5, "aliases": ["Prizm", "Panini Prizm"]},
    "Bowman's Best":   {"weight": 3.5, "aliases": ["Bowmans Best", "BB"]},
    "Topps Heritage":  {"weight": 2.5, "aliases": ["Heritage"]},
    "Bowman U Now":    {"weight": 4.0, "aliases": ["Bowman U", "BU Now"]},
    "National Treasures": {"weight": 3.0, "aliases": ["NT", "National Treasures"]},
}

# Sentiment-bearing templates. Mix of positive, neutral, negative.
POS = [
    "{p} {s} just hit a new high, this market is on fire",
    "Picked up my {p} {s} auto today, absolutely love this card",
    "Anyone else loading up on {p}? The PC is growing fast",
    "This {p} {s} refractor is gorgeous, the print run is tiny",
    "Huge W on my {p} {s} auction, came in way under comps",
    "{p} is going to be a generational talent, all in on the {s}",
    "PSA 10 hit on my {p} {s}, very happy with the grade",
]
NEU = [
    "Looking for comps on a {p} {s} base auto, anyone seen recent sales?",
    "What is the print run on the {p} {s} gold refractor?",
    "Is {p} {s} a better long term hold than the {s} flagship?",
    "Just got my submission back, {p} {s} came back a 9",
    "Selling my {p} {s}, will post comps later",
    "Bought a {p} {s} from a LCS, paid retail",
    "Comparing {p} rookie cards across {s} and the flagship",
]
NEG = [
    "{p} {s} prices are tanking, I am out",
    "Bad break on my {p} {s}, terrible centering",
    "Cant believe what {p} {s} is going for right now, this is a bubble",
    "Got burned on a {p} {s}, will not buy raw again",
    "The market for {p} {s} is dead, no bids on anything",
    "Overpaid on my {p} {s}, regretting it",
    "{p} is going to bust, dumping my {s} now",
]

# Generic chatter (no entity mention)
GENERIC = [
    "What is everyone buying this week?",
    "Mail day! Got some great cards in.",
    "PSA turnaround times are brutal lately.",
    "Anyone else hitting the card show this weekend?",
    "Just venting, hobby is wild right now.",
    "First post here, what should I know?",
    "Show me your latest PC additions.",
    "Lol at the prices at retail.",
    "Hobby is dead, change my mind.",
    "Hobby is back, change my mind.",
]

START = datetime(2025, 11, 1)
WEEKS = 26  # ~6 months
END = START + timedelta(weeks=WEEKS)

# User population, Zipf-like activity
N_USERS = 2200
user_ids = [f"u_{i:05d}" for i in range(N_USERS)]
# Activity weight ~ 1/rank^0.9
activity = np.array([1.0/((i+1)**0.9) for i in range(N_USERS)])
activity = activity / activity.sum()

# Some users are "fanboys" for specific players, biases who mentions whom
player_list = list(PLAYERS.keys())
fan_assignment = {}
for u in user_ids[:300]:  # top 300 users get a fan tilt
    fan_assignment[u] = random.choice(player_list)

def pick_player():
    weights = np.array([PLAYERS[p]["weight"] for p in player_list])
    weights = weights / weights.sum()
    return np.random.choice(player_list, p=weights)

def pick_set():
    sets = list(SETS.keys())
    weights = np.array([SETS[s]["weight"] for s in sets])
    weights = weights / weights.sum()
    return np.random.choice(sets, p=weights)

def render_comment(user, week_idx):
    # 70% mention a player + set, 30% generic
    if random.random() < 0.70:
        if user in fan_assignment and random.random() < 0.55:
            p = fan_assignment[user]
        else:
            p = pick_player()
        s = pick_set()
        # Hype week boost for top 3 players around weeks 8 to 14
        is_hype = p in ["Cooper Flagg", "Dylan Harper", "Paul Skenes"] and 8 <= week_idx <= 14
        if is_hype:
            tone = np.random.choice(["pos","neu","neg"], p=[0.65, 0.30, 0.05])
        else:
            tone = np.random.choice(["pos","neu","neg"], p=[0.40, 0.45, 0.15])
        bucket = {"pos": POS, "neu": NEU, "neg": NEG}[tone]
        # Use canonical name 60% of the time, alias 40%
        if random.random() < 0.4 and PLAYERS[p]["aliases"]:
            p_str = random.choice(PLAYERS[p]["aliases"])
        else:
            p_str = p
        if random.random() < 0.4 and SETS[s]["aliases"]:
            s_str = random.choice(SETS[s]["aliases"])
        else:
            s_str = s
        return random.choice(bucket).format(p=p_str, s=s_str)
    return random.choice(GENERIC)

def sample_user(prev_user=None):
    # 15% chance to reply to same recent user, to seed reply clusters
    u = np.random.choice(user_ids, p=activity)
    return u

# Generate threads + comments
TARGET_COMMENTS = 50000
posts, comments = [], []
post_id = 0
comment_id = 0

while len(comments) < TARGET_COMMENTS:
    # New post
    post_id += 1
    sub = random.choice(SUBS)
    # Random timestamp within window
    week_idx = random.randint(0, WEEKS-1)
    t_post = START + timedelta(weeks=week_idx, hours=random.randint(0,167))
    author = np.random.choice(user_ids, p=activity)
    posts.append({
        "post_id": f"p_{post_id:06d}",
        "subreddit": sub,
        "author": author,
        "created_utc": t_post.isoformat(),
        "week": week_idx,
        "title": render_comment(author, week_idx),
    })
    # Comment count per post, fat-tailed
    n_c = max(1, int(np.random.lognormal(mean=1.8, sigma=1.0)))
    n_c = min(n_c, 60)
    thread_users = [author]
    for j in range(n_c):
        comment_id += 1
        if j == 0 or random.random() < 0.4:
            parent_author = author
        else:
            parent_author = random.choice(thread_users)
        author_c = np.random.choice(user_ids, p=activity)
        t_c = t_post + timedelta(minutes=random.randint(1, 60*72))
        comments.append({
            "comment_id": f"c_{comment_id:07d}",
            "post_id": f"p_{post_id:06d}",
            "subreddit": sub,
            "author": author_c,
            "parent_author": parent_author,
            "created_utc": t_c.isoformat(),
            "week": (t_c - START).days // 7,
            "body": render_comment(author_c, week_idx),
        })
        thread_users.append(author_c)
        if len(comments) >= TARGET_COMMENTS:
            break

posts_df = pd.DataFrame(posts)
comments_df = pd.DataFrame(comments)
# Clip week to valid range
comments_df["week"] = comments_df["week"].clip(0, WEEKS-1)

posts_df.to_csv("/home/claude/data620_final/posts.csv", index=False)
comments_df.to_csv("/home/claude/data620_final/comments.csv", index=False)

# Synthetic auction price series per player, weekly
price_records = []
for p, meta in PLAYERS.items():
    base = 100.0
    series = [base]
    for w in range(1, WEEKS):
        # Underlying drift + noise
        drift = np.random.normal(0, 0.03)
        # Hype-driven price catches up to chatter with a 2 week lag for some players
        hype_lag = 0
        if p in ["Cooper Flagg", "Dylan Harper", "Paul Skenes"] and 10 <= w <= 16:
            hype_lag = 0.04
        if p in ["Caitlin Clark"] and 4 <= w <= 8:
            hype_lag = 0.03
        series.append(series[-1] * (1 + drift + hype_lag))
    for w, v in enumerate(series):
        price_records.append({"player": p, "week": w, "price_index": round(v, 3)})
prices_df = pd.DataFrame(price_records)
prices_df.to_csv("/home/claude/data620_final/prices.csv", index=False)

print(f"posts: {len(posts_df):,}")
print(f"comments: {len(comments_df):,}")
print(f"unique users in comments: {comments_df['author'].nunique():,}")
print(f"weeks: {WEEKS}, span: {START.date()} -> {END.date()}")
print(f"price rows: {len(prices_df):,}")
