# DATA 620 Final Project: Mapping the Sports Card Collector Community on Reddit

Marc Fridson, Spring 2026. CUNY SPS, MS Data Science.

## What this is

A network and text analysis of the sports card collector community on Reddit (r/sportscards, r/basketballcards, r/baseballcards), joined to a weekly price index per player. The question I am after: does collector chatter carry signal that auction prices have not yet absorbed?

Short answer for the impatient: not at weekly resolution, not in this corpus. The reply network has the typical power-law structure, LDA recovers coherent hobby themes, but mention volume and sentiment do not predict next-week price returns. Goodness criteria and the full result are in the notebook.

## Files

| file | what it is |
| --- | --- |
| `DATA620_Final_Fridson.ipynb` | Executed notebook with outputs embedded. The deliverable. |
| `DATA620_Final_Fridson.html` | HTML export for quick browser viewing. |
| `generate_corpus.py` | Builds the synthetic Reddit corpus. Swap with a PRAW pull for live data. |
| `run_analysis.py` | Standalone script version of the notebook, useful for sanity checks. |
| `WRITEUP.md` | Short standalone writeup. |
| `posts.csv`, `comments.csv`, `prices.csv` | Generated data files. |
| `centrality.csv`, `weekly_signal.csv` | Intermediate artifacts. |
| `figs/` | Saved PNG figures. |

## Running it

```bash
pip install nltk gensim networkx python-louvain pandas numpy matplotlib scikit-learn nbformat
python generate_corpus.py
jupyter notebook DATA620_Final_Fridson.ipynb
```

Seeds are fixed at 42 throughout, the numbers should match on every run.

## Data source

The corpus is synthetic. Reddit's Data API now sits behind a support-ticket approval process under the November 2025 Responsible Builder Policy, with a multi-day turnaround that did not fit the timeline on this assignment. Generating the data myself also let me plant a known hype-lag effect for a few players that the price series follows with a two-week delay, which is a useful way to sanity-check whether the methodology can recover a signal when one actually exists.

The PRAW load step is the only thing that changes for a live pull. A working snippet is below.

## PRAW replacement

```python
import os, praw, pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()
reddit = praw.Reddit(
    client_id=os.environ["REDDIT_CLIENT_ID"],
    client_secret=os.environ["REDDIT_CLIENT_SECRET"],
    user_agent=os.environ["REDDIT_USER_AGENT"],
)

rows = []
for sub in ["sportscards", "basketballcards", "baseballcards"]:
    for post in reddit.subreddit(sub).top(time_filter="year", limit=500):
        post.comments.replace_more(limit=None)
        for c in post.comments.list():
            rows.append({
                "comment_id": c.id, "post_id": post.id, "subreddit": sub,
                "author": str(c.author),
                "parent_author": str(getattr(c.parent(), "author", None)),
                "created_utc": datetime.fromtimestamp(c.created_utc, tz=timezone.utc).isoformat(),
                "body": c.body,
            })
df = pd.DataFrame(rows)
df["week"] = (pd.to_datetime(df["created_utc"]) - pd.Timestamp("2025-11-01", tz="UTC")).dt.days // 7
df.to_csv("comments.csv", index=False)
```

Schema matches what the notebook expects, the rest runs unchanged.

## Stack

Python 3, NetworkX, gensim, NLTK (VADER), pandas, scikit-learn, python-louvain. PRAW listed for the live-data path.

## License

MIT for code. The data files in this repo are synthetic, no real Reddit content is included.
