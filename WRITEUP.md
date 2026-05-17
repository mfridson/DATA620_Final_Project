# Mapping the Sports Card Collector Community on Reddit

Marc Fridson, DATA 620 Final, Spring 2026

## The question

Does collector chatter on r/sportscards, r/basketballcards, and r/baseballcards carry signal that auction prices have not yet absorbed, and does the reply network surface users worth watching for that signal?

## What I built

Two tracks that meet at the end. The network track works on the user reply graph (directed, edges weighted by reply count) and a two-mode bipartite user-by-player graph. The user reply graph gets the standard treatment: undirected projection over the giant component, degree and betweenness and eigenvector centrality, Louvain community detection with modularity as the structural diagnostic. The bipartite graph gets projected onto players to produce a co-mention network.

The text track is VADER sentiment on every comment plus an LDA topic model (six topics, four passes, vocabulary filtered to terms in 20 to 50 percent document frequency) on a 20,000-comment sample.

The two tracks meet at a weekly per-player aggregate: mention volume, mean compound sentiment, positive-comment share. That gets joined to a per-player weekly price index, and I look at cross-correlations between mention volume and returns at lags `k` in `[-3, +3]`, then fit a ridge regression of next-week return on current-week mention z-score, sentiment z-score, and momentum.

## How I judge goodness

Held-out R² and MAE on the last 30 percent of weeks. The Reddit features have to add value over two baselines: a zero-prediction baseline and a momentum-only baseline. A 4-fold TimeSeriesSplit CV checks that the held-out R² is not a one-off.

## What I found

The reply graph has the shape Reddit graphs usually have. About 2,200 users, 36,000 weighted edges, a handful of users carrying most of the centrality across all three measures. Louvain modularity comes in around 0.11 across roughly a dozen communities, which is the right signature for a single-topic forum where most participants overlap. LDA recovers coherent hobby themes without much tuning, the topics map cleanly to comps and sales, refractors and parallels, retail and Prizm chatter, grading submissions, and broader market mood. The player co-mention graph is dense, which means co-mention by itself does not give a sharp clustering signal, active commenters talk about a lot of players.

The headline is that mention volume and sentiment do not predict next-week price returns at weekly resolution. The cross-correlations across players sit near zero at every lag. The ridge model lands at R² near zero on the held-out weeks, slightly worse than the zero-prediction baseline and noticeably better than the momentum-only baseline. The TimeSeriesSplit CV confirms negative R² across folds, so this is not a single-fold artifact.

That tracks with what I expected going in. On the high end of the hobby, chatter and prices co-move because the chatter is reacting to auction results that just printed, which is the wrong shape for a clean weekly lead.

## What would change the verdict

A real PRAW pull over 12 months instead of synthetic over 6 (sample size per player), daily resolution rather than weekly (chatter probably leads price by hours or days, not weeks), better entity resolution on set names where aliases collide, and narrowing the analysis to a single product cycle like Bowman Chrome rookies instead of pooling across sets.

## Limitations

The Reddit slice misses live-auction buyers, Whatnot and IG Live, and the Japanese and Chinese markets, so anything I find here is about Reddit's collector community, not the hobby in general. Entity resolution is rules-based, a production version would want a fine-tuned NER step plus the manual alias map. Even a clean signal would be descriptive of price movement, not a trading recommendation.

The corpus is synthetic. Reddit's Data API now requires support-ticket approval under the November 2025 Responsible Builder Policy, with several days of turnaround that did not fit this timeline. Generating the data myself also let me plant a known hype-lag effect so I could check whether the methodology recovers a signal when one is there. The PRAW load step is the only change for a live pull, the rest of the pipeline is identical.

## Reproducing

Code, data, and outputs are all in the repo. Run `python generate_corpus.py`, then execute the notebook top to bottom. Seeds are fixed at 42.
