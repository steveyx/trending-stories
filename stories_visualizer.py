import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import numpy as np
from functools import reduce
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 100


def plot_trending_news(news_by_period):
    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_axes([0.03, 0.02, 0.94, 0.94])
    ax.grid(False)

    top_stories = 3
    news_by_period = [day for day in news_by_period if day]
    x_date = [news[0]['date'] for news in news_by_period]
    print(x_date)
    stories = [[st['keywords'] for st in news[:top_stories]] for news in news_by_period]
    for period in stories:
        for i, story in enumerate(period):
            wds = story[:4]
            u, v = "{} {}", "{}\n{}"
            _text = reduce(lambda a, b:
                           u.format(a, b) if len(a.split("\n")[-1]) < 15 and b.find(" ") < 0 else v.format(a, b), wds)
            period[i] = _text
    stories_size = [[st['cluster_size'] for st in news[:top_stories]] for news in news_by_period]
    df_topics = pd.DataFrame(stories, columns=['topic' + str(st) for st in range(top_stories)])
    stories_size = np.log2(stories_size) + 0.3
    df_topics_size = pd.DataFrame(stories_size, columns=['size' + str(st) for st in range(top_stories)])
    pos_y = [0.2 * (ith % 4) - 0.3 for ith, _ in enumerate(stories_size)]
    pos = [[(-1) ** ith * (1.3 + jth * 1.2) + pos_y[ith] for jth, size in enumerate(day)] for ith, day in
           enumerate(stories_size)]
    df_pos = pd.DataFrame(pos, columns=['pos' + str(st) for st in range(top_stories)])
    df = pd.concat([df_topics, df_topics_size, df_pos], axis=1)
    df['date'] = x_date

    x = list(range(len(x_date)))
    for i in range(top_stories):
        ax.scatter(x, df_pos['pos' + str(i)], s=df_topics_size['size' + str(i)] * 400,
                   facecolors="steelblue", alpha=0.4, edgecolors="grey", linewidth=2)
    for x, r in df_pos.iterrows():
        for idy, y in enumerate(pos[x]):
            ax.text(x, y, stories[x][idy],
                    horizontalalignment='center', verticalalignment='center', wrap=True, fontsize=7)
    ax.set_title("Trending Stories", fontsize=10)
    ax.set_ylim(-top_stories-1.4, top_stories+1.4)
    ax.set_xlim(-1, len(x_date))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.tick_params(labelsize=8, length=0)
    ax.tick_params(axis="x", direction="in", pad=0)
    ax.yaxis.set_ticks([])
    ax.set_xticks(list(range(len(x_date))))
    ax.set_xticklabels([d.strftime("%Y\n%m-%d") for d in x_date], rotation=0, va="center")
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    hw, hl = 1./40.*(ymax-ymin), 1./40.*(xmax-xmin)  # arrowhead width and length
    lw, ohg = 0.5, 0.3  # axis line width and arrow overhang
    # draw x axis arrow
    ax.arrow(-1, 0, len(x_date)+1, 0., fc='k', ec='k', lw=lw,
             head_width=hw, head_length=hl, overhang=ohg, length_includes_head=True, clip_on=False)
    plt.savefig("data/trending_stories.png", dpi=200)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()


def visualize_trending_stories(processed_news):
    _pubDates = [_news['pubDate'] for _news in processed_news]
    _min_time, _max_time = min(_pubDates), max(_pubDates)
    period_days = 14
    # update every period_days
    periods = int((_max_time - _min_time).total_seconds() / (3600 * 24 * period_days))
    print("news are between {} and {}, number of periods {}".format(_min_time, _max_time, periods))
    news_by_period = []
    for p_i in range(periods, -1, -1):
        _end_time = _max_time - dt.timedelta(days=p_i*period_days)
        _start_time = _end_time - dt.timedelta(days=period_days)
        _news_in = [_d for _d in processed_news if _start_time < _d['pubDate'] <= _end_time]
        if not _news_in:
            print("no news between {} and {}".format(_start_time, _end_time))
            continue
        _data = [[i, _d['cluster_id'], _d['kwords'], 1] for i, _d in enumerate(_news_in)]
        _df = pd.DataFrame(_data, columns=['i', 'cluster_id', 'kwords', 'cluster_size'])
        _df_g = _df.groupby(by='cluster_id').agg({"cluster_size": sum, "kwords": lambda x: list(x)[0]})
        _df_g.sort_values(by="cluster_size", ascending=False, inplace=True)
        group_titles = [
            {
                'date': _end_time.date(),
                'keywords': [_k['keyword'] for _k in _r['kwords']],
                'cluster_size': _r['cluster_size']
            }
            for idx, _r in _df_g.iterrows()
        ]
        news_by_period.append(group_titles)
    plot_trending_news(news_by_period)
