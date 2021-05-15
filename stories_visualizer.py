import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt


def plot_trending_news(news_by_period):
    fig = plt.figure(figsize=(19.2, 10.8))
    ax = fig.add_axes([0.03, 0.08, 0.94, 0.88])
    ax.grid()

    top_stories = 3
    news_by_period = [day for day in news_by_period if day]
    x_date = [news[0]['date'] for news in news_by_period]
    stories = [["\n".join(st['keywords']) for st in news[:top_stories]] for news in news_by_period]
    stories_size = [[st['cluster_size'] for st in news[:top_stories]] for news in news_by_period]
    df_topics = pd.DataFrame(stories, columns=['topic' + str(st) for st in range(top_stories)])
    df_topics_size = pd.DataFrame(stories_size, columns=['size' + str(st) for st in range(top_stories)])
    pos_y = [0.2 * (ith % 4) - 0.3 for ith, day in enumerate(stories_size)]
    pos = [[(-1) ** ith * (1 + jth * 1.2) + pos_y[ith] for jth, size in enumerate(day)] for ith, day in
           enumerate(stories_size)]
    df_pos = pd.DataFrame(pos, columns=['pos' + str(st) for st in range(top_stories)])
    df = pd.concat([df_topics, df_topics_size, df_pos], axis=1)
    df['date'] = x_date

    x = list(range(len(x_date)))
    for i in range(top_stories):
        ax.scatter(x, df_pos['pos' + str(i)],
                   s=df_topics_size['size' + str(i)] * 500,
                   c=x,
                   cmap="Blues", alpha=0.4, edgecolors="grey", linewidth=2)

    for x, r in df_pos.iterrows():
        for idy, y in enumerate(pos[x]):
            ax.text(x, y, stories[x][idy],
                    horizontalalignment='center',
                    verticalalignment='center',
                    wrap=True,
                    fontsize=8)
    ax.set_xticks(list(range(len(x_date))))
    ax.set_xticklabels([d.strftime("%Y\n%m-%d") for d in x_date], rotation=0)
    ax.set_title("Trending topics")
    plt.show()


def visualize_trending_stories(processed_news):
    _pubDates = [_news['pubDate'] for _news in processed_news]
    period_days = 15
    _min_time, _max_time = min(_pubDates), max(_pubDates)
    # update every 6 hours
    _periods = int((_max_time - _min_time).total_seconds() / (3600 * 6 * period_days))
    news_by_period = []
    for i in range(_periods, -1, -1):
        _end_time = _max_time - dt.timedelta(hours=i * 6)
        _start_time = _end_time - dt.timedelta(days=period_days)
        _news_in = [_d for _d in processed_news if _start_time < _d['pubDate'] <= _end_time]
        if not _news_in:
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
