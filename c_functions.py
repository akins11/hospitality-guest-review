from pandas import DataFrame, Series, concat
from numpy import nan

import nltk
from nltk import bigrams
from nltk.corpus import wordnet, stopwords, words
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from collections import Counter
from string import punctuation, digits

from wordcloud import WordCloud
from matplotlib.pyplot import imshow, axis, figure, tick_params, tight_layout, ylabel, xlabel, title
from seaborn import barplot
from plotly.subplots import make_subplots
from plotly.graph_objects import Bar, Pie, Scatter
from plotly.express import colors, bar, histogram, scatter

c_plt_config = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["pan2d", "lasso2d", "zoomIn2d", "zoomOut2d", "zoom2d", "toImage", "select2d", "autoScale2d"]
    }

agg_label = {"min": "Minimum", "mean": "Average", "median": "Median", "max": "Maximum", "sum": "Total"}

def clean_num(str_num: str) -> float:
    """
    parameter
    ---------
    str_num: The string to clean.

    """
    if str_num is not nan:
        for string in ["$", ","]:
            str_num =  str.replace(str_num, string, "")

        return float(str_num)
    else:
        return str_num


def unique_group_count(df, gp_var: str, unique_var: str) -> DataFrame:
    """
    parameter
    ---------
    df: pandas DataFrame.
    gp_var:
    unique_var:

    return
    ------
    """
    f_df = (
        df.groupby(gp_var)[unique_var]
        .nunique()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={unique_var: "count"})
    )
    f_df["proportion"] = round((f_df["count"] / df["host_id"].nunique()) * 100, 2)

    return f_df 


def num_description(df: DataFrame, var: str):
    """
    parameter
    ---------
    df: 
    var:

    return
    ------
    """
    f_df = (
        df[var]
        .describe()
        .reset_index()
        .rename(columns={"index": "stats"})
        .query("stats != 'count'")
    )
    f_df[var] = f_df[var].round(2)

    return f_df


def num_distribution(df: DataFrame, var: str, nbin: int | None=None, format_dollar: bool=True):
    c_lab = str.title(str.replace(var, "_", " "))

    tt = f"{c_lab} Distribution"
    xt = "Booking Price" if var == "price" else c_lab

    fig = histogram(
        data_frame=df,
        x=var,
        nbins=nbin,
        marginal="box",
        labels={"count": "Count", var: xt},
        title=tt,
        template="plotly_white",
        width=1000, height=450
    )

    fig.update_traces(marker_color="#E000FF", hovertemplate=f"{c_lab}: <b>%{{x}}</b><br>count: <b>%{{y:,}}</b><extra></extra>")

    fig.update_layout(
        hoverlabel={
        "bgcolor": "#D100D1",
        "font_size": 15,
        "font_color": "#FFFFFF"
    })

    fig.update_yaxes(tickformat="~s")
    if format_dollar:
        fig.update_xaxes(tickformat="~s", tickprefix="$")
    else:
        fig.update_xaxes(tickformat="~s")

    return fig.show(config=c_plt_config)



def get_price_group_summary(df, gp_var: str, num_var: str, add_agg_fun: list[str]=None, rm_agg_fun: list[str]=None) -> DataFrame:
    """
    parameter
    ---------
    df:
    gp_var:
    num_var:
    add_agg_fun:
    rm_agg_fun:

    return
    ------
    """
    agg_fun = ["min", "mean", "median", "max"]

    if add_agg_fun is not None:
        agg_fun.append(add_agg_fun)

    if rm_agg_fun is not None:
        agg_fun = [fun for fun in agg_fun if fun not in rm_agg_fun]

    agg_fun = list(set(agg_fun))

    return (
        df
        .groupby(gp_var)[num_var]
        .agg(agg_fun)
        .reset_index()
        .sort_values(by="mean", ascending=False)
    )


def price_loc_summary(df: DataFrame, var: str, count_threshold: int=5, output_typ: str="med"):
    """
    parameter
    ---------

    df:
    var:
    count_threshold:
    output_typ:

    return
    ------
    """
    # med, count
    cf_df = (
        df[["host_id", "host_location"]]
        .drop_duplicates()["host_location"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "host_location", "host_location": "count"})
    )

    if output_typ == "count":
        cf_df["proportion"] = round((cf_df["count"] / cf_df["count"].sum()) * 100, 2)

        return cf_df

    else:
        valid_loc = cf_df.loc[cf_df["count"] >= count_threshold]["host_location"].values

        cf_df = df.loc[df["host_location"].isin(valid_loc)]

        cf_df = cf_df.groupby("host_location")[var].median().reset_index()

        # cf_df["proportion"] = round((cf_df[var] / cf_df[var].sum()) * 100, 2)

        return cf_df.sort_values(by=var, ascending=False)


def price_loc_plt(df: DataFrame, x_var: str, y_var: str="host_location", add_title=""):
    """
    parameter
    ---------
    df:
    x_var:
    y_var:
    add_title:

    return
    ------

    """
    fig = bar(
        data_frame=df.sort_values(by=x_var, ascending=True),
        x=x_var,
        y=y_var,
        hover_name="host_location",
        hover_data={"host_location": False, x_var: True},
        title=f"Top 10 Locations By {add_title}",
        template="plotly_white",
        width=1000, height=450
    )

    fig.update_xaxes(title_text=None, tickformat="~s", tickprefix="$")
    fig.update_yaxes(title_text=None)

    fig.update_traces(
        hovertemplate=f"<b>%{{hovertext}}</b><br><br>{str.title(str.replace(x_var, '_', ' '))}: <b>%{{x}}</b><extra></extra>",
        marker_color="#7209B7"
    )
    fig.update_layout(hoverlabel={
        "bgcolor": "#FFFFFF",
        "font_size": 15,
        "font_color": "#480CA8"
    })
  
    return  fig.show(config=c_plt_config)





def clean_review(df: DataFrame, review_variable: str="comments") -> DataFrame:
    """
    parameter
    ---------
    df: pandas DataFrame.
    review_variable: a variable from the data `df` which serve as the customer review column.

    return
    ------
    A pandas DataFrame.
    """
    f_df = df.copy()

    # Remove automated post
    f_df["find_index"] = f_df[review_variable].str.contains("day arrival automate post")
    
    f_df = f_df.query("find_index == False")
    f_df = f_df.drop(columns="find_index")

    # Remove non english reviews
    wordz = set(words.words())

    f_df[review_variable] = f_df[review_variable].apply(lambda x: " ".join(w for w in word_tokenize(x) if w.lower() in wordz or not w.isalpha()))

    # f_df = f_df[review_variable].reset_index(drop= True).to_frame()
    f_df = f_df.reset_index(drop=True)

    return f_df




def plt_loc_price_summary(df: DataFrame, agg_var: str, gp_var: str, price_name: str, n_obs: int=10):
    dir_bl = False if agg_var == "min" else True

    f_df = df.head(n_obs).sort_values(by=agg_var, ascending=dir_bl)

    fig = bar(
        data_frame=f_df,
        x=agg_var,
        y=gp_var,
        hover_name="host_location",
        hover_data={"host_location": False, agg_var: True},
        title=f"{agg_label[agg_var]} {price_name} By Top {n_obs} Host Loaction",
        template="plotly_white",
        width=1000, height=450
    )   

    fig.update_xaxes(title_text=None, tickformat="~s", tickprefix="$")
    fig.update_yaxes(title_text=None)

    fig.update_traces(
        hovertemplate="<b>%{hovertext}</b><br><br>Median: <b>%{x}</b><extra></extra>",
        marker_color="#A100F2"
        )

    fig.update_layout(hoverlabel={
        "bgcolor": "#FFFFFF",
        "font_size": 15,
        "font_color": "#480CA8"
    })

    return fig.show(config=c_plt_config)


def num_point_plt(df, var):
    """
    parameter
    ---------
    df:
    var:

    return
    ------

    """
    p_lab = str.title(str.replace(var, "_", " "))

    fig = scatter(
        data_frame=df,
        x="host_since_days",
        y=var,
        labels={"host_since_days": "Number of Days After Listing Property", var: p_lab},
        title=f"Days vs Median {str.title(str.replace(var, '_', ' '))}",
        template="plotly_white",
        width=1000, height=450
    )
    
    fig.update_xaxes(tickformat="~s")
    fig.update_yaxes(tickformat="~s", tickprefix="$")

    fig.update_traces(
        hovertemplate=f"Period (In Days): <b>%{{x:,}}</b><br>{p_lab}: <b>%{{y:,}}</b><extra></extra>",
        marker_color="#A100F2"
    )   

    fig.update_layout(hoverlabel={
        "bgcolor": "#FFFFFF",
        "font_size": 16,
        "font_color": "#A100F2"
    })  

    return fig.show(config=c_plt_config)


def gp_stay_count(df: DataFrame, sumry_var: str, gp_var: str="host_id") -> DataFrame:
    """
    parameter
    ---------
    df:
    sumry_var:
    gp_var:

    return
    ------
    """
    f_df = df.groupby(gp_var)[sumry_var].mean().reset_index()

    f_df[sumry_var] = f_df[sumry_var].astype("int64")

    f_df = (
        f_df[sumry_var]
        .value_counts()
        .reset_index()
        .rename(columns={"index": sumry_var, sumry_var: "count"})
    )

    f_df[sumry_var] = f_df[sumry_var].astype("object")

    f_df["proportion"] = round((f_df["count"] / f_df["count"].sum()) * 100)

    return f_df



def stay_count_plt(df: DataFrame, var: str, data_label=False):
    """
    parameter
    ---------
    df:
    var:
    data_label:

    return
    ------

    """
    tt = "Provided Accommodation" if var == "accommodates" else "Minimum Nights"

    fig = bar(
        data_frame=df, #.sort_values(by="count", ascending=True),
        x=var,
        y="count",
        title=f"Average Number of {tt}",
         hover_data={var: True, "count": True, "proportion": True},
        template="plotly_white",
        width=1000, height=450
    )

    fig.update_xaxes(title_text="Number of Guests")
    fig.update_yaxes(title_text="Count", tickformat="~s")

    if data_label:
        fig.data[0].text = df["count"] 
        fig.data[0].textposition = "outside"
        fig.data[0].textfont = {"color": "#858585"}

    fig.update_traces(
        hovertemplate=f"<b>{str.title(str.replace(var, '_', ' '))}: %{{x}}</b><br><br>count: <b>%{{y}}</b><br>proportion: <b>%{{customdata[0]}}</b><extra></extra>",
        marker_color="#A100F2"
    )

    fig.update_layout(
        hoverlabel={
            "bgcolor": "#FFFFFF",
            "font_size": 15,
            "font_color": "#480CA8"
        },
        xaxis={"tickmode": "linear", "type": "category"}
    )
    
    return  fig.show(config=c_plt_config)


def price_stays_plt(df: DataFrame, var: str, p_var: str="price"):
    if var == "accommodates":
        a_tt = "Number of Guests Accommodated" 
        a_xt = "Persons"
        a_tl = "Accommodates"
    else:
        a_tt = "Minimum Number of Nights"
        a_xt = "Minimum nights"
        a_tl = "Minimum Night(s)"

    fig = bar(
        data_frame=df,
        x=var,
        y=p_var,
        title=f"Average Price vs {a_tt}",
        template="plotly_white",
        width=1000, height=450
    )

    fig.update_xaxes(title_text=f"Number of {a_xt}")
    fig.update_yaxes(title_text="Price", tickformat="~s", tickprefix="$")

    fig.update_traces(
        hovertemplate=f"{a_tl}: <b>%{{x}}</b><br>price: <b>%{{y}}</b><extra></extra>",
        marker_color="#A100F2"
    )

    fig.update_layout(
        hoverlabel={
            "bgcolor": "#FFFFFF",
            "font_size": 17,
            "font_color": "#480CA8"
        }, 
        xaxis={"tickmode": "linear", "type": "category"}
    )

    fig.show(config=c_plt_config)


def get_price_group_summary(df, gp_var: str, num_var: str, add_agg_fun: list[str]=None, rm_agg_fun: list[str]=None) -> DataFrame:
    """
    df:
    gp_var:
    num_var:
    add_agg_fun:
    rm_agg_fun:

    """
    agg_fun = ["min", "mean", "median", "max"]

    if add_agg_fun is not None:
        agg_fun.append(add_agg_fun)

    if rm_agg_fun is not None:
        agg_fun = [fun for fun in agg_fun if fun not in rm_agg_fun]

    agg_fun = list(set(agg_fun))

    return (
        df
        .groupby(gp_var)[num_var]
        .agg(agg_fun)
        .reset_index()
        .sort_values(by="mean", ascending=False)
    )



def grouped_freq(df: DataFrame, use: str, top_n: int, group_col: str="host_id") -> DataFrame:
    """
    parameter
    ---------
    df: pandas DataFrame.
    use: How to get the frequency either 'series' or 'counter'.
    top_n: Top number of frequency.
    group_col: The variable in the data df that serve as the group column.

    return
    ------
    A pandas DataFrame 
    """
    g_frequency = {}

    for group in list(df[group_col].unique()):
        joined_text = " ".join(df.query(f"{group_col} == {group}")["comments"]).lower().split(" ")
        
        if use == "counter":
            counts = Counter(joined_text).most_common(top_n)
            counts = DataFrame(counts).rename(columns = {0: "word", 1: "frequency"})

        elif use == "series":
            counts = Series(joined_text).value_counts()[:5]
            counts = counts.to_frame().reset_index().rename(columns = {"index": "word", 0: "frequency"})

        g_frequency[group] = counts

    out_df = DataFrame()

    for key, value in g_frequency.items():
        frequency_df = value
        frequency_df[group_col] = key

        out_df = concat([out_df, frequency_df])

    out_df = out_df.reset_index(drop=True)
    
    return out_df[[group_col, "word", "frequency"]]



def grouped_freq2(df: DataFrame, use: str, top_n: int, group_col: str="host_id") -> DataFrame:
    """
    parameter
    ---------
    df: pandas DataFrame.
    use: How to get the frequency either 'series' or 'counter'.
    top_n: Top number of frequency.
    group_col: The variable in the data df that serve as the group column.

    return
    ------
    A pandas DataFrame.
    """
    g_frequency = {}

    for group in list(df[group_col].unique()):
        if group_col == "host_id":
            text = " ".join(df.query(f"{group_col} == {group}")["comments"]).lower()
        else:
            text = " ".join(df.query(f"{group_col} == '{group}'")["comments"]).lower()

        tokens = list(bigrams(word_tokenize(text)))

        if use == "counter":
            count =  Counter([" ".join(bi) for bi in tokens]).most_common(top_n)
            unique_vals = DataFrame(count).rename(columns = {0: "bi_gram", 1: "frequency"})

        elif use == "series":
            count = Series([" ".join(bi) for bi in tokens]).value_counts()[:top_n]
            unique_vals = count.to_frame().reset_index().rename(columns = {"index": "bi_gram", 0: "frequency"})

        g_frequency[group] = unique_vals

    out_df = DataFrame()

    for key, value in g_frequency.items():
        frequency_df = value
        frequency_df[group_col] = key

        out_df = concat([out_df, frequency_df])
        
    out_df = out_df.reset_index(drop=True)

    return out_df[[group_col, "bi_gram", "frequency"]]


def most_frequent_words(df: DataFrame, 
                        gram: int=1, 
                        use: str="counter", 
                        top_n: int=20, 
                        review_variable: str="comments", 
                        group: bool=False, 
                        group_col: str="host_id") -> DataFrame:
    """
    parameter
    ---------
    df: pandas DataFrame
    gram: number of word grams either 1 or 2.
    use: How to get the frequency either 'series' or 'counter'.
    top_n: Top number of frequency.
    review_variable: a variable from the data df which serve as the customer review column.
    group: Whether to use grouped reviews or not.
    group_col: if group is True the variable to group by.

    return
    ------
    A pandas DataFrame.
    """
    if use not in ["counter", "series"]:
        raise ValueError(f"`use` can only be either 'counter' or 'series' and not '{use}'")

    if gram < 1 or gram > 2:
        raise ValueError(f"`gram` must be either 1 or 2 and not {gram}")

    if gram == 1:
        if group:
            output = grouped_freq(df=df, use=use, top_n=top_n, group_col=group_col)
        else:
            if use == "counter":
                output = Counter(" ".join(df[review_variable]).lower().split()).most_common(top_n)
                output = DataFrame(output).rename(columns = {0: "word", 1: "frequency"})

            elif use == "series":
                output = Series(" ".join(df[review_variable]).lower().split(" ")).value_counts()[:top_n]
                output = output.to_frame().reset_index().rename(columns = {"index": "word", 0: "frequency"})

    elif gram == 2:
        if group:
            output = grouped_freq2(df=df, use=use, top_n=top_n, group_col=group_col)
        else:
            text = " ".join(df[review_variable]).lower()
            tokens = list(bigrams(word_tokenize(text)))

            if use == "counter":
                output = Counter([" ".join(bi) for bi in tokens]).most_common(top_n)
                output = DataFrame(output).rename(columns = {0: "bi_gram", 1: "frequency"})
            
            elif use == "series":
                output = Series([" ".join(bi) for bi in tokens]).value_counts()[:top_n]
                output = output.to_frame().reset_index().rename(columns = {"index": "bi_gram", 0: "frequency"})

    return output



def get_wordnet_pos(pos_tag):
    """
    parameter
    ---------
    pos_tag

    return
    ------

    """
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
        


def clean_text(text: str, not_useful: list) -> Series:
    """
    parameter
    ---------
    text: A pandas series or a string
    not_useful: words to remove when cleaning the text.

    return
    ------
    pandas series.
    """
    # Convert all text to lowercase
    text = text.lower()

    # tokenize
    text = word_tokenize(text)

    # remove punctuation
    text = [tx for tx in text if tx not in punctuation] 

    # remove words that contain numbers
    text = [tx for tx in text if tx not in digits]

    # remove stop words
    not_useful_words = stopwords.words("english") + not_useful
    text = [tx for tx in text if tx not in not_useful_words]

    # remove empty tokens
    text = [tx for tx in text if len(tx) > 1]

    # pos tag text
    pos_tags = pos_tag(text)

    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]

    # remove words with only one letter
    text = [tx for tx in text if len(tx) > 1]

    # join all
    text = " ".join(text)

    return(text)



def create_polarity_score(df: DataFrame, review_variable: str="comments") -> DataFrame:
    """
    parameter
    ---------
    df: pandas DataFrame.
    review_variable: a variable from the data df which serve as the customer review column.

    return
    ------
    pandas DataFrame.
    """
    f_df = df.copy()

    sid = SentimentIntensityAnalyzer()

    f_df["polarity_scores"] = f_df[review_variable].apply(lambda x: sid.polarity_scores(x))
    f_df = concat([f_df.drop(['polarity_scores'], axis=1), f_df['polarity_scores'].apply(Series)], axis=1)

    f_df["sentiment"] = nan

    f_df.loc[(f_df["pos"] > f_df["neu"]) & (f_df["pos"] > f_df["neg"]), "sentiment"] = "Positive"
    f_df.loc[(f_df["neg"] > f_df["pos"]) & (f_df["neg"] > f_df["neu"]), "sentiment"] = "Negative"
    f_df.loc[(f_df["neu"] > f_df["pos"]) & (f_df["neu"] > f_df["neg"]), "sentiment"] = "Neutral"
    f_df.loc[(f_df["neg"] == f_df["pos"]) | (f_df["neu"] == f_df["neg"]) | (f_df["neu"] == f_df["pos"]), "sentiment"] = "Neutral"

    return f_df



def add_character_count(df: DataFrame) -> DataFrame:
    """
    parameter
    ---------
    df: pandas DataFrame
    """
    f_df = df.copy()

    f_df["number_char"] = f_df["comments"].apply(lambda x: len(x))

    f_df["number_words"] = f_df["comments"].apply(lambda x: len(x.split(" ")))

    return f_df



def wrangle(df: DataFrame, not_useful: list) -> DataFrame:
    """
    parameter
    ---------
    df: cleaned DataFrame
    not_useful: a list of words to drop

    return
    ------
    pandas Dataframe.
    """
    f_df = df.copy()

    # Clean Text
    f_df["comments"] = f_df["comments"].apply(lambda x: clean_text(text=x, not_useful=not_useful))

    # Add Polarity Scores
    f_df = create_polarity_score(df=f_df)

    # Add Character Count
    f_df = add_character_count(df=f_df)

    return f_df



def polarity_count(df: DataFrame, sentiment_variable: str="sentiment") -> DataFrame:
    """
    parameter
    ---------
    df: text cleaned Dataframe.
    sentiment_variable: a variable from the data df which contains the sentiments.

    returns
    -------
    summary pandas DataFrame.
    """
    f_sentiment = (
        df[sentiment_variable]
        .value_counts()
        .to_frame()
        .reset_index()
        .rename(columns={"index": sentiment_variable, sentiment_variable: "count"})
    )

    f_sentiment["percentage"] = round(f_sentiment["count"] / f_sentiment["count"].sum()*100, 2)

    return f_sentiment


def check_words(words: list, check_words: list) -> str:
    """
    parameter
    ---------
    words: A list of words to check from.
    check_words: A word or list of words to check.

    return
    ------
    A string of word(s)
    """
    words = list(set(words))
    check_words = check_words if isinstance(check_words, list) else [check_words]
    
    output = []

    for word in check_words:
        if word in words:
            output.append("1")
        else:
            output.append("0")
    
    output = ", ".join(output) if len(output) > 1 else "".join(output)

    return output


def convert_to_list(df, var, rm_punt: list=["{", "}", '"']):
    """
    parameter
    ---------
    df:
    var:
    rm_punt:

    return
    ------

    """
    def rmv_punt(x, punt_list: list=rm_punt):
        for punt in punt_list:
            x = str.replace(x, punt, "")
        
        return str.split(x, ",")

    df[var] = df[var].apply(lambda x: rmv_punt(x))

    return df

    

def amenities_summary(df: DataFrame, amenities: list, am_name):
    """
    parameter
    ---------
    df:
    amenities:

    return
    ------

    """
    f_df = df[["host_id", "amenities"]]

    f_df = f_df.drop_duplicates()

    nrows = f_df.shape[0]

    f_df = convert_to_list(f_df, "amenities")

    f_df = add_present_words(f_df, amenities, search_variable="amenities", keep_edited_search_cols=False)

    f_df = f_df[amenities].agg("sum").reset_index().rename(columns={"index": am_name, 0: "count"})

    f_df["proportion"] = round((f_df["count"] / nrows)*100, 2)

    return f_df.sort_values(by="count", ascending=False)


def plt_amenities_summary(df, am_name):
    am_lab = str.title(str.replace(am_name, "_", " "))

    fig = bar(
        data_frame=df,
        x=am_name,
        y="count",
        hover_name=am_name,
        hover_data={am_name: False, "count": True, "proportion": True},
        title=f"Number Of Hosts Provisions Based On {am_lab}",
        template="plotly_white",
        width=1000, height=450
    )

    fig.update_xaxes(title_text=None)
    fig.update_yaxes(title_text=None, tickformat="~s")

    fig.update_traces(
        hovertemplate="<b>%{hovertext}</b><br><br>count: <b>%{y}</b><br>proportion: <b>%{customdata[0]}%</b><extra></extra>",
        marker_color="#A100F2"
    )

    fig.update_layout(hoverlabel={
        "bgcolor": "#FFFFFF",
        "font_size": 16,
        "font_color": "#000000"
    })

    return fig.show(config=c_plt_config)


def freq_polarity_plot(df_frq: DataFrame, seti_frq: DataFrame, title: str, subplot_titles: tuple):
    """
    parameter
    ---------
    df_frq: a summarised frequency dataframe.
    seti_frq: a summarised sentiment count dataframe.
    title: title of the plot.
    subplot_titles: titles of each subplot. e.g ("subplot title1", "subplot title2")

    return
    ------
    plotly graph object.
    """
    f_fig = make_subplots(
        rows=1, 
        cols=2, 
        horizontal_spacing = 0.1,
        specs=[[{"type": "bar"}, {"type": "domain"}]], 
        subplot_titles=subplot_titles
    )

    f_fig.add_trace(
        Bar(
            x=df_frq["frequency"], 
            y=df_frq["word"], 
            orientation="h", 
            marker=dict(color="blue"), 
            marker_color=colors.sequential.Plotly3_r,
            hovertemplate= "<b>%{y}</b><br><br>Frequency : <b>%{x}</b><extra></extra>"
        ),
        row=1, col=1
    )

    f_fig.add_trace(
        Pie(
            labels=seti_frq["sentiment"], 
            values=seti_frq["count"], 
            hole=0.6, 
            hovertemplate="<b>%{label}</b><br><br>Count : <b>%{value}</b><extra></extra>",
            showlegend=False,
            textposition="outside",
            textinfo="label+percent",
            marker=dict(colors=["#9A32CD", "#EE30A7", "#FF7F50"])
        ),
        row=1, col=2
    )
    
    f_fig.update_layout(
        height=500, 
        width=960, 
        title=title,
        showlegend=False, 
        template="plotly_white",
        # annotations=[dict(text="look", x=0.18, y=0.5, font_size=20, showarrow=False)]
    )

    f_fig.update_traces(marker={"line": {"color": "#FFFFFF", "width": 1}})

    return  f_fig.show(config=c_plt_config)


def plt_featured_grams(df: DataFrame, search_grams: list, title: str):

    feat_words = add_present_words(df, search_grams, "bi_gram")       
    feat_words = feat_words[search_grams]
    feat_words = feat_words.sum().to_frame().reset_index().rename(columns={"index": "word", 0: "count"})
    feat_words["percentage"] = round(feat_words["count"] / df.shape[0]*100, 2)

    fig = bar(                                                        
        data_frame=feat_words.sort_values(by="count", ascending=False),
        x="count",
        y="word",
        color="word",
        hover_name="word",
        color_discrete_sequence=colors.sequential.Plotly3,
        hover_data={"word": False, "count": True, "percentage": True},
        title=f"Number Of Reviews Containing Selected Words {title}",
        template="plotly_white",
        width=1000, height=450
    )

    fig.update_xaxes(title_text=None, tickformat="~s")
    fig.update_yaxes(title_text=None)

    fig.update_traces(
        hovertemplate="<b>%{hovertext}</b><br><br>Count: <b>%{x}</b><br>Percentage: <b>%{customdata[0]}%</b><extra></extra>"
    )

    fig.update_layout(
        hoverlabel={"font_size": 15},
        showlegend=False
    )

    fig.show(config = c_plt_config)



def add_present_words(df: DataFrame, 
                      search_words: list, 
                      search_variable: str="comments", 
                      keep_edited_search_cols: bool=False) -> DataFrame:
    """
    parameter
    ---------
    df: pandas DataFrame.
    search_words: A word or list of words to check.
    search_variable: A variable in the data df to search from.
    keep_edited_search_cols: Whether to keep The columns used for the search.

    return
    ------
    pandas DataFrame
    """
    f_df = df.copy()
    f_search_variable = f"f_{search_variable}"

    # tokenize reviews
    try:
        s_index = f_df[search_variable].index.start
    except:
        s_index = f_df[search_variable].index[0]

    if isinstance(f_df[search_variable][s_index], str):
        f_df[f_search_variable] = DataFrame(f_df[search_variable].apply(lambda x: word_tokenize(x)))
    else:
        f_df[f_search_variable] = f_df[search_variable]

    # check if words are present
    f_df["searched_words"] = f_df[f_search_variable].apply(lambda x: check_words(words=x, check_words=search_words))

    # add check to data.
    f_df[search_words] = f_df["searched_words"].str.split(", ", expand=True).astype("int64")

    if keep_edited_search_cols:
        return f_df
    else:
        f_df = f_df.drop(["searched_words", f_search_variable], axis=1)
        return f_df



def plot_present_words(present_word_df: DataFrame, search_words: list, add_title: str="", color=None):
    """
    parameter
    ---------
    present_word_df: DataFrame with `search_words` variable in it.
    search_words: word variables present in the `present_word_df` table.
    add_title: plot title.

    return
    ------
    plotly graph object
    """
    f_df = present_word_df[search_words]

    fs_df = f_df.sum().to_frame().reset_index().rename(columns={"index": "word", 0: "count"})
    
    fs_df["percentage"] = round(fs_df["count"] / f_df.shape[0]*100, 2)

    color = colors.sequential.Plotly3 if color is None else color

    f_fig = bar(
            data_frame=fs_df, 
            x="word" if len(search_words) <= 5 else "count", 
            y="count" if len(search_words) <= 5 else "word", 
            color= "word",
            color_discrete_sequence = color,
            labels={"word": "", "count": "Number Of Reviews"},
            title=f"Number Of Reviews Which Contain Selected {add_title} Words",
            template="plotly_white",
            custom_data=["percentage"],
            # hover_data = {"word": False, "count": ":.2f", "percentage": True},
            width=800
        )
        
    f_fig.update_traces(hovertemplate="Count : <b>%{value:,.0f}</b><br><br>Percentage : <b>%{customdata:.2f}%</b><extra></extra>", showlegend=False)

    return  f_fig.show(config = c_plt_config)



def create_bigram(df: DataFrame, review_variable: str="comments") -> DataFrame:
    """
    parameter
    ---------
    df: pandas DataFrame.
    review_variable: a variable from the data df which serve as the customer review column.

    return
    -------
    pandas DataFrame with a `bi_gram` variable.
    """
    f_df = df.copy()

    # Tokenize by single words
    f_df["bi_gram"] = f_df[review_variable].apply(lambda x: word_tokenize(x))

    # Create bigram
    f_df["bi_gram"] = f_df["bi_gram"].apply(lambda x: list(bigrams(x)))

    # joins bi words
    f_df["bi_gram"] = f_df["bi_gram"].apply(lambda x: [" ".join(bi) for bi in x])

    return f_df



def create_word_cloud(df: DataFrame, random: bool=False, review_variable: str="comments"):
    """
    parameter
    ---------
    df: pandas DataFrame
    random: reproducible.
    review_variable: a variable from the data df which serve as the customer review column.

    return
    ------
    """
    f_df = df.copy()

    text = " ".join(f_df[review_variable])
    random = None if random else 11

    word_cloud = WordCloud(
        max_font_size=100, 
        max_words=100, 
        background_color="#EEE9E9", 
        width=800, height=400,
        random_state=random,
        colormap="cool"
    ).generate(text)

    figure(figsize=(18, 8)) # facecolor="k"
    tight_layout(pad=0)
    imshow(word_cloud, interpolation='bilinear')
    axis("off");



def freq_bar_plot(df: DataFrame, gram: str="word"):
    """
    df: pandas DataFrame
    """        
    fig = figure(1, figsize=(12, 7))

    barplot(data = df, x = "frequency", y = gram, palette = "cool_r")
    tick_params(labelsize = 13)
    ylabel("")

    gram_lab = "Word" if gram == "word" else "Bigram"
    xlabel(f"{gram_lab} Frequency", fontsize=13)
    title(f"Top {df.shape[0]} Customer Review Words", loc="left", fontsize=15);



def plot_featured_grams(df: DataFrame, search_grams: list, orientation: str="v"):
    """
    parameters
    ----------
    df:
    search_grams: a list of bigrams to search for.
    orientation: orientation of the plot. either 'v' or 'h'

    return
    ------
    plotly graph object.
    """
    feat_words = add_present_words(df, search_grams, "bi_gram")
    feat_words = feat_words[search_grams]
    feat_words = feat_words.sum().to_frame().reset_index().rename(columns={"index": "word", 0: "count"})
    feat_words["percentage"] = round(feat_words["count"] / df.shape[0]*100, 2)

    if orientation == "h":
        feat_words = feat_words.sort_values(by="count")
    elif orientation == "v":
        feat_words = feat_words.sort_values(by="count", ascending=False)
    
    def go_bar(x, y, colr, orient=orientation):
        go_fig = Bar(
            x=feat_words[x],
            y=feat_words[y],
            orientation=orient,
            marker_color=colr,
            customdata = feat_words["percentage"],
            hoverinfo= "y+text",
            hovertemplate=f"count : %{{value:,.0f}}<br>percentage : %{{customdata:.2f}}%<extra></extra>"
        )
        return go_fig

    if len(search_grams) <= 5 and orientation == "v":
        f_fig = go_bar(x="word", y="count", colr=colors.sequential.Plotly3)
    else:
        f_fig = go_bar(x="count", y="word", colr=colors.sequential.Plotly3)

    return f_fig



def get_host_wb_freq(df: DataFrame, host_info: DataFrame=None, filter_words: str=None, get: str="host_freq"):
    """
    parameter
    ---------
    df: dataframe with the most frequent words or bigram.
    host_info: dataframe with host names.
    filter_words: filter the table.
    get: the type of output. either 'host_freq' or 'host_count'.

    return
    ------
    if `get` is 'host_freq' pandas DataFrame else number of host.
    """
    word_col = "bi_gram" if "bi_gram" in df.columns else "word"

    if get == "host_freq":
        df_freq = df.query(f"{word_col} == '{filter_words}'").sort_values(by="frequency", ascending=False)

        df_freq = df_freq.join(host_info.set_index("host_id"), on="host_id", how="left").drop_duplicates()

        return df_freq[["host_id", "frequency", "host_name"]]
        
    elif get == "host_count":
        return df.query(f"{word_col} == '{filter_words}'").sort_values(by="frequency", ascending=False)["host_id"].nunique()



def host_freq_plot(df: DataFrame, host_info: DataFrame, filter_words: str=None):
    """
    parameter
    ---------
    df: DataFrame with frequent bigram words for a host.
    filter_words: filter the table.

    return
    -------
    plotly graph object.
    """
    
    f_df = get_host_wb_freq(df, host_info=host_info, filter_words=filter_words, get="host_freq").head(10).sort_values(by="frequency")

    f_fig = Scatter(
            x=f_df["frequency"],
            y=f_df["host_name"],
            mode="markers",
            marker_color=colors.sequential.Plotly3_r,
            marker=dict(size=f_df["frequency"], color=f_df["frequency"]),
            hovertemplate = "<b>%{y}</b><br><br>Count: <b>%{x}</b><extra></extra>",
            textfont=dict(color="white")
        )

    return f_fig



def loc_freq_plot(df, filter_bigram):
    """
    parameter
    ---------
    df: DataFrame with frequent bigram words for a host.
    filter_bigram: Two Specific words to filter by.

    return
    ------
    plotly graph object.
    """

    f_df = df.query(f"bi_gram == '{filter_bigram}'").sort_values(by="frequency", ascending=True).head(10)

    f_fig = Scatter(
            x=f_df["frequency"],
            y=f_df["host_location"],
            mode="markers",
            marker_color=colors.sequential.Plotly3_r,
            marker=dict(size=f_df["frequency"]*6, color=f_df["frequency"]),
            hovertemplate = "<b>%{y}</b><br><br>Count: <b>%{x}</b><extra></extra>",
            textfont=dict(color="white", size=20)
        )
        
    return f_fig