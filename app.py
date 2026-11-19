import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(
    page_title="Mobility Stories | India",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"

# Cinematic, high-contrast OTT-inspired styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');
:root{--paper:#F3F0E8;--green:#183B35;--clay:#D56C4E;--ink:#20312D;--muted:#68746E;--white:#FFFEFA;--line:#D9DED5}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:var(--paper);color:var(--ink)}
[data-testid="stHeader"]{background:rgba(243,240,232,.94)}
.block-container{max-width:1440px;padding-top:2rem;padding-bottom:4rem}
h1,h2,h3,h4{font-family:'Playfair Display',Georgia,serif!important;color:var(--green)!important;letter-spacing:-.3px}
h1{font-size:clamp(2.5rem,5vw,4.5rem)!important;line-height:1.05!important}
h2{font-size:2rem!important} p,li,label,.stMarkdown{color:var(--ink)}
hr{border-color:var(--line)}
[data-testid="stSidebar"]{background:var(--green);border-right:0}
[data-testid="stSidebar"] *{color:#F7F4EC}
[data-testid="stSidebar"] .block-container{padding:2rem 1.15rem}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] hr{border-color:rgba(255,255,255,.18);margin:1.3rem 0}
[data-testid="stSidebar"] [data-testid="stRadio"]>label{color:#D2DDD4!important;font-size:10px!important;font-weight:700!important;letter-spacing:1.6px;margin-bottom:12px}
[data-testid="stSidebar"] [role="radiogroup"]{gap:5px}
[data-testid="stSidebar"] [role="radiogroup"] label{padding:8px 10px;border-radius:4px;border:1px solid transparent;margin:0}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{background:rgba(255,255,255,.09)}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){background:rgba(255,255,255,.13);border-left:3px solid #E28A6D}
[data-testid="stSidebar"] [role="radiogroup"] label p{color:#F7F4EC!important;font-size:13px;font-weight:500}
.side-mark{font-family:'Playfair Display',Georgia,serif;font-size:27px;line-height:1.05;color:#fff}
.side-kicker{color:#C5D3C9;font-size:10px;letter-spacing:2px;text-transform:uppercase;margin-top:9px}
.side-note{color:#C5D3C9;font-size:12px;line-height:1.8}
.kicker{color:var(--clay);font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase;margin-bottom:10px}
.hero-editorial{background:var(--green);color:#F7F4EC;padding:clamp(28px,5vw,64px);border-radius:3px;min-height:350px;display:flex;flex-direction:column;justify-content:center;position:relative;overflow:hidden}
.hero-editorial:after{content:"";position:absolute;right:-100px;top:-120px;width:390px;height:390px;border:1px solid rgba(255,255,255,.18);border-radius:50%;box-shadow:0 0 0 35px rgba(255,255,255,.035),0 0 0 75px rgba(255,255,255,.025)}
.hero-editorial .kicker{color:#E7A18A}
.hero-editorial h1{color:#FFFDF6!important;max-width:800px;position:relative;z-index:1}
.hero-editorial p{color:#D8E1D9;font-size:17px;line-height:1.7;max-width:700px;position:relative;z-index:1}
.hero-meta{color:#D9CFC0;font-size:11px;letter-spacing:1.2px;text-transform:uppercase;margin-top:20px;position:relative;z-index:1}
.number-card{background:var(--white);border:1px solid var(--line);padding:20px 22px;min-height:112px}
.number-card .n{font-family:'Playfair Display',Georgia,serif;color:var(--green);font-size:35px;line-height:1}
.number-card .l{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:1.2px;margin-top:10px}
.editorial-card{background:var(--white);border:1px solid var(--line);padding:23px;height:100%}
.editorial-card .index{color:var(--clay);font-size:11px;letter-spacing:1.6px;font-weight:700}
.editorial-card p,.caption-editorial{color:var(--muted);line-height:1.65;font-size:13px}
.intro-copy{font-family:'Playfair Display',Georgia,serif;font-size:23px;line-height:1.5;color:var(--green)}
.callout{background:#E4EAE3;border-left:4px solid var(--clay);padding:18px 21px;color:var(--green)}
div[data-testid="stMetric"]{background:var(--white);border:1px solid var(--line);padding:14px 17px;border-radius:3px}
div[data-testid="stMetric"] label{color:var(--muted)!important}
div[data-testid="stMetric"] [data-testid="stMetricValue"]{color:var(--green)!important}
.stButton>button,.stDownloadButton>button{background:var(--clay);color:#fff;border:1px solid var(--clay);border-radius:3px;font-weight:700;min-height:42px}
.stButton>button:hover,.stDownloadButton>button:hover{background:#B9583F;border-color:#B9583F;color:#fff}
.stSelectbox div[data-baseweb="select"]>div,.stTextInput input{background:var(--white);color:var(--ink);border-color:#C8D0C7;border-radius:3px}
[data-testid="stDataFrame"]{border:1px solid var(--line)}
[data-testid="stExpander"]{background:var(--white);border:1px solid var(--line);border-radius:3px}
footer{visibility:hidden}

/* Midnight blue / black / restrained neon-red redesign */
:root{
 --paper:#080B12; --green:#111827; --clay:#FF3355; --ink:#E8EEF8;
 --muted:#9AA9BF; --white:#111827; --line:#263449;
 --blue:#2878F0; --blue-bright:#4B9BFF;
}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:radial-gradient(ellipse at 50% -20%,#142849 0%,#0A1220 42%,#070A10 82%);color:#E8EEF8}
[data-testid="stHeader"]{background:rgba(8,11,18,.92)}
.block-container{max-width:1480px;padding-top:1.1rem;padding-bottom:3.5rem}
h1,h2,h3,h4{font-family:'Playfair Display',Georgia,serif!important;color:#F4F7FC!important}
p,li,label,.stMarkdown{color:#D5DFEF}
hr{border-color:#263449}
/* Hide the old left rail completely; navigation now lives above the page. */
[data-testid="stSidebar"], [data-testid="collapsedControl"]{display:none!important}
.top-brand{display:flex;align-items:center;justify-content:space-between;gap:18px;padding:5px 2px 16px;border-bottom:1px solid #263449;margin-bottom:12px}
.top-brand-name{font-family:'Playfair Display',Georgia,serif;font-size:25px;line-height:1.1;color:#F5F7FC}
.top-brand-name span{color:#4B9BFF}
.top-brand-note{font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#91A5C2}
.top-nav-label{font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#8196B5;margin:7px 0 5px}
div[data-testid="stRadio"]>label{display:none}
div[data-testid="stRadio"] [role="radiogroup"]{display:flex;flex-wrap:wrap;gap:7px;padding:5px 0 15px;border-bottom:1px solid #263449;margin-bottom:26px}
div[data-testid="stRadio"] [role="radiogroup"] label{background:#101827;border:1px solid #263449;border-radius:9px;padding:8px 13px;margin:0;transition:all .15s ease}
div[data-testid="stRadio"] [role="radiogroup"] label:hover{border-color:#4B9BFF;background:#14233A}
div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked){background:#17345B;border-color:#2878F0;box-shadow:inset 0 -2px 0 #FF3355}
div[data-testid="stRadio"] [role="radiogroup"] label p{color:#DCE7F7!important;font-size:13px;font-weight:600}
div[data-testid="stRadio"] [role="radiogroup"] label>div:first-child{display:none}
.hero-editorial,.hero{background:linear-gradient(115deg,#101D32 0%,#102B50 58%,#101827 100%);color:#F5F7FC;border:1px solid #29466D;border-radius:15px;padding:clamp(28px,5vw,58px);min-height:320px;position:relative;overflow:hidden}
.hero-editorial:after{border-color:rgba(75,155,255,.22);box-shadow:0 0 0 35px rgba(40,120,240,.06),0 0 0 75px rgba(40,120,240,.035)}
.hero-editorial h1,.hero h1{color:#F5F7FC!important}
.hero-editorial p,.hero p{color:#C5D4E9}
.kicker,.section-kicker,.eyebrow{color:#4B9BFF!important}
.hero-editorial .kicker,.hero .eyebrow{color:#FF5570!important}
.number-card,.editorial-card,.story-card,.metric{background:linear-gradient(145deg,#121D2D,#0E1521)!important;border:1px solid #263449!important;color:#E8EEF8}
.number-card .n,.metric-value{color:#F4F7FC!important}
.number-card .l,.metric-label,.metric-note,.editorial-card p,.story-card p,.caption-editorial{color:#9AAAC1!important}
.editorial-card .index{color:#FF5570!important}
.callout{background:#12233A;border-left:4px solid #FF3355;color:#DCE7F7}
div[data-testid="stMetric"]{background:#111C2B;border:1px solid #263449}
div[data-testid="stMetric"] label{color:#9AAAC1!important}
div[data-testid="stMetric"] [data-testid="stMetricValue"]{color:#F4F7FC!important}
.stButton>button,.stDownloadButton>button{background:#1E63C5;color:white;border:1px solid #347FE5;border-radius:9px;font-weight:700;min-height:41px}
.stButton>button:hover,.stDownloadButton>button:hover{background:#2878F0;border-color:#FF3355;color:white}
.stSelectbox div[data-baseweb="select"]>div,.stTextInput input,.stMultiSelect div[data-baseweb="select"]>div{background:#111C2B;color:#E8EEF8;border-color:#31445F;border-radius:8px}
[data-testid="stDataFrame"]{border:1px solid #263449;border-radius:9px}
[data-testid="stExpander"]{background:#111827;border:1px solid #263449;border-radius:9px}
div[data-testid="stAlert"]{background:#111E30;border-color:#2A4B73;color:#DCE7F7}
[data-testid="stCaptionContainer"]{color:#91A2BB}
footer{visibility:hidden}
</style>
""", unsafe_allow_html=True)

# =========================================================
# DATA
# =========================================================
@st.cache_data
def load_csv(filename, index_col=None):
    path = OUT / filename
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, index_col=index_col)
    except Exception:
        return None

regional = load_csv("regional_mobility.csv", index_col=0)
scores = load_csv("model_scores.csv")
full_clusters = load_csv("final_full_dataset_clusters.csv", index_col=0)
sensitivity_clusters = load_csv("final_sensitivity_clusters.csv", index_col=0)
full_profiles = load_csv("final_full_cluster_profiles.csv", index_col=0)
sensitivity_profiles = load_csv("final_sensitivity_profiles.csv", index_col=0)
pca_coordinates = load_csv("final_pca_coordinates.csv")
selected_metrics = load_csv("final_selected_metrics.csv")
manhattan = load_csv("manhattan_comparison.csv")

if regional is None:
    st.error("I couldn't find outputs/regional_mobility.csv. Run preprocess.py from your project folder first.")
    st.stop()

regional.index = regional.index.astype(str)
FEATURE_LABELS = {
    "retail_and_recreation_percent_change_from_baseline": "Shopping & recreation",
    "grocery_and_pharmacy_percent_change_from_baseline": "Grocery & pharmacy",
    "parks_percent_change_from_baseline": "Parks",
    "transit_stations_percent_change_from_baseline": "Public transport",
    "workplaces_percent_change_from_baseline": "Workplaces",
    "residential_percent_change_from_baseline": "Time at home"
}
friendly = {c: FEATURE_LABELS.get(c, c.replace("_percent_change_from_baseline","").replace("_"," ").title()) for c in regional.columns}

def metric_box(label, value, note=""):
    st.markdown(f"""<div class="metric"><div class="metric-label">{label}</div>
    <div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>""", unsafe_allow_html=True)

def heading(kicker, title, description=None):
    st.markdown(f'<div class="section-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.markdown(f"## {title}")
    if description:
        st.markdown(f"<p style='color:#adbed5;margin-top:-8px'>{description}</p>", unsafe_allow_html=True)

def chart_theme(fig, height=430):
    fig.update_layout(
        template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#dce7f7"),
        margin=dict(l=20,r=20,t=55,b=25), legend_title_text=""
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,.08)", zerolinecolor="rgba(255,255,255,.15)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,.08)", zerolinecolor="rgba(255,255,255,.15)")
    return fig

# =========================================================
# TOP NAVIGATION — moved from the left sidebar into a horizontal menu
# =========================================================
PAGES = [
    "01  The story", "02  The measures", "03  The patterns",
    "04  Compare regions", "05  Check the evidence",
    "06  Sensitivity check", "07  Distance methods", "08  Research notes"
]
st.markdown(
    '<div class="top-brand"><div><div class="top-brand-name">Mobility <span>Stories.</span></div>'
    '<div class="top-brand-note">An India mobility study · 2022</div></div>'
    '<div class="top-brand-note">Regional mobility · Clustering research</div></div>',
    unsafe_allow_html=True
)
st.markdown('<div class="top-nav-label">Explore the study</div>', unsafe_allow_html=True)
page = st.radio("Explore the study", PAGES, horizontal=True, label_visibility="collapsed")

# =========================================================
# DISCOVER
# =========================================================
if page == "01  The story":
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">A data story · India · 2022</div>
      <h1>How did everyday movement<br>change across India?</h1>
      <p>Explore patterns in shopping, travel, parks, workplaces and time spent at home.
      This project uses clustering to group regions with similar mobility-change profiles.</p>
      <div><span class="pill">Unsupervised learning</span><span class="pill">Google Community Mobility Reports</span><span class="pill">Interactive research</span></div>
    </div>""", unsafe_allow_html=True)
    st.write("")
    a,b,c,d = st.columns(4)
    with a: metric_box("Regions studied", f"{len(regional)}", "State/territory-level averages")
    with b: metric_box("Mobility signals", f"{len(regional.columns)}", "Six everyday activity categories")
    with c: metric_box("PCA summary", "98.14%", "Variance in first two components")
    with d: metric_box("Cluster settings tested", "2–6", "Five K-Means choices")
    st.write("")
    heading("START HERE", "The story in three stops", "No machine-learning background needed.")
    x,y,z = st.columns(3)
    with x:
        st.markdown('<div class="story-card"><div class="section-kicker">01 · Understand</div><h3>What was measured?</h3><p>Learn what each mobility indicator means and how the Google baseline works.</p></div>', unsafe_allow_html=True)
    with y:
        st.markdown('<div class="story-card"><div class="section-kicker">02 · Explore</div><h3>Which regions look alike?</h3><p>See how clustering groups regions with similar overall mobility-change patterns.</p></div>', unsafe_allow_html=True)
    with z:
        st.markdown('<div class="story-card"><div class="section-kicker">03 · Question</div><h3>Can we trust the pattern?</h3><p>Explore model scores, unusual observations and sensitivity checks.</p></div>', unsafe_allow_html=True)
    st.write("")
    st.info("**The headline finding:** the full-data three-cluster experiment separates Jammu and Kashmir as a singleton group, while the other 35 regions form two larger groups. This unusual observation is investigated rather than hidden.")
    with st.expander("What does “percent change from baseline” mean?"):
        st.write("A value above zero indicates activity was above its Google-defined baseline for that category and date; a value below zero indicates it was below baseline. It is a percentage change from a reference period, not a percentage of people or an absolute movement count.")

# =========================================================
# THE DATA
# =========================================================
elif page == "02  The measures":
    heading("01 · UNDERSTAND THE EVIDENCE", "Six signals of everyday life", "Choose an indicator to see what it represents in ordinary language.")
    descriptions = {
        "Shopping & recreation":"Visits and activity around places such as restaurants, cafes, shopping centres and entertainment venues.",
        "Grocery & pharmacy":"Activity around grocery markets, food warehouses, farmers markets and pharmacies.",
        "Parks":"Activity around parks, public beaches, marinas, dog parks and similar outdoor spaces.",
        "Public transport":"Activity around public transport hubs such as subway, bus and train stations.",
        "Workplaces":"Activity around places of work.",
        "Time at home":"Time spent in residential areas. This indicator often moves in the opposite direction to activity outside the home."
    }
    cols = st.columns(3)
    icons = ["🛍️","🛒","🌳","🚆","🏢","🏠"]
    for i,(col,(name,desc)) in enumerate(zip(cols*2, descriptions.items())):
        with col:
            st.markdown(f'<div class="story-card"><div style="font-size:30px">{icons[i]}</div><h3>{name}</h3><p>{desc}</p></div>', unsafe_allow_html=True)
    st.write("")
    heading("THE DATASET", "A quick look at the study")
    a,b,c = st.columns(3)
    with a: metric_box("Regional profiles",len(regional),"One aggregated row per region")
    with b: metric_box("Indicators",len(regional.columns),"Mobility categories")
    with c: metric_box("Study window","Jan–Oct 2022","Through October 15")
    st.markdown("#### Explore regional averages")
    selected = st.selectbox("Choose a region", regional.index.tolist())
    vals = regional.loc[selected].rename(index=friendly)
    fig = px.bar(x=vals.values,y=vals.index,orientation="h",labels={"x":"Average change from baseline (%)","y":""},title=f"Mobility profile · {selected}")
    st.plotly_chart(chart_theme(fig,360),use_container_width=True)
    with st.expander("Show the regional data table"):
        st.dataframe(regional.rename(columns=friendly).round(2),use_container_width=True)
        st.download_button("Download regional data",regional.to_csv().encode("utf-8"),"regional_mobility.csv","text/csv")
    st.warning("These are aggregated Google mobility indicators based on opted-in location data. They are not a census of all residents or direct counts of trips.")

# =========================================================
# MOBILITY PATTERNS
# =========================================================
elif page == "03  The patterns":
    heading("02 · EXPLORE THE GROUPS", "Regions with similar mobility profiles", "Clustering is a way of grouping regions whose six mobility indicators look similar after scaling.")
    mode = st.radio("Which view would you like?",["Main study · all 36 regions","Sensitivity view · without Jammu and Kashmir"],horizontal=True)
    data = regional.copy() if mode.startswith("Main") else regional.drop(index="Jammu and Kashmir",errors="ignore")
    k = st.select_slider("Choose how many groups to create",options=[2,3,4,5,6],value=3 if mode.startswith("Main") else 2)
    X = StandardScaler().fit_transform(data)
    labels = KMeans(n_clusters=k,random_state=42,n_init=50).fit_predict(X)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    plot = pd.DataFrame({"Region":data.index,"PC1":coords[:,0],"PC2":coords[:,1],"Group":labels.astype(str)})
    ev = pca.explained_variance_ratio_.sum()*100
    a,b,c = st.columns(3)
    with a: metric_box("Groups shown",k,"Selected by you")
    with b: metric_box("Regions included",len(data),"Current view")
    with c: metric_box("2D map coverage",f"{ev:.1f}%","Variation represented by PCA axes")
    fig = px.scatter(plot,x="PC1",y="PC2",color="Group",hover_name="Region",hover_data={"PC1":":.2f","PC2":":.2f"},color_discrete_sequence=["#4B9BFF","#FF3355","#76B7FF","#8B9BB5","#2D65B0","#D5E5FF"],title="Each dot is one region · nearby dots have more similar profiles")
    fig.update_traces(marker=dict(size=14,line=dict(width=1,color="#f4f7ff")))
    st.plotly_chart(chart_theme(fig,530),use_container_width=True)
    dist = plot["Group"].value_counts().sort_index().rename_axis("Group").reset_index(name="Regions")
    fig2 = px.bar(dist,x="Group",y="Regions",color="Group",text="Regions",title="How many regions are in each group",color_discrete_sequence=px.colors.qualitative.Bold)
    st.plotly_chart(chart_theme(fig2,300),use_container_width=True)
    chosen_group = st.selectbox("Inspect a group",sorted(plot.Group.unique(),key=int))
    members = plot.loc[plot.Group==chosen_group,"Region"].tolist()
    st.markdown(f"**Group {chosen_group} includes {len(members)} region(s).**")
    st.write(", ".join(members))
    if mode.startswith("Main") and k==3:
        st.info("In the saved primary experiment, the three groups contain 17, 1 and 18 regions. Jammu and Kashmir forms the one-region group. The plot above is recalculated interactively with the selected settings.")
    with st.expander("How to read this chart"):
        st.write("PCA compresses the six indicators into two plotted dimensions. Each dot represents one region. Dots close together have more similar profiles in this two-dimensional view; separation on this chart does not prove that regions are identical or that one caused another's mobility changes.")

# =========================================================
# COMPARE REGIONS
# =========================================================
elif page == "04  Compare regions":
    heading("03 · LOOK CLOSER", "Put two regions side by side", "Choose any two regions and compare their average change across the six indicators.")
    left,right = st.columns(2)
    options = regional.index.tolist()
    with left: r1 = st.selectbox("Region A",options,index=options.index("Punjab") if "Punjab" in options else 0)
    with right: r2 = st.selectbox("Region B",options,index=options.index("Delhi") if "Delhi" in options else min(1,len(options)-1))
    compare = pd.DataFrame({"Indicator":[friendly[x] for x in regional.columns],r1:regional.loc[r1].values,r2:regional.loc[r2].values})
    melted = compare.melt("Indicator",var_name="Region",value_name="Average change (%)")
    fig = px.bar(melted,x="Indicator",y="Average change (%)",color="Region",barmode="group",title="Average mobility change by category",color_discrete_sequence=["#4B9BFF","#FF3355"])
    fig.update_xaxes(tickangle=-20)
    st.plotly_chart(chart_theme(fig,440),use_container_width=True)
    st.dataframe(compare.set_index("Indicator").round(2),use_container_width=True)
    delta = regional.loc[r1]-regional.loc[r2]
    biggest = friendly[delta.abs().idxmax()]
    st.markdown(f'<div class="story-card"><div class="section-kicker">BIGGEST DIFFERENCE</div><h3>{biggest}</h3><p>The largest absolute gap between {r1} and {r2} appears in this indicator: {delta.abs().max():.2f} percentage points.</p></div>',unsafe_allow_html=True)
    st.caption("These are differences between regional averages in the selected study period, not a live comparison or a measure of individual behaviour.")

# =========================================================
# RELIABILITY
# =========================================================
elif page == "05  Check the evidence":
    heading("04 · CHECK THE EVIDENCE", "How convincing are the groups?", "No single score can prove a clustering result is correct. We look at several checks together.")
    if scores is not None:
        scores = scores.copy()
        st.markdown("#### Try a cluster count")
        if "Clusters" in scores.columns:
            kpick = st.select_slider("Number of groups tested",options=sorted(scores["Clusters"].astype(int).unique()),value=3)
            row = scores[scores["Clusters"].astype(int)==kpick].iloc[0]
            a,b,c = st.columns(3)
            with a: metric_box("Silhouette",f"{row['Silhouette']:.3f}","Higher is generally better")
            with b: metric_box("Davies–Bouldin",f"{row['Davies_Bouldin']:.3f}","Lower is generally better")
            with c: metric_box("Within-group spread",f"{row['WCSS']:.2f}","Lower with more groups")
        col1,col2 = st.columns(2)
        with col1:
            fig=px.line(scores,x="Clusters",y="WCSS",markers=True,title="Elbow view · within-group spread")
            st.plotly_chart(chart_theme(fig,350),use_container_width=True)
        with col2:
            fig=px.line(scores,x="Clusters",y="Silhouette",markers=True,title="Silhouette · separation and cohesion")
            st.plotly_chart(chart_theme(fig,350),use_container_width=True)
    st.markdown("#### What did the research checks tell us?")
    st.markdown('<div class="story-card"><h3>🔁 Repeatability across random starts</h3><p>For StandardScaler K-Means, the tested solutions had ARI stability of 1.0 across the tested seeds. This means the same partition was recovered in those runs; it does not prove that the groups are objectively true.</p></div>',unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="story-card"><h3>🧭 PCA gives a compact view</h3><p>The first two principal components explain 98.14% of the variance in the standardized regional features. This makes a two-dimensional overview useful, while still being a compressed representation.</p></div>',unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="story-card"><h3>⚠️ One unusual observation matters</h3><p>Jammu and Kashmir has much larger average values than the other regions in the analysed features and forms a singleton cluster in the full-data three-group result. The sensitivity analysis checks how the picture changes when it is excluded; the full-data result still retains it.</p></div>',unsafe_allow_html=True)
    with st.expander("Technical detail: PCA loadings and evaluation tables"):
        if selected_metrics is not None: st.dataframe(selected_metrics.round(4),use_container_width=True)
        st.caption("PCA: PC1 81.76%, PC2 16.37%; combined 98.14%. PC1 broadly reflects non-residential mobility indicators; PC2 is strongly associated with residential mobility.")

# =========================================================
# RESEARCH LAB
# =========================================================
elif page == "08  Research notes":
    heading("05 · FOR THE CURIOUS", "Research lab & methodology", "The technical details are here for readers who want to inspect how the analysis was conducted.")
    with st.expander("Dataset and preprocessing",expanded=True):
        st.write("Google COVID-19 Community Mobility Reports for India, January 1 to October 15, 2022. The analysis aggregates daily observations into one average profile for each of 36 regions using six mobility indicators. Missing values were treated during preprocessing and the features were scaled before clustering.")
    with st.expander("Main clustering result"):
        st.write("The provisional primary descriptive solution uses StandardScaler and K-Means with k=3 across all 36 regions. Cluster sizes: 17, 1 and 18. Silhouette: 0.4334. Davies–Bouldin: 0.5349. The singleton group is Jammu and Kashmir.")
        if full_clusters is not None:
            st.dataframe(full_clusters,use_container_width=True)
            st.download_button("Download full-data cluster assignments",full_clusters.to_csv().encode("utf-8"),"final_full_dataset_clusters.csv","text/csv")
    with st.expander("Sensitivity experiment"):
        st.write("A separate analysis excludes Jammu and Kashmir to check its influence; it does not permanently remove the region from the primary dataset. The 35-region k=2 result has cluster sizes 17 and 18, silhouette 0.3452 and Davies–Bouldin 1.0686.")
        if sensitivity_clusters is not None:
            st.dataframe(sensitivity_clusters,use_container_width=True)
            st.download_button("Download sensitivity assignments",sensitivity_clusters.to_csv().encode("utf-8"),"final_sensitivity_clusters.csv","text/csv")
    with st.expander("Euclidean vs Manhattan distance"):
        st.write("The Manhattan comparison uses a custom K-Medians approach. Standard K-Means minimizes squared Euclidean distances; K-Medians is a more appropriate counterpart for Manhattan distance. Silhouette scores are calculated with their corresponding distance metrics. Euclidean WCSS and Manhattan objective values should not be compared as if they were the same quantity.")
        if manhattan is not None:
            st.dataframe(manhattan.round(4),use_container_width=True)
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=manhattan["K"],y=manhattan["Euclidean_Silhouette"],mode="lines+markers",name="Euclidean K-Means"))
            fig.add_trace(go.Scatter(x=manhattan["K"],y=manhattan["Manhattan_Silhouette"],mode="lines+markers",name="Manhattan K-Medians"))
            st.plotly_chart(chart_theme(fig,370),use_container_width=True)
    with st.expander("Limitations, responsible use and interpretation"):
        st.markdown("""
        - Google mobility data is aggregated and based on opted-in location data; coverage may be uneven.
        - Percentage changes depend on the baseline period and are not absolute counts of people or trips.
        - The final clustering uses 36 regional averages, so daily variation within each region is not represented in the final profiles.
        - The analysis is exploratory and does not establish causation.
        - Jammu and Kashmir has an unusual profile that strongly influences the clustering; it is retained in the main result and examined separately in sensitivity analysis.
        - October 2022 contains data only through October 15.
        - Clusters should not be used to rank regions or make consequential decisions about residents.
        """)
    st.markdown("#### Download the regional dataset")
    st.download_button("Download regional mobility CSV",regional.to_csv().encode("utf-8"),"regional_mobility.csv","text/csv")

elif page == "06  Sensitivity check":
    st.markdown('<div class="kicker">RESEARCH CHECK / SENSITIVITY</div>',unsafe_allow_html=True)
    st.markdown("## What changes when we examine the unusual profile?")
    st.write("Jammu and Kashmir is retained in the main result. This separate experiment checks how clustering changes when it is excluded.")
    if full_clusters is not None and sensitivity_clusters is not None:
        left,right=st.columns(2)
        with left:
            st.markdown("### All 36 regions")
            st.dataframe(full_clusters["Cluster"].value_counts().rename("Regions"),use_container_width=True)
        with right:
            st.markdown("### Without Jammu and Kashmir")
            st.dataframe(sensitivity_clusters["Cluster"].value_counts().rename("Regions"),use_container_width=True)
        comparison=pd.DataFrame({"Experiment":["All 36 regions","Without Jammu and Kashmir"],"Regions":[36,35],"K":[3,2],"Silhouette":[0.4334,0.3452],"Davies-Bouldin":[0.5349,1.0686]})
        st.dataframe(comparison,use_container_width=True)
    else:
        st.warning("Final cluster files are missing. Run final_results.py first.")

elif page == "07  Distance methods":
    st.markdown('<div class="kicker">METHODS / DISTANCE COMPARISON</div>',unsafe_allow_html=True)
    st.markdown("## Two ways to measure similarity")
    if manhattan is not None:
        st.dataframe(manhattan.round(4),use_container_width=True)
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=manhattan["K"],y=manhattan["Euclidean_Silhouette"],mode="lines+markers",name="Euclidean K-Means"))
        fig.add_trace(go.Scatter(x=manhattan["K"],y=manhattan["Manhattan_Silhouette"],mode="lines+markers",name="Manhattan K-Medians"))
        st.plotly_chart(chart_theme(fig,370),use_container_width=True)
        st.caption("Each silhouette score uses its corresponding distance metric. Euclidean WCSS and Manhattan objective values are not directly comparable.")
    else:
        st.warning("manhattan_comparison.csv is missing.")

st.markdown("---")
st.markdown("<p style='text-align:center;color:#7f91aa;font-size:12px'>MOBILITY STORIES · Regional Mobility-Change Pattern Segmentation · India · 2022</p>",unsafe_allow_html=True)
