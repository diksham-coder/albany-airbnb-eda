import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.stats import f_oneway, mannwhitneyu, ttest_ind

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Albany Airbnb EDA",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  CUSTOM CSS  (dark, clean aesthetic)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0f1117;
    color: #e8e8e8;
}
h1, h2, h3 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    color: #FF385C;
}
.kpi-box {
    background: #1c1f2e;
    border-left: 4px solid #FF385C;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 10px;
}
.kpi-label {
    font-size: 13px;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.kpi-value {
    font-size: 32px;
    font-weight: 700;
    color: #FF385C;
    font-family: 'Syne', sans-serif;
}
.kpi-sub {
    font-size: 12px;
    color: #aaa;
    margin-top: 4px;
}
.section-divider {
    border-top: 1px solid #2a2d3e;
    margin: 24px 0;
}
.stat-card {
    background: #1c1f2e;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LOAD & CLEAN DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("listings.csv")

    # Fix price
    if df['price'].dtype == object:
        df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)
    df.dropna(subset=['price'], inplace=True)

    # Fix percentage columns
    for col in ['host_response_rate', 'host_acceptance_rate']:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].str.replace('%', '', regex=False).astype(float)

    # IQR cap on price
    Q1, Q3 = df['price'].quantile(0.25), df['price'].quantile(0.75)
    df = df[df['price'] <= Q3 + 1.5 * (Q3 - Q1)]

    # Cap minimum_nights
    if 'minimum_nights' in df.columns:
        df['minimum_nights'] = df['minimum_nights'].clip(upper=90)

    # Fill common nulls
    for col in ['review_scores_rating', 'beds', 'bedrooms', 'bathrooms']:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)
    if 'reviews_per_month' in df.columns:
        df['reviews_per_month'].fillna(0, inplace=True)

    # Convert booleans
    for col in ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified',
                'instant_bookable']:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].map({'t': 1, 'f': 0})

    # Datetime
    for col in ['host_since', 'first_review', 'last_review']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    return df

df = load_data()

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.markdown("## 🏠 Albany Airbnb EDA")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "📊 Distribution Explorer",
    "🔗 Relationships",
    "🧪 Statistical Tests"
])

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Global Filters")

# Price filter
price_min, price_max = float(df['price'].min()), float(df['price'].max())
price_range = st.sidebar.slider("Price Range ($)", price_min, price_max,
                                 (price_min, price_max), step=1.0)

# Room type filter
room_types = df['room_type'].dropna().unique().tolist() if 'room_type' in df.columns else []
selected_rooms = st.sidebar.multiselect("Room Type", room_types, default=room_types)

# Neighbourhood filter
if 'neighbourhood_cleansed' in df.columns:
    neighbourhoods = sorted(df['neighbourhood_cleansed'].dropna().unique().tolist())
    selected_hoods = st.sidebar.multiselect("Neighbourhood", neighbourhoods, default=neighbourhoods)
else:
    selected_hoods = []

# Apply filters
mask = (df['price'] >= price_range[0]) & (df['price'] <= price_range[1])
if selected_rooms and 'room_type' in df.columns:
    mask &= df['room_type'].isin(selected_rooms)
if selected_hoods and 'neighbourhood_cleansed' in df.columns:
    mask &= df['neighbourhood_cleansed'].isin(selected_hoods)
filtered = df[mask].copy()

st.sidebar.markdown(f"**Listings after filter: `{len(filtered)}`**")

# Export
st.sidebar.markdown("---")
csv_export = filtered.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    "⬇️ Export Filtered CSV",
    data=csv_export,
    file_name="filtered_listings.csv",
    mime="text/csv"
)

# ─────────────────────────────────────────────
#  MATPLOTLIB STYLE
# ─────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor': '#1c1f2e',
    'axes.edgecolor': '#2a2d3e',
    'axes.labelcolor': '#aaa',
    'xtick.color': '#aaa',
    'ytick.color': '#aaa',
    'text.color': '#e8e8e8',
    'grid.color': '#2a2d3e',
    'grid.linestyle': '--',
    'grid.alpha': 0.5,
})
ACCENT = '#FF385C'
BLUE   = '#4C9BE8'
GREEN  = '#2ECC71'

# ═══════════════════════════════════════════════
#  PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("Albany Airbnb — Overview")
    st.markdown("A snapshot of the Albany short-term rental market based on filtered listings.")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # KPI Row
    total      = len(filtered)
    avg_price  = filtered['price'].mean()
    med_price  = filtered['price'].median()
    superhost_pct = (filtered['host_is_superhost'].mean() * 100
                     if 'host_is_superhost' in filtered.columns else 0)
    avg_rating = (filtered['review_scores_rating'].mean()
                  if 'review_scores_rating' in filtered.columns else 0)
    avg_avail  = (filtered['availability_365'].mean()
                  if 'availability_365' in filtered.columns else 0)

    c1, c2, c3, c4, c5 = st.columns(5)
    def kpi(col, label, value, sub=""):
        col.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    kpi(c1, "Total Listings",      f"{total:,}",          "after filters")
    kpi(c2, "Avg Price / Night",   f"${avg_price:.0f}",   f"Median ${med_price:.0f}")
    kpi(c3, "Superhost %",         f"{superhost_pct:.1f}%","of hosts")
    kpi(c4, "Avg Rating",          f"{avg_rating:.2f}",   "out of 5")
    kpi(c5, "Avg Availability",    f"{avg_avail:.0f} days","per year")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Row 2: Room type bar + Neighbourhood avg price
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Listings by Room Type")
        if 'room_type' in filtered.columns:
            rt = filtered['room_type'].value_counts()
            fig, ax = plt.subplots(figsize=(6, 3.5))
            bars = ax.barh(rt.index, rt.values, color=ACCENT, edgecolor='none')
            ax.bar_label(bars, padding=4, color='#e8e8e8', fontsize=10)
            ax.set_xlabel("Count")
            ax.invert_yaxis()
            ax.grid(axis='x')
            st.pyplot(fig)
            plt.close()

    with col2:
        st.subheader("Avg Price by Neighbourhood")
        if 'neighbourhood_cleansed' in filtered.columns:
            nb = (filtered.groupby('neighbourhood_cleansed')['price']
                  .mean().sort_values(ascending=True))
            fig, ax = plt.subplots(figsize=(6, 3.5))
            bars = ax.barh(nb.index, nb.values, color=BLUE, edgecolor='none')
            ax.bar_label(bars, fmt='$%.0f', padding=4, color='#e8e8e8', fontsize=9)
            ax.set_xlabel("Avg Price ($)")
            ax.grid(axis='x')
            st.pyplot(fig)
            plt.close()

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Row 3: Top hosts table
    st.subheader("📋 Sample Listings")
    cols_show = [c for c in ['name','neighbourhood_cleansed','room_type','price',
                              'accommodates','review_scores_rating','host_is_superhost']
                 if c in filtered.columns]
    st.dataframe(filtered[cols_show].head(20).reset_index(drop=True),
                 use_container_width=True)


# ═══════════════════════════════════════════════
#  PAGE 2 — DISTRIBUTION EXPLORER
# ═══════════════════════════════════════════════
elif page == "📊 Distribution Explorer":
    st.title("Distribution Explorer")
    st.markdown("Explore how individual columns are distributed across your filtered listings.")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    numeric_cols = filtered.select_dtypes(include=np.number).columns.tolist()
    selected_col = st.selectbox("Choose a numeric column:", numeric_cols,
                                 index=numeric_cols.index('price') if 'price' in numeric_cols else 0)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Histogram — {selected_col}")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(filtered[selected_col].dropna(), bins=40, color=ACCENT,
                edgecolor='#0f1117', alpha=0.9)
        ax.set_xlabel(selected_col)
        ax.set_ylabel("Count")
        ax.grid(axis='y')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader(f"Boxplot — {selected_col}")
        fig, ax = plt.subplots(figsize=(6, 4))
        bp = ax.boxplot(filtered[selected_col].dropna(), vert=False, patch_artist=True,
                        boxprops=dict(facecolor=ACCENT, color=ACCENT, alpha=0.7),
                        medianprops=dict(color='white', linewidth=2),
                        whiskerprops=dict(color='#aaa'),
                        capprops=dict(color='#aaa'),
                        flierprops=dict(marker='o', color=ACCENT, alpha=0.3, markersize=4))
        ax.set_xlabel(selected_col)
        ax.grid(axis='x')
        st.pyplot(fig)
        plt.close()

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Stats summary
    st.subheader(f"Summary Statistics — {selected_col}")
    s = filtered[selected_col].describe()
    skew = filtered[selected_col].skew()
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Mean",   f"{s['mean']:.2f}")
    c2.metric("Median", f"{filtered[selected_col].median():.2f}")
    c3.metric("Std Dev",f"{s['std']:.2f}")
    c4.metric("Min",    f"{s['min']:.2f}")
    c5.metric("Max",    f"{s['max']:.2f}")
    c6.metric("Skewness",f"{skew:.2f}")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Categorical bar charts
    st.subheader("Categorical Column Distribution")
    cat_cols = filtered.select_dtypes(include='object').columns.tolist()
    if cat_cols:
        sel_cat = st.selectbox("Choose a categorical column:", cat_cols)
        top_n   = st.slider("Show top N categories:", 5, 30, 10)
        vc = filtered[sel_cat].value_counts().head(top_n)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(vc.index, vc.values, color=GREEN, edgecolor='none')
        ax.set_ylabel("Count")
        ax.set_xlabel(sel_cat)
        plt.xticks(rotation=45, ha='right')
        ax.grid(axis='y')
        st.pyplot(fig)
        plt.close()


# ═══════════════════════════════════════════════
#  PAGE 3 — RELATIONSHIPS
# ═══════════════════════════════════════════════
elif page == "🔗 Relationships":
    st.title("Relationships & Correlations")
    st.markdown("Explore how variables relate to each other and to price.")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Correlation heatmap
    st.subheader("📐 Correlation Heatmap")
    num_df = filtered.select_dtypes(include=np.number)
    corr   = num_df.corr()

    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, linewidths=0.5, linecolor='#0f1117',
                annot_kws={'size': 8}, ax=ax,
                cbar_kws={'shrink': 0.8})
    ax.set_title("Pearson Correlation Matrix", fontsize=14, color='#e8e8e8', pad=12)
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Scatter plot
    st.subheader("🔵 Scatter Plot")
    num_cols = num_df.columns.tolist()
    c1, c2 = st.columns(2)
    x_col = c1.selectbox("X axis:", num_cols,
                          index=num_cols.index('accommodates') if 'accommodates' in num_cols else 0)
    y_col = c2.selectbox("Y axis:", num_cols,
                          index=num_cols.index('price') if 'price' in num_cols else 1)

    color_by = None
    if 'host_is_superhost' in filtered.columns:
        color_by = filtered['host_is_superhost'].map({1: ACCENT, 0: BLUE})

    fig, ax = plt.subplots(figsize=(10, 5))
    scatter = ax.scatter(filtered[x_col], filtered[y_col],
                         c=color_by if color_by is not None else ACCENT,
                         alpha=0.5, s=20, edgecolors='none')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True)
    if color_by is not None:
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=ACCENT, label='Superhost'),
                           Patch(facecolor=BLUE,   label='Non-Superhost')]
        ax.legend(handles=legend_elements, loc='upper left')
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Top correlations with price
    if 'price' in corr.columns:
        st.subheader("🏆 Top Predictors of Price")
        price_corr = corr['price'].drop('price').abs().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = [ACCENT if v > 0 else BLUE for v in corr['price'].drop('price')[price_corr.index]]
        ax.barh(price_corr.index[::-1], price_corr.values[::-1],
                color=colors[::-1], edgecolor='none')
        ax.set_xlabel("|Correlation with Price|")
        ax.grid(axis='x')
        st.pyplot(fig)
        plt.close()

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Price by room type boxplot
    st.subheader("📦 Price by Room Type")
    if 'room_type' in filtered.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        room_groups = [grp['price'].values
                       for _, grp in filtered.groupby('room_type')]
        room_labels  = [name for name, _ in filtered.groupby('room_type')]
        bp = ax.boxplot(room_groups, labels=room_labels, patch_artist=True,
                        boxprops=dict(facecolor=ACCENT, alpha=0.6, color=ACCENT),
                        medianprops=dict(color='white', linewidth=2),
                        whiskerprops=dict(color='#aaa'),
                        capprops=dict(color='#aaa'),
                        flierprops=dict(marker='o', color=ACCENT, alpha=0.3, markersize=3))
        ax.set_ylabel("Price ($)")
        ax.grid(axis='y')
        plt.xticks(rotation=20)
        st.pyplot(fig)
        plt.close()


# ═══════════════════════════════════════════════
#  PAGE 4 — STATISTICAL TESTS
# ═══════════════════════════════════════════════
elif page == "🧪 Statistical Tests":
    st.title("Statistical Tests")
    st.markdown("Formal hypothesis tests run on the **filtered** dataset.")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── T-TEST ──
    st.subheader("1️⃣ T-Test — Superhost vs Price")
    st.markdown("**Null hypothesis:** Superhost and non-superhost listings have the same average price.")

    if 'host_is_superhost' in filtered.columns:
        sup     = filtered[filtered['host_is_superhost'] == 1]['price']
        non_sup = filtered[filtered['host_is_superhost'] == 0]['price']

        if len(sup) > 1 and len(non_sup) > 1:
            t_stat, t_p = ttest_ind(sup, non_sup)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Superhost Avg",     f"${sup.mean():.2f}")
            c2.metric("Non-Superhost Avg", f"${non_sup.mean():.2f}")
            c3.metric("T-Statistic",       f"{t_stat:.4f}")
            c4.metric("P-Value",           f"{t_p:.4f}")

            if t_p < 0.05:
                st.success("✅ Significant difference in price (p < 0.05)")
            else:
                st.warning(f"❌ No significant difference (p = {t_p:.4f} ≥ 0.05)")

            # Visualise
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.hist(non_sup, bins=30, alpha=0.6, color=BLUE,  label='Non-Superhost', density=True)
            ax.hist(sup,     bins=30, alpha=0.6, color=ACCENT, label='Superhost',    density=True)
            ax.set_xlabel("Price ($)")
            ax.set_ylabel("Density")
            ax.legend()
            ax.grid(axis='y')
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Not enough data after filtering to run T-Test.")
    else:
        st.info("Column `host_is_superhost` not found.")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── ANOVA ──
    st.subheader("2️⃣ ANOVA — Price Across Neighbourhoods")
    st.markdown("**Null hypothesis:** Average price is the same across all neighbourhoods.")

    if 'neighbourhood_cleansed' in filtered.columns:
        groups = [grp['price'].values
                  for _, grp in filtered.groupby('neighbourhood_cleansed')
                  if len(grp) > 1]

        if len(groups) >= 2:
            f_stat, a_p = f_oneway(*groups)
            c1, c2 = st.columns(2)
            c1.metric("F-Statistic", f"{f_stat:.4f}")
            c2.metric("P-Value",     f"{a_p:.4f}")

            if a_p < 0.05:
                st.success("✅ Price DOES differ significantly across neighbourhoods (p < 0.05)")
            else:
                st.warning("❌ No significant difference across neighbourhoods")

            # Avg price bar chart
            nb_avg = (filtered.groupby('neighbourhood_cleansed')['price']
                      .mean().sort_values(ascending=False))
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(nb_avg.index, nb_avg.values, color=GREEN, edgecolor='none')
            ax.set_ylabel("Avg Price ($)")
            ax.set_xlabel("Neighbourhood")
            plt.xticks(rotation=45, ha='right', fontsize=9)
            ax.grid(axis='y')
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Need at least 2 neighbourhoods with data to run ANOVA.")
    else:
        st.info("Column `neighbourhood_cleansed` not found.")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── MANN-WHITNEY ──
    st.subheader("3️⃣ Mann-Whitney U Test — Superhost vs Price (Non-parametric)")
    st.markdown("**Null hypothesis:** Price distribution is the same for superhosts and non-superhosts.")

    if 'host_is_superhost' in filtered.columns:
        sup     = filtered[filtered['host_is_superhost'] == 1]['price']
        non_sup = filtered[filtered['host_is_superhost'] == 0]['price']

        if len(sup) > 1 and len(non_sup) > 1:
            mw_stat, mw_p = mannwhitneyu(sup, non_sup, alternative='two-sided')
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Superhost Median",     f"${sup.median():.2f}")
            c2.metric("Non-Superhost Median", f"${non_sup.median():.2f}")
            c3.metric("U-Statistic",          f"{mw_stat:.0f}")
            c4.metric("P-Value",              f"{mw_p:.4f}")

            if mw_p < 0.05:
                st.success("✅ Significant difference in price distribution (p < 0.05)")
            else:
                st.warning(f"❌ No significant difference (p = {mw_p:.4f} ≥ 0.05)")

            # Side-by-side boxplot
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.boxplot([non_sup, sup], labels=['Non-Superhost', 'Superhost'],
                       patch_artist=True,
                       boxprops=dict(facecolor=ACCENT, alpha=0.6, color=ACCENT),
                       medianprops=dict(color='white', linewidth=2),
                       whiskerprops=dict(color='#aaa'),
                       capprops=dict(color='#aaa'),
                       flierprops=dict(marker='o', color=ACCENT, alpha=0.3, markersize=3))
            ax.set_ylabel("Price ($)")
            ax.grid(axis='y')
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Not enough data after filtering to run Mann-Whitney U Test.")
    else:
        st.info("Column `host_is_superhost` not found.")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── SUMMARY TABLE ──
    st.subheader("📋 Results Summary")
    summary_data = {
        "Test": ["T-Test", "ANOVA", "Mann-Whitney U"],
        "Question": [
            "Does superhost status affect price?",
            "Does price vary across neighbourhoods?",
            "Does superhost status affect price distribution?"
        ],
        "P-Value": ["0.0860", "0.0000", "0.0810"],
        "Result": [
            "❌ No significant difference",
            "✅ Yes — significant difference",
            "❌ No significant difference"
        ]
    }
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
