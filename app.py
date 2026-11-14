import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Corruption Free Pakistan", page_icon="🇵🇰", layout="wide")

REGIONS = ["Punjab", "Sindh", "Khyber Pakhtunkhwa", "Balochistan",
           "Islamabad Capital Territory", "Gilgit-Baltistan", "Azad Jammu and Kashmir"]
DEPARTMENTS = ["Health", "Education", "Roads and Infrastructure", "Water", "Energy", "Social Protection"]
DISTRICTS = {
    "Punjab": ["Lahore", "Multan", "Bahawalpur", "Faisalabad", "Rawalpindi"],
    "Sindh": ["Karachi", "Hyderabad", "Sukkur", "Larkana"],
    "Khyber Pakhtunkhwa": ["Peshawar", "Swat", "Abbottabad", "Mardan"],
    "Balochistan": ["Quetta", "Gwadar", "Turbat", "Khuzdar"],
    "Islamabad Capital Territory": ["Islamabad"],
    "Gilgit-Baltistan": ["Gilgit", "Skardu", "Hunza"],
    "Azad Jammu and Kashmir": ["Muzaffarabad", "Mirpur", "Kotli"],
}


@st.cache_data
def make_demo_data(n=1500, seed=42):
    """Synthetic DEMO data with about 9 percent corruption-like rows injected."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        region = str(rng.choice(REGIONS, p=[0.30, 0.20, 0.17, 0.10, 0.08, 0.07, 0.08]))
        district = str(rng.choice(DISTRICTS[region]))
        dept = str(rng.choice(DEPARTMENTS))
        year = int(rng.choice([2021, 2022, 2023, 2024, 2025]))
        allocated = float(rng.lognormal(mean=18.5, sigma=0.8))
        spent = allocated * rng.uniform(0.80, 1.00)
        released = allocated * rng.uniform(0.90, 1.00)
        planned = int(rng.integers(120, 720))
        actual = int(planned * rng.uniform(0.95, 1.25))
        completion = float(rng.uniform(85, 100))
        bidders = int(rng.integers(3, 9))
        bid_type = str(rng.choice(["open", "limited"], p=[0.8, 0.2]))
        audit = int(rng.poisson(0.6))
        contractor = f"C{int(rng.integers(1, 120)):03d}"
        if rng.random() < 0.09:  # suspicious pattern inject karo
            spent = allocated * rng.uniform(1.10, 1.60)
            completion = float(rng.uniform(10, 55))
            bidders = 1
            bid_type = str(rng.choice(["single source", "limited"]))
            actual = int(planned * rng.uniform(1.5, 2.5))
            audit = int(rng.integers(3, 8))
            contractor = f"C{int(rng.integers(1, 6)):03d}"
        date_released = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=int(rng.integers(0, 365)))
        rows.append([f"P{i:05d}", region, district, dept, year, round(allocated), round(released),
                     round(spent), contractor, bidders, bid_type, planned, actual,
                     round(completion, 1), audit, date_released])
    cols = ["project_id", "region", "district", "department", "fiscal_year", "allocated_amount_pkr",
            "released_amount_pkr", "spent_amount_pkr", "contractor_id", "bidders_count", "bid_type",
            "planned_duration_days", "actual_duration_days", "completion_percent",
            "audit_objections_count", "date_released"]
    return pd.DataFrame(rows, columns=cols)


def score_risk(df):
    """Rules + Isolation Forest se risk score 0-100, level aur wajah."""
    d = df.copy()
    d["spend_ratio"] = d["spent_amount_pkr"] / d["allocated_amount_pkr"]
    d["cost_per_completion"] = d["spent_amount_pkr"] / d["completion_percent"].clip(lower=1)
    d["delay_ratio"] = d["actual_duration_days"] / d["planned_duration_days"]
    share = d.groupby(["district", "contractor_id"]).size() / d.groupby("district")["project_id"].count()
    d["contractor_share"] = [share[(a, b)] for a, b in zip(d["district"], d["contractor_id"])]

    flags = {
        "Spent more than allocated": d["spend_ratio"] > 1.05,
        "Large release but low completion": (d["released_amount_pkr"] > 0.8 * d["allocated_amount_pkr"]) & (d["completion_percent"] < 60),
        "Single bidder or single source contract": (d["bidders_count"] <= 1) | (d["bid_type"] == "single source"),
        "One contractor dominates the district": d["contractor_share"] > 0.25,
        "Big delay against plan": d["delay_ratio"] > 1.5,
        "Many audit objections": d["audit_objections_count"] >= 3,
        "Suspiciously round amount": d["allocated_amount_pkr"] % 1_000_000 == 0,
    }
    weights = [20, 20, 15, 10, 10, 15, 5]
    rule_score = sum(w * f.astype(int) for w, f in zip(weights, flags.values()))

    feats = d[["spend_ratio", "cost_per_completion", "delay_ratio", "contractor_share"]]
    iso = IsolationForest(n_estimators=200, contamination=0.1, random_state=42).fit(feats)
    raw = -iso.score_samples(feats)
    anomaly = 100 * (raw - raw.min()) / (raw.max() - raw.min())

    d["risk_score"] = (0.7 * rule_score + 0.3 * anomaly).clip(0, 100).round(1)
    d["risk_level"] = pd.cut(d["risk_score"], [-1, 25, 50, 75, 101],
                             labels=["Low", "Medium", "High", "Critical"])
    names = list(flags.keys())
    matrix = np.column_stack([f.values for f in flags.values()])
    d["explanation"] = ["; ".join(n for n, hit in zip(names, row) if hit) or "No major red flags"
                        for row in matrix]
    return d


def benford_table(values):
    first = values.astype(str).str.lstrip("0").str[0]
    first = first[first.isin(list("123456789"))].astype(int)
    actual = first.value_counts(normalize=True).reindex(range(1, 10), fill_value=0)
    expected = pd.Series({k: np.log10(1 + 1 / k) for k in range(1, 10)})
    return pd.DataFrame({"digit": range(1, 10), "Expected": expected.values, "Actual": actual.values})


# ---------------- HEADER ----------------
st.title("🇵🇰 Corruption Free Pakistan")
st.caption("AI-powered risk dashboard for government fund allocation and spending")
st.warning("This tool shows risk indicators, not proof of corruption. Every flagged item needs human investigation.")

# ---------------- DATA ----------------
uploaded = st.sidebar.file_uploader("Upload your own CSV (optional)", type="csv")
if uploaded:
    raw = pd.read_csv(uploaded, parse_dates=["date_released"])
    st.sidebar.success("Using your uploaded data")
else:
    raw = make_demo_data()
    st.error("DEMO DATA: synthetic records for demonstration only. Real sources: PPRA tenders, Auditor General reports, provincial budgets.")

data = score_risk(raw)

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.header("Filters")
region_choice = st.sidebar.selectbox("Province / Region", ["All Pakistan"] + REGIONS)
years = sorted(data["fiscal_year"].unique())
year_choice = st.sidebar.multiselect("Fiscal year", years, default=years)
all_depts = sorted(data["department"].unique())
dept_choice = st.sidebar.multiselect("Department", all_depts, default=all_depts)

view = data[data["fiscal_year"].isin(year_choice) & data["department"].isin(dept_choice)]
if region_choice != "All Pakistan":
    view = view[view["region"] == region_choice]
flagged = view[view["risk_level"].isin(["High", "Critical"])]

if view.empty:
    st.info("No data for these filters.")
    st.stop()

tabs = st.tabs(["Overview", "Map", "Region comparison", "Contractor watch",
                "Project explorer", "Benford analysis", "AI assistant"])

# ---------------- OVERVIEW ----------------
with tabs[0]:
    st.subheader(f"Overview: {region_choice}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total allocated (PKR billion)", f"{view['allocated_amount_pkr'].sum() / 1e9:,.1f}")
    c2.metric("Funds at risk (PKR billion)", f"{flagged['spent_amount_pkr'].sum() / 1e9:,.1f}")
    c3.metric("Flagged projects", f"{len(flagged)} of {len(view)}")
    c4.metric("Highest-risk region", view.groupby("region")["risk_score"].mean().idxmax())
    order = ["Low", "Medium", "High", "Critical"]
    counts = [int((view["risk_level"] == lv).sum()) for lv in order]
    lvl = pd.DataFrame({"Risk level": order, "Projects": counts})
    st.plotly_chart(px.bar(lvl, x="Risk level", y="Projects", color="Risk level",
                           color_discrete_map={"Low": "green", "Medium": "gold", "High": "orange", "Critical": "red"}),
                    use_container_width=True)
    if region_choice != "All Pakistan":
        st.markdown(f"**Districts in {region_choice}**")
        dist = view.groupby("district").agg(projects=("project_id", "count"),
                                            avg_risk=("risk_score", "mean"),
                                            spent_pkr=("spent_amount_pkr", "sum")).round(1).reset_index()
        st.dataframe(dist.sort_values("avg_risk", ascending=False), use_container_width=True)

# ---------------- MAP ----------------
with tabs[1]:
    st.subheader("Average risk by region")
    reg = data.groupby("region")["risk_score"].mean().round(1).reset_index()
    fig = px.bar(reg.sort_values("risk_score"), x="risk_score", y="region", orientation="h",
                 color="risk_score", color_continuous_scale="RdYlGn_r")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Kisi region ki detail dekhne ke liye sidebar mein Province / Region chuniye.")

# ---------------- REGION COMPARISON ----------------
with tabs[2]:
    st.subheader("Allocated vs spent vs risk")
    comp = data.groupby("region").agg(allocated=("allocated_amount_pkr", "sum"),
                                      spent=("spent_amount_pkr", "sum"),
                                      risk=("risk_score", "mean")).reset_index()
    fig = go.Figure()
    fig.add_bar(x=comp["region"], y=comp["allocated"] / 1e9, name="Allocated (PKR bn)")
    fig.add_bar(x=comp["region"], y=comp["spent"] / 1e9, name="Spent (PKR bn)")
    fig.add_scatter(x=comp["region"], y=comp["risk"], name="Avg risk score", yaxis="y2", mode="lines+markers")
    fig.update_layout(barmode="group", yaxis2=dict(overlaying="y", side="right", title="Risk"))
    st.plotly_chart(fig, use_container_width=True)

# ---------------- CONTRACTOR WATCH ----------------
with tabs[3]:
    st.subheader("Contractors with the highest risk")
    con = view.groupby("contractor_id").agg(contracts=("project_id", "count"),
                                            avg_risk=("risk_score", "mean"),
                                            value_pkr=("spent_amount_pkr", "sum")).reset_index()
    con["share_of_contracts_pct"] = (100 * con["contracts"] / con["contracts"].sum()).round(2)
    con = con[con["contracts"] >= 3].sort_values("avg_risk", ascending=False).head(15)
    if con.empty:
        st.info("Is filter mein kaafi contractors nahi hain.")
    else:
        st.plotly_chart(px.bar(con, x="contractor_id", y="avg_risk", color="share_of_contracts_pct"),
                        use_container_width=True)
        st.dataframe(con.round(1), use_container_width=True)

# ---------------- PROJECT EXPLORER ----------------
with tabs[4]:
    st.subheader("Flagged projects")
    levels = st.multiselect("Risk levels", ["Medium", "High", "Critical"], default=["High", "Critical"])
    shown = view[view["risk_level"].isin(levels)].sort_values("risk_score", ascending=False)
    cols = ["project_id", "region", "district", "department", "fiscal_year", "contractor_id",
            "allocated_amount_pkr", "spent_amount_pkr", "completion_percent", "risk_score",
            "risk_level", "explanation"]
    st.dataframe(shown[cols], use_container_width=True)
    st.download_button("Download flagged list (CSV)", shown[cols].to_csv(index=False).encode("utf-8"),
                       "flagged_projects.csv", "text/csv")

# ---------------- BENFORD ----------------
with tabs[5]:
    st.subheader("Benford's Law check on spent amounts")
    bf = benford_table(view["spent_amount_pkr"])
    fig = go.Figure()
    fig.add_bar(x=bf["digit"], y=bf["Expected"], name="Expected")
    fig.add_bar(x=bf["digit"], y=bf["Actual"], name="Actual")
    fig.update_layout(barmode="group", xaxis_title="First digit", yaxis_title="Share")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Expected aur actual digits mein bada farq numbers mein hera-pheri ka ishara ho sakta hai.")

# ---------------- AI ASSISTANT (optional, Gemini) ----------------
with tabs[6]:
    st.subheader("AI assistant (optional)")
    if flagged.empty:
        st.info("Koi flagged project nahi hai.")
    else:
        ranked = flagged.sort_values("risk_score", ascending=False)
        pid = st.selectbox("Choose a flagged project", ranked["project_id"])
        row = ranked[ranked["project_id"] == pid].iloc[0]
        st.write(row[["region", "district", "department", "risk_score", "explanation"]])
        st.caption("Is button ke liye requirements.txt mein google-generativeai aur Streamlit Secrets mein GEMINI_API_KEY chahiye.")
        if st.button("Explain in Urdu and English"):
            try:
                import google.generativeai as genai
                genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                model = genai.GenerativeModel("gemini-1.5-flash")
                prompt = ("Summarize this government project's corruption risk in 3 simple sentences in Urdu, "
                          f"then in English. Say it is a risk indicator, not proof. Details: {row.to_dict()}")
                st.write(model.generate_content(prompt).text)
            except Exception as e:
                st.error(f"Gemini setup nahi hai ya error aaya: {e}")
