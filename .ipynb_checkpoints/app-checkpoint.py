# streamlit_app.py
"""
Streamlit dashboard for the ATOM CEO's Challenge FinAI agents.
Run: streamlit run streamlit_app.py
Requires: models/scaler.pkl, models/behavioral_risk_model.pkl,
          models/behavioral_risk_label_encoder.pkl, models/priority_model.pkl,
          models/health_score_model.pkl, models/lead_model.pkl
"""

import json
from dataclasses import dataclass, field
from datetime import date
from typing import List, Dict
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import numpy_financial as npf

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(page_title="ATOM FinAI Agents", layout="wide")

# ----------------------------------------------------------------------
# Load trained models (cached so they load once per session)
# ----------------------------------------------------------------------
@st.cache_resource
def load_models():
    scaler = joblib.load("models/scaler.pkl")
    behavior_model = joblib.load("models/behavioral_risk_model.pkl")
    le_behavior = joblib.load("models/behavioral_risk_label_encoder.pkl")
    priority_model = joblib.load("models/priority_model.pkl")
    health_model = joblib.load("models/health_score_model.pkl")
    lead_model = joblib.load("models/lead_model.pkl")
    return scaler, behavior_model, le_behavior, priority_model, health_model, lead_model

try:
    scaler, behavior_model, le_behavior, priority_model, health_model, lead_model = load_models()
except FileNotFoundError as e:
    st.error(
        "Model files not found. Make sure the `models/` folder "
        "(scaler.pkl, behavioral_risk_model.pkl, behavioral_risk_label_encoder.pkl, "
        "priority_model.pkl, health_score_model.pkl, lead_model.pkl) is in the same "
        f"directory as this script.\n\nDetails: {e}"
    )
    st.stop()

FEATURES = [
    "age", "income", "savings_rate", "n_holdings", "pnl_pct",
    "redemptions_90d", "sip_missed_3m", "sip_active", "idle_cash",
    "has_insurance", "has_elss", "has_debt", "large_cap_pct", "is_hni"
]

# ----------------------------------------------------------------------
# Data models
# ----------------------------------------------------------------------
@dataclass
class Holding:
    fund_name: str
    category: str
    invested_value: float
    current_value: float

@dataclass
class Transaction:
    txn_date: date
    txn_type: str
    fund_name: str
    amount: float

@dataclass
class Client:
    client_id: str
    name: str
    age: int
    monthly_income: float
    risk_profile: str
    goals: List[Dict]
    holdings: List[Holding] = field(default_factory=list)
    transactions: List[Transaction] = field(default_factory=list)
    bank_idle_cash: float = 0.0
    is_hni: bool = False
    monthly_spending: float = 0.0
    has_insurance: bool = False
    sip_active: bool = True
    sip_missed_count_3m: int = 0

def client_to_feature_vector(client: Client) -> np.ndarray:
    invested = sum(h.invested_value for h in client.holdings) or 0
    current = sum(h.current_value for h in client.holdings) or 0
    pnl_pct = (current - invested) / invested if invested else 0
    large_cap_val = sum(h.current_value for h in client.holdings if h.category == "Large Cap")
    large_cap_pct = large_cap_val / current if current else 0
    redemptions_90d = sum(1 for t in client.transactions if t.txn_type == "Redemption")
    has_elss = any(h.category == "ELSS" for h in client.holdings)
    has_debt = any(h.category == "Debt" for h in client.holdings)
    savings_rate = 1 - (client.monthly_spending / client.monthly_income) if client.monthly_income else 0

    raw = np.array([[
        client.age, client.monthly_income, savings_rate, len(client.holdings), pnl_pct,
        redemptions_90d, client.sip_missed_count_3m, int(client.sip_active), client.bank_idle_cash,
        int(client.has_insurance), int(has_elss), int(has_debt), large_cap_pct, int(client.is_hni)
    ]])
    return scaler.transform(raw)

# ----------------------------------------------------------------------
# Base agent + agents (same logic as the notebook)
# ----------------------------------------------------------------------
class BaseAgent(ABC):
    name: str = "base_agent"

    @abstractmethod
    def fetch_context(self, client) -> dict: ...
    @abstractmethod
    def score(self, context: dict) -> dict: ...
    @abstractmethod
    def explain(self, context: dict, score: dict) -> str: ...

    def run(self, client) -> dict:
        context = self.fetch_context(client)
        score = self.score(context)
        explanation = self.explain(context, score)
        return {"agent": self.name, "context": context, "score": score, "explanation": explanation}


class GoalPlannerAgent(BaseAgent):
    name = "customer_goal_planner"
    RETURN_ASSUMPTIONS = {"Equity": 0.12, "Debt": 0.07, "Hybrid": 0.09}

    def fetch_context(self, client) -> dict:
        goal = client.goals[0]
        return {"goal_name": goal["name"], "target_amount": goal["target_amount"],
                "years": goal["years"], "risk_profile": client.risk_profile}

    def _allocation(self, risk, years):
        if risk == "Aggressive" and years >= 7:
            return {"Equity": 0.75, "Debt": 0.15, "Hybrid": 0.10}
        if risk == "Moderate" or years < 7:
            return {"Equity": 0.55, "Debt": 0.35, "Hybrid": 0.10}
        return {"Equity": 0.35, "Debt": 0.55, "Hybrid": 0.10}

    def score(self, context: dict) -> dict:
        alloc = self._allocation(context["risk_profile"], context["years"])
        blended = sum(alloc[k] * self.RETURN_ASSUMPTIONS[k] for k in alloc)
        n_months = max(context["years"] * 12, 1)
        required_sip = -npf.pmt(blended / 12, n_months, 0, context["target_amount"])
        return {"suggested_allocation": alloc, "assumed_return_pct": round(blended * 100, 2),
                "required_monthly_sip": round(required_sip, 0)}

    def explain(self, context, score) -> str:
        alloc_str = ", ".join(f"{k}: {v*100:.0f}%" for k, v in score["suggested_allocation"].items())
        return (f"To reach your {context['goal_name']} goal of ₹{context['target_amount']:,} in "
                f"{context['years']} years, a suggested allocation is {alloc_str}, requiring "
                f"an estimated SIP of ₹{score['required_monthly_sip']:,.0f}/month "
                f"(illustrative {score['assumed_return_pct']}% return, not guaranteed).")


class BehavioralAIAgent(BaseAgent):
    name = "customer_behavioral_ai"

    def fetch_context(self, client) -> dict:
        return {"feature_vector": client_to_feature_vector(client)}

    def score(self, context: dict) -> dict:
        proba = behavior_model.predict_proba(context["feature_vector"])[0]
        pred_idx = proba.argmax()
        label = le_behavior.inverse_transform([pred_idx])[0]
        return {"severity": label, "confidence": round(float(proba[pred_idx]), 3),
                "class_probabilities": dict(zip(le_behavior.classes_, proba.round(3)))}

    def explain(self, context, score) -> str:
        templates = {
            "high": "Model flags HIGH behavioral risk (confidence {c}%) - possible panic-selling or SIP lapse pattern.",
            "medium": "Model flags MEDIUM behavioral risk (confidence {c}%) - worth a check-in.",
            "low": "Model shows LOW behavioral risk (confidence {c}%) - client on track.",
        }
        return templates[score["severity"]].format(c=round(score["confidence"] * 100, 1))


class HyperPersonalizedInsightsAgent(BaseAgent):
    name = "customer_insights"

    def fetch_context(self, client) -> dict:
        savings_rate = 1 - (client.monthly_spending / client.monthly_income) if client.monthly_income else 0
        mix = {}
        for h in client.holdings:
            mix[h.category] = mix.get(h.category, 0) + h.current_value
        return {"age": client.age, "savings_rate": round(savings_rate, 2),
                "category_mix": mix, "has_insurance": client.has_insurance}

    def score(self, context: dict) -> dict:
        insights = []
        if context["savings_rate"] < 0.2:
            insights.append("low_savings_rate")
        if "Debt" not in context["category_mix"]:
            insights.append("no_debt_allocation")
        if not context["has_insurance"]:
            insights.append("insurance_gap")
        if "ELSS" not in context["category_mix"] and context["age"] < 45:
            insights.append("elss_missing")
        return {"insight_codes": insights}

    def explain(self, context, score) -> str:
        MSG = {
            "low_savings_rate": "Your savings rate is below 20% of income - a budget review could help.",
            "no_debt_allocation": "No debt fund allocation - portfolio may be more volatile than needed.",
            "insurance_gap": "No insurance on record - consider closing this protection gap.",
            "elss_missing": "No ELSS holding - you may be missing a tax-saving opportunity under 80C.",
        }
        if not score["insight_codes"]:
            return "Your portfolio and financial habits look on track - no gaps flagged."
        return " ".join(MSG[c] for c in score["insight_codes"])


class ClientPrioritizationAgent(BaseAgent):
    name = "distributor_client_prioritization"

    def fetch_context(self, client): raise NotImplementedError("use run_batch")
    def score(self, context): pass
    def explain(self, context, score): pass

    def run_batch(self, clients) -> dict:
        rows = []
        for c in clients:
            fv = client_to_feature_vector(c)
            proba = priority_model.predict_proba(fv)[0][1]
            rows.append({"client_id": c.client_id, "name": c.name, "is_hni": c.is_hni,
                         "priority_probability": round(float(proba), 3)})
        ranked = sorted(rows, key=lambda r: r["priority_probability"], reverse=True)
        for r in ranked:
            r["reason"] = ("Model score high - review HNI/behavioral/loss signals"
                            if r["priority_probability"] > 0.5 else "Routine")
        return {"agent": self.name, "ranked_clients": ranked}


class MeetingIntelligenceAgent(BaseAgent):
    name = "distributor_meeting_intelligence"

    def fetch_context(self, client) -> dict:
        recent = sorted(client.transactions, key=lambda t: t.txn_date, reverse=True)[:5]
        return {"name": client.name, "age": client.age, "goals": client.goals,
                "risk_profile": client.risk_profile,
                "recent_transactions": [(t.txn_date.isoformat(), t.txn_type, t.amount) for t in recent],
                "sip_missed_3m": client.sip_missed_count_3m}

    def score(self, context: dict) -> dict:
        objections = []
        if context["sip_missed_3m"] > 0:
            objections.append("cash_flow_constraint")
        if any(t[1] == "Redemption" for t in context["recent_transactions"]):
            objections.append("post_redemption_skepticism")
        return {"objection_codes": objections or ["none"]}

    def explain(self, context, score) -> str:
        MSG = {"cash_flow_constraint": "May cite cash-flow issues for missed SIPs.",
               "post_redemption_skepticism": "May be cautious after a recent redemption.",
               "none": "No major objections flagged."}
        starters = "; ".join(f"Ask about progress on {g['name']}" for g in context["goals"])
        objections = " ".join(MSG[c] for c in score["objection_codes"])
        return f"{objections} Conversation starter: {starters}."


class PortfolioHealthScoreAgent(BaseAgent):
    name = "distributor_portfolio_health"

    def fetch_context(self, client) -> dict:
        total = sum(h.current_value for h in client.holdings) or 1
        mix_pct = {}
        for h in client.holdings:
            mix_pct[h.category] = mix_pct.get(h.category, 0) + h.current_value / total
        return {"feature_vector": client_to_feature_vector(client), "mix_pct": mix_pct,
                "has_insurance": client.has_insurance, "age": client.age}

    def score(self, context: dict) -> dict:
        predicted_score = float(health_model.predict(context["feature_vector"])[0])
        reasons = []
        if context["mix_pct"].get("Large Cap", 0) > 0.6:
            reasons.append("Too much large cap concentration")
        if "Debt" not in context["mix_pct"]:
            reasons.append("No debt allocation")
        if not context["has_insurance"]:
            reasons.append("Insurance gap")
        if "ELSS" not in context["mix_pct"] and context["age"] < 45:
            reasons.append("ELSS missing")
        return {"health_score": round(predicted_score, 1), "reasons": reasons or ["Well diversified"]}

    def explain(self, context, score) -> str:
        return f"Health score: {score['health_score']}/100. Key drivers: {', '.join(score['reasons'])}."


class VerifiedRecommendationAgent(BaseAgent):
    name = "distributor_verified_recommendations"

    def fetch_context(self, client) -> dict:
        health_agent = PortfolioHealthScoreAgent()
        h_ctx = health_agent.fetch_context(client)
        h_score = health_agent.score(h_ctx)
        savings_rate = 1 - (client.monthly_spending / client.monthly_income) if client.monthly_income else 0
        return {"reasons": h_score["reasons"], "savings_rate": savings_rate,
                "sip_active": client.sip_active, "has_insurance": client.has_insurance}

    def score(self, context: dict) -> dict:
        actions = []
        if context["savings_rate"] > 0.3 and context["sip_active"]:
            actions.append("Increase SIP")
        if "Too much large cap concentration" in context["reasons"]:
            actions.append("Switch Fund")
        if "No debt allocation" in context["reasons"]:
            actions.append("Goal Review")
        if "ELSS missing" in context["reasons"]:
            actions.append("Tax Saving")
        if not context["has_insurance"]:
            actions.append("Insurance Discussion")
        return {"recommended_actions": actions, "requires_distributor_signoff": True}

    def explain(self, context, score) -> str:
        if not score["recommended_actions"]:
            return "No actions flagged this cycle - pending distributor review."
        return "Suggested (pending your approval): " + ", ".join(score["recommended_actions"]) + "."


class LeadGenerationAgent(BaseAgent):
    name = "distributor_lead_generation"

    def fetch_context(self, client): raise NotImplementedError("use run_batch")
    def score(self, context): pass
    def explain(self, context, score): pass

    def run_batch(self, clients) -> dict:
        leads = []
        for c in clients:
            fv = client_to_feature_vector(c)
            proba = lead_model.predict_proba(fv)[0][1]
            if proba > 0.5:
                opp = []
                if not c.sip_active:
                    opp.append("Start SIP")
                elif proba > 0.7:
                    opp.append("Increase SIP")
                if not any(h.category == "ELSS" for h in c.holdings) and c.age < 45:
                    opp.append("Buy ELSS")
                if c.bank_idle_cash > 2 * c.monthly_income:
                    opp.append("Deploy idle cash")
                leads.append({"client_id": c.client_id, "name": c.name,
                              "conversion_probability": round(float(proba), 3),
                              "opportunities": opp or ["General review"]})
        leads.sort(key=lambda x: x["conversion_probability"], reverse=True)
        return {"agent": self.name, "leads": leads}


class Orchestrator:
    def __init__(self):
        self.goal_planner = GoalPlannerAgent()
        self.behavioral = BehavioralAIAgent()
        self.insights = HyperPersonalizedInsightsAgent()
        self.prioritization = ClientPrioritizationAgent()
        self.meeting_intel = MeetingIntelligenceAgent()
        self.health = PortfolioHealthScoreAgent()
        self.verified_recs = VerifiedRecommendationAgent()
        self.lead_gen = LeadGenerationAgent()

    def run_customer_dashboard(self, client) -> dict:
        return {"goal_plan": self.goal_planner.run(client),
                "behavioral_alerts": self.behavioral.run(client),
                "insights": self.insights.run(client)}

    def run_distributor_client_view(self, client) -> dict:
        return {"meeting_brief": self.meeting_intel.run(client),
                "portfolio_health": self.health.run(client),
                "verified_recommendations": self.verified_recs.run(client)}

    def run_distributor_book_view(self, clients) -> dict:
        return {"prioritized_clients": self.prioritization.run_batch(clients),
                "leads": self.lead_gen.run_batch(clients)}

orchestrator = Orchestrator()

# ----------------------------------------------------------------------
# Sample book of clients (replace with real/uploaded data later)
# ----------------------------------------------------------------------
@st.cache_data
def load_clients_from_csv(csv_path: str = "fabricated_client_dataset.csv") -> List[Client]:
    """
    Loads the full fabricated dataset and reconstructs Client objects.
    Since the CSV stores aggregated features (not raw holdings/transactions),
    we build minimal synthetic holdings/transactions that reproduce the same
    aggregate values, so client_to_feature_vector() still computes correctly.
    """
    df = pd.read_csv(csv_path)
    clients = []

    for _, row in df.iterrows():
        holdings = []

        # Reconstruct a Large Cap holding if large_cap_pct > 0
        invested_total = float(row["invested_total"])
        current_total = float(row["current_total"])

        if row["large_cap_pct"] > 0 and current_total > 0:
            lc_current = current_total * row["large_cap_pct"]
            lc_invested = invested_total * row["large_cap_pct"] if invested_total else lc_current
            holdings.append(Holding("Synthetic Large Cap Fund", "Large Cap", lc_invested, lc_current))

        if row["has_debt"] == 1:
            holdings.append(Holding("Synthetic Debt Fund", "Debt", 50000, 51000))

        if row["has_elss"] == 1:
            holdings.append(Holding("Synthetic ELSS Fund", "ELSS", 40000, 42000))

        # Pad remaining holding slots (n_holdings) with a generic "Other" category
        # so len(holdings) roughly matches n_holdings from the dataset
        while len(holdings) < int(row["n_holdings"]):
            holdings.append(Holding("Synthetic Other Fund", "Other", 30000, 30500))

        # Reconstruct transactions to reproduce redemptions_90d and sip_missed_3m
        transactions = []
        for i in range(int(row["redemptions_90d"])):
            transactions.append(Transaction(date(2026, 6, 1), "Redemption", "Synthetic Fund", 20000))
        if row["sip_active"] == 1:
            transactions.append(Transaction(date(2026, 6, 1), "SIP", "Synthetic Fund", 10000))

        client = Client(
            client_id=row["client_id"],
            name=f"Client {row['client_id']}",
            age=int(row["age"]),
            monthly_income=float(row["income"]),
            monthly_spending=float(row["income"]) * (1 - float(row["savings_rate"])),
            risk_profile=row["risk_profile"],
            goals=[{"name": "Wealth Creation", "target_amount": 1000000, "years": 5}],
            holdings=holdings,
            transactions=transactions,
            bank_idle_cash=float(row["idle_cash"]),
            is_hni=bool(row["is_hni"]),
            has_insurance=bool(row["has_insurance"]),
            sip_active=bool(row["sip_active"]),
            sip_missed_count_3m=int(row["sip_missed_3m"]),
        )
        clients.append(client)

    return clients

if "clients" not in st.session_state:
    st.session_state.clients = load_clients_from_csv("fabricated_client_dataset.csv")

clients = st.session_state.clients

# ----------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------
st.sidebar.title("ATOM FinAI Agents")
view = st.sidebar.radio("View", ["Customer Dashboard", "Distributor — Client View", "Distributor — Book View", "Add / Edit Client"])

clients = st.session_state.clients

search = st.sidebar.text_input("Search client ID")
filtered_clients = [c for c in clients if search.upper() in c.client_id.upper()] if search else clients
client_names = {c.client_id: c.name for c in filtered_clients}

if not client_names:
    st.sidebar.warning("No clients match that search.")
    st.stop()

selected_id = st.sidebar.selectbox(
    "Select client",
    options=list(client_names.keys()),
    format_func=lambda cid: f"{client_names[cid]} ({cid})",
    key="selected_client_id",
)
selected_client = next(c for c in clients if c.client_id == selected_id)

st.sidebar.caption(f"{len(clients)} clients in book ({len(filtered_clients)} shown)")

# ----------------------------------------------------------------------
# View: Customer Dashboard
# ----------------------------------------------------------------------
if view == "Customer Dashboard":
    st.title(f"Customer Dashboard — {selected_client.name}")
    result = orchestrator.run_customer_dashboard(selected_client)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🎯 Goal Plan")
        gp = result["goal_plan"]
        st.metric("Required Monthly SIP", f"₹{gp['score']['required_monthly_sip']:,.0f}")
        st.metric("Assumed Return", f"{gp['score']['assumed_return_pct']}%")
        st.write(pd.DataFrame([gp["score"]["suggested_allocation"]]).T.rename(columns={0: "Allocation %"}))
        st.info(gp["explanation"])

    with col2:
        st.subheader("🧠 Behavioral Risk")
        ba = result["behavioral_alerts"]
        severity = ba["score"]["severity"]
        color = {"high": "🔴", "medium": "🟡", "low": "🟢"}[severity]
        st.metric("Risk Level", f"{color} {severity.upper()}", f"{ba['score']['confidence']*100:.1f}% confidence")
        st.bar_chart(pd.Series(ba["score"]["class_probabilities"]))
        if severity == "high":
            st.warning(bal["explanation"])
        else:
            st.info(ba["explanation"])

    with col3:
        st.subheader("💡 Personalized Insights")
        ins = result["insights"]
        if ins["score"]["insight_codes"]:
            for code in ins["score"]["insight_codes"]:
                st.write(f"- `{code}`")
        else:
            st.success("No gaps flagged")
        st.info(ins["explanation"])

# ----------------------------------------------------------------------
# View: Distributor — Client View
# ----------------------------------------------------------------------
elif view == "Distributor — Client View":
    st.title(f"Distributor View — {selected_client.name}")
    result = orchestrator.run_distributor_client_view(selected_client)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📋 Meeting Brief")
        mb = result["meeting_brief"]
        st.write(mb["explanation"])
        st.caption(f"Objection codes: {', '.join(mb['score']['objection_codes'])}")

        st.subheader("✅ Verified Recommendations (pending sign-off)")
        vr = result["verified_recommendations"]
        if vr["score"]["recommended_actions"]:
            for action in vr["score"]["recommended_actions"]:
                st.checkbox(action, key=f"action_{selected_client.client_id}_{action}")
        else:
            st.write(vr["explanation"])

    with col2:
        st.subheader("📊 Portfolio Health")
        ph = result["portfolio_health"]
        score = ph["score"]["health_score"]
        st.metric("Health Score", f"{score}/100")
        st.progress(min(max(score / 100, 0.0), 1.0))
        for reason in ph["score"]["reasons"]:
            st.write(f"- {reason}")

# ----------------------------------------------------------------------
# View: Distributor — Book View (batch across all clients)
# ----------------------------------------------------------------------
elif view == "Distributor — Book View":
    st.title("Distributor — Full Book View")
    result = orchestrator.run_distributor_book_view(clients)

    st.subheader("🔝 Prioritized Clients")
    ranked_df = pd.DataFrame(result["prioritized_clients"]["ranked_clients"])
    st.dataframe(ranked_df, use_container_width=True)

    st.subheader("📈 Lead Opportunities")
    leads = result["leads"]["leads"]
    if leads:
        leads_df = pd.DataFrame(leads)
        leads_df["opportunities"] = leads_df["opportunities"].apply(lambda x: ", ".join(x))
        st.dataframe(leads_df, use_container_width=True)
    else:
        st.write("No leads above threshold in current book.")

# ----------------------------------------------------------------------
# View: Add / Edit Client (simple form to extend the book)
# ----------------------------------------------------------------------
elif view == "Add / Edit Client":
    st.title("Add a New Client")
    with st.form("add_client_form"):
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("Name")
            age = st.number_input("Age", 18, 80, 30)
            income = st.number_input("Monthly Income (₹)", 0, 5000000, 100000, step=5000)
            spending = st.number_input("Monthly Spending (₹)", 0, 5000000, 50000, step=5000)
            risk_profile = st.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"])
        with c2:
            idle_cash = st.number_input("Bank Idle Cash (₹)", 0, 10000000, 50000, step=5000)
            has_insurance = st.checkbox("Has Insurance")
            sip_active = st.checkbox("SIP Active", value=True)
            sip_missed = st.number_input("SIPs Missed (last 3 months)", 0, 12, 0)
            goal_name = st.text_input("Primary Goal Name", "Wealth Creation")
            goal_amount = st.number_input("Goal Target Amount (₹)", 0, 100000000, 1000000, step=10000)
            goal_years = st.number_input("Goal Horizon (years)", 1, 40, 5)

        submitted = st.form_submit_button("Add Client")
        if submitted:
            new_id = f"NEWCL{len(st.session_state.clients) + 1}"
            new_client = Client(
                client_id=new_id, name=name or "Unnamed Client", age=age,
                monthly_income=income, monthly_spending=spending, risk_profile=risk_profile,
                goals=[{"name": goal_name, "target_amount": goal_amount, "years": goal_years}],
                holdings=[], transactions=[],
                bank_idle_cash=idle_cash, is_hni=(income >= 300000),
                has_insurance=has_insurance, sip_active=sip_active, sip_missed_count_3m=sip_missed,
            )
            st.session_state.clients.append(new_client)
            st.success(f"Added {new_client.name} ({new_id}). Select them from the sidebar.")