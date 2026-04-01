import io
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="ASCE 7-22 Seismic Coefficient Plotter", layout="wide")


@dataclass
class SeismicInputs:
    risk_category: str
    Ie: float
    R: float
    omega_0: float
    Cd: float
    Ct: float
    x: float
    hn: float
    Ss: float
    S1: float
    Sms: float
    Sm1: float
    Sds: float
    Sd1: float
    TL: float


# -----------------------------------------------------------------------------
# Core calculations
# -----------------------------------------------------------------------------
def calc_Ta(Ct: float, x: float, hn: float) -> float:
    """Approximate fundamental period per ASCE 7 expression Ta = Ct * hn^x."""
    return Ct * (hn ** x)


def design_spectrum_sa(T: np.ndarray, Sds: float, Sd1: float, TL: float) -> np.ndarray:
    """
    ASCE 7-22 design response spectrum ordinates using SDS, SD1, and TL.

    Assumes the usual piecewise design spectrum:
      - 0 <= T <= T0   : ramp from 0.4*SDS to SDS
      - T0 < T <= Ts   : SDS plateau
      - Ts < T <= TL   : SD1 / T
      - T > TL         : SD1 * TL / T^2

    where:
      Ts = SD1 / SDS
      T0 = 0.2 * Ts
    """
    T = np.asarray(T, dtype=float)
    Ts = Sd1 / Sds if Sds != 0 else np.inf
    T0 = 0.2 * Ts

    Sa = np.zeros_like(T, dtype=float)

    mask1 = T <= T0
    mask2 = (T > T0) & (T <= Ts)
    mask3 = (T > Ts) & (T <= TL)
    mask4 = T > TL

    if np.any(mask1):
        Sa[mask1] = Sds * (0.4 + 0.6 * (T[mask1] / T0)) if T0 > 0 else Sds
    if np.any(mask2):
        Sa[mask2] = Sds
    if np.any(mask3):
        Sa[mask3] = Sd1 / T[mask3]
    if np.any(mask4):
        Sa[mask4] = (Sd1 * TL) / (T[mask4] ** 2)

    return Sa


def coefficient_from_sa(Sa: np.ndarray, R: float, Ie: float) -> np.ndarray:
    """Convert spectral acceleration to seismic response coefficient C = Sa / (R / Ie)."""
    return np.asarray(Sa, dtype=float) * Ie / R


def lower_bound_coefficient(Sds: float, Ie: float, S1: float, R: float) -> float:
    """
    ELF lower-bound checks often referenced for Cs:
      max(0.044*SDS*Ie, 0.01), and if S1 >= 0.6 then not less than 0.5*S1/(R/Ie)

    This app shows the spectrum-based coefficients directly and reports the governing
    value at the selected period with optional ELF lower-bound enforcement.
    """
    lb = max(0.044 * Sds * Ie, 0.01)
    if S1 >= 0.6:
        lb = max(lb, 0.5 * S1 * Ie / R)
    return lb


def interpolate_uploaded_spectrum(T_query: np.ndarray, spec_df: pd.DataFrame, t_col: str, sa_col: str) -> np.ndarray:
    """Linearly interpolate uploaded multi-period spectrum to the requested periods."""
    df = spec_df[[t_col, sa_col]].dropna().copy()
    df = df.sort_values(t_col)
    df = df.drop_duplicates(subset=t_col, keep="first")

    T_data = df[t_col].to_numpy(dtype=float)
    Sa_data = df[sa_col].to_numpy(dtype=float)

    return np.interp(T_query, T_data, Sa_data)


# -----------------------------------------------------------------------------
# Sidebar inputs
# -----------------------------------------------------------------------------
st.title("ASCE 7-22 Seismic Coefficient Plotter")
st.caption(
    "Plots seismic coefficient versus period for: "
    "(1) the ASCE 7-22 design spectrum using SDS/SD1/TL, and "
    "(2) an uploaded multi-period design spectrum CSV."
)

with st.sidebar:
    st.header("Seismic inputs")

    risk_category = st.selectbox("Risk category", ["I", "II", "III", "IV"], index=3)

    importance_factor_map = {
        "I": 1.00,
        "II": 1.00,
        "III": 1.25,
        "IV": 1.50,
    }
    Ie = importance_factor_map[risk_category]
    st.number_input(
        "Importance factor, Ie",
        value=float(Ie),
        min_value=0.1,
        step=0.1,
        disabled=True,
    )

    st.subheader("System factors")
    R = st.number_input("Response modification coefficient, R", value=8.0, min_value=0.1, step=0.1)
    omega_0 = st.number_input("Overstrength factor, Ω₀", value=2.5, min_value=0.1, step=0.1)
    Cd = st.number_input("Deflection amplification factor, Cd", value=3.0, min_value=0.1, step=0.1)

    st.subheader("Approximate period parameters")
    period_parameter_options = {
        "0.028, 0.80": (0.028, 0.80),
        "0.016, 0.90": (0.016, 0.90),
        "0.030, 0.75": (0.030, 0.75),
        "0.020, 0.75": (0.020, 0.75),
    }
    period_parameter_label = st.selectbox(
        "Select Ct and x",
        list(period_parameter_options.keys()),
        index=2,
    )
    Ct, x = period_parameter_options[period_parameter_label]
    left_period, right_period = st.columns(2)
    with left_period:
        st.number_input("Ct", value=float(Ct), min_value=0.0, step=0.001, format="%.3f", disabled=True)
    with right_period:
        st.number_input("x", value=float(x), min_value=0.0, step=0.01, format="%.2f", disabled=True)
    hn = st.number_input("Structural height, hn (ft)", value=30.0, min_value=0.0, step=1.0)

    st.subheader("Seismic hazard")
    Ss = st.number_input("Ss", value=2.28, min_value=0.0, step=0.01)
    S1 = st.number_input("S1", value=0.76, min_value=0.0, step=0.01)
    Sms = st.number_input("Sms", value=2.41, min_value=0.0, step=0.01)
    Sm1 = st.number_input("Sm1", value=1.74, min_value=0.0, step=0.01)
    Sds = st.number_input("Sds", value=1.61, min_value=0.0, step=0.01)
    Sd1 = st.number_input("Sd1", value=1.16, min_value=0.0, step=0.01)
    TL = st.number_input("TL (s)", value=8.0, min_value=0.01, step=0.1)

    Ta = calc_Ta(Ct, x, hn)

    enforce_lower_bound = st.checkbox("Apply ELF lower-bound coefficient check", value=True)

    st.markdown("---")
    st.header("Multi-period spectrum CSV")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])


inputs = SeismicInputs(
    risk_category=risk_category,
    Ie=Ie,
    R=R,
    omega_0=omega_0,
    Cd=Cd,
    Ct=Ct,
    x=x,
    hn=hn,
    Ss=Ss,
    S1=S1,
    Sms=Sms,
    Sm1=Sm1,
    Sds=Sds,
    Sd1=Sd1,
    TL=TL,
)


# -----------------------------------------------------------------------------
# Input summary
# -----------------------------------------------------------------------------
col_a, col_b = st.columns([1.3, 1])
with col_a:
    st.subheader("Calculated values")
    lb = lower_bound_coefficient(inputs.Sds, inputs.Ie, inputs.S1, inputs.R)
    Ts = inputs.Sd1 / inputs.Sds if inputs.Sds != 0 else np.nan
    T0 = 0.2 * Ts if np.isfinite(Ts) else np.nan

    summary_df = pd.DataFrame(
        {
            "Parameter": ["Ta", "T0", "Ts", "TL", "ELF lower bound"],
            "Value": [Ta, T0, Ts, inputs.TL, lb],
            "Units": ["s", "s", "s", "s", "-"]
        }
    )
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

with col_b:
    st.subheader("CSV format expected")
    st.markdown(
        "Use a CSV with at least two columns: one for **period** and one for **spectral acceleration**.\n\n"
        "Example:\n"
        "```csv\n"
        "Period,Sa\n"
        "0.00,0.64\n"
        "0.10,1.20\n"
        "0.20,1.55\n"
        "0.50,1.10\n"
        "1.00,0.62\n"
        "```"
    )


# -----------------------------------------------------------------------------
# Build the two methods
# -----------------------------------------------------------------------------
T_max_for_plot = max(10.0, inputs.TL * 1.2, Ta * 1.2)
T_plot = np.linspace(0.0, T_max_for_plot, 1201)

Sa_asce = design_spectrum_sa(T_plot, inputs.Sds, inputs.Sd1, inputs.TL)
C_asce = coefficient_from_sa(Sa_asce, inputs.R, inputs.Ie)

if enforce_lower_bound:
    C_asce = np.maximum(C_asce, lb)

csv_ready = False
csv_df = None
C_csv = None
Sa_csv = None
csv_t_col = None
csv_sa_col = None

if uploaded_file is not None:
    try:
        csv_df = pd.read_csv(uploaded_file)
        if csv_df.shape[1] < 2:
            st.error("The uploaded CSV needs at least two columns.")
        else:
            st.subheader("Uploaded spectrum preview")
            st.dataframe(csv_df.head(10), use_container_width=True)

            cols = list(csv_df.columns)
            left, right = st.columns(2)
            with left:
                csv_t_col = st.selectbox("Period column", cols, index=0)
            with right:
                csv_sa_col = st.selectbox("Spectral acceleration column", cols, index=1)

            Sa_csv = interpolate_uploaded_spectrum(T_plot, csv_df, csv_t_col, csv_sa_col)
            C_csv = coefficient_from_sa(Sa_csv, inputs.R, inputs.Ie)
            if enforce_lower_bound:
                C_csv = np.maximum(C_csv, lb)
            csv_ready = True
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")


# -----------------------------------------------------------------------------
# Evaluate at selected period
# -----------------------------------------------------------------------------
Sa_asce_sel = float(design_spectrum_sa(np.array([Ta]), inputs.Sds, inputs.Sd1, inputs.TL)[0])
C_asce_sel = float(coefficient_from_sa(np.array([Sa_asce_sel]), inputs.R, inputs.Ie)[0])
if enforce_lower_bound:
    C_asce_sel = max(C_asce_sel, lb)

C_csv_sel = None
Sa_csv_sel = None
if csv_ready:
    Sa_csv_sel = float(interpolate_uploaded_spectrum(np.array([Ta]), csv_df, csv_t_col, csv_sa_col)[0])
    C_csv_sel = float(coefficient_from_sa(np.array([Sa_csv_sel]), inputs.R, inputs.Ie)[0])
    if enforce_lower_bound:
        C_csv_sel = max(C_csv_sel, lb)

if C_csv_sel is None:
    governing_method = "ASCE 7-22 design spectrum"
    governing_value = C_asce_sel
else:
    if C_asce_sel >= C_csv_sel:
        governing_method = "ASCE 7-22 design spectrum"
        governing_value = C_asce_sel
    else:
        governing_method = "Uploaded multi-period spectrum"
        governing_value = C_csv_sel


# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=T_plot,
        y=C_asce,
        mode="lines",
        name="ASCE 7-22 design spectrum",
        hovertemplate="Method: ASCE 7-22<br>Period: %{x:.3f} s<br>Coefficient: %{y:.4f}<extra></extra>",
    )
)

if csv_ready:
    fig.add_trace(
        go.Scatter(
            x=T_plot,
            y=C_csv,
            mode="lines",
            name="Uploaded multi-period spectrum",
            hovertemplate="Method: CSV spectrum<br>Period: %{x:.3f} s<br>Coefficient: %{y:.4f}<extra></extra>",
        )
    )

# Highlight selected-period values
fig.add_trace(
    go.Scatter(
        x=[Ta],
        y=[C_asce_sel],
        mode="markers+text",
        name="ASCE @ selected T",
        text=[f"ASCE = {C_asce_sel:.4f}"],
        textposition="top center",
        marker=dict(size=10),
        hovertemplate="Approximate period Ta<br>Method: ASCE 7-22<br>Period: %{x:.3f} s<br>Coefficient: %{y:.4f}<extra></extra>",
        showlegend=False,
    )
)

if C_csv_sel is not None:
    fig.add_trace(
        go.Scatter(
            x=[Ta],
            y=[C_csv_sel],
            mode="markers+text",
            name="CSV @ selected T",
            text=[f"CSV = {C_csv_sel:.4f}"],
            textposition="bottom center",
            marker=dict(size=10),
            hovertemplate="Approximate period Ta<br>Method: CSV spectrum<br>Period: %{x:.3f} s<br>Coefficient: %{y:.4f}<extra></extra>",
            showlegend=False,
        )
    )

# Governing point marker
fig.add_vline(x=Ta, line_dash="dash", annotation_text=f"Ta = {Ta:.3f} s")
fig.add_trace(
    go.Scatter(
        x=[Ta],
        y=[governing_value],
        mode="markers+text",
        name="Governing at selected T",
        text=[f"Governing at Ta: {governing_method}\nC = {governing_value:.4f}"],
        textposition="middle right",
        marker=dict(size=15, symbol="star"),
        hovertemplate="Governing at Ta<br>Method: 
        showlegend=False,
    )
)

fig.update_layout(
    title="Seismic coefficient vs. structure period",
    xaxis_title="Period, T (s)",
    yaxis_title="Seismic coefficient",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    yaxis=dict(rangemode="tozero"),
    height=650,
)

st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# Results table
# -----------------------------------------------------------------------------
st.subheader("Approximate-period results")
results_rows = [
    {
        "Method": "ASCE 7-22 design spectrum",
        "Approximate period Ta (s)": Ta,
        "Sa(T)": Sa_asce_sel,
        "Coefficient": C_asce_sel,
        "Governing at Ta": governing_method == "ASCE 7-22 design spectrum",
    }
]

if C_csv_sel is not None:
    results_rows.append(
        {
            "Method": "Uploaded multi-period spectrum",
            "Approximate period Ta (s)": Ta,
            "Sa(T)": Sa_csv_sel,
            "Coefficient": C_csv_sel,
            "Governing at Ta": governing_method == "Uploaded multi-period spectrum",
        }
    )

st.dataframe(pd.DataFrame(results_rows), use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# Downloadable plot data
# -----------------------------------------------------------------------------
st.subheader("Export plot data")
export_df = pd.DataFrame({
    "T": T_plot,
    "C_asce_design_spectrum": C_asce,
})
if csv_ready:
    export_df["C_uploaded_multi_period"] = C_csv

csv_bytes = export_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download plotted data as CSV",
    data=csv_bytes,
    file_name="seismic_coefficient_plot_data.csv",
    mime="text/csv",
)


# -----------------------------------------------------------------------------
# Notes
# -----------------------------------------------------------------------------
with st.expander("Notes / assumptions used in this app"):
    st.markdown(
        "- The **ASCE 7-22 design spectrum** curve is built from **SDS**, **SD1**, and **TL** using the standard piecewise design spectrum shape.\n"
        "- The uploaded CSV is assumed to contain **period** and **design spectral acceleration** values.\n"
        "- Both curves are converted to a plotted **seismic coefficient** using `Coefficient = Sa * Ie / R`.\n"
        "- An optional **ELF lower-bound** check can be applied to both curves for comparison.\n"
        "- The point highlighted on the plot is the **approximate period Ta = Ct × hn^x** calculated directly from the selected Ct/x pair and the entered height.\n"
        "- If you want this app to follow a different house standard for the governing seismic coefficient, adjust the helper functions at the top of the file."
    )
