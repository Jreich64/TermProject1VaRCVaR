import streamlit as st
import numpy as np
import plotly.graph_objects as go
from ComputeHelper import shock_returns, load_regular_var_and_cvar, _get_thread_pool


def build_animated_3d(frames_dict, title, color, z_label="Value", frame_label="Frame", start_key=None):
    """Build an animated 3D surface plot.

    Performance notes:
      - z arrays are downcast to float32 and rounded to 4 decimals → ~3x smaller
        JSON payload sent to the browser (this is what was causing the 300MB
        graph timeouts).
      - Frames are constructed as raw dicts to bypass plotly's per-trace
        validators, which are slow when there are many frames.
      - x/y are only sent on the base trace (frames inherit them), eliminating
        repeated coordinate arrays in every frame.
    """
    sorted_keys = sorted(frames_dict.keys())
    if start_key is not None and start_key in frames_dict:
        active_idx = sorted_keys.index(start_key)
    else:
        active_idx = 0

    first_df = frames_dict[sorted_keys[0]]
    tau_vals = first_df.index.values
    delta_vals = first_df.columns.values

    # Pre-convert all z arrays once: float32 + rounded keeps JSON small without
    # visible precision loss for surface plots.
    z_arrays = {
        k: np.round(frames_dict[k].values, 4).astype(np.float32)
        for k in sorted_keys
    }

    # Raw-dict frames: skips plotly's per-trace validation (much faster build).
    frames = [
        dict(
            name=str(k),
            data=[dict(
                type="surface",
                z=z_arrays[k],
                colorscale=color,
                showscale=False,  # only base trace shows the colorbar
            )],
        )
        for k in sorted_keys
    ]

    base = go.Surface(
        x=delta_vals, y=tau_vals, z=z_arrays[sorted_keys[active_idx]],
        colorscale=color, showscale=True,
        colorbar=dict(title=z_label),
    )
    fig = go.Figure(data=[base], frames=frames)

    sliders = [dict(
        active=active_idx,
        steps=[dict(
            method="animate",
            args=[[str(key)], dict(mode="immediate", frame=dict(duration=500, redraw=True), transition=dict(duration=300))],
            label=str(key)
        ) for key in sorted_keys],
        currentvalue=dict(prefix=f"{frame_label}: "),
        pad=dict(t=50)
    )]

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Delta",
            yaxis_title="Tau",
            zaxis_title=z_label
        ),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=1.15,
            x=0.5,
            xanchor="center",
            buttons=[
                dict(label="Play",
                     method="animate",
                     args=[None, dict(frame=dict(duration=500, redraw=True),
                                      fromcurrent=True,
                                      transition=dict(duration=300))]),
                dict(label="Pause",
                     method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate",
                                        transition=dict(duration=0))])
            ]
        )],
        sliders=sliders,
        template="plotly_white"
    )
    return fig


# ---------- Cached wrappers ----------
# `_frames_dict` has a leading underscore so Streamlit skips hashing it (dict
# of DataFrames is expensive to hash). `frames_id = id(frames_dict)` is the
# real cache key — it changes whenever upstream caches return a new object,
# which is exactly what we want for invalidation.

@st.cache_data(show_spinner=False, max_entries=8, ttl=1800)
def _build_animated_3d_cached(_frames_dict, frames_id, title, color, z_label, frame_label, start_key):
    return build_animated_3d(_frames_dict, title, color, z_label=z_label, frame_label=frame_label, start_key=start_key)


@st.cache_data(show_spinner=False, max_entries=8, ttl=1800)
def _figure_to_html_cached(_fig, fig_id):
    return _fig.to_html(include_plotlyjs=True, full_html=True)


def _build_one(job):
    """Worker for ThreadPoolExecutor: returns (name, figure)."""
    name, frames_dict, title, color, z_label, frame_label, start_key = job
    fig = _build_animated_3d_cached(frames_dict, id(frames_dict), title, color, z_label, frame_label, start_key)
    return name, fig


# ---------- Page ----------
st.title("3D VaR / CVaR Surface Plots")

if "regular_var" not in st.session_state:
    st.warning("No results yet. Run the model on the main page first.")
else:
    selected_funds = st.session_state.get("_saved_funds", [])
    st.write(f"**Selected Funds:** {', '.join(selected_funds) if selected_funds else 'None'}")

    regular_var = st.session_state["regular_var"]
    regular_cvar = st.session_state["regular_cvar"]
    selected_quantile = st.session_state["selected_quantile"]
    selected_sectors = st.session_state["selected_sectors"]
    selected_devalue = st.session_state.get("_saved_max_devalue", st.session_state.get("selected_devalue", 0.3))
    slider_devalue = float(round(st.session_state.get("selected_devalue", 0.0), 2))

    q = round(selected_quantile, 2)
    quantiles = sorted(regular_var.keys())
    min_quantile = min(quantiles)
    max_quantile = max(quantiles)
    normalize_returns = st.session_state.get("normalize_returns", True)
    norm_label = "Normalized" if normalize_returns else "Non-Normalized"

    # ---- Build the regular VaR + CVaR figures in parallel ----
    base_jobs = [
        ("var",  regular_var,  f"VaR vs Tau & Delta ({norm_label})",  "Blues", "VaR",  "Quantile", q),
        ("cvar", regular_cvar, f"CVaR vs Tau & Delta ({norm_label})", "Reds",  "CVaR", "Quantile", q),
    ]

    pool = _get_thread_pool()
    figs = {}
    with st.spinner("Building 3D surfaces..."):
        for name, fig in pool.map(_build_one, base_jobs):
            figs[name] = fig

    # --- Regular VaR ---
    st.subheader("VaR Surface (animated over Quantile)")
    st.plotly_chart(figs["var"], use_container_width=True)
    st.download_button(
        "Download VaR 3D",
        _figure_to_html_cached(figs["var"], id(figs["var"])),
        "var_3d.html", "text/html",
    )

    # --- Regular CVaR ---
    st.subheader("CVaR Surface (animated over Quantile)")
    st.plotly_chart(figs["cvar"], use_container_width=True)
    st.download_button(
        "Download CVaR 3D",
        _figure_to_html_cached(figs["cvar"], id(figs["cvar"])),
        "cvar_3d.html", "text/html",
    )

    # --- Shocked VaR/CVaR animated over devalue percent ---
    if selected_sectors and "returns" in st.session_state and "sector_df_dict" in st.session_state:
        returns_dict = st.session_state["returns"]
        sectors_dict = st.session_state["sector_df_dict"]

        # Check if cached results are still valid (q excluded — all quantiles are cached)
        cache_key = (tuple(selected_sectors), selected_devalue, min_quantile, max_quantile, normalize_returns)
        cached_key = st.session_state.get("_3d_shock_cache_key")

        if cached_key == cache_key and "_3d_shocked_var_all" in st.session_state:
            shocked_var_all = st.session_state["_3d_shocked_var_all"]
            shocked_cvar_all = st.session_state["_3d_shocked_cvar_all"]
        else:
            devalue_steps = np.round(np.arange(0.0, selected_devalue + 0.01, 0.01), 2)
            shocked_var_all = {}
            shocked_cvar_all = {}
            with st.spinner("Computing shocked surfaces for each devalue step..."):
                for dv in devalue_steps:
                    shocked = shock_returns(returns_dict, sectors_dict, selected_sectors, float(dv))
                    sv, sc = load_regular_var_and_cvar(shocked, min_quantile, max_quantile, normalize_returns)
                    shocked_var_all[float(dv)] = sv
                    shocked_cvar_all[float(dv)] = sc
            st.session_state["_3d_shocked_var_all"] = shocked_var_all
            st.session_state["_3d_shocked_cvar_all"] = shocked_cvar_all
            st.session_state["_3d_shock_cache_key"] = cache_key

        # Extract the selected quantile slice for display
        shocked_var_frames = {dv: sv[q] for dv, sv in shocked_var_all.items()}
        shocked_cvar_frames = {dv: sc[q] for dv, sc in shocked_cvar_all.items()}

        # Selected-devalue slice for the "animated over Quantile" pair
        shocked_var_q_frames = {q_key: shocked_var_all[slider_devalue][q_key] for q_key in quantiles if q_key in shocked_var_all.get(slider_devalue, {})}
        shocked_cvar_q_frames = {q_key: shocked_cvar_all[slider_devalue][q_key] for q_key in quantiles if q_key in shocked_cvar_all.get(slider_devalue, {})}

        # ---- Build all (up to 4) shocked figures in parallel ----
        shocked_jobs = [
            ("svar",  shocked_var_frames,  f"Shocked VaR ({norm_label})",  "Greens",  "Shocked VaR",  "Devalue %", slider_devalue),
            ("scvar", shocked_cvar_frames, f"Shocked CVaR ({norm_label})", "Purples", "Shocked CVaR", "Devalue %", slider_devalue),
        ]
        if shocked_var_q_frames:
            shocked_jobs.append(("svar_q",  shocked_var_q_frames,  f"Shocked VaR ({norm_label})",  "Greens",  "Shocked VaR",  "Quantile", q))
            shocked_jobs.append(("scvar_q", shocked_cvar_q_frames, f"Shocked CVaR ({norm_label})", "Purples", "Shocked CVaR", "Quantile", q))

        with st.spinner("Building shocked 3D surfaces..."):
            shocked_figs = {}
            for name, fig in pool.map(_build_one, shocked_jobs):
                shocked_figs[name] = fig

        st.subheader(f"Shocked VaR Surface at Quantile {q} (animated over Devalue %)")
        st.write(f"Sectors Shocked: {selected_sectors}")
        st.plotly_chart(shocked_figs["svar"], use_container_width=True)
        st.download_button(
            "Download Shocked VaR 3D (Devalue)",
            _figure_to_html_cached(shocked_figs["svar"], id(shocked_figs["svar"])),
            "shocked_var_3d_devalue.html", "text/html",
        )

        st.subheader(f"Shocked CVaR Surface at Quantile {q} (animated over Devalue %)")
        st.plotly_chart(shocked_figs["scvar"], use_container_width=True)
        st.download_button(
            "Download Shocked CVaR 3D (Devalue)",
            _figure_to_html_cached(shocked_figs["scvar"], id(shocked_figs["scvar"])),
            "shocked_cvar_3d_devalue.html", "text/html",
        )

        if "svar_q" in shocked_figs:
            st.subheader(f"Shocked VaR Surface at Devalue {slider_devalue} (animated over Quantile)")
            st.write(f"Sectors Shocked: {selected_sectors}")
            st.plotly_chart(shocked_figs["svar_q"], use_container_width=True)
            st.download_button(
                "Download Shocked VaR 3D (Quantile)",
                _figure_to_html_cached(shocked_figs["svar_q"], id(shocked_figs["svar_q"])),
                "shocked_var_3d_quantile.html", "text/html",
            )

            st.subheader(f"Shocked CVaR Surface at Devalue {slider_devalue} (animated over Quantile)")
            st.plotly_chart(shocked_figs["scvar_q"], use_container_width=True)
            st.download_button(
                "Download Shocked CVaR 3D (Quantile)",
                _figure_to_html_cached(shocked_figs["scvar_q"], id(shocked_figs["scvar_q"])),
                "shocked_cvar_3d_quantile.html", "text/html",
            )
    else:
        st.info("Select sectors to shock on the main page to see shocked 3D plots.")
