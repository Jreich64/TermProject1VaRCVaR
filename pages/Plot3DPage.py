import streamlit as st
import numpy as np
import plotly.graph_objects as go
from ComputeHelper import shock_returns, load_regular_var_and_cvar


def build_animated_3d(frames_dict, title, color, z_label="Value", frame_label="Frame", start_key=None):
    sorted_keys = sorted(frames_dict.keys())
    if start_key is not None and start_key in frames_dict:
        active_idx = sorted_keys.index(start_key)
    else:
        active_idx = 0
    first_df = frames_dict[sorted_keys[0]]
    tau_vals = first_df.index.values
    delta_vals = first_df.columns.values

    frames = []
    for key in sorted_keys:
        df = frames_dict[key]
        frames.append(go.Frame(
            data=[go.Surface(
                x=delta_vals, y=tau_vals, z=df.values,
                colorscale=color, showscale=True,
                colorbar=dict(title=z_label)
            )],
            name=str(key)
        ))

    first_z = frames_dict[sorted_keys[active_idx]].values
    fig = go.Figure(
        data=[go.Surface(
            x=delta_vals, y=tau_vals, z=first_z,
            colorscale=color, showscale=True,
            colorbar=dict(title=z_label)
        )],
        frames=frames
    )

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


st.title("3D VaR / CVaR Surface Plots")

if "regular_var" not in st.session_state:
    st.warning("No results yet. Run the model on the main page first.")
else:
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

    # --- Regular VaR animated over quantiles ---
    st.subheader("VaR Surface (animated over Quantile)")
    fig_var = build_animated_3d(regular_var, f"VaR vs Tau & Delta ({norm_label})", "Blues", z_label="VaR", frame_label="Quantile", start_key=q)
    st.plotly_chart(fig_var, use_container_width=True)
    html_var = fig_var.to_html(include_plotlyjs=True, full_html=True)
    st.download_button("Download VaR 3D", html_var, "var_3d.html", "text/html")

    # --- Regular CVaR animated over quantiles ---
    st.subheader("CVaR Surface (animated over Quantile)")
    fig_cvar = build_animated_3d(regular_cvar, f"CVaR vs Tau & Delta ({norm_label})", "Reds", z_label="CVaR", frame_label="Quantile", start_key=q)
    st.plotly_chart(fig_cvar, use_container_width=True)
    html_cvar = fig_cvar.to_html(include_plotlyjs=True, full_html=True)
    st.download_button("Download CVaR 3D", html_cvar, "cvar_3d.html", "text/html")

    # --- Shocked VaR/CVaR animated over devalue percent ---
    if selected_sectors and "returns" in st.session_state and "sector_df_dict" in st.session_state:
        returns_dict = st.session_state["returns"]
        sectors_dict = st.session_state["sector_df_dict"]

        # Check if cached results are still valid
        cache_key = (tuple(selected_sectors), selected_devalue, q, min_quantile, max_quantile, normalize_returns)
        cached_key = st.session_state.get("_3d_shock_cache_key")

        if cached_key == cache_key and "_3d_shocked_var_frames" in st.session_state:
            shocked_var_frames = st.session_state["_3d_shocked_var_frames"]
            shocked_cvar_frames = st.session_state["_3d_shocked_cvar_frames"]
        else:
            devalue_steps = np.round(np.arange(0.0, selected_devalue + 0.01, 0.01), 2)
            shocked_var_frames = {}
            shocked_cvar_frames = {}
            with st.spinner("Computing shocked surfaces for each devalue step..."):
                for dv in devalue_steps:
                    shocked = shock_returns(returns_dict, sectors_dict, selected_sectors, float(dv))
                    sv, sc = load_regular_var_and_cvar(shocked, min_quantile, max_quantile, normalize_returns)
                    shocked_var_frames[float(dv)] = sv[q]
                    shocked_cvar_frames[float(dv)] = sc[q]
            st.session_state["_3d_shocked_var_frames"] = shocked_var_frames
            st.session_state["_3d_shocked_cvar_frames"] = shocked_cvar_frames
            st.session_state["_3d_shock_cache_key"] = cache_key

        st.subheader(f"Shocked VaR Surface at Quantile {q} (animated over Devalue %)")
        st.write(f"Sectors Shocked: {selected_sectors}")
        fig_svar = build_animated_3d(shocked_var_frames, f"Shocked VaR ({norm_label})", "Greens", z_label="Shocked VaR", frame_label="Devalue %", start_key=slider_devalue)
        st.plotly_chart(fig_svar, use_container_width=True)
        html_svar = fig_svar.to_html(include_plotlyjs=True, full_html=True)
        st.download_button("Download Shocked VaR 3D", html_svar, "shocked_var_3d.html", "text/html")

        st.subheader(f"Shocked CVaR Surface at Quantile {q} (animated over Devalue %)")
        fig_scvar = build_animated_3d(shocked_cvar_frames, f"Shocked CVaR ({norm_label})", "Purples", z_label="Shocked CVaR", frame_label="Devalue %", start_key=slider_devalue)
        st.plotly_chart(fig_scvar, use_container_width=True)
        html_scvar = fig_scvar.to_html(include_plotlyjs=True, full_html=True)
        st.download_button("Download Shocked CVaR 3D", html_scvar, "shocked_cvar_3d.html", "text/html")
    else:
        st.info("Select sectors to shock on the main page to see shocked 3D plots.")
