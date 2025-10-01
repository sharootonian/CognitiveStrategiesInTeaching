"""
========================================================
Author: Sevan Harootonian
Affiliation: Princeton University
========================================================
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import digamma, gammaln
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
from matplotlib.transforms import Bbox

def softmax_probs(utilities,inversetemp):
    probs = {e:np.exp(inversetemp*val) for e,val in utilities.items()}
    norm = sum(probs.values())
    probs = {e: p/norm for e,p in probs.items()}
    return probs
            
def sample_multinomial_logit(feature_weights,feature_value_dict):
    edge_logits = defaultdict(float)
    for w, feature in zip(feature_weights,feature_value_dict):
        for edge, val in feature.items():
            edge_logits[edge] += val*w
    probs = softmax_probs(edge_logits, 1)
    edges = list(feature)
    p = np.array([probs[e] for e in edges], dtype=float)
    idx = np.random.choice(len(edges),size= 1, p=p)  # sample indices, not tuples
    return edges[idx[0]]


def vb_random_effects_model_comparison(df_bic, alpha0_value=1.0, max_iter=1000, tol=1e-6, use_digamma=True):
    """
    VB-EM for random-effects model comparison.
    Returns:
      model_cols: list of model names
      r: estimated model frequencies
      alpha: posterior Dirichlet parameters
      u: responsibilities (N x K)
      log_ev: log-evidence matrix (N x K)
      alpha0: prior counts
    """
    model_cols = list(df_bic.columns.drop('subjID'))
    log_ev = -0.5 * df_bic[model_cols].values  # ℓₙₖ
    N, K = log_ev.shape

    alpha0 = np.ones(K) * alpha0_value
    alpha = alpha0.copy()
    r = alpha / alpha.sum()

    for _ in range(max_iter):
        # E-step: compute E[log m]
        if use_digamma:
            eq_logm = digamma(alpha) - digamma(alpha.sum())
        else:
            eq_logm = np.log(r + 1e-16)

        # responsibilities
        log_u = log_ev + eq_logm
        log_u -= log_u.max(axis=1, keepdims=True)
        u = np.exp(log_u)
        u /= u.sum(axis=1, keepdims=True)

        # M-step: update α and r
        alpha_new = alpha0 + u.sum(axis=0)
        r_new = alpha_new / alpha_new.sum()

        if np.linalg.norm(r_new - r) < tol:
            alpha, r = alpha_new, r_new
            break
        alpha, r = alpha_new, r_new

    return model_cols, r, alpha, u, log_ev, alpha0

def compute_elbo(log_ev, u, alpha0, alpha):
    """
    Compute the Variational Bayes ELBO (lower bound on log evidence).
    """
    # Data term: E_q[log p(D|z)]
    term1 = np.sum(u * log_ev)
    # Prior term: E_q[log p(z|m)]
    eq_logm = digamma(alpha) - digamma(alpha.sum())
    term2 = np.sum(u * eq_logm)
    # Entropy of q(z)
    term3 = -np.sum(u * np.log(u + 1e-16))
    # KL divergence KL[q(m)||p(m)]
    logB0 = np.sum(gammaln(alpha0)) - gammaln(alpha0.sum())
    logB  = np.sum(gammaln(alpha))  - gammaln(alpha.sum())
    kl_qm_pm = logB0 - logB + np.sum((alpha - alpha0) * eq_logm)
    return term1 + term2 + term3 - kl_qm_pm

def compute_exceedance(alpha, n_draws=20000):
    """
    Monte Carlo estimate of exceedance probability φ_k.
    """
    samples = np.random.default_rng().dirichlet(alpha, size=n_draws)
    winners = np.argmax(samples, axis=1)
    return np.bincount(winners, minlength=len(alpha)) / n_draws


def plot_model_comparison(model_names, frequencies, protected_exceedance, 
                          order=None, figsize=(12, 5)):
    # Determine ordering
    if order is not None:
        # Create index mapping from model_names to order
        order_idx = [model_names.index(m) for m in order]
        names_ord = [model_names[i] for i in order_idx]
        freq_ord = frequencies[order_idx]
        phi_ord = protected_exceedance[order_idx]
    else:
        names_ord = model_names
        freq_ord = frequencies
        phi_ord = protected_exceedance

    x = np.arange(len(names_ord))
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # exceedance
    axes[0].bar(x, phi_ord, color='skyblue')
    axes[0].set_title('Exceedance Probability', fontsize=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names_ord, rotation=45, ha='right')
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel('Probability')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(phi_ord):
        axes[0].text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Model frequencies
    axes[1].bar(x, freq_ord, color='salmon')
    axes[1].set_title('Model Frequencies', fontsize=14)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names_ord, rotation=45, ha='right')
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel('Frequency')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(freq_ord):
        axes[1].text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    return fig, axes


def plot_model_comparison_horizontal(model_names, frequencies, exceedance, 
                                     order=None, figsize=(10, 3)):
    if order is not None:
        order_idx = [model_names.index(m) for m in order]
        names_ord = [model_names[i] for i in order_idx]
        freq_ord = frequencies[order_idx]
        phi_ord = exceedance[order_idx]
    else:
        names_ord = model_names
        freq_ord = frequencies
        phi_ord = exceedance
    
    y = np.arange(len(names_ord))
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    fig.subplots_adjust(left=0.25, right=0.95, top=0.9, bottom=0.2, wspace=0.4)
    
    # exceedance (horizontal)
    bars1 = axes[0].barh(y, phi_ord, height=0.5, color='skyblue')
    axes[0].set_title('Exceedance Probability', fontsize=18)
    axes[0].set_xlabel('Probability')
    axes[0].invert_yaxis()
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names_ord)
    axes[0].set_xlim(0, 1)
    axes[0].tick_params(axis='x', labelsize=12)
    axes[0].grid(axis='x', linestyle=':', color='gray', linewidth=0.7)
    # Annotate values
    for bar in bars1:
        width = bar.get_width()
        y_pos = bar.get_y() + bar.get_height()/2
        axes[0].text(width + 0.02, y_pos, f'{width:.2f}', va='center', fontsize=11)

    # Model frequencies (horizontal)
    bars2 = axes[1].barh(y, freq_ord, height=0.5, color='salmon')
    axes[1].set_title('Model Frequency', fontsize=18)
    axes[1].set_xlabel('Frequency')
    axes[1].invert_yaxis()
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(names_ord)  # ensure labels appear
    axes[1].set_xlim(0, 1)
    axes[1].tick_params(axis='x', labelsize=12)
    axes[1].grid(axis='x', linestyle=':', color='gray', linewidth=0.7)
    # Annotate values
    for bar in bars2:
        width = bar.get_width()
        y_pos = bar.get_y() + bar.get_height()/2
        axes[1].text(width + 0.02, y_pos, f'{width:.2f}', va='center', fontsize=11)
    
    plt.tight_layout()
    return fig, axes

def model_recovery_plot(df, metric="BIC", order=None, cmap="Oranges"):
    sns.set(style="white", font_scale=1)

    # 1) winner per (sample, subjID, sim_model)
    idx = df.groupby(["sample", "subjID", "sim_model"])[metric].idxmin()
    winners = df.loc[idx, ["subjID", "sim_model", "fit_model"]].copy()

    # 2) counts and proportions
    counts = pd.crosstab(winners["sim_model"], winners["fit_model"])

    # If an explicit order is provided, reindex rows and columns to that order
    if order is None:
        rlabels = counts.index.tolist()
        clabels = counts.columns.tolist()
    else:
        rlabels = list(order)  # y-axis (True / sim_model)
        clabels = list(order)  # x-axis (Recovered / fit_model)
        counts = counts.reindex(index=rlabels, columns=clabels, fill_value=0)

    props = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    # 3) plot
    fig_w = max(6, 1 * len(clabels))
    fig_h = max(5, 1 * len(rlabels))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    mat = props.values
    im = ax.imshow(mat, aspect="equal", cmap=cmap, vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proportion recovered", rotation=90)

    ax.set_xticks(np.arange(len(clabels)))
    ax.set_yticks(np.arange(len(rlabels)))
    ax.set_xticklabels(clabels, rotation=45, ha="right")
    ax.set_yticklabels(rlabels)
    ax.set_xlabel("Recovered")
    ax.set_ylabel("True")
    ax.set_title(f"Model Recovery (winner = min {metric})")

    # annotate with % and counts
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            cnt = counts.iloc[i, j]
            pct = mat[i, j]
            ax.text(j, i, f"{pct:.0%}\n({cnt})", ha="center", va="center")

    # diagonal accuracy
    diag = counts.reindex(index=rlabels, columns=clabels, fill_value=0).to_numpy()
    diag_count = int(np.trace(diag))
    total = int(counts.to_numpy().sum())
    acc = diag_count / total if total else float("nan")
    print(f"Overall recovery (diagonal) = {acc:.3f}  [{diag_count}/{total}]")

    plt.tight_layout()
    return counts, props, fig

def prepare_ppc_data(best_models, merged, *,
                     score_col="teaching_score",
                     model_col_best="model",
                     sim_model_col="sim_model",
                     subj_col="subjID",
                     palette_name="Set1"):
    
    # Order by best-fit frequency; include any models that appear in merged
    counts = best_models[model_col_best].value_counts()
    model_order = counts.index.tolist()

    per_subj = (
        merged.groupby([subj_col, sim_model_col], as_index=False)[score_col]
              .mean()
              .rename(columns={score_col: "mean_score"})
    )
    missing = [m for m in per_subj[sim_model_col].unique() if m not in model_order]
    model_order += sorted(missing)

    # Palette keyed by model name
    palette_list = sns.color_palette(palette_name, n_colors=len(model_order))
    model_palette = dict(zip(model_order, palette_list))

    # Best-fit counts (fill zeros for models only in per_subj)
    counts_df = (
        counts.reindex(model_order, fill_value=0)
              .rename_axis('model').reset_index(name='count')
    )

    # Categoricals help consistent ordering in plots later
    per_subj[sim_model_col] = pd.Categorical(per_subj[sim_model_col],
                                             categories=model_order, ordered=True)

    return per_subj, counts_df, model_order, model_palette


def plot_bestfit_ppc_corr(per_subj, counts_df, model_order, model_palette,
                          real_df=None, real_score_col="real_score",
                          subj_col="subjID", sim_model_col="sim_model",
                          bins=30, iter_count=75, figsize=(14,10)):
    """
    2x2 figure:
      [0,0] barplot of best fits
      [0,1] stacked histogram of per-subject simulated means
      [1,0] correlation scatterplot (if real_df given)
      [1,1] legend only
    """

    fig, axes = plt.subplots(2, 2, figsize=figsize,
                             gridspec_kw={'width_ratios':[1,1.2],
                                          'height_ratios':[1,1]})
    sns.set(style="white")

    # ---------- subplot [0,0]: barplot ----------
    sns.barplot(
        data=counts_df, y='model', x='count',
        hue='model', dodge=False,
        order=model_order, palette=model_palette,
        edgecolor='black', legend=False, ax=axes[0,0]
    )
    axes[0,0].set_ylabel('')
    axes[0,0].set_xlabel('Number Best Fit', fontsize=14)
    axes[0,0].tick_params(axis="x", labelsize=12)
    axes[0,0].tick_params(axis="y", labelsize=12)
    axes[0,0].grid(axis="x", linestyle="--", alpha=0.6)

    # ---------- subplot [0,1]: stacked histogram ----------
    vmin, vmax = per_subj["mean_score"].min(), per_subj["mean_score"].max()
    bin_edges = np.linspace(vmin, vmax, bins)
    sns.histplot(
        data=per_subj, x="mean_score",
        hue=sim_model_col, hue_order=model_order,
        bins=bin_edges, multiple="stack",
        edgecolor="black", alpha=0.85,
        palette=model_palette, legend=False, ax=axes[0,1]
    )
    axes[0,1].set_xlabel("Average Teaching Score", fontsize=14)
    axes[0,1].set_ylabel("Simulated Participants", fontsize=14)
    axes[0,1].tick_params(axis="x", labelsize=12)
    axes[0,1].tick_params(axis="y", labelsize=12)
    axes[0,1].grid(axis="y", linestyle="--", alpha=0.6)
    axes[0,1].grid(axis="x", visible=False)

    # ---------- subplot [1,0]: correlation ----------
    if real_df is not None:
        if real_df.groupby(subj_col).size().max() > 1:
            real_df = real_df.groupby(subj_col, as_index=False)[real_score_col].mean()

        joined = per_subj.merge(real_df[[subj_col, real_score_col]], on=subj_col, how="inner")

        sns.scatterplot(
            data=joined, x="mean_score", y=real_score_col,
            hue=sim_model_col, palette=model_palette,
            ax=axes[1,0], alpha=0.8,
            edgecolor="black", linewidth=0.5,
            legend=False
        )
        sns.regplot(data=joined, x="mean_score", y=real_score_col,color= "black",
                    scatter=False, ci=95, line_kws={"lw":2}, ax=axes[1,0])

        # labels
        axes[1,0].set_xlabel("Simulated Avg Teaching Score", fontsize=14)
        axes[1,0].set_ylabel("Real Avg Teaching Score", fontsize=14)
        axes[1,0].tick_params(axis="x", labelsize=12)
        axes[1,0].tick_params(axis="y", labelsize=12)
        axes[1,0].grid(False)

        # add gray diagonal line (y=x)
        lims = [
            min(axes[1,0].get_xlim()[0], axes[1,0].get_ylim()[0]),
            max(axes[1,0].get_xlim()[1], axes[1,0].get_ylim()[1])
        ]
        axes[1,0].plot(lims, lims, ls="--", color="gray")
        axes[1,0].set_xlim(lims)
        axes[1,0].set_ylim(lims)

        # correlation stats
        r, p = pearsonr(joined["mean_score"], joined[real_score_col])
        axes[1,0].text(0.05, 0.95,
                       f"r = {r:.2f}, p = {p:.3g}",
                       transform=axes[1,0].transAxes,
                       ha="left", va="top", fontsize=12,
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.5))
    else:
        axes[1,0].axis("off")

    # ---------- subplot [1,1]: legend only ----------
    axes[1,1].axis("off")
    handles = [plt.Line2D([0],[0], marker='o', linestyle='',
                          markerfacecolor=model_palette[m],
                          markeredgecolor="black", markeredgewidth=0.5,
                          label=m)
               for m in model_order]
    axes[1,1].legend(
        handles=handles, labels=model_order,
        title="Model", loc="center",
        fontsize=14, title_fontsize=16, markerscale=2.0
    )

    # ---------- big figure title ----------
    fig.suptitle(f"Posterior Predictive Check with {iter_count} iterations",
                 fontsize=18, fontweight="bold")

    sns.despine(fig)
    plt.tight_layout(rect=[0,0,1,0.95])
    return fig, axes

def model_recovery_fitting(posterior_sim, models,exp):

    model_fits = []
    subj_list = posterior_sim['subjID'].unique().tolist()
    samples = posterior_sim['iteration'].unique().tolist()
    total = len(samples) * len(subj_list)
    pbar = tqdm(total=total, desc="samples×subj")
    for s in samples:
        posterior_sim_s= posterior_sim[posterior_sim['iteration']== s]
        for subj in subj_list:
            trials = posterior_sim_s[posterior_sim_s['subjID']== subj]

            for sim_model in trials.sim_model.unique():
                trial = trials[trials.sim_model == sim_model].reset_index(drop=True)
                for model_name, feature_names in models.items():
                    feature_weights = [0]*(len(feature_names))
                    nll_func = create_nll_features(trial, feature_names, "choice", fit_randchoose=False)
                    min_res = minimize(nll_func,feature_weights,method = 'SLSQP')
                    
                    betas = list(np.atleast_1d(min_res.x))
                    res = {
                        "sample": s,
                        "subjID": subj,
                        "sim_model": sim_model,     # the model that generated the data (if sim)
                        "fit_model": model_name,      # the model used to fit
                        "nparam": len(betas),
                        "nll": float(min_res.fun),
                        "BIC": np.log(5)*len(betas) + 2*float(min_res.fun)
                    }
                    row.update({f"beta{i+1}": b for i, b in enumerate(betas)})

                    if exp >=2:
                        res['group'] = list(trials.group)[0]
                        if exp == 3:
                            res['condition'] = list(trials.condition)[0]
                            res['subjectId'] = list(trials.subjectId)[0]
                            
                    model_fits.append(res)
                    
                    pbar.update(1)
        df_fits= pd.DataFrame(model_fits)
        
    return df_fits

def vb_from_long(plotdf_long, alpha0_std=1.0, alpha0_null=1000.0):
    df_BIC = plotdf_long.pivot(index='subjID', columns='model', values='BIC').reset_index()
    model_cols, r_std, alpha_std, u_std, log_ev, _ = vb_random_effects_model_comparison(df_BIC, alpha0_value=alpha0_std)
    _, _, alpha_null, u_null, _, _ = vb_random_effects_model_comparison(df_BIC, alpha0_value=alpha0_null)
    phi = compute_exceedance(alpha_std)
    return model_cols, r_std, phi

def plot_vb_exp(plotdf, exp=1, *, group_col='group', condition_col=None,
                model_order=None, figsize_per_row=(10,4), bar_height=0.7,
                row_title_fontsize=14, row_title_pad_pts=4,
                hide_top_xlabels=False):           # ⬅️ NEW ARG
    """
    Exp 1: single figure (all subjects, no groups).
    Exp 2: one row per group (Exceedance | Frequency).
    Exp 3: one row per group×condition (Exceedance | Frequency).

    Parameters
    ----------
    hide_top_xlabels : bool, default False
        If True, removes the x-axis label and x tick labels from the top row
        of subplots (both panels). Titles remain.
    """
    # decide rows
    if exp == 1:
        groups = [("", plotdf[['subjID','model','BIC']].copy())]
    elif exp == 2:
        groups = [(f"Test Trials — {str(g)} Training",
                   gdf[['subjID','model','BIC']].copy())
                  for g, gdf in plotdf.groupby(group_col, dropna=False)]
    elif exp == 3:
        if condition_col is None:
            raise ValueError("exp=3 requires condition_col.")
        groups = [(f"Test Trials — {str(g)} Training — {str(c)} Condition",
                   gdf[['subjID','model','BIC']].copy())
                  for (g, c), gdf in plotdf.groupby([group_col, condition_col], dropna=False)]
    else:
        raise ValueError("exp must be 1, 2, or 3")

    if model_order is None:
        model_order = list(pd.unique(plotdf['model']))

    n = len(groups)
    fig, axes = plt.subplots(n, 2, figsize=(figsize_per_row[0], figsize_per_row[1]*n), sharey=True)
    if n == 1:
        axes = np.atleast_2d(axes)

    # we'll place row titles AFTER drawing, using tight bboxes
    _row_headers = []  # list of (ax_left, ax_right, title)

    for r, (row_title, df_long) in enumerate(groups):
        if df_long.empty:
            axes[r,0].axis('off'); axes[r,1].axis('off'); continue

        model_cols, r_std, phi = vb_from_long(df_long)
        freq = pd.Series(r_std, index=model_cols).reindex(model_order, fill_value=0.0).values
        exph = pd.Series(phi,   index=model_cols).reindex(model_order, fill_value=0.0).values
        y = np.arange(len(model_order))

        # left: exceedance
        ax0 = axes[r, 0]
        bars1 = ax0.barh(y, exph, height=bar_height, color='skyblue')
        ax0.set_title("Exceedance Probability", fontsize=12)
        ax0.set_xlabel('Probability'); ax0.set_xlim(0, 1)
        ax0.set_yticks(y); ax0.set_yticklabels(model_order); ax0.invert_yaxis()
        ax0.grid(axis='x', linestyle=':', color='gray', linewidth=0.7)
        for b in bars1:
            w = b.get_width(); yy = b.get_y() + b.get_height()/2
            if w > 0.9:
                ax0.text(min(w - 0.2, 0.98), yy, f"{w:.2f}", va='center', fontsize=11)
            else:
                ax0.text(min(w + 0.02, 0.98), yy, f"{w:.2f}", va='center', fontsize=11)

        # right: model frequency
        ax1 = axes[r, 1]
        bars2 = ax1.barh(y, freq, height=bar_height, color='salmon')
        ax1.set_title("Model Frequency", fontsize=12)
        ax1.set_xlabel('Frequency'); ax1.set_xlim(0, 1)
        ax1.set_yticks(y); ax1.set_yticklabels(model_order); ax1.invert_yaxis()
        ax1.grid(axis='x', linestyle=':', color='gray', linewidth=0.7)
        for b in bars2:
            w = b.get_width(); yy = b.get_y() + b.get_height()/2
            if w > 0.9:
                ax1.text(min(w - 0.2, 0.98), yy, f"{w:.2f}", va='center', fontsize=11)
            else:
                ax1.text(min(w + 0.02, 0.98), yy, f"{w:.2f}", va='center', fontsize=11)

        if row_title:
            _row_headers.append((ax0, ax1, row_title))

    # do normal spacing first (tune these if you want more/less space between rows)
    fig.subplots_adjust(left=0.25, right=0.95, top=0.92, bottom=0.1, hspace=0.6, wspace=0.4)

    # now place row titles dynamically just above each row’s tight bbox
    fig.canvas.draw()  # ensure tightbbox is available
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    pad_fig_y = row_title_pad_pts / fig.bbox.height  # convert pad in points to figure fraction

    for axL, axR, title in _row_headers:
        bbL = axL.get_tightbbox(renderer)
        bbR = axR.get_tightbbox(renderer)
        bb  = Bbox.union([bbL, bbR])  # union of both panels in the row

        # to figure coords
        (x0, y0) = inv.transform((bb.x0, bb.y0))
        (x1, y1) = inv.transform((bb.x1, bb.y1))
        xmid = 0.5 * (x0 + x1)
        ytxt = y1 + pad_fig_y

        fig.text(xmid, ytxt, title, ha="center", va="bottom", #fontweight="bold",
                fontsize=row_title_fontsize)

    if hide_top_xlabels:
        for ax in np.atleast_2d(axes)[0, :].ravel():
            ax.set_xlabel(None)                  # remove x-axis label text
            ax.xaxis.label.set_visible(False)    # ensure label itself is hidden
            ax.tick_params(axis='x', labelbottom=False)  # hide tick labels (keeps ticks)
            # If you also want to hide the tick marks themselves, uncomment:
            # ax.tick_params(axis='x', bottom=False)

    return fig, axes


def run_ppc_for_ids(best_models_df, merged_df, subj_ids, *,
                     real_df=None, real_score_col="edge_normresp_U_obm",
                     title_prefix=None, figsize=(10,8)):
    bm_sub  = best_models_df[best_models_df['subjID'].isin(subj_ids)]
    mrg_sub = merged_df[merged_df['subjID'].isin(subj_ids)]
    real_sub = None if real_df is None else real_df[real_df['subjID'].isin(subj_ids)]

    per_subj, counts_df, model_order, model_palette = prepare_ppc_data(bm_sub, mrg_sub)

    iters = mrg_sub['iteration'].nunique() if 'iteration' in mrg_sub.columns else None
    fig, axes = plot_bestfit_ppc_corr(
        per_subj, counts_df, model_order, model_palette,
        real_df=real_sub, real_score_col=real_score_col,
        iter_count=iters, figsize=figsize
    )

    if title_prefix:
        # prepend your context to the suptitle the plot function already sets
        st = fig._suptitle
        if st:
            st.set_text(f"{title_prefix} \n {st.get_text()}")
        else:
            fig.suptitle(title_prefix, fontsize=18, fontweight="bold")
        plt.tight_layout(rect=[0,0,1,0.95])

    return fig, axes


def extract_true_param_long(true_df, model_col="model", subj_col="subjID", minres_col="min_res"):
    """
    Build a long table of true parameters with columns:
      ['subjID','sim_model','param_name','true_value']
    Pulls params from `min_res['x']` when available; otherwise, infers from columns
    whose names match the model (e.g., 'Level,Reward' -> ['Level','Reward']).
    """
    rows = []
    for _, r in true_df.iterrows():
        m = r[model_col]
        subj = r[subj_col]

        # parse min_res safely if it's a string
        mr = r.get(minres_col, None)
        if isinstance(mr, str):
            try:
                mr = ast.literal_eval(mr)
            except Exception:
                pass

        if isinstance(mr, dict) and "x" in mr:
            params = list(mr["x"])
        else:
            # fallback: read from columns named after the model
            if "," in m:
                parts = [p.strip() for p in m.split(",")]
                params = [r.get(parts[0], np.nan), r.get(parts[1], np.nan)]
            else:
                params = [r.get(m, np.nan)]

        # emit one row per parameter as beta1, beta2, ...
        for i, val in enumerate(params, start=1):
            if pd.isna(val):
                continue
            rows.append({
                subj_col: subj,
                "sim_model": m,
                "param_name": f"beta{i}",
                "true_value": float(val)
            })

    return pd.DataFrame(rows)


def extract_est_param_long(sim_fits_all,
                           subj_col="subjID",
                           sim_model_col="sim_model",
                           fit_model_col="fit_model",
                           extra_id_cols=("sample",)):
    """
    Keep within-model fits (fit_model == sim_model) and melt beta columns to long:
      ['sample','subjID','sim_model','param_name','est_value']
    """
    df = sim_fits_all.loc[sim_fits_all[fit_model_col] == sim_fits_all[sim_model_col]].copy()
    id_vars = [c for c in extra_id_cols if c in df.columns] + [subj_col, sim_model_col]

    beta_cols = [c for c in df.columns if c.startswith("beta")]
    est_long = df.melt(id_vars=id_vars, value_vars=beta_cols,
                       var_name="param_name", value_name="est_value")
    est_long = est_long.dropna(subset=["est_value"])
    return est_long


def build_param_recovery_long(true_df, sim_fits_all,
                              subj_col="subjID", sim_model_col="sim_model"):
    """
    Returns a tidy table ready to plot/join stats:
      ['sample','subjID','sim_model','param_name','true_value','est_value']
    """
    truth_long = extract_true_param_long(true_df, subj_col=subj_col)
    est_long   = extract_est_param_long(sim_fits_all, subj_col=subj_col, sim_model_col=sim_model_col)

    # join on subjID + sim_model + param_name (truth usually has no 'sample')
    param_long = est_long.merge(
        truth_long, on=[subj_col, sim_model_col, "param_name"], how="inner"
    )
    return param_long




def plot_param_recovery_grid(param_long,
                             model_order=None,
                             ncols=3,
                             pad=0.07,                 # padding fraction per subplot
                             figsize_per=(5, 4),
                             param_palette_name="colorblind",
                             clip_quantiles=None,       # e.g., (0.01, 0.99) to tame outliers
                             model_title_map=None):     # dict: raw_name -> display_name
    """
    param_long columns expected:
      ['subjID','sim_model','param_name','true_value','est_value']

    Creates a grid of subplots: one subplot per sim_model, overlaying each parameter.
    X-axis: Recovered (est_value), Y-axis: True (true_value).
    """
    sns.set(style="white")
    models = model_order or sorted(param_long['sim_model'].unique(), key=str)
    params = sorted(param_long['param_name'].unique(), key=str)

    # consistent colors/markers per parameter
    colors  = sns.color_palette(param_palette_name, n_colors=len(params))
    markers = ['o','s','^','D','P','X']
    param_palette = dict(zip(params, colors))

    n = len(models)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per[0]*ncols, figsize_per[1]*nrows),
                             squeeze=False)

    for idx, model in enumerate(models):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        d = param_long[param_long['sim_model'] == model]
        if d.empty:
            ax.axis("off")
            continue

        # per-model limits (optionally robust via quantiles), with padding
        if clip_quantiles:
            qlo_t, qhi_t = np.quantile(d['true_value'], clip_quantiles)
            qlo_e, qhi_e = np.quantile(d['est_value'],  clip_quantiles)
            lo = min(qlo_t, qlo_e); hi = max(qhi_t, qhi_e)
        else:
            lo = min(d['true_value'].min(), d['est_value'].min())
            hi = max(d['true_value'].max(), d['est_value'].max())
        span = max(1e-9, hi - lo)
        lo_p, hi_p = lo - pad*span, hi + pad*span

        # plot per-parameter
        stats_lines = []
        for i, pname in enumerate(params):
            g = d[d['param_name'] == pname]
            if g.empty:
                continue

            # NOTE: x = recovered, y = true
            x = g['est_value'].to_numpy()
            y = g['true_value'].to_numpy()

            ax.scatter(x, y,
                       s=35,
                       marker=markers[i % len(markers)],
                       facecolor=param_palette[pname],
                       edgecolor="black", linewidth=0.6,
                       alpha=0.9, label=pname, zorder=3)

            if len(g) >= 2:
                # OLS of y ~ x (calibration line)
                m, b = np.polyfit(x, y, 1)
                xs = np.linspace(lo_p, hi_p, 200)
                ax.plot(xs, m*xs + b, color=param_palette[pname], lw=2, zorder=2)

                # r, p
                r_val, p_val = pearsonr(x, y)
                p_txt = "p<0.001" if (p_val is not None and p_val < 1e-3) else f"p={p_val:.3f}"
                stats_lines.append(f"{pname}: r={r_val:.2f}")

        # identity line (perfect recovery: x == y)
        ax.plot([lo_p, hi_p], [lo_p, hi_p], ls=":", color="gray", zorder=1)

        ax.set_xlim(lo_p, hi_p)
        ax.set_ylim(lo_p, hi_p)

        # axis labels (only left/bottom to reduce clutter)
        if c == 0:
            ax.set_ylabel("True parameter")
        else:
            ax.set_ylabel("")
        if r == nrows - 1:
            ax.set_xlabel("Recovered parameter")
        else:
            ax.set_xlabel("")

        # title with optional mapping
        display_title = model_title_map.get(model, model) if model_title_map else model
        ax.set_title(display_title)
        ax.grid(False)

        # stats box
        if stats_lines:
            ax.text(0.02, 0.98, "\n".join(stats_lines),
                    transform=ax.transAxes, ha="left", va="top", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.5))

    # hide unused axes
    for j in range(n, nrows*ncols):
        rr, cc = divmod(j, ncols)
        axes[rr, cc].axis("off")

    # common legend (parameters) on the right
    handles = [
        plt.Line2D([], [], marker=markers[i % len(markers)], linestyle="",
                   markerfacecolor=param_palette[p], markeredgecolor="black",
                   markeredgewidth=0.6, label=p)
        for i, p in enumerate(params)
    ]
    fig.legend(handles=handles, title="Parameter", loc="center right",
               frameon=False, fontsize=12, title_fontsize=13)

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    return fig, axes


def _mad_mask(x, thresh=4.0):
    """Keep points within |robust z| <= thresh using MAD."""
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0 or not np.isfinite(mad):
        return None  # fall back to IQR upstream
    z = 0.6745 * (x - med) / mad  # approx z-score
    return np.isfinite(z) & (np.abs(z) <= thresh)

def _iqr_mask(x, k=3.0):  # k=3.0 = 'extreme' Tukey fence
    x = np.asarray(x, float)
    q1, q3 = np.nanpercentile(x, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - k*iqr, q3 + k*iqr
    return np.isfinite(x) & (x >= lo) & (x <= hi)

def filter_param_outliers(param_long,
                          groupby=("sim_model","param_name"),
                          method="mad", mad_thresh=4.0, iqr_k=3.0,
                          apply_to=("true_value","est_value")):
    """
    Returns a filtered copy where points are dropped if they are outliers
    on *either* true or est within each (sim_model, param_name) group.
    """
    kept = []
    for _, g in param_long.groupby(list(groupby), dropna=False):
        if g.empty:
            continue
        mask = np.ones(len(g), dtype=bool)
        for col in apply_to:
            if method == "mad":
                m = _mad_mask(g[col].values, thresh=mad_thresh)
                if m is None:
                    m = _iqr_mask(g[col].values, k=iqr_k)  # fallback
            elif method == "iqr":
                m = _iqr_mask(g[col].values, k=iqr_k)
            else:
                raise ValueError("method must be 'mad' or 'iqr'")
            mask &= m
        kept.append(g.loc[mask])
    return pd.concat(kept, ignore_index=True) if kept else param_long.iloc[0:0].copy()
