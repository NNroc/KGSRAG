"""
Hyperparameter sensitivity analysis figure for KGSRAG on BioASQ-QA.
2×2 broken y-axis: upper band for Yes/No Acc, lower band for Factoid MRR & List F1.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from matplotlib.patches import FancyBboxPatch

# ============================================================
# Data (percentage)
# ============================================================
data = {
    'T_min': {
        'x': [0.1, 0.2, 0.3],
        'Yes/No Acc':  [81.37, 81.37, 81.37],
        'Factoid MRR': [38.82, 33.24, 35.69],
        'List F1':     [32.92, 32.21, 33.86],
        'default_x': 0.1,
        'xlabel': r'$T_{\min}$',
    },
    'T_max': {
        'x': [0.4, 0.5, 0.6],
        'Yes/No Acc':  [81.37, 81.37, 82.35],
        'Factoid MRR': [36.47, 38.82, 34.71],
        'List F1':     [33.40, 32.92, 31.84],
        'default_x': 0.5,
        'xlabel': r'$T_{\max}$',
    },
    'alpha': {
        'x': [3, 5, 10, 15],
        'Yes/No Acc':  [80.39, 80.39, 81.37, 81.37],
        'Factoid MRR': [36.27, 34.51, 38.82, 37.06],
        'List F1':     [33.96, 31.57, 32.92, 32.47],
        'default_x': 10,
        'xlabel': r'$\alpha$',
    },
    'beta': {
        'x': [3, 5, 7, 10],
        'Yes/No Acc':  [80.39, 81.37, 82.35, 81.37],
        'Factoid MRR': [35.29, 38.82, 35.88, 34.12],
        'List F1':     [32.61, 32.92, 31.82, 30.76],
        'default_x': 5,
        'xlabel': r'$\beta$',
    },
}

panel_labels = {
    'T_min': r'(a) $T_{\min}$',
    'T_max': r'(b) $T_{\max}$',
    'alpha': r'(c) $\alpha$',
    'beta':  r'(d) $\beta$',
}

# ============================================================
# Style — muted colors, equal visual weight
# ============================================================
styles = {
    'Yes/No Acc':  dict(color='#C0392B', marker='o', linestyle='-',  linewidth=1.8, markersize=6.5, label='Yes/No Acc'),
    'Factoid MRR': dict(color='#2471A3', marker='s', linestyle='--', linewidth=1.8, markersize=6.5, label='Factoid MRR'),
    'List F1':     dict(color='#1E8449', marker='^', linestyle='-.', linewidth=1.8, markersize=6.5, label='List F1'),
}

# Broken axis ranges
upper_ylim = (78, 84)
lower_ylim = (29, 42)

# ============================================================
# Helper: draw one panel with broken y-axis
# ============================================================
def draw_panel(ax_upper, ax_lower, param_name, d):
    x = d['x']
    default_x = d['default_x']
    xlabel = d['xlabel']

    for metric, s in styles.items():
        y = d[metric]
        ax_upper.plot(x, y, **s, zorder=3)
        ax_lower.plot(x, y, **s, zorder=3)

    # --- Upper axis (Yes/No Acc range) ---
    ax_upper.set_ylim(*upper_ylim)
    ax_upper.set_yticks([79, 81, 83])
    ax_upper.tick_params(axis='x', labelbottom=False)
    ax_upper.tick_params(axis='y', labelsize=9)
    ax_upper.grid(True, axis='y', alpha=0.25, linestyle='-')
    ax_upper.spines['top'].set_visible(False)
    ax_upper.spines['right'].set_visible(False)
    ax_upper.spines['bottom'].set_visible(False)
    ax_upper.axvline(x=default_x, color='#cccccc', linestyle='--', linewidth=0.8, zorder=1)

    # --- Lower axis (Factoid / List range) ---
    ax_lower.set_ylim(*lower_ylim)
    ax_lower.set_yticks([30, 33, 36, 39, 42])
    ax_lower.set_xticks(x)
    ax_lower.set_xticklabels([str(v) for v in x], fontsize=9)
    ax_lower.set_xlabel(xlabel, fontsize=11)
    ax_lower.tick_params(axis='y', labelsize=9)
    ax_lower.grid(True, axis='y', alpha=0.25, linestyle='-')
    ax_lower.spines['top'].set_visible(False)
    ax_lower.spines['right'].set_visible(False)
    ax_lower.spines['left'].set_visible(False)
    ax_lower.tick_params(axis='y', which='both', length=0)
    ax_lower.axvline(x=default_x, color='#cccccc', linestyle='--', linewidth=0.8, zorder=1)

    # --- Break symbol (diagonal marks on left spine) ---
    d_top = upper_ylim[1]
    d_bot = lower_ylim[0]
    # Use same transform for both axes (they share the same figure y-positioning)
    for ax in [ax_upper, ax_lower]:
        # Small diagonal slashes at the break
        pass
    # Draw break marks on upper axis bottom
    kwargs = dict(transform=ax_upper.transAxes, color='0.3', clip_on=False, linewidth=0.8)
    ax_upper.plot((-0.02, 0.02), (0, 0), **kwargs)        # top of lower break
    ax_upper.plot((-0.02, 0.02), (-0.02, 0.02), **kwargs) # slant
    # Draw break marks on lower axis top
    kwargs2 = dict(transform=ax_lower.transAxes, color='0.3', clip_on=False, linewidth=0.8)
    ax_lower.plot((-0.02, 0.02), (1, 1), **kwargs2)
    ax_lower.plot((-0.02, 0.02), (1.02, 1.02), **kwargs2)


# ============================================================
# Figure: 2×2
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(9, 7.5), dpi=300,
                         gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.45, 'wspace': 0.35})

param_list = ['T_min', 'T_max', 'alpha', 'beta']
positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

for (row, col), param_name in zip(positions, param_list):
    d = data[param_name]
    # Each panel gets two sub-axes: upper and lower
    # We use gridspec to split each cell
    gs = axes[row, col].get_subplotspec()
    # Create twin axes within the same grid cell
    ax_upper = fig.add_subplot(gs)
    # Use a portion of the same area for the lower axis
    ax_lower = fig.add_subplot(gs)

    # Remove the original axes
    axes[row, col].set_visible(False)

    # Position upper and lower within the cell using set_position
    # Upper: top 35% of cell
    # Lower: bottom 65% of cell
    bbox = gs.get_position(fig)
    cell_x = bbox.x0
    cell_y = bbox.y0
    cell_w = bbox.width
    cell_h = bbox.height

    gap = 0.015  # gap between upper and lower as fraction of figure height

    upper_h = cell_h * 0.30
    lower_h = cell_h * 0.65
    lower_y = cell_y
    upper_y = cell_y + lower_h + gap * fig.get_size_inches()[1] / fig.dpi * 72

    ax_upper.set_position([cell_x, upper_y, cell_w, upper_h])
    ax_lower.set_position([cell_x, lower_y, cell_w, lower_h])

    draw_panel(ax_upper, ax_lower, param_name, d)

    # Panel title above upper axis
    ax_upper.set_title(panel_labels[param_name], fontsize=12, fontweight='bold', pad=6)

# Y-axis label only on left panels
axes[0, 0].set_visible(True)
axes[0, 0].axis('off')
axes[0, 0].set_ylabel('')  # placeholder
axes[1, 0].set_visible(True)
axes[1, 0].axis('off')

# Add a shared y-label using a dummy axes
fig.text(0.02, 0.5, 'Performance (%)', va='center', rotation='vertical', fontsize=12)

# Shared legend at bottom
handles = [Line2D([0], [0], **{k: v for k, v in s.items() if k != 'label'})
           for s in [styles['Yes/No Acc'], styles['Factoid MRR'], styles['List F1']]]
labels = ['Yes/No Acc', 'Factoid MRR', 'List F1']
fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=10,
           frameon=False, bbox_to_anchor=(0.52, -0.01),
           handletextpad=0.4, columnspacing=1.5)

plt.savefig('/data2/npy/projects/KGSRAG/figure/hyperparameter.pdf',
            bbox_inches='tight', pad_inches=0.15)
plt.savefig('/data2/npy/projects/KGSRAG/figure/hyperparameter.png',
            bbox_inches='tight', pad_inches=0.15)
print("Saved: figure/hyperparameter.pdf")
print("Saved: figure/hyperparameter.png")
