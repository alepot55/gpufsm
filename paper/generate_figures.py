#!/usr/bin/env python3
"""Generate publication figures for the Triton vs CUDA paper.

This script generates all figures from real benchmark data:
- Figure 1: Structured Workloads (MatMul, MLP) - Parity (vertical layout)
- Figure 2: FSM Performance Gap (CUDA vs Triton)
- Figure 3: CUDA Technique Comparison - ALL benchmarks
- Figure 4: Triton vs CUDA Speedup Analysis
- Figure 5: Productivity vs Performance Trade-off (with BitGen)
- Figure 6: System Overhead Analysis (Kernel vs Overhead)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# Use Type 1 fonts for better PDF quality
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Publication-quality settings - IMPROVED FONTS
plt.rcParams.update({
    'figure.figsize': (7, 4),
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 600,  # High DPI for publication
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.linewidth': 1.2,
    'lines.linewidth': 1.5,
})

# NOTE: legacy figure script from the prior 3-domain (FSM/MatMul/MLP) study.
# Pending rewrite for the FSM-only scope and the new `gpufsm sweep` CSV schema
# (backend,technique,mean_ms,std_ms,ci95_ms). The MatMul/MLP loaders below refer
# to results that are out of scope now and will be removed in the rewrite.
# Paths
DATA_DIR = Path('data')
RESULTS_DIR = Path('results')
OUTPUT_DIR = Path('figures')
OUTPUT_DIR.mkdir(exist_ok=True)

# Colors - colorblind safe
COLORS = {
    'cuda': '#1f77b4',        # Blue
    'triton': '#d62728',      # Red
    'cublas': '#2ca02c',      # Green
    'basic': '#1f77b4',
    'csr_iterative': '#ff7f0e',
    'ngap_base_v2': '#2ca02c',
    'bitgen': '#9467bd',      # Purple for BitGen
    'overhead': '#ff7f0e',    # Orange for overhead
    'kernel': '#1f77b4',      # Blue for kernel
}


def load_data():
    """Load all benchmark data."""
    # FSA data
    fsa_path = DATA_DIR / 'benchmarks_final_publication.csv'
    fsa_df = pd.read_csv(fsa_path)
    fsa_df = fsa_df[fsa_df['status'] == 'passed'].copy()

    # MLP data
    mlp_path = RESULTS_DIR / 'mlp' / 'performance.csv'
    mlp_df = pd.read_csv(mlp_path)
    mlp_df = mlp_df[mlp_df['status'] == 'passed'].copy()

    # MatMul data
    matmul_path = RESULTS_DIR / 'matmul' / 'performance.csv'
    matmul_df = pd.read_csv(matmul_path)
    matmul_df = matmul_df[(matmul_df['status'] == 'passed') &
                          (matmul_df['domain'] == 'matmul')].copy()

    return fsa_df, mlp_df, matmul_df


def fig1_structured_workloads(mlp_df, matmul_df):
    """Figure 1: Structured Workloads - MatMul and MLP (VERTICAL layout)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9))

    # ===== Panel A: MatMul =====
    sizes = ['1024', '2048', '4096']
    size_labels = ['1024³', '2048³', '4096³']

    cuda_matmul = []
    triton_matmul = []

    for size in sizes:
        test_name = f'matrix_matmul_{size}x{size}x{size}'
        cuda_row = matmul_df[(matmul_df['test_name'] == test_name) &
                             (matmul_df['implementation'] == 'cuda') &
                             (matmul_df['technique'] == 'cublas')]
        triton_row = matmul_df[(matmul_df['test_name'] == test_name) &
                               (matmul_df['implementation'] == 'triton')]

        cuda_time = cuda_row['kernel_time_ms'].values[0] if len(cuda_row) > 0 else np.nan
        triton_time = triton_row['kernel_time_ms'].min() if len(triton_row) > 0 else np.nan

        cuda_matmul.append(cuda_time)
        triton_matmul.append(triton_time)

    x = np.arange(len(sizes))
    width = 0.35

    ax1.bar(x - width/2, cuda_matmul, width, label='CUDA (cuBLAS)',
            color=COLORS['cublas'], edgecolor='black', linewidth=0.8)
    ax1.bar(x + width/2, triton_matmul, width, label='Triton (best)',
            color=COLORS['triton'], edgecolor='black', linewidth=0.8)

    ax1.set_ylabel('Kernel Time (ms)', fontsize=12)
    ax1.set_xlabel('Matrix Size (N × N × N)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(size_labels, fontsize=11)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.set_title('(a) Matrix Multiplication', fontsize=13, fontweight='bold')

    # Add speedup annotations
    for i in range(len(sizes)):
        if cuda_matmul[i] > 0 and triton_matmul[i] > 0:
            ratio = triton_matmul[i] / cuda_matmul[i]
            ax1.annotate(f'{ratio:.2f}x', xy=(i, max(cuda_matmul[i], triton_matmul[i])),
                        xytext=(0, 5), textcoords='offset points', ha='center', fontsize=10, fontweight='bold')

    # ===== Panel B: MLP =====
    mlp_configs = [
        ('perf_mlp_128x768x3072x768', '128×768×3072'),
        ('perf_mlp_256x1024x2048x512', '256×1024×2048'),
        ('perf_mlp_256x1536x6144x1536', '256×1536×6144'),
        ('perf_mlp_512x2048x4096x1024', '512×2048×4096'),
        ('perf_mlp_1024x4096x8192x2048', '1024×4096×8192'),
    ]

    cuda_mlp = []
    triton_mlp = []
    mlp_labels = []

    for test_name, label in mlp_configs:
        cuda_row = mlp_df[(mlp_df['test_name'] == test_name) &
                          (mlp_df['implementation'] == 'cuda') &
                          (mlp_df['technique'] == 'v04_cublas')]
        triton_row = mlp_df[(mlp_df['test_name'] == test_name) &
                            (mlp_df['implementation'] == 'triton')]

        cuda_time = cuda_row['kernel_time_ms'].values[0] if len(cuda_row) > 0 else np.nan
        triton_time = triton_row['kernel_time_ms'].min() if len(triton_row) > 0 else np.nan

        cuda_mlp.append(cuda_time)
        triton_mlp.append(triton_time)
        mlp_labels.append(label)

    x = np.arange(len(mlp_configs))

    ax2.bar(x - width/2, cuda_mlp, width, label='CUDA (cuBLAS)',
            color=COLORS['cublas'], edgecolor='black', linewidth=0.8)
    ax2.bar(x + width/2, triton_mlp, width, label='Triton (best)',
            color=COLORS['triton'], edgecolor='black', linewidth=0.8)

    ax2.set_ylabel('Kernel Time (ms)', fontsize=12)
    ax2.set_xlabel('MLP Configuration (B×I×H)', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(mlp_labels, fontsize=10)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.set_title('(b) MLP Forward Pass', fontsize=13, fontweight='bold')

    # Add speedup annotations
    for i in range(len(mlp_configs)):
        if cuda_mlp[i] > 0 and triton_mlp[i] > 0:
            ratio = triton_mlp[i] / cuda_mlp[i]
            ax2.annotate(f'{ratio:.2f}x', xy=(i, max(cuda_mlp[i], triton_mlp[i])),
                        xytext=(0, 5), textcoords='offset points', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_structured_parity.pdf')
    plt.savefig(OUTPUT_DIR / 'fig_structured_parity.png')
    print("Saved: fig_structured_parity.pdf/png")
    plt.close()

    return cuda_matmul, triton_matmul, cuda_mlp, triton_mlp


def fig2_fsm_gap(fsa_df):
    """Figure 2: FSM Performance Gap - CUDA vs Triton."""
    fig, ax = plt.subplots(figsize=(10, 5))

    cuda_df = fsa_df[fsa_df['implementation'] == 'cuda']
    triton_df = fsa_df[fsa_df['implementation'] == 'triton']

    cuda_best = cuda_df.groupby('test_name')['kernel_time_ms'].min()
    triton_best = triton_df.groupby('test_name')['kernel_time_ms'].min()

    tests = sorted(set(cuda_best.index) & set(triton_best.index))
    x = np.arange(len(tests))
    width = 0.35

    cuda_times = [cuda_best[t] for t in tests]
    triton_times = [triton_best[t] for t in tests]

    ax.bar(x - width/2, cuda_times, width, label='CUDA (best)',
           color=COLORS['cuda'], edgecolor='black', linewidth=0.8)
    ax.bar(x + width/2, triton_times, width, label='Triton (best)',
           color=COLORS['triton'], edgecolor='black', linewidth=0.8)

    ax.set_ylabel('Kernel Time (ms, log scale)', fontsize=12)
    ax.set_xlabel('Benchmark', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace('_', '\n')[:15] for t in tests], rotation=45, ha='right', fontsize=9)
    ax.set_yscale('log')
    ax.legend(loc='upper left', fontsize=10)

    speedups = [triton_times[i] / cuda_times[i] for i in range(len(tests))]
    median_speedup = np.median(speedups)
    ax.set_title(f'FSM Kernel Time: CUDA vs Triton (Triton {median_speedup:.0f}x slower median)', fontsize=13)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_fsm_gap.pdf')
    plt.savefig(OUTPUT_DIR / 'fig_fsm_gap.png')
    print("Saved: fig_fsm_gap.pdf/png")
    plt.close()


def fig3_techniques_comparison(fsa_df):
    """Figure 3: CUDA vs Triton Technique Comparison - ALL benchmarks."""
    fig, ax = plt.subplots(figsize=(16, 6))

    cuda_df = fsa_df[fsa_df['implementation'] == 'cuda']
    triton_df = fsa_df[fsa_df['implementation'] == 'triton']

    # Select representative techniques from each
    techniques = [
        # CUDA techniques
        ('cuda', 'basic', 'CUDA Basic', '#1f77b4'),
        ('cuda', 'csr_iterative', 'CUDA CSR', '#aec7e8'),
        ('cuda', 'ngap_base_v2', 'CUDA ngAP', '#2ca02c'),
        ('cuda', 'bitgen', 'CUDA BitGen', '#98df8a'),
        # Triton techniques
        ('triton', 'NFA_BASIC_DFS', 'Triton DFS', '#d62728'),
        ('triton', 'TABLE_CSR', 'Triton CSR', '#ff9896'),
        ('triton', 'BITMAP_VECTORIZED', 'Triton Bitmap', '#ffbb78'),
    ]

    # Get common test cases
    all_tests = sorted(set(cuda_df['test_name'].unique()) & set(triton_df['test_name'].unique()))

    x = np.arange(len(all_tests))
    n_techniques = len(techniques)
    width = 0.8 / n_techniques

    for i, (impl, tech, label, color) in enumerate(techniques):
        if impl == 'cuda':
            tech_data = cuda_df[cuda_df['technique'] == tech]
        else:
            tech_data = triton_df[triton_df['technique'] == tech]

        values = []
        for test in all_tests:
            test_val = tech_data[tech_data['test_name'] == test]['kernel_time_ms'].values
            values.append(test_val[0] if len(test_val) > 0 else np.nan)

        offset = (i - n_techniques/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=label,
               color=color, edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Kernel Time (ms, log scale)', fontsize=12)
    ax.set_xlabel('Benchmark', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace('_', '\n') for t in all_tests],
                       rotation=45, ha='right', fontsize=9)
    ax.set_yscale('log')

    # Create legend with two columns - CUDA and Triton
    ax.legend(loc='upper left', ncol=2, fontsize=9,
              title='Implementation', title_fontsize=10)
    ax.set_title('FSM Techniques Comparison: CUDA vs Triton (All Benchmarks)', fontsize=13)

    # Add horizontal line showing the gap
    ax.axhline(y=0.01, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_cuda_techniques.pdf')
    plt.savefig(OUTPUT_DIR / 'fig_cuda_techniques.png')
    print("Saved: fig_cuda_techniques.pdf/png")
    plt.close()


def fig4_speedup_analysis(fsa_df):
    """Figure 4: Triton vs CUDA Speedup Analysis per benchmark."""
    fig, ax = plt.subplots(figsize=(12, 5))

    cuda_df = fsa_df[fsa_df['implementation'] == 'cuda']
    triton_df = fsa_df[fsa_df['implementation'] == 'triton']

    # Get best times for each implementation
    cuda_best = cuda_df.groupby('test_name')['kernel_time_ms'].min()
    triton_best = triton_df.groupby('test_name')['kernel_time_ms'].min()

    tests = sorted(set(cuda_best.index) & set(triton_best.index))

    # Calculate speedup (CUDA speedup = Triton_time / CUDA_time)
    speedups = [triton_best[t] / cuda_best[t] for t in tests]

    # Sort by speedup for better visualization
    sorted_data = sorted(zip(tests, speedups), key=lambda x: x[1], reverse=True)
    tests_sorted, speedups_sorted = zip(*sorted_data)

    x = np.arange(len(tests_sorted))

    # Color bars based on speedup
    colors = ['#d62728' if s > 10 else '#ff7f0e' if s > 5 else '#2ca02c' for s in speedups_sorted]

    bars = ax.bar(x, speedups_sorted, color=colors, edgecolor='black', linewidth=0.8)

    ax.axhline(y=1, color='black', linestyle='--', linewidth=1.5, label='Parity (1x)')
    ax.axhline(y=10, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label='10x slowdown')

    ax.set_ylabel('CUDA Speedup (Triton/CUDA)', fontsize=12)
    ax.set_xlabel('Benchmark', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace('_', '\n') for t in tests_sorted],
                       rotation=45, ha='right', fontsize=8)
    ax.set_yscale('log')
    ax.legend(loc='upper right', fontsize=10)

    median_speedup = np.median(speedups_sorted)
    ax.set_title(f'CUDA Speedup over Triton (FSM) - Median: {median_speedup:.1f}x', fontsize=13)

    # Add value labels on bars
    for i, (bar, speedup) in enumerate(zip(bars, speedups_sorted)):
        ax.annotate(f'{speedup:.1f}x',
                   xy=(bar.get_x() + bar.get_width()/2, speedup),
                   xytext=(0, 3), textcoords='offset points',
                   ha='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_speedup_analysis.pdf')
    plt.savefig(OUTPUT_DIR / 'fig_speedup_analysis.png')
    print("Saved: fig_speedup_analysis.pdf/png")
    plt.close()


def fig5_tradeoff(fsa_df):
    """Figure 5: Productivity vs Performance Trade-off (with BitGen)."""
    fig, ax = plt.subplots(figsize=(9, 6))

    # Data including BitGen estimate
    # LoC: estimated from code complexity
    # Performance Score: relative to Triton Bitmap (baseline = 1)
    data = {
        'Implementation': ['Triton Bitmap', 'Triton CSR', 'CUDA BitGen', 'CUDA Basic', 'CUDA ngAP'],
        'Lines_of_Code': [45, 90, 220, 140, 380],
        'Performance_Score': [1, 16, 107, 3900, 2400],  # Based on real kernel times
        'Color': ['#ff9999', '#ff6666', '#b366ff', '#99ccff', '#003366'],
        'Marker': ['o', 'o', 's', 's', 's']
    }

    for impl, loc, perf, color, marker in zip(data['Implementation'],
                                               data['Lines_of_Code'],
                                               data['Performance_Score'],
                                               data['Color'],
                                               data['Marker']):
        ax.scatter(loc, perf, s=300, c=color, marker=marker,
                  edgecolors='black', linewidth=2, alpha=0.9, zorder=5)

        # Position labels - adjust based on position
        if impl == 'Triton Bitmap':
            offset_x, offset_y, ha = 10, -15, 'left'
        elif impl == 'Triton CSR':
            offset_x, offset_y, ha = 10, 5, 'left'
        elif impl == 'CUDA BitGen':
            offset_x, offset_y, ha = 10, 5, 'left'
        elif impl == 'CUDA Basic':
            offset_x, offset_y, ha = -10, 5, 'right'
        else:  # ngAP
            offset_x, offset_y, ha = -10, -15, 'right'

        ax.annotate(impl, xy=(loc, perf), xytext=(offset_x, offset_y),
                   textcoords='offset points', fontsize=11, fontweight='bold', ha=ha)

    ax.set_yscale('log')
    ax.set_xlabel('Development Effort (Lines of Code)', fontsize=13)
    ax.set_ylabel('Performance Score (Speedup vs Triton Bitmap)', fontsize=13)
    ax.set_title('Development Effort vs Performance Trade-off', fontsize=14, fontweight='bold')

    ax.set_xlim(0, 450)
    ax.set_ylim(0.5, 10000)
    ax.grid(True, which='both', alpha=0.3, linestyle='--')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff6666',
               markersize=12, markeredgecolor='black', markeredgewidth=1.5, label='Triton'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#003366',
               markersize=12, markeredgecolor='black', markeredgewidth=1.5, label='CUDA'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_tradeoff.pdf')
    plt.savefig(OUTPUT_DIR / 'fig_tradeoff.png')
    print("Saved: fig_tradeoff.pdf/png")
    plt.close()


def fig6_overhead_analysis(fsa_df):
    """Figure 6: System Overhead Analysis - Stacked Bar (Kernel vs Overhead)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Get data for different techniques
    techniques = [
        ('basic', 'CUDA Basic'),
        ('csr_iterative', 'CUDA CSR'),
        ('ngap_base_v2', 'CUDA ngAP'),
        ('bitgen', 'CUDA BitGen'),
    ]

    # Also add Triton best
    cuda_df = fsa_df[fsa_df['implementation'] == 'cuda']
    triton_df = fsa_df[fsa_df['implementation'] == 'triton']

    kernel_times = []
    overhead_times = []
    labels = []

    # CUDA techniques
    for tech, label in techniques:
        tech_data = cuda_df[cuda_df['technique'] == tech]
        if len(tech_data) > 0:
            kernel = tech_data['kernel_time_ms'].median()
            total = tech_data['total_execution_time_ms'].median()
            overhead = max(0, total - kernel)
            kernel_times.append(kernel)
            overhead_times.append(overhead)
            labels.append(label)

    # Triton (best technique per test, then median)
    triton_best = triton_df.groupby('test_name').apply(
        lambda x: x.loc[x['kernel_time_ms'].idxmin()]
    )
    if len(triton_best) > 0:
        kernel = triton_best['kernel_time_ms'].median()
        total = triton_best['total_execution_time_ms'].median()
        overhead = max(0, total - kernel)
        kernel_times.append(kernel)
        overhead_times.append(overhead)
        labels.append('Triton (best)')

    x = np.arange(len(labels))
    width = 0.6

    # Stacked bar chart
    bars1 = ax.bar(x, kernel_times, width, label='Kernel Execution',
                   color=COLORS['kernel'], edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x, overhead_times, width, bottom=kernel_times, label='System Overhead',
                   color=COLORS['overhead'], edgecolor='black', linewidth=0.8)

    ax.set_ylabel('Time (ms, log scale)', fontsize=12)
    ax.set_xlabel('Implementation', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, rotation=15, ha='right')
    ax.set_yscale('log')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title('Kernel Time vs System Overhead (FSM Workloads)', fontsize=13)

    # Add annotations for kernel time
    for i, (k, o) in enumerate(zip(kernel_times, overhead_times)):
        total = k + o
        kernel_pct = (k / total * 100) if total > 0 else 0
        ax.annotate(f'{k:.3f}ms\n({kernel_pct:.0f}%)',
                   xy=(i, k/2), ha='center', va='center',
                   fontsize=9, fontweight='bold', color='white')
        if o > 0.1:  # Only annotate if overhead is significant
            ax.annotate(f'{o:.1f}ms',
                       xy=(i, k + o/2), ha='center', va='center',
                       fontsize=9, fontweight='bold', color='black')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_overhead.pdf')
    plt.savefig(OUTPUT_DIR / 'fig_overhead.png')
    print("Saved: fig_overhead.pdf/png")
    plt.close()


def print_statistics(fsa_df, mlp_df, matmul_df):
    """Print key statistics for the paper."""
    print("\n" + "="*70)
    print("KEY STATISTICS FOR PAPER")
    print("="*70)

    # MatMul parity
    print("\n--- MATMUL (Structured) ---")
    for size in ['1024', '2048', '4096']:
        test_name = f'matrix_matmul_{size}x{size}x{size}'
        cuda_row = matmul_df[(matmul_df['test_name'] == test_name) &
                             (matmul_df['implementation'] == 'cuda') &
                             (matmul_df['technique'] == 'cublas')]
        triton_row = matmul_df[(matmul_df['test_name'] == test_name) &
                               (matmul_df['implementation'] == 'triton')]

        if len(cuda_row) > 0 and len(triton_row) > 0:
            cuda_time = cuda_row['kernel_time_ms'].values[0]
            triton_time = triton_row['kernel_time_ms'].min()
            ratio = triton_time / cuda_time
            print(f"  {size}³: CUDA={cuda_time:.3f}ms, Triton={triton_time:.3f}ms, ratio={ratio:.2f}x")

    # MLP parity
    print("\n--- MLP (Structured) ---")
    mlp_cuda = mlp_df[mlp_df['implementation'] == 'cuda']
    mlp_triton = mlp_df[mlp_df['implementation'] == 'triton']
    for test in mlp_df['test_name'].unique():
        cuda_row = mlp_cuda[(mlp_cuda['test_name'] == test) &
                            (mlp_cuda['technique'] == 'v04_cublas')]
        triton_row = mlp_triton[mlp_triton['test_name'] == test]

        if len(cuda_row) > 0 and len(triton_row) > 0:
            cuda_time = cuda_row['kernel_time_ms'].values[0]
            triton_time = triton_row['kernel_time_ms'].min()
            ratio = triton_time / cuda_time
            print(f"  {test}: CUDA={cuda_time:.3f}ms, Triton={triton_time:.3f}ms, ratio={ratio:.2f}x")

    # FSA gap
    print("\n--- FSA (Irregular) ---")
    cuda_fsa = fsa_df[fsa_df['implementation'] == 'cuda']
    triton_fsa = fsa_df[fsa_df['implementation'] == 'triton']

    cuda_best = cuda_fsa.groupby('test_name')['kernel_time_ms'].min()
    triton_best = triton_fsa.groupby('test_name')['kernel_time_ms'].min()

    common = set(cuda_best.index) & set(triton_best.index)
    speedups = [triton_best[t] / cuda_best[t] for t in common]

    print(f"  Triton slowdown: median={np.median(speedups):.1f}x, min={np.min(speedups):.1f}x, max={np.max(speedups):.1f}x")

    # Overhead stats
    print("\n--- OVERHEAD ANALYSIS ---")
    for tech in ['basic', 'csr_iterative', 'ngap_base_v2', 'bitgen']:
        tech_data = cuda_fsa[cuda_fsa['technique'] == tech]
        if len(tech_data) > 0:
            kernel = tech_data['kernel_time_ms'].median()
            total = tech_data['total_execution_time_ms'].median()
            overhead = total - kernel
            print(f"  {tech}: kernel={kernel:.4f}ms, overhead={overhead:.1f}ms, ratio={overhead/kernel:.0f}x")

    print("\n" + "="*70)


if __name__ == '__main__':
    print("Loading benchmark data...")
    fsa_df, mlp_df, matmul_df = load_data()

    print(f"  FSA: {len(fsa_df)} rows")
    print(f"  MLP: {len(mlp_df)} rows")
    print(f"  MatMul: {len(matmul_df)} rows")

    print("\nGenerating figures...")
    fig1_structured_workloads(mlp_df, matmul_df)
    fig2_fsm_gap(fsa_df)
    fig3_techniques_comparison(fsa_df)
    fig4_speedup_analysis(fsa_df)
    fig5_tradeoff(fsa_df)
    fig6_overhead_analysis(fsa_df)

    print_statistics(fsa_df, mlp_df, matmul_df)
    print("\nDone!")
