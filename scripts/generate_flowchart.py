"""
Generate Research Workflow Flowchart for Chapter 3
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def create_flowchart():
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 14)
    ax.axis('off')

    # Colors
    start_color = '#4CAF50'  # Green
    process_color = '#2196F3'  # Blue
    decision_color = '#FF9800'  # Orange
    end_color = '#F44336'  # Red
    text_color = 'white'

    # Box style
    box_style = dict(boxstyle="round,pad=0.3", facecolor=process_color, edgecolor='black', linewidth=2)
    start_style = dict(boxstyle="round,pad=0.3", facecolor=start_color, edgecolor='black', linewidth=2)
    decision_style = dict(boxstyle="round,pad=0.3", facecolor=decision_color, edgecolor='black', linewidth=2)
    end_style = dict(boxstyle="round,pad=0.3", facecolor=end_color, edgecolor='black', linewidth=2)

    # Y positions (top to bottom)
    positions = {
        'start': (6, 13),
        'define': (6, 11.5),
        'fem': (6, 10),
        'decision': (6, 8.5),
        'pristine': (3, 7),
        'corrosion': (6, 7),
        'crack': (9, 7),
        'modal': (6, 5.5),
        'extract': (6, 4),
        'dataset': (6, 2.5),
        'preprocess': (3, 1),
        'ml': (6, 1),
        'eval': (9, 1),
        'end': (6, -0.5)
    }

    # Draw boxes
    def draw_box(pos, text, style, fontsize=10):
        ax.text(pos[0], pos[1], text, ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color='white', bbox=style, wrap=True)

    draw_box(positions['start'], 'Start', start_style, 11)
    draw_box(positions['define'], 'Define Beam Parameters\n(L, b, h, f\'c, damage)', box_style)
    draw_box(positions['fem'], 'Finite Element Modeling\n(Euler-Bernoulli)', box_style)
    draw_box(positions['decision'], 'Damage Scenario?', decision_style)
    draw_box(positions['pristine'], 'Pristine\n(No damage)', box_style)
    draw_box(positions['corrosion'], 'Corrosion\n(Uniform reduction)', box_style)
    draw_box(positions['crack'], 'Crack\n(Localized damage)', box_style)
    draw_box(positions['modal'], 'Modal Analysis\n(Eigenvalue problem)', box_style)
    draw_box(positions['extract'], 'Extract Natural\nFrequencies', box_style)
    draw_box(positions['dataset'], 'Generate Dataset\n(3,000 samples)', box_style)
    draw_box(positions['preprocess'], 'Data\nPreprocessing', box_style)
    draw_box(positions['ml'], 'Machine Learning\nModels (5 algorithms)', box_style)
    draw_box(positions['eval'], 'Model\nEvaluation', box_style)
    draw_box(positions['end'], 'End', end_style, 11)

    # Draw arrows
    arrow_style = dict(arrowstyle='->', color='black', linewidth=2, mutation_scale=15)

    def draw_arrow(start, end):
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Main flow
    draw_arrow((6, 12.7), (6, 12))  # Start to Define
    draw_arrow((6, 11), (6, 10.5))  # Define to FEM
    draw_arrow((6, 9.5), (6, 9))    # FEM to Decision

    # Decision branches
    draw_arrow((4.5, 8.5), (3, 7.5))   # Decision to Pristine
    draw_arrow((6, 8), (6, 7.5))       # Decision to Corrosion
    draw_arrow((7.5, 8.5), (9, 7.5))   # Decision to Crack

    # Converge to Modal
    draw_arrow((3, 6.5), (5, 5.9))     # Pristine to Modal
    draw_arrow((6, 6.5), (6, 6))       # Corrosion to Modal
    draw_arrow((9, 6.5), (7, 5.9))     # Crack to Modal

    # Continue flow
    draw_arrow((6, 5), (6, 4.5))       # Modal to Extract
    draw_arrow((6, 3.5), (6, 3))       # Extract to Dataset

    # Final branches
    draw_arrow((4.5, 2.5), (3, 1.5))   # Dataset to Preprocess
    draw_arrow((6, 2), (6, 1.5))       # Dataset to ML
    draw_arrow((7.5, 2.5), (9, 1.5))   # Dataset to Eval

    # Converge to End
    draw_arrow((3, 0.5), (5, -0.2))    # Preprocess to End
    draw_arrow((6, 0.5), (6, 0))       # ML to End
    draw_arrow((9, 0.5), (7, -0.2))    # Eval to End

    # Title
    ax.set_title('Figure 3.1: Research Workflow', fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig('/Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/docs/figures/research_workflow.png',
                dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    print("Flowchart saved to docs/figures/research_workflow.png")
    plt.close()

if __name__ == "__main__":
    create_flowchart()
