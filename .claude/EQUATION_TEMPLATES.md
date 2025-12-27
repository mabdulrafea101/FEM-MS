# LaTeX Equation Templates for Word Documents

This file contains standardized LaTeX equation templates for use in generating Word documents (.docx).
These equations follow the format used manually in the thesis and should be used for consistency.

## Equation Format Pattern
```
f_n = \frac{1}{2\pi}\sqrt{\frac{k}{m}} \quad \text{(Eq. 1)}
```

## All Thesis Equations in LaTeX Format

### Eq. 1: Basic Frequency-Stiffness Relationship
```latex
f_n = \frac{1}{2\pi}\sqrt{\frac{k}{m}} \quad \text{(Eq. 1)}
```

### Eq. 2: Euler-Bernoulli Frequency Equation
```latex
f_n = \frac{\lambda_n^2}{2\pi L^2}\sqrt{\frac{EI}{\rho A}} \quad \text{(Eq. 2)}
```

### Eq. 3: ACI Concrete Elastic Modulus
```latex
E_c = 4700\sqrt{f'_c} \text{ MPa} \quad \text{(Eq. 3)}
```

### Eq. 4: Frequency-Stiffness Relationship
```latex
\frac{\Delta f}{f} \approx \frac{1}{2}\frac{\Delta K}{K} \quad \text{(Eq. 4)}
```

### Eq. 5: Generalized Eigenvalue Problem
```latex
[K] \{u\} = \omega^2 [M] \{u\} \quad \text{(Eq. 5)}
```

### Eq. 6: Damaged Stiffness Reduction
```latex
EI_{\text{damaged}} = EI_{\text{original}} \times (1 - \alpha) \quad \text{(Eq. 6)}
```

### Eq. 7: Natural Frequency from Angular Frequency
```latex
f = \frac{\omega}{2\pi} = \frac{\sqrt{\lambda}}{2\pi} \quad \text{(Eq. 7)}
```

### Eq. 8: Moment of Inertia (Rectangular Section)
```latex
I = \frac{bh^3}{12} \quad \text{(Eq. 8)}
```

### Eq. 9: Element Stiffness Matrix
```latex
[k]_e = \frac{EI}{L_e^3} \begin{bmatrix}
12 & 6L_e & -12 & 6L_e \\
6L_e & 4L_e^2 & -6L_e & 2L_e^2 \\
-12 & -6L_e & 12 & -6L_e \\
6L_e & 2L_e^2 & -6L_e & 4L_e^2
\end{bmatrix} \quad \text{(Eq. 9)}
```

### Eq. 10: Element Mass Matrix
```latex
[m]_e = \frac{\rho A L_e}{420} \begin{bmatrix}
156 & 22L_e & 54 & -13L_e \\
22L_e & 4L_e^2 & 13L_e & -3L_e^2 \\
54 & 13L_e & 156 & -22L_e \\
-13L_e & -3L_e^2 & -22L_e & 4L_e^2
\end{bmatrix} \quad \text{(Eq. 10)}
```

### Eq. 11: Damage Factor for Corrosion
```latex
\alpha = \min\left(1.6 \times \frac{C}{100}, 0.9\right) \quad \text{(Eq. 11)}
```

### Eq. 12: Feature Scaling (StandardScaler)
```latex
X_{\text{scaled}} = \frac{X - \mu}{\sigma} \quad \text{(Eq. 12)}
```

### Eq. 13: Frequency-Parameter Proportionality
```latex
f \propto \frac{1}{L^2}\sqrt{\frac{EI}{\rho A}} \propto \frac{h}{L^2}\sqrt{f'_c} \quad \text{(Eq. 13)}
```

### Eq. 14: Corrosion-Induced Frequency Reduction
```latex
\frac{f_{\text{corroded}}}{f_{\text{pristine}}} \approx \sqrt{1 - \alpha} = \sqrt{1 - 1.6 \times \frac{C}{100}} \quad \text{(Eq. 14)}
```

### Eq. 15: Localized Damage Frequency Impact
```latex
\Delta f \approx -k_1 \beta - k_2 \beta^2 \quad \text{(Eq. 15)}
```

### Eq. 16: Sensitivity Coefficient
```latex
S_i = \frac{\partial f}{\partial p_i} \times \frac{p_i}{f} \quad \text{(Eq. 16)}
```

## Usage in Script

When generating equations in the Word document script, use the `createEquationLatex()` function with these LaTeX strings:

```javascript
createEquationLatex("f_n = \\frac{1}{2\\pi}\\sqrt{\\frac{k}{m}} \\quad \\text{(Eq. 1)}")
```

Note: In JavaScript strings, backslashes must be escaped (doubled).

## Key LaTeX Elements

- `\frac{a}{b}` - Fractions
- `\sqrt{x}` - Square root
- `\text{...}` - Text in math mode
- `\quad` - Large space
- `\times` - Multiplication symbol
- `\propto` - Proportional to
- `\approx` - Approximately equal
- `\min()` - Minimum function
- `\begin{bmatrix}...\end{bmatrix}` - Matrix
- `\partial` - Partial derivative
- `_{\text{subscript}}` - Subscripts with text
- `\alpha, \beta, \lambda, \omega, \rho, \sigma, \mu, \Delta` - Greek letters
