# Teoria epistemica computazionale a classi di equivalenza
# Concetti: v3 ("3"), vIII ("III"), nuova direzione concettuale con espansione
# Punto 10 – Secondo Principio: Azione esterna come cambio di base (fine-tuning)

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)

# Sigmoid per attivazione epistemica
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def epistemic_error(y_expected, y_output):
    return norm(y_expected - y_output)

# === Base epistemica originale ===
k = 10
A = np.eye(k)
print("\n=== Matrice epistemica A ===")
print(A)

# === Concetti equivalenti: v3 e vIII ===
v3 = np.zeros(k); v3[2] = 1
vIII = np.full(k, 0.5)

print("\n=== Vettori concettuali ===")
print("v3:", v3)
print("vIII:", vIII)

# === Punto 10: Azione esterna come cambio di base ===
M_finetune = np.outer(v3, vIII) / np.dot(vIII, vIII)
vIII_translated = M_finetune @ vIII
out_finetuned = sigmoid(A @ vIII_translated)
out_expected = sigmoid(A @ v3)
err_finetuned = epistemic_error(out_expected, out_finetuned)

print("\n=== Punto 10 – Azione esterna come cambio di base ===")
print("vIII trasformato (M_finetune @ vIII):", vIII_translated.round(3))
print("Output sigmoid A @ v3:", out_expected.round(3))
print("Output sigmoid A @ M_finetune @ vIII:", out_finetuned.round(3))
print("Errore epistemico (fine-tuning):", round(err_finetuned, 4))

# === Classe di equivalenza semantica (sistema modificato) ===
M_equiv = M_finetune
W_equiv = A @ M_equiv
print("\n=== Matrice di trasformazione M_equiv ===")
print(M_equiv)
print("\n=== Sistema epistemico modificato W_equiv = A @ M_equiv ===")
print(W_equiv)

out_3 = sigmoid(W_equiv @ v3)
out_III = sigmoid(W_equiv @ vIII)
err_equiv = epistemic_error(out_3, out_III)

print("\n=== Classe di equivalenza semantica (v3 ~ vIII) ===")
print("Output v3:", out_3.round(3))
print("Output vIII:", out_III.round(3))
print("Errore epistemico v3 ~ vIII:", round(err_equiv, 4))

# === Nuovo concetto indipendente: somma ("6" = 3 + 3) ===
v6 = np.zeros(k); v6[5] = 1
v_sum = v3 + v3
print("\n=== Nuovo concetto (somma): v_sum = v3 + v3 ===")
print("v_sum:", v_sum)
print("v6 (target):", v6)

# Caso 1: sistema chiuso
out_closed = sigmoid(A @ v_sum)
err_closed = epistemic_error(sigmoid(v6), out_closed)

# Caso 2: espansione epistemica
K_expand = np.outer(v6, v_sum) / np.dot(v_sum, v_sum)
W_expand = A + K_expand
out_expanded = sigmoid(W_expand @ v_sum)
err_expanded = epistemic_error(sigmoid(v6), out_expanded)

print("\n=== Matrice di espansione K_expand ===")
print(K_expand)
print("\n=== Sistema espanso W_expand = A + K_expand ===")
print(W_expand)

print("\n=== Espansione epistemica (nuovo concetto: 'somma') ===")
print("Output sistema chiuso:", out_closed.round(3))
print("Errore epistemico (chiuso):", round(err_closed, 4))
print("Output sistema espanso:", out_expanded.round(3))
print("Errore epistemico (espanso):", round(err_expanded, 4))

# === PLOT 1: Detailed Activation Comparison ===
fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(out_expected, 'o-', label='A @ v3')
ax1.plot(out_finetuned, 's--', label='A @ MvIII')
ax1.plot(out_III, 'd-.', label='W @ vIII')
ax1.plot(out_expanded, 'x-', label='W+K @ v_sum')
ax1.set_title('Epistemic Activations (Sigmoid Output)')
ax1.set_ylabel('Activation Level')
ax1.set_xlabel('Concept Component')
ax1.legend()
ax1.grid(True)
plt.tight_layout()
plt.savefig('activation_comparison.png')

# === PLOT 2: Epistemic Error Summary ===
errors = [err_finetuned, err_equiv, err_closed, err_expanded]
labels = ['A vs MvIII', 'W v3 vs vIII', 'Closed: v_sum vs v6', 'Expanded: v_sum vs v6']
colors = ['#ff7f0e', '#1f77b4', '#d62728', '#2ca02c']

fig, ax2 = plt.subplots(figsize=(8, 4))
ax2.bar(labels, errors, color=colors)
ax2.set_ylabel('Epistemic Error')
ax2.set_title('Error in Concept Recognition')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('error_summary.png')

# === PLOT 3: Focus on Concept “3” vs “6” Activation ===
fig, ax3 = plt.subplots(figsize=(6, 4))
bar_width = 0.3
index = np.arange(2)

# Activations on component 2 (concept "3")
a3 = out_closed[2]
a3_exp = out_expanded[2]
# Activations on component 5 (concept "6")
a6 = out_closed[5]
a6_exp = out_expanded[5]

ax3.bar(index[0] - bar_width/2, a3, bar_width, label='Closed (3)', color='#1f77b4')
ax3.bar(index[0] + bar_width/2, a3_exp, bar_width, label='Expanded (3)', color='#aec7e8')
ax3.bar(index[1] - bar_width/2, a6, bar_width, label='Closed (6)', color='#d62728')
ax3.bar(index[1] + bar_width/2, a6_exp, bar_width, label='Expanded (6)', color='#ff9896')

ax3.set_xticks(index)
ax3.set_xticklabels(['3', '6'])
ax3.set_ylabel('Activation')
ax3.set_title('Effect of Expansion on Concept Activation')
ax3.legend()
plt.tight_layout()
plt.savefig('concept_focus.png')

