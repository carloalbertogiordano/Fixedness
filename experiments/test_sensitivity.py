import yaml
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import itertools
import numpy as np
from datetime import datetime

def run_experiment(bk_frac, k):
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.yaml'), 'r') as f:
        config = yaml.safe_load(f)
    config['experiment']['anonymization']['background_knowledge_frac'] = float(bk_frac)
    config['experiment']['anonymization']['k'] = k
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    subprocess.run(["python3", "tests/fixedness_test/main.py"], check=True)
    
    res_dir = "tests/fixedness_test/results"
    latest_run = sorted([d for d in os.listdir(res_dir) if os.path.isdir(os.path.join(res_dir, d))])[-1]
    df = pd.read_csv(os.path.join(res_dir, latest_run, "full_audit.csv"))
    df_valid = df[df['fixedness'] >= 0]
    
    return {
        'k': k, 
        'bk_frac': bk_frac, 
        'run_dir': os.path.join(res_dir, latest_run),
        'avg_fixedness': df_valid['fixedness'].mean(),
        'avg_sponginess': df_valid['sponginess'].mean()
    }

def main():
    # Aumentiamo la risoluzione: 10 punti per BK, 3 per K
    ks = [2, 5, 10]
    bks = np.linspace(0.0, 0.9, 10) 
    history = [run_experiment(bk, k) for k, bk in itertools.product(ks, bks)]
    df = pd.DataFrame(history)
    
    latest_run_dir = df.iloc[-1]['run_dir']
    
    # Plotting fluido
    plt.figure(figsize=(12, 6))
    for k in ks:
        sub = df[df['k'] == k]
        # Usiamo una curva interpolata per vedere meglio il trend
        plt.plot(sub['bk_frac'], sub['avg_fixedness'], marker='o', label=f'Fixedness k={k}')
    
    plt.title('Fixedness Phase Transition (High Resolution)')
    plt.xlabel('Background Knowledge (0.0 to 0.9)')
    plt.ylabel('Fixedness (Certainty)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(latest_run_dir, 'fixedness_curve.png'))
    
    plt.figure(figsize=(12, 6))
    for k in ks:
        sub = df[df['k'] == k]
        plt.plot(sub['bk_frac'], sub['avg_sponginess'], marker='s', label=f'Sponginess k={k}')
    
    plt.title('Sponginess Phase Transition (High Resolution)')
    plt.xlabel('Background Knowledge (0.0 to 0.9)')
    plt.ylabel('Sponginess (Inference Potential)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(latest_run_dir, 'sponginess_curve.png'))
    
    print(f"\nGrafici ad alta risoluzione salvati in: {latest_run_dir}")

if __name__ == "__main__":
    main()
