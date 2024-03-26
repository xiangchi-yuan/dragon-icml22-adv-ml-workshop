# Guide for MITIGATING SEVERE ROBUSTNESS DEGRADATION ON GRAPHS (ICLR 24, ICML Workshop 23)


## Training Without Differential Privacy

1. Run the `training.py` script.
2. In the `training.py` script, you have the option to choose a model with or without DP. Uncomment the appropriate line based on your choice.
   - Without DP:
     ```python
     $cd autotip
     from gatdp import GAT
     ```
3. After training, run the `injection_leaderboard.py` script.

## Training With Differential Privacy

1. Run the `training.py` script.
2. In the `training.py` script, make sure to select the raw GCN model initially to create a surrogate model for injection.
3. After training the GCN model, run the `adv_training.py` script.
4. In the `adv_training.py` script, you have the option to choose a model with or without DP. Uncomment the appropriate line based on your choice.
   - With DP:
     ```python
     # from grb.model.dgl import GAT
     from gatdp import GAT
     ```
5. Finally, run the `injection_leaderboard.py` script with an additional note: change the line 53 in that script to "final_model.pt".

