## model_validation

   **regression:**  contains two folders large_mol and loo to run the validation for the large_mol validation task and leave-one-out task. These validations and the plot of the figures can obtained using the following command lines for the large molecule in *large_mol* and *loo* respectively.
```      
python perf_bm.py
python figures_bm.py --run average --desc Custom --model RF2 --rxn dioxirane
```   
Generates this type of figures for TOP-1, TOP-2, TOP-3, TOP-5 and TOP-AVG:
       
![TOP-1 for the predictions on large molecules](model_validation/regression/large_mol/dioxirane/average/heatmap_TOP1.png)

```   
python perf_loo.py
python figure_loo.py --run run_01 --desc Selected --model RF2
```   

Generates this type of figures for TOP-1, TOP-2, TOP-3, TOP-5 and TOP-AVG:
       
![TOP-1 for the predictions on loo](model_validation/regression/loo/dioxirane/average/heatmap_TOP-1.png)