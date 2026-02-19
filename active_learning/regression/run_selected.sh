smis=('smi_to_calc')

acqfs=('random' 'acqf_1' 'acqf_2' 'acqf_2-1' 'acqf_3' 'acqf_4' 'acqf_4-1' 'acqf_5' 'acqf_6' 'acqf_7' 'acqf_8' 'acqf_9' 'acqf_10')
n_runs=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)
res='bde-zinc'

b=1
start='cold'
feat='selected'
n_repet=10
db=0.01
n_est=250
max_feat=0.5
max_depth=10
min_samples=3
model="regression_rf"
strat="simple"
folder="ZINC_cat_norm_bde"
alpha=1
obs="bde_avg"
max_tset=135

for smi in ${smis[@]}; do
  for acqf in ${acqfs[@]}; do
    for idx in ${n_runs[@]}; do #$(seq$n_runs); do
        python ./main.py --smi $smi --acqf $acqf --batch $b --start $start --n_repet $n_repet --db $db --feat $feat --n_est $n_est --max_feats $max_feat --max_depth $max_depth --min_samples_leaf $min_samples --model $model --selection_strat $strat --res $res --run $idx --df_folder $folder --alpha $alpha --y $obs --max_tset $max_tset
    done
  done
done

