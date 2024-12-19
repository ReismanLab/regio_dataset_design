# distribution shift analysis

## 1. Tanimoto similarities between the small molecules dataset and the complex molecules dataset

The results reported in the SI can be obtained running the notebook: *tanimoto_distances.ipynb*.

## 2. BRICS and pBRICS analysis

The decomposition of the smiles in the dioxirane dataset are in the folder *decomposition_data*. The BRICS decomposition was obtained using RDKit Chem.BRICS.BRICSDecompose function and the pBRICS decomposition was obtained thanks to the code provided by the authors of the [original publication](https://pubs.acs.org/doi/10.1021/acs.jcim.3c00689). 

The results reported in the SI can be obtained running the notebook: *pbrics_comparison.ipynb*.

## 3. Fréchet ChemNet Distance (FCD)

The Fréchet ChemNet Distance can be computed using the [fcd package](https://github.com/insilicomedicine/fcd_torch) and the python script provided in this folder.

### installation of the FCD environment:

'conda create -n fcd -y
conda activate fcd
git clone https://github.com/insilicomedicine/fcd_torch.git
cd fcd_torch   
python3 setup.py install'  

### running the results:

'mv ../fcd_small_vs_complex.py .
python3 fcd_small_vs_complex.py'

