# Nucleic Acid Conditioned Protein Language Model
GitHub repo for PNLM, a protein language model for Abe generation.

## Install requirements 

PNLM is developed under environment with: Pytorch

To install the required packages for running PNLM, please use the following command:

1. Create conda environment

```bash
conda create -n <env_name> python==3.9
conda activate <env_name>
```

2. Install the package

```bash
git clone http://github.com/yao-jiawei/PNLM.git
pip install -t requirement.txt
```



## usage

1. Clone this repository by git clone http://github.com/yao-jiawei/PNLM.git

2. Install the packages required by PNLM. See the **Install Requirements** section.

3. Model checkpoint can load from 

   ```bash
   cd checkpoint
   wget https://drive.google.com/file/d/149PKklWYpmqkvECyMvVBeUIMauNy_3KZ/view?usp=sharing
   tar xfz weight.tar.gz
   ```

4. Sample

   ```bash
   python sample.py --t 1.8 --p 0.9 --num-samples 2 --context "1"
   ```

   

5. Example for Scoring

   The notebook calculates various sequence- and structure-based quality scores for proteins, such as those produced by generative models. Many different kinds of metrics can be calculated. For all of the metrics, proteins with higher scores (closer to zero for negative numbers) are predicted to be more likely to fold and have function than proteins with lower scores.
   
   **inputs**: Please provide protein sequences in fasta format, AlphaFold2-predicted structures, and/or reference protein sequences in fasta format as appropriate.
   
   **outputs**: A comma-separated values (csv) text file with the calculated metrics.





## Reference

If you find our code or paper useful, please cite:

```
@articale{
	title={},
	author={},
	journal={},
	year={2024}
}
```

