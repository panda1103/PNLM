## Usage

Before finetuning the model, we need to process the data. 

This script will use ESM2 to generate the embedding and save as .npy files. 

```
python3 prepare_data.py \
    --input_files xxx.fasta \
    --train_split_ratio=0.8
```

We can also specify the ratio of train and test data split (default is 0.8) and using a boolean flag `--bidirectional` we can save the sequences also in reverse, if we want to train a bidirectional model.



The notebook provides a simple example showing how to use ESM-2 and RNA-FM



## Reference

```
@article{lin2022language,
  title={Language models of protein sequences at the scale of evolution enable accurate structure prediction},
  author={Lin, Zeming and Akin, Halil and Rao, Roshan and Hie, Brian and Zhu, Zhongkai and Lu, Wenting and Smetanin, Nikita and dos Santos Costa, Allan and Fazel-Zarandi, Maryam and Sercu, Tom and Candido, Sal and others},
  journal={bioRxiv},
  year={2022},
  publisher={Cold Spring Harbor Laboratory}
}
```



```
@article{chen2022interpretable,
  title={Interpretable rna foundation model from unannotated data for highly accurate rna structure and function predictions},
  author={Chen, Jiayang and Hu, Zhihang and Sun, Siqi and Tan, Qingxiong and Wang, Yixuan and Yu, Qinze and Zong, Licheng and Hong, Liang and Xiao, Jin and King, Irwin and others},
  journal={arXiv preprint arXiv:2204.00300,
  year={2022}
}
```

