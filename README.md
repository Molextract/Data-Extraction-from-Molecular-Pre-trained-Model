# Anonymous code for Molextract

## About

Anonymous Code Submission for NeurIPS'24: 'Data Extraction from Molecular Pre-trained Model'

## Brief Introduction 
Most of the existing methods are not suitable for data extraction in the context of graphs due to the implicit semantics present in the graph structures.

- We design a scoring function and employ an auxiliary dataset to further refine and learn the scoring function, enabling the filtration of potential training molecules.
- To efficiently extract molecules from the molecular  pre-trained model, we propose a reinforcement learning-based extraction method, utilizing the scoring function as the reward mechanism.




## File Structure 

```
├── README.md
├── core_motif.py
├── core_motif_mc.py
├── descriptors.py
├── docking_score
│   ├── ReLeaSE_Vina
│   │   └── docking
│   │       ├── 5ht1b
│   │       │   ├── datasets
│   │       │   │   └── 5ht1b.csv
│   │       │   ├── metadata.json
│   │       │   ├── receptor.pdbqt
│   │       │   └── receptor_copy.pdbqt
│   │       ├── fa7
│   │       │   └── receptor.pdbqt
│   │       └── parp1
│   │           └── receptor.pdbqt
│   ├── __init__.py
│   ├── bin
│   │   └── qvina02
│   ├── config_5ht1b.yaml
│   ├── config_fa7.yaml
│   ├── config_parp1.yaml
│   ├── docking_score.py
│   ├── docking_simple.py
│   ├── fpscores.pkl.gz
│   ├── sascorer.py
│   └── tmp
│       ├── dock_0.pdbqt
│       ├── ligand_0.mol
│       └── ligand_0.pdbqt
├── evaluator.py
├── gym_molecule
│   ├── __init__.py
│   ├── __pycache__
│   │   └── __init__.cpython-37.pyc
│   ├── dataset
│   │   ├── descriptors_list.txt
│   │   ├── motif_cleaned.txt
│   │   ├── motif_cleaned2.txt
│   │   ├── motifs_1k.txt
│   │   ├── motifs_350.txt
│   │   ├── motifs_66.txt
│   │   ├── motifs_91.txt
│   │   ├── motifs_91_all_att.txt
│   │   ├── motifs_zinc_random_92.txt
│   │   ├── opt.test.logP-SA
│   │   ├── ring.txt
│   │   ├── ring_84.txt
│   │   ├── scaffold_top8.txt
│   │   └── side_chain_bank.txt
│   └── envs
│       ├── __init__.py
│       ├── __pycache__
│       │   ├── __init__.cpython-37.pyc
│       │   ├── docking_simple.cpython-37.pyc
│       │   ├── env_utils_graph.cpython-37.pyc
│       │   └── molecule_graph.cpython-37.pyc
│       ├── docking_simple.py
│       ├── env_utils_graph.py
│       ├── fpscores.pkl.gz
│       ├── molecule_graph.py
│       ├── opt.test.logP-SA
│       ├── pretrained_models
│       │   ├── GNN_aux.py
│       │   ├── GNN_auxv1.py
│       │   ├── GNN_simple.py
│       │   ├── __init__.py
│       │   ├── __pycache__
│       │   │   ├── GNN_aux.cpython-37.pyc
│       │   │   ├── GNN_simple.cpython-37.pyc
│       │   │   └── __init__.cpython-37.pyc
│       │   ├── aux_saved
│       │   │   ├── reg_stre_0.0
│       │   │   │   └── scaffold_0.pth
│       │   │   └── reg_stre_10.0
│       │   │       ├── scaffold_0.pth
│       │   │       ├── scaffold_1.pth
│       │   │       ├── scaffold_2.pth
│       │   │       ├── scaffold_3.pth
│       │   │       └── scaffold_4.pth
│       │   └── context_pred
│       │       ├── __init__.py
│       │       ├── __pycache__
│       │       │   ├── __init__.cpython-37.pyc
│       │       │   ├── loader.cpython-37.pyc
│       │       │   └── model.cpython-37.pyc
│       │       ├── contextpred.pth
│       │       ├── loader.py
│       │       ├── model.py
│       │       ├── tmp.ipynb
│       │       └── tmp.pkl
│       └── sascorer.py
├── model.py
├── molecule_generation
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-310.pyc
│   │   ├── __init__.cpython-37.pyc
│   │   ├── molecule_graph.cpython-37.pyc
│   │   ├── utils_graph.cpython-310.pyc
│   │   └── utils_graph.cpython-37.pyc
│   ├── dataset
│   │   ├── motifs_91.txt
│   │   ├── ring.txt
│   │   ├── ring_84.txt
│   │   ├── ring_84_old.txt
│   │   ├── scaffold_special_with_core_freq_100.txt
│   │   ├── scaffold_top8.txt
│   │   └── side_chain_bank.txt
│   ├── molecule_graph.py
│   └── utils_graph.py
├── records_generator.py
├── records_process.py
├── records_training.py
├── requirements.txt
├── run_rl.py
└── saved
    ├── Mole-BERT.pth
    ├── contextpred.pth
    ├── edgepred.pth
    ├── infomax.pth
    ├── masking.pth
    ├── model.pth
    ├── model_2023_05_01_21_35_14.ep50
    ├── pretrained.pth
    ├── simgrace_100.pth
    ├── supervised.pth
    ├── supervised_contextpred.pth
    ├── supervised_edgepred.pth
    ├── supervised_infomax.pth
    └── supervised_masking.pth
```

*****

Below, we will specifically explain the meaning of important file folders to help the user better understand the file structure.

`saved`: the directory for pre-trained model.

`gym_moleculars & molecule_generation`: contains data of motif bank

`docking_score`: contains calculation of docking score.

`model.py`: contains the reinforcement learning code.

## Requirements

Python module dependencies are listed in requirements.txt, which can be easily installed with pip:

`pip install -r requirements.txt`


## Run the code
Please run the following command:
```
python -u records_generator.py --scaffold_idx 0 --cpu_num 100 --savefile result
python -u records_process.py --scaffold_idx 0 --device $device --savefile result --model_type gcl
python -u records_training.py --device 0 --scaffold_idx 0 --model_type gcl --savefile result
```

The results will be stored in the `result` directory.