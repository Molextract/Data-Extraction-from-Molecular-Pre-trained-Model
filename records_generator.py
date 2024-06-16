import os
import sys

sys.path.insert(0, os.getcwd())
import argparse
from multiprocessing import Pool, Manager, Process

import pickle as pkl
from tqdm import tqdm
from rdkit import Chem

from molecule_generation.utils_graph import *
from molecule_generation.molecule_graph import Molecule_Graph


def process_records_freq(records, mol_bank, mol_bank_aux, return_records):
    for record in tqdm(records):
        freq = record.potential_subgraph_mg.get_freq(mol_bank)
        record.other['freq'] = freq
        freq_aux = record.potential_subgraph_mg.get_freq(mol_bank_aux)
        record.other['freq_aux'] = freq_aux

    return_records.extend(records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("mol_generation")
    parser.add_argument('--scaffold_idx', default=0, type=int)
    parser.add_argument('--cpu_num', default=100, type=int)
    parser.add_argument('--model_type', type=str, default='grover')
    parser.add_argument('--savefile', type=str, default='grover')
    args = parser.parse_args()

    scaffold_smi = SCAFFOLD_VOCAB[args.scaffold_idx]

    mg = Molecule_Graph(scaffold_smi)
    try:
        core2id = pkl.load(open(r'./data_preprocessing/zinc/core2mol_bank/core2id.pkl', 'rb'))
        mol_bank_id = core2id[Chem.MolToSmiles(mg.mol_core)]
        mol_bank_filtered = pkl.load(
            open(f'./data_preprocessing/zinc/core2mol_bank/{mol_bank_id}.pkl', 'rb'))
    except:
        mol_bank = pkl.load(open(r'./dataset/zinc_standard_agent/processed/mols.pkl', 'rb'))

        mol_core = mg.mol_core
        mol_bank_filtered = [mol for mol in tqdm(mol_bank) if mol.HasSubstructMatch(mol_core)]

    mol_core = mg.mol_core
    mol_bank_aux = pkl.load(
        open('./dataset/zinc_standard_agent_aux_20k/processed/mols.pkl', 'rb'))
    mol_bank_aux_filtered = [mol for mol in tqdm(mol_bank_aux) if mol.HasSubstructMatch(mol_core)]

    attach_points = mg.attach_points
    motif_bank_size = len(FRAG_VOCAB)

    records = []
    for p in tqdm(range(len(mg.attach_points))):
        for i in range(motif_bank_size):
            for np in range(len(FRAG_VOCAB_ATT[i])):
                potential_subgraph = copy.deepcopy(mg)
                ac = [p, i, np]
                potential_subgraph.add_motif(ac)
                records.append(record(scaffold_mg=copy.deepcopy(mg),
                                      side_chain_mg=Molecule_Graph(FRAG_VOCAB[i][0]),
                                      potential_subgraph_mg=potential_subgraph,
                                      scaffold_att_pos=p,
                                      side_chain_idx=i,
                                      side_chain_att_pos=np,
                                      other=dict()))

    pool = Pool(args.cpu_num)

    manager = Manager()
    return_records = manager.list()
    print("----------")

    size = int(len(records) / args.cpu_num) + 1

    for idx, i in enumerate(range(0, len(records), size)):
        pool.apply_async(func=process_records_freq,
                         args=(records[i:i + size], mol_bank_filtered, mol_bank_aux_filtered, return_records))

    pool.close()
    pool.join()

    records = [r for r in return_records]

    if not os.path.exists(f'./records/{args.savefile}'):
        os.makedirs(f'./records/{args.savefile}')
    pkl.dump(records, open(f'./records/{args.savefile}/scaffold_{args.scaffold_idx}.pkl', 'wb'))

    print(len(records))
