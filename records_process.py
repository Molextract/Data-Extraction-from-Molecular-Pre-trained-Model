import torch
import pickle as pkl
import torch.nn.functional as F
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.QED import qed
from model import GNN, GNN_graphpred
import argparse
from molecule_generation.utils_graph import *
from molecule_generation.molecule_graph import Molecule_Graph
import docking_score.sascorer as sascorer
from docking_score.docking_score import docking_sc

if __name__ == '__main__':
    parser = argparse.ArgumentParser("mol_generation")
    parser.add_argument('--scaffold_idx', default=0, type=int)
    parser.add_argument('--device', default=1, type=int)
    parser.add_argument('--model_type', type=str, default='contextpred')
    parser.add_argument('--savefile', type=str, default='phase1')
    args = parser.parse_args()

    records = pkl.load(open(f'./records/{args.savefile}/scaffold_{args.scaffold_idx}.pkl', 'rb'))

    records_new = []
    for r in records:
        r = record(scaffold_mg=r.scaffold_mg,
                   side_chain_mg=Molecule_Graph(FRAG_VOCAB[r.side_chain_idx][0]),
                   potential_subgraph_mg=r.potential_subgraph_mg,
                   scaffold_att_pos=r.scaffold_att_pos,
                   side_chain_idx=r.side_chain_idx,
                   side_chain_att_pos=r.side_chain_att_pos,
                   other=r.other)
        records_new.append(r)
    records = records_new

    for r in tqdm(records):
        r.other['qed'] = qed(r.potential_subgraph_mg.mol)
        r.other['sa'] = -1 * sascorer.calculateScore(r.potential_subgraph_mg.mol)

    smiles = [Chem.MolToSmiles(r.potential_subgraph_mg.mol_core) for r in records]
    for protein in ['fa7', 'parp1', '5ht1b']:
        scores = docking_sc([Chem.MolToSmiles(r.potential_subgraph_mg.mol_core) for r in records], protein)
        for r, s in zip(records, scores):
            r.other[f"docking_{protein}"] = -1 * s

    device = args.device if args.device is not None else 'cpu'
    model = GNN_graphpred(5, 300, 1, JK='last', drop_ratio=0.5, graph_pooling='mean').to(device)
    if args.model_type == 'contextpred':
        state = torch.load('./saved/contextpred.pth')
        model.gnn.load_state_dict(state)
    elif args.model_type == 'gcl':
        state = torch.load('./saved/graphcl_80.pth')
        model.gnn.load_state_dict(state)
    elif args.model_type == 'masking':
        state = torch.load('./saved/masking.pth')
    elif args.model_type == 'infomax':
        state = torch.load('./saved/infomax.pth')
        model.gnn.load_state_dict(state)
    elif args.model_type == 'mae':
        state = torch.load('./saved/pretrained.pth')
        model.gnn.load_state_dict(state)
    elif args.model_type == 'simgrace':
        state = torch.load('./saved/simgrace_100.pth')
        model.gnn.load_state_dict(state)
    elif args.model_type == 'molebert':
        state = torch.load('./saved/Mole-BERT.pth')
        model.gnn.load_state_dict(state)
    elif args.model_type == 'supercont':
        state = torch.load('./sav ed/supervised_contextpred.pth')
        model.gnn.load_state_dict(state)
    elif args.model_type == 'superedge':
        state = torch.load('./saved/supervised_edgepred.pth')
        model.gnn.load_state_dict(state)
    elif args.model_type == 'superinfo':
        state = torch.load('./saved/supervised_infomax.pth')
        model.gnn.load_state_dict(state)
    elif args.model_type == 'supermasking':
        state = torch.load('./saved/supervised_masking.pth')
        model.gnn.load_state_dict(state)
    elif args.model_type == 'supervised':
        state = torch.load('./saved/supervised.pth')
        model.gnn.load_state_dict(state)
    else:
        raise NotImplementedError
    model.eval()

    for alpha in [0.0, 0.1, 0.5, 0.9, 1.0]:
        for r in tqdm(records):
            M = r.scaffold_mg.atom_num
            N = r.side_chain_mg.atom_num
            representation_M = r.scaffold_mg.representation(model, device)
            representation_N = r.side_chain_mg.representation(model, device)
            representation_estimate = (alpha * M * representation_M + (1 - alpha) * N * representation_N) / (
                    alpha * M + (1 - alpha) * N)

            representation_total = r.potential_subgraph_mg.representation(model, device)
            score = (representation_estimate @ representation_total.T).item()
            r.other[f'score_{alpha}'] = score
            score = F.cosine_similarity(representation_estimate, representation_total).item()
            r.other[f'cos_score_{alpha}'] = score

            representation_M_normed = representation_M / representation_M.norm()
            representation_N_normed = representation_N / representation_N.norm()
            representation_estimated_normed = (alpha * M * representation_M_normed + (
                    1 - alpha) * N * representation_N_normed) / (alpha * M + (1 - alpha) * N)

            score = (representation_estimated_normed @ representation_total.T).item()
            r.other[f'score_{alpha}_normed'] = score
            score = F.cosine_similarity(representation_estimate, representation_total).item()
            r.other[f'cos_score_{alpha}_normed'] = score

            r.other['representation_scaffold'] = representation_M.detach().cpu().squeeze().numpy()
            r.other['representation_side_chain'] = representation_N.detach().cpu().squeeze().numpy()
            r.other['representation_merged_mol'] = representation_total.detach().cpu().squeeze().numpy()

    if not os.path.exists(f'./records/records_{args.model_type}_ring'):
        os.makedirs(f'./records/records_{args.model_type}_ring')
    pkl.dump(records, open(f'./records/records_{args.model_type}_ring/scaffold_{args.scaffold_idx}.pkl', 'wb'))
