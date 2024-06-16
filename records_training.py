import numpy as np
import argparse
from tqdm import tqdm
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rdkit import Chem
import matplotlib.pyplot as plt
from molecule_generation.utils_graph import *
import seaborn as sns
from molecule_generation.molecule_graph import Molecule_Graph
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, roc_auc_score
from fcd import get_fcd, load_ref_model,canonical_smiles
import evaluator

def calculate_kl_divergence(records):
    generated_smiles_list = [Chem.MolToSmiles(mol) for mol in
                             [record.potential_subgraph_mg.mol for record in records]]
    training_smiles_list = mol_bank_smiles_list
    return evaluator.kl_divergence(generated_smiles_list, training_smiles_list)

def calculate_fcd(records):
    generated_smiles_list = [Chem.MolToSmiles(mol) for mol in
                             [record.potential_subgraph_mg.mol for record in records]]

    # Load chemnet model
    model = load_ref_model()
    can_sample1 = [w for w in canonical_smiles(generated_smiles_list) if w is not None]
    can_sample2 = [w for w in canonical_smiles(mol_bank_smiles_list) if w is not None]
    fcd_score = get_fcd(can_sample1, can_sample2, model)
    print(f'{args.model_type},{args.scaffold_idx}', 'FCD: ', fcd_score)
    return fcd_score

def TP_with_fixed_FP(y_true, y_pred, fp):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    return tpr[fpr <= fp].max()


def calculate_extraction(records, scaffold_idx):
    scaffold_smi = SCAFFOLD_VOCAB[scaffold_idx]
    core2id = pkl.load(open(r'./data_preprocessing/zinc/core2mol_bank/core2id.pkl', 'rb'))
    mol_bank_id = core2id[Chem.MolToSmiles(Molecule_Graph(scaffold_smi).mol_core)]

    att = Chem.MolFromSmiles('*')
    H = Chem.MolFromSmiles('[H]')

    freq_list = []
    ratio_list = []
    mol_bank_filtered = pkl.load(
        open(f'./data_preprocessing/zinc/core2mol_bank/{mol_bank_id}.pkl', 'rb'))
    for i in tqdm(range(len(records))):
        print()
        mg = records[i].potential_subgraph_mg.mol
        print('scaffold',Chem.MolToSmiles(records[i].scaffold_mg.mol))
        print(Chem.MolToSmiles(mg))
        mg1 = sanitize(Chem.ReplaceSubstructs(mg, att, H, replaceAll=True)[0])
        print(Chem.MolToSmiles(mg1))
        matched_mols = [mol for mol in mol_bank_filtered if mol.HasSubstructMatch(mg1)]
        freq = len(matched_mols)
        if freq > 0:
            freq_list.append(1)
        else:
            freq_list.append(0)
        ratio_list.append(sum(freq_list) / len(freq_list))
    return sum(freq_list) / len(freq_list), freq_list, ratio_list


def get_representations(records):
    scaffold_representations = []
    side_chain_representations = []
    merged_mol_representations = []
    scaffold_sizes = []
    side_chain_sizes = []

    labels = []
    labels_ground = []

    invalid_count = 0
    for r in tqdm(records):
        if np.abs(r.other['representation_merged_mol']).max() > 1e4:
            invalid_count += 1
            continue
        scaffold_representations.append(r.other['representation_scaffold'])
        side_chain_representations.append(r.other['representation_side_chain'])
        merged_mol_representations.append(r.other['representation_merged_mol'])
        scaffold_sizes.append(r.scaffold_mg.atom_num)
        side_chain_sizes.append(r.side_chain_mg.atom_num)

        labels.append(1. if r.other['freq_aux'] > 0 else 0.)
        labels_ground.append(1. if r.other['freq'] > 0 else 0.)

    scaffold_representations = torch.tensor(scaffold_representations)
    scaffold_sizes = torch.tensor(scaffold_sizes)[:, np.newaxis]
    side_chain_representations = torch.tensor(side_chain_representations)
    side_chain_sizes = torch.tensor(side_chain_sizes)[:, np.newaxis]
    merged_mol_representations = torch.tensor(merged_mol_representations)
    labels = torch.tensor(labels)
    labels_ground = torch.tensor(labels_ground)

    return scaffold_representations, scaffold_sizes, side_chain_representations, side_chain_sizes, merged_mol_representations, labels, labels_ground


class LearnAlphaNew(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.dim = 100
        self.transform_motif = nn.Sequential(nn.Linear(emb_dim, self.dim),)
        self.transform_scaffold = nn.Sequential(nn.Linear(emb_dim, self.dim),)
        self.transform_merged = nn.Sequential(nn.Linear(emb_dim, self.dim),)
        self.transform_alpha = nn.Sequential( nn.Linear(3 * emb_dim, 1), )

    def forward(self, motif_representations, scaffold_representations, merged_mol_representations, scaffold_sizes,
                motif_sizes):
        x = torch.cat([scaffold_representations, motif_representations, merged_mol_representations], dim=-1)
        alpha = self.transform_alpha(x)

        side_chain_representations = self.transform_motif(motif_representations)
        scaffold_representations = self.transform_scaffold(scaffold_representations)
        merged_mol_representations = self.transform_merged(merged_mol_representations)
        estimated_representations = (
                                            alpha * scaffold_representations * scaffold_sizes + (
                                            1 - alpha) * side_chain_representations * motif_sizes) / (
                                            scaffold_sizes + motif_sizes)
        score = F.cosine_similarity(estimated_representations, merged_mol_representations)
        return alpha.cpu().detach().numpy(), score.cpu().detach().numpy(), estimated_representations.cpu().detach().numpy(), side_chain_representations.cpu().detach().numpy(), scaffold_representations.cpu().detach().numpy(), merged_mol_representations.cpu().detach().numpy()

    def loss(self, labels, motif_representations, scaffold_representations, merged_mol_representations,
             scaffold_sizes, motif_sizes):
        x = torch.cat([scaffold_representations, motif_representations, merged_mol_representations], dim=-1)
        alpha = self.transform_alpha(x)

        motif_representations = self.transform_motif(motif_representations)
        scaffold_representations = self.transform_scaffold(scaffold_representations)
        merged_mol_representations = self.transform_merged(merged_mol_representations)
        estimated_representations = (
                                            alpha * scaffold_representations * scaffold_sizes + (
                                            1 - alpha) * motif_representations * motif_sizes) / (
                                            scaffold_sizes + motif_sizes)
        score1 = F.cosine_similarity(estimated_representations, merged_mol_representations)
        weight = labels.sum() / labels.size(0)

        loss = torch.where(labels >= 1.0, -1 * score1, weight * score1).sum()
        loss_alpha = torch.where((0.0 < alpha) & (alpha <= 1.0), torch.tensor([0.0]), torch.abs(alpha)).sum()
        loss = loss + 50 * loss_alpha
        return loss


def train(args):
    records = pkl.load(open(f'./records/records_{args.model_type}_ring/scaffold_{args.scaffold_idx}.pkl',
            'rb'))

    scaffold_representations, scaffold_sizes, motif_representations, motif_sizes, merged_mol_representations, labels, labels_ground = get_representations(
        records)

    model = LearnAlphaNew(300)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    model.train()
    loss_list = []
    for e in tqdm(range(200)):
        optimizer.zero_grad()
        loss = model.loss(labels, motif_representations, scaffold_representations, merged_mol_representations,
                          scaffold_sizes, motif_sizes)
        loss.backward()
        loss_list.append(loss.item())
        optimizer.step()

    os.makedirs(f'./saved_model/{args.savefile}', exist_ok=True)
    torch.save({'model': model.state_dict()},
               f'./saved_model/{args.savefile}/scaffold_{args.scaffold_idx}.pth')

    model.eval()

    alpha, scores, estimated_representations, side_chain_representations, scaffold_representations, merged_mol_representations = model(
        motif_representations, scaffold_representations, merged_mol_representations,
        scaffold_sizes, motif_sizes)

    sorted_indexes = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top_mol = [index for index, _ in sorted_indexes[:1000]]
    top_records = [records[i] for i in top_mol]
    extraction_result, extraction_result_list, ratio_list = calculate_extraction(top_records, args.scaffold_idx)

    kl_divergence = calculate_kl_divergence(top_records)

    fcd_score = calculate_fcd(top_records)

    auc_train = roc_auc_score(labels, scores)
    tp_train = TP_with_fixed_FP(labels, scores, 0.01)
    auc_ground = roc_auc_score(labels_ground, scores)
    tp_ground = TP_with_fixed_FP(labels_ground, scores, 0.01)

    print(
        f"train_auc: {round(auc_train, 4)}, train_tp:  {round(tp_train, 4)}, ground_auc: {round(auc_ground, 4)}, ground_tp: {round(tp_ground, 4)}, avg_alpha: {round(np.mean(alpha), 4)}, extraction result:{extraction_result}")
    print(
        f"kl_divergence: {round(kl_divergence,4)}, fcd_score:  {round(fcd_score, 4)}")


    if not os.path.exists(f'./result/{args.savefile}'):
        os.makedirs(f'./result/{args.savefile}')

    with open(f'./result/{args.savefile}/scaffold_{args.scaffold_idx}.txt', 'w') as f:
        f.write(f"train_auc: {auc_train}\n")
        f.write(f"train_tp: {tp_train}\n")
        f.write(f"ground_auc: {auc_ground}\n")
        f.write(f"ground_tp: {tp_ground}\n")
        f.write(f"avg_alpha: {np.mean(alpha)}\n")
        f.write(f"extraction result:{extraction_result}\n")

    np.save(f'./result/{args.savefile}/array_{args.scaffold_idx}.npy', np.array(ratio_list))
    print("done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=5e-4,
                        help='weight decay (default: 0)')
    parser.add_argument("--scaffold_idx", type=int, default=0)
    parser.add_argument('--model_type', type=str, default='gcl')
    parser.add_argument('--savefile', type=str, default='gcl')
    parser.add_argument('--plotdist', type=bool, default=False)
    parser.add_argument('--plottsne', type=bool, default=True)
    parser.add_argument('--plotratio', type=bool, default=True)
    args = parser.parse_args()

    scaffold_idx = args.scaffold_idx

    scaffold_smi = SCAFFOLD_VOCAB[scaffold_idx]
    core2id = pkl.load(open(r'./data_preprocessing/zinc/core2mol_bank/core2id.pkl', 'rb'))
    mol_bank_id = core2id[Chem.MolToSmiles(Molecule_Graph(scaffold_smi).mol_core)]
    mol_bank_filtered = pkl.load(
        open(f'./data_preprocessing/zinc/core2mol_bank/{mol_bank_id}.pkl', 'rb'))
    mol_bank_smiles_list = [Chem.MolToSmiles(mol) for mol in mol_bank_filtered]


    train(args)
