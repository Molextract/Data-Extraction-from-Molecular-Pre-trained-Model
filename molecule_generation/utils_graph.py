import os
import copy
import csv
import numpy as np
from rdkit import Chem
import collections


def sanitize(mol):
    try:
        smiles = Chem.MolToSmiles(mol, kekuleSmiles=False)
        mol = Chem.MolFromSmiles(smiles)
    except Exception as e:
        return None
    return mol


def get_att_points(mol):
    att_points = []
    for a in mol.GetAtoms():
        if a.GetSymbol() == '*':
            att_points.append(a.GetIdx())
    return att_points


def remove_att(s):
    s = Chem.MolFromSmiles(s)
    s = Chem.ReplaceSubstructs(s, att, H, replaceAll=True)[0]
    s = sanitize(s)
    s = Chem.MolToSmiles(s)
    return s


att = Chem.MolFromSmiles('*')
H = Chem.MolFromSmiles('[H]')

ATOM_VOCAB = ['C', 'N', 'O', 'S', 'P', 'F', 'I', 'Cl', 'Br', '*']

FRAG_VOCAB = open('/data/renhong/mol-attack/molecule_generation/dataset/motifs_91.txt', 'r').readlines()  # n=132
# SCAFFOLD_VOCAB = open('molecule_generation/dataset/scaffold_top8.txt', 'r').readlines()
SCAFFOLD_VOCAB = open('/data/renhong/mol-attack/molecule_generation/dataset/ring_84.txt', 'r').readlines()

# FRAG_VOCAB = open('gym_molecule/dataset/motifs_66.txt','r').readlines() # n=66
# FRAG_VOCAB = open('gym_molecule/dataset/motifs_350.txt','r').readlines() # n=350
# FRAG_VOCAB = open('gym_molecule/dataset/motifs_1k.txt','r').readlines() # n=1k
# FRAG_VOCAB = open('gym_molecule/dataset/motifs_91_all_att.txt','r').readlines() # n=91 with all possible attachment sites

FRAG_VOCAB = [s.strip('\n').split(',') for s in FRAG_VOCAB]
FRAG_VOCAB_WO_ATT = [remove_att(s[0]) for s in FRAG_VOCAB]
FRAG_VOCAB_MOL = [Chem.MolFromSmiles(s[0]) for s in FRAG_VOCAB]
FRAG_VOCAB_ATT = [get_att_points(m) for m in FRAG_VOCAB_MOL]
SCAFFOLD_VOCAB = [s.strip('\n') for s in SCAFFOLD_VOCAB]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: float(x == s), allowable_set))


def edge_feature(bond):
    bt = bond.GetBondType()
    return np.asarray([
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()])


def atom_feature(atom, use_atom_meta):
    if use_atom_meta == False:
        return np.asarray(
            one_of_k_encoding_unk(atom.GetSymbol(), ATOM_VOCAB)
        )
    else:
        return np.asarray(
            one_of_k_encoding_unk(atom.GetSymbol(), ATOM_VOCAB) +
            one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
            one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
            one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
            [atom.GetIsAromatic()])


# From GCPN
def convert_radical_electrons_to_hydrogens(mol):
    """
    Converts radical electrons in a molecule into bonds to hydrogens. Only
    use this if molecule is valid. Results a new mol object
    :param mol: rdkit mol object
    :return: rdkit mol object
    """
    m = copy.deepcopy(mol)
    if Chem.Descriptors.NumRadicalElectrons(m) == 0:  # not a radical
        return m
    else:  # a radical
        for a in m.GetAtoms():
            num_radical_e = a.GetNumRadicalElectrons()
            if num_radical_e > 0:
                a.SetNumRadicalElectrons(0)
                a.SetNumExplicitHs(num_radical_e)
    return m


def _CreateNamedOnMain(*args):
    import __main__
    namedtupleClass = collections.namedtuple(*args)
    setattr(__main__, namedtupleClass.__name__, namedtupleClass)
    namedtupleClass.__module__ = "__main__"
    return namedtupleClass


_CreateNamedOnMain('record', ['scaffold_mg',
                              'side_chain_mg',
                              'potential_subgraph_mg',
                              'scaffold_att_pos',
                              'side_chain_idx',
                              'side_chain_att_pos',
                              'other'])
