import argparse
import os
import sys
import yaml
# sys.path.insert(0, '/data/renhong/mol-attack')

from docking_score.docking_simple import DockingVina


def docking_sc(smiles, target):
    with open(f'./docking_score/config_{target}.yaml', 'r') as f:
        docking_config = yaml.load(f, Loader=yaml.FullLoader)

    docking_config['vina_program'] = './' + docking_config['vina_program']
    predictor = DockingVina(docking_config)

    return predictor.predict(smiles)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='fa7', help='fa7, parp1, 5ht1b')

    args = parser.parse_args()

    smiles = ['Cc1cc(-c2csc(=O)[nH]2)cc(C)c1O']

    print(docking_sc(smiles, args.target))
