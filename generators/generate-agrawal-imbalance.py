#!/usr/bin/env python3

import os
import sys
import logging
from random import randrange

path = r'../'
if path not in sys.path:
    sys.path.append(path)

# from third_party.PEARL.src.stream_generator import RecurrentDriftStream
from stream_generator import RecurrentDriftStream

formatter = logging.Formatter('%(message)s')

def setup_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file, mode='w')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def generate(exp_code, concepts, imbalance_ratio):
    max_samples = 15000
    generator = 'agrawal'
    drift_type = 'abrupt'
    data_dir_prefix = f'../data/{exp_code}/'
    concepts_str = ''.join([str(v) for v in concepts])
    imbalance_ratio_str = str(imbalance_ratio)
    data_dir_suffix = f'{concepts_str}/{imbalance_ratio_str}/'

    for param in [(-1, "uniform")]:
        data_dir = f'{data_dir_prefix}/{data_dir_suffix}/'

        for seed in range(0, 10):
            print(f"generating {param[0]} seed {seed}")


            if not os.path.exists(data_dir):
                os.makedirs(data_dir)

            logger = setup_logger(f'seq-{param[0]}-{seed}', f'{data_dir}/drift-{seed}.log')

            if drift_type == "abrupt":
                stream = RecurrentDriftStream(generator=generator,
                                              # concepts=[0, 3, 7],
                                              concepts=concepts,
                                              has_noise=False,
                                              # noise_level=noise_level,
                                              # stable_period_lam=param[0],
                                              # stable_period_start=1000,
                                              # stable_period_base=200,
                                              stable_period=8000,
                                              position=8000,
                                              stable_period_logger=logger,
                                              random_state=seed)
            elif drift_type == "gradual":
                stream = RecurrentDriftStream(generator=generator,
                                              width=1000,
                                              concepts=[4, 0, 8, 6, 2, 1, 3, 5, 7, 9],
                                              has_noise=False,
                                              stable_period=6000,
                                              position=5000,
                                              stable_period_logger=logger,
                                              random_state=seed)
            else:
                print(f"Unknown drift type {drift_type}")
                exit()


            stream.prepare_for_use()
            print(stream.get_data_info())

            output_filename = os.path.join(data_dir, f'{seed}.arff')
            print(f'generating {output_filename}...')

            with open(output_filename, 'w') as out:
                out.write(stream.get_arff_header())

                for _ in range(max_samples):
                    X, y = stream.next_sample()

                    out.write(','.join(str(v) for v in X[0]))
                    out.write(f',{y[0]}')
                    out.write('\n')

exp_code = 'imbalance'
generate(exp_code, concepts=[7,0], imbalance_ratio=0.1)
generate(exp_code, concepts=[7,0], imbalance_ratio=0.9)
generate(exp_code, concepts=[7,0], imbalance_ratio=0.3)
generate(exp_code, concepts=[7,0], imbalance_ratio=0.7)
