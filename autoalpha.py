#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import qlib

# 1. Initialize Qlib
qlib.init(
    provider_uri="path/to/your/qlib_data",
    region="cn"
)

from qlib.data import D
from AutoAlpha.genetic import SymbolicTransformer
from AutoAlpha.config import functions_arity, FEATURE_LIST
from scipy.stats import spearmanr

# import warnings
# warnings.filterwarnings('ignore', category=RuntimeWarning)


def main(args):
    # 2. Prepare Qlib configuration
    qlib_config = {
        "data_client": D,
        "instruments": D.instruments(market="all"),
        "start_time": args.start_time,
        "end_time": args.end_time,
        "freq": "day",
    }

    # 3. Create and run SymbolicTransformer
    transformer = SymbolicTransformer(
        population_size=args.population_size,
        hall_of_fame=args.hall_of_fame,
        n_components=args.n_components,
        generations=args.generations,
        function_set=functions_arity.keys(),
        metric="pearson",
        parsimony_coefficient=0.0,
        qlib_config=qlib_config,
        feature_names=FEATURE_LIST,
        random_state=0
    )

    # 4. Collect best programs and compute metrics
    programs = transformer._best_programs
    records = []
    for idx, prog in enumerate(programs):
        expr = str(prog)
        ic = prog.fitness_
        records.append({"formula": expr, "IC": ic})

    # 5. Create DataFrame and save results
    result_df = pd.DataFrame(records)
    print(result_df)
    result_df.to_parquet("autoalpha_results.parquet")
    print("Results saved to autoalpha_results.parquet")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoAlpha SymbolicTransformer on Qlib data")
    parser.add_argument("--start_time", required=True, help="Data start time in YYYY-MM-DD format")
    parser.add_argument("--end_time", required=True, help="Data end time in YYYY-MM-DD format")
    parser.add_argument("--population_size", type=int, default=30, help="Population size")
    parser.add_argument("--hall_of_fame", type=int, default=10, help="Hall of fame size")
    parser.add_argument("--n_components", type=int, default=6, help="Number of components")
    parser.add_argument("--generations", type=int, default=3, help="Number of generations")
    args = parser.parse_args()
    main(args)
