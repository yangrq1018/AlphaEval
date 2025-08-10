from openai import OpenAI
import os
from typing import Optional
from FAMA.backtester import FactorBacktester
from FAMA.experience_chain import ExperienceChainSet, ExperienceChain
from FAMA.selection import CrossSampleSelection
from qlib.data import D
import numpy as np
from copy import deepcopy
import json
from typing import Literal

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

function_definition = "\
            Abs(x), Log(x), Sign(x) = standard definitions; same for the operators “+”, “-”, “*”, “/”, “**”\n\
            Ref(x, d) = value of x d days ago\n\
            Corr(x, y, d) = time-serial correlation of x and y for the past d days\n\
            Cov(x, y, d) = time-serial covariance of x and y for the past d days\n\
            Delta(x, d) = today’s value of x minus the value of x d days ago\n\
            WMA(x, d) = weighted moving average over the past d days with linearly decaying weights d, d – 1, …, 1 (rescaled to sum up to 1)\n\
            Min(x, d) = time-series min over the past d days\n\
            Max(x, d) = time-series max over the past d days\n\
            IdxMax(x, d) = which day Max(x, d) occurred on\n\
            IdxMin(x, d) = which day Min(x, d) occurred on\n\
            Rank(x, d) = time-series rank in the past d days\n\
            Sum(x, d) = time-series sum over the past d days\n\
            Std(x, d) = moving time-series standard deviation over the past d days\n\
            Greater(x, y) = 1 if x > y, else 0\n\
            Less(x, y) = 1 if x < y, else 0" 


SYSTEMPROMPT = f"""You are an alpha generator. You should follow the following rules:\n\
    1. The inputs are the alpha factors that are currently performing well, \
        and you are required to output a new alpha factor that is generated from the fusion of these factors, \
        and your factor must be different from the input factor.\n\
    2. Do not repeat example answer.\n\
    3. You should return new different factors in a json array.\n\
    4. The specific function is defined as follows: {function_definition}\n\
    5. Follow the path in "improve_path". -> Indicates that the following factors have better performance than the previous factors. \
        You should refer it to build new alpha."""

def train_model(
    last_expressions: list[str],
    last_datas: np.ndarray,
    last_chain_set: ExperienceChainSet,
    start_date: str,
    end_date: str,
    instruments: Optional[list[str]] = None,
    freq: str = "day",
    mode: Literal["normal", "unittest"] = "normal",
    unitttest_factors: Optional[list[str]] = None):
    """
    Train a model using the specified parameters.
    Parameters
    ----------
    start_date : str
        The start date for the training data.
    end_date : str
        The end date for the training data.
    instruments : Optional[list[str]]
        List of instruments to train on. If None, defaults to all instruments.
    freq : str
        Frequency of the data (e.g., 'day').
    """
    new_factors = []
    new_datas = []
    new_ics = []
    new_chain_set = ExperienceChainSet()
    generate_factor_num = 3
    cross_sample_selection = CrossSampleSelection(10)
    print("Fitting cross sample selection with last datas.")
    print(f"last_datas shape: {np.array(last_datas).shape}", flush=True)
    cross_sample_selection.fit(np.array(last_datas.T))
    selected_indices = cross_sample_selection.sample_select(10)
    print(f"Selected indices for new factor generation: {selected_indices}")
    for idx in selected_indices:
        chain = last_chain_set.match(last_expressions[idx], last_datas[:,idx])
        if chain is not None:
            print(f"Selected chain for expression {idx}, {last_expressions[idx]}: {chain}")
        if mode == "unittest":
            if unitttest_factors is None:
                raise ValueError("unitttest_factors must be provided in unittest mode.")
            print(f"Using unittest factors: {unitttest_factors}")
            gen_new_factors = unitttest_factors
        else:
            gen_new_factors = gen_factor(
                example_factor=last_expressions[idx],
                chain=chain,
                generate_factor_num=generate_factor_num
            )
            print(f"Generated new factors: {gen_new_factors}")
        for new_factor in gen_new_factors:
            ic, data = evaluate_factor(
                factor_expr=new_factor,
                start_date=start_date,
                end_date=end_date,
                instruments=instruments,
                freq=freq
            )
            new_factors.append(new_factor)
            new_datas.append(data)
            new_ics.append(ic)
            if ic > max(chain.ics):
                print(f"New factor {new_factor} has better IC {ic} than chain's max IC {max(chain.ics)}")
                chain = deepcopy(chain)  # Create a copy of the chain to avoid modifying the original
                chain.insert(new_factor, data, ic)
                new_chain_set.insert(chain)
            else:
                print(f"Skipping factor {new_factor} with IC {ic} as it does not improve over the chain's max IC {max(chain.ics)}")
    return new_factors, np.stack(new_datas, axis=1), new_chain_set
        

def gen_factor(example_factor: str, chain: ExperienceChain, generate_factor_num: int) -> str:
    """
    Generate a factor using the OpenAI API.
    This function uses the OpenAI API to generate a factor expression based on the system prompt.
    """
    prompt = f"""alphas: ["{example_factor}"]\n
                 generate_factor_num: {generate_factor_num}\n
                 improve_path: "{chain}"\n"""
    try_times = 0
    while try_times < 10:
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEMPROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.5
            )
            factor_exprs = response.choices[0].message.content.strip()
            print(f"Generated factor expression: {factor_exprs}")
            factor_exprs = json.loads(factor_exprs[7:-3])
            return factor_exprs
        except Exception as e:
            print(f"Error generating factor: {e}")
            try_times += 1
            if try_times >= 10:
                raise RuntimeError("Failed to generate factor after 10 attempts.")
            print(f"Retrying... ({try_times}/10)")

def evaluate_factor(
    factor_expr: str,
    start_date: str,
    end_date: str,
    instruments: Optional[list[str]] = None,
    freq: str = "day",
):
    """
    Evaluate a factor using the specified parameters.
    
    Parameters
    ----------
    factor_expr : str
        The factor expression to evaluate.
    start_date : str
        The start date for the evaluation data.
    end_date : str
        The end date for the evaluation data.
    instruments : Optional[list[str]]
        List of instruments to evaluate on. If None, defaults to all instruments.
    freq : str
        Frequency of the data (e.g., 'day').
    """
    tester = FactorBacktester(factor_expr=factor_expr,
                              start_date=start_date,
                              end_date=end_date,
                              instruments=instruments,
                              freq=freq)
    tester.load_data()
    tester.calculate_rank_ic()
    return np.abs(tester.rank_ic), np.nan_to_num(tester.factor_data.values, nan=0.0).reshape(-1)

def train(
    m: int,
    start_date: str,
    end_date: str,
    instruments: Optional[list[str]] = None,
    freq: str = "day"):
    # init
    # 选取alpha158中的20个alpha作为初始因子
    last_expressions = [
        '($close-$open)/$open',
        '($high-$low)/$open',
        '($close-$open)/($high-$low+1e-12)',
        '($high-Greater($open, $close))/$open',
        '($high-Greater($open, $close))/($high-$low+1e-12)',
        '(Less($open, $close)-$low)/$open',
        '(Less($open, $close)-$low)/($high-$low+1e-12)',
        '(2*$close-$high-$low)/$open',
        '(2*$close-$high-$low)/($high-$low+1e-12)',
        '$open/$close',
        '$high/$close',
        '$low/$close',
        '$vwap/$close',
        'Ref($close, 5)/$close',
        'Mean($close, 60)/$close',
        'Std($close, 10)/$close',
        'Max($high, 20)/$close',
        'Min($low, 30)/$close',
        'Rank($close, 5)',
        '($close-Min($low, 10))/(Max($high, 10)-Min($low, 10)+1e-12)',
        'IdxMax($high, 20)/20',
        'IdxMin($low, 30)/30',
        '(IdxMax($high, 60)-IdxMin($low, 60))/60',
        'Corr($close, Log($volume+1), 5)',
        'Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 10)',
        'Mean($close>Ref($close, 1), 20)',
        'Mean($close<Ref($close, 1), 30)',
        'Mean($close>Ref($close, 1), 60)-Mean($close<Ref($close, 1), 60)',
        'Sum(Greater($close-Ref($close, 1), 0), 5)/(Sum(Abs($close-Ref($close, 1)), 5)+1e-12)',
        'Sum(Greater(Ref($close, 1)-$close, 0), 10)/(Sum(Abs($close-Ref($close, 1)), 10)+1e-12)',
        '(Sum(Greater($close-Ref($close, 1), 0), 20)-Sum(Greater(Ref($close, 1)-$close, 0), 20))/(Sum(Abs($close-Ref($close, 1)), 20)+1e-12)',
        'Mean($volume, 30)/($volume+1e-12)',
        'Std($volume, 60)/($volume+1e-12)',
        'Std(Abs($close/Ref($close, 1)-1)*$volume, 5)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, 5)+1e-12)',
        'Sum(Greater($volume-Ref($volume, 1), 0), 10)/(Sum(Abs($volume-Ref($volume, 1)), 10)+1e-12)',
        'Sum(Greater(Ref($volume, 1)-$volume, 0), 20)/(Sum(Abs($volume-Ref($volume, 1)), 20)+1e-12)',
        '(Sum(Greater($volume-Ref($volume, 1), 0), 60)-Sum(Greater(Ref($volume, 1)-$volume, 0), 60))/(Sum(Abs($volume-Ref($volume, 1)), 60)+1e-12)'
    ]
    last_datas = []
    last_ics = []
    last_chain_set = ExperienceChainSet()
    for expr in last_expressions:
        print(f"Evaluating factor: {expr}")
        ic, data = evaluate_factor(
            factor_expr=expr,
            start_date=start_date,
            end_date=end_date,
            instruments=instruments,
            freq=freq
        )
        print(f"IC for {expr}: {ic}")
        last_datas.append(data)
        last_ics.append(ic)
    # stack (n_samples,) datas to (n_samples, n_factors)
    last_datas = np.stack(last_datas, axis=1)
    print(f"data shape: {last_datas.shape}")
    last_ics = np.array(last_ics)
    print("Warmup the experience chain set with initial factors.")
    last_chain_set.warmup(
        last_expressions, last_datas, last_ics,
        CrossSampleSelection(10).fit(last_datas.T), k=10
    )
    for i in range(m):
        print(f"Training iteration {i + 1}/{m}")
        last_factors, last_datas, last_chain_set = train_model(
            last_expressions,
            last_datas,
            last_chain_set,
            start_date,
            end_date,
            instruments=instruments,
            freq=freq
        )
    return last_factors, last_chain_set