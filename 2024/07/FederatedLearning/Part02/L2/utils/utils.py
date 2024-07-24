import math
import os
import json
import re
import warnings
import logging
import pandas as pd
from typing import List
from collections import OrderedDict
from typing import Callable, Dict, Tuple

import textwrap
from omegaconf import OmegaConf
from logging import WARNING, ERROR, LogRecord
import flwr as fl
from flwr_datasets import FederatedDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from flwr.common import Context
from flwr.common.typing import NDArrays, Scalar
from flwr.common.logger import ConsoleHandler, console_handler, FLOWER_LOGGER, LOG_COLORS
from hydra import compose, initialize
from omegaconf import DictConfig
from datasets import load_dataset, Dataset, load_from_disk
from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from peft.utils import prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def format_string(msg, char_width: int=50) -> str:
    return textwrap.fill(msg, char_width, subsequent_indent="\t")


######### print Hydra config as yaml ##################
def print_config(config: DictConfig):
    print(OmegaConf.to_yaml(config))

########## console logger with less white spaces #############
FLOWER_LOGGER.removeHandler(console_handler) # remove default handler
class ConsoleHandlerV2(ConsoleHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def format(self, record: LogRecord) -> str:
        """Format function that adds colors to log level."""
        if self.json:
            log_fmt = "{lvl='%(levelname)s', time='%(asctime)s', msg='%(message)s'}"
        else:
            log_fmt = (
                f"{LOG_COLORS[record.levelname] if self.colored else ''}"
                f"%(levelname)s {'%(asctime)s' if self.timestamps else ''}"
                f"{LOG_COLORS['RESET'] if self.colored else ''}"
                f": %(message)s"
            )
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

# Configure console logger
console_handlerv2 = ConsoleHandlerV2(
    timestamps=False,
    json=False,
    colored=True,
)
console_handlerv2.setLevel(logging.INFO)
FLOWER_LOGGER.addHandler(console_handlerv2)


########## Format dataset ###########

def format_dataset(dataset):
    dataset = dataset.remove_columns(['instruction'])
    dataset = dataset.rename_column("output", "response")
    dataset = dataset.rename_column("input", "instruction")
    return dataset


########### to filter out all warnigns from HF coming from client side #########
backend_setup = {"logging_level": ERROR, "log_to_driver": False}

################################ configs components #############################


def get_config(config_name: str):
    with initialize(config_path="../conf", version_base="1.1"):
        cfg = compose(config_name=config_name)

    return cfg


############################# visualize data partitions #######################
def visualize_partitions(fed_dataset: FederatedDataset):
    _ = fed_dataset.load_partition(0)
    num_partitions = fed_dataset.partitioners['train'].num_partitions
    
    plt.bar(range(num_partitions), [len(fed_dataset.load_partition(i)) for i in range(num_partitions)])
    plt.xticks(range(num_partitions))
    plt.xlabel("Partition ID")
    plt.ylabel("Number of examples")
    plt.title(f"IID partitioning into {num_partitions} partitions")


############################### Report communication costs #################

def compute_communication_costs(config, comm_bw_mbps: float = 20):
    model = get_model(config.model)

    trainable, all_parameters = model.get_nb_trainable_parameters()

    total_size = 4*all_parameters/(1024**2)
    trainable_size = 4*trainable/(1024**2)

    upload_time_total = total_size/(comm_bw_mbps/8)
    upload_time_finetune = trainable_size/(comm_bw_mbps/8)
    
    print(f"Full model:\n\t{all_parameters/1e6:.3f} M parameters\n\t{total_size:.2f} MB --> upload in {upload_time_total:.2f}s @ {comm_bw_mbps}Mbps")
    print(f"Finetuned model:\n\t{trainable/1e6:.3f} M parameters\n\t{trainable_size:.2f} MB --> upload in {upload_time_finetune:.2f}s @ {comm_bw_mbps}Mbps")
    # print(f"In a {comm_bw_mbps} Mbps channel --> {}")

    num_rounds = config.flower.num_rounds
    num_clients_per_round = int(config.flower.num_clients * config.flower.fraction_fit)
    print(f"Federated Learning setting: "
          f"\n\tNumber of rounds: {num_rounds}"
          f"\n\tNumber of clients per round: {num_clients_per_round}")
    
    print(f"-----------------------------------------------")
    print(f"Total Communication costs (Full model): {2*num_rounds*num_clients_per_round*total_size/1024:.1f} GB")
    print(f"Total Communication costs (Finetuning): {2*num_rounds*num_clients_per_round*trainable_size} MB")
    print(f"Communication savings: {all_parameters/trainable:.1f}x")


################################ model components #############################


def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Implement cosine annealing learning rate schedule."""

    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))


def get_model(model_cfg: DictConfig):
    """Load model with appropiate quantization config and
    other optimizations."""

    use_cuda = torch.cuda.is_available()
    quantization_config = None
    model_name = model_cfg.name
    if use_cuda:
        if model_cfg.quantization == 4:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        elif model_cfg.quantization == 8:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError(
                f"Use 4-bit or 8-bit quantization. You passed: {model_cfg.quantization}/"
            )

        model_name = model_cfg.name

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    if use_cuda:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=model_cfg.gradient_checkpointing
        )

    target_modules = model_cfg.lora.target_modules
    if target_modules:
        target_modules = list(target_modules)
    peft_config = LoraConfig(
        r=model_cfg.lora.peft_lora_r,
        lora_alpha=model_cfg.lora.peft_lora_alpha,
        lora_dropout=0.075,
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    peft_model = get_peft_model(model, peft_config)
    if not (use_cuda):
        peft_model.enable_input_require_grads()

    if model_cfg.gradient_checkpointing:
        model.config.use_cache = False

    return peft_model


################################ dataset components #############################


def formatting_prompts_func(example):
    output_texts = []
    # Constructing a standard Alpaca (https://github.com/tatsu-lab/stanford_alpaca#data-release) prompt
    mssg = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    for i in range(len(example["instruction"])):
        text = f"{mssg}\n### Instruction:\n{example['instruction'][i]}\n### Response: {example['response'][i]}"
        output_texts.append(text)
    return output_texts


def get_tokenizer_and_data_collator_and_propt_formatting(
    model_name: str, use_fast: bool, padding_side: str
):

    # From: https://huggingface.co/docs/trl/en/sft_trainer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=use_fast, padding_side=padding_side
    )

    tokenizer.pad_token = (
        tokenizer.bos_token if padding_side == "left" else tokenizer.eos_token
    )
    response_template_with_context = "\n### Response:"  # alpaca response tag
    response_template_ids = tokenizer.encode(
        response_template_with_context, add_special_tokens=False
    )[2:]
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, tokenizer=tokenizer
    )

    return tokenizer, data_collator, formatting_prompts_func


################################ client components #############################
# pylint: disable=too-many-arguments
class FlowerClient(
    fl.client.NumPyClient
):  # pylint: disable=too-many-instance-attributes
    """Standard Flower client for CNN training."""

    def __init__(
        self,
        model_cfg: DictConfig,
        train_cfg: DictConfig,
        trainset,
        tokenizer,
        formatting_prompts_func,
        data_collator,
        save_path,
    ):  # pylint: disable=too-many-arguments
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_cfg = train_cfg
        self.training_argumnets = TrainingArguments(**train_cfg.training_arguments)
        self.tokenizer = tokenizer
        self.formatting_prompts_func = formatting_prompts_func
        self.data_collator = data_collator
        self.save_path = save_path

        # instantiate model
        self.model = get_model(model_cfg)

        self.trainset = trainset

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the parameters of the current net."""

        state_dict = get_peft_model_state_dict(self.model)
        return [val.cpu().numpy() for _, val in state_dict.items()]

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implement distributed fit function for a given client."""
        set_parameters(self.model, parameters)

        new_lr = cosine_annealing(
            int(config["current_round"]),
            self.train_cfg.num_rounds,
            self.train_cfg.learning_rate_max,
            self.train_cfg.learning_rate_min,
        )

        self.training_argumnets.learning_rate = new_lr
        self.training_argumnets.output_dir = self.save_path

        evalset = None
        if self.train_cfg.evaluate_split:
            train_test = self.trainset.train_test_split(test_size=0.1, seed=1234)
            trainset = train_test['train']
            evalset = train_test['test']
        else:
            trainset = self.trainset

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_argumnets,
            max_seq_length=self.train_cfg.seq_length,
            train_dataset=trainset,
            eval_dataset=evalset,
            formatting_func=self.formatting_prompts_func,
            data_collator=self.data_collator,
        )

        metrics = {}
        if self.train_cfg.evaluate_split:
            eval_res = trainer.evaluate()
            metrics['eval_loss'] = eval_res['eval_loss']
            print(eval_res)

        # Do local training
        results = trainer.train()

        metrics = {**metrics, "train_loss": results.training_loss}

        return (
            self.get_parameters({}),
            len(self.trainset),
            metrics,
        )


def set_parameters(model, parameters: NDArrays) -> None:
    """Change the parameters of the model using the given ones."""
    peft_state_dict_keys = get_peft_model_state_dict(model).keys()
    params_dict = zip(peft_state_dict_keys, parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    set_peft_model_state_dict(model, state_dict)


def gen_client_fn(
    fds,
    tokenizer,
    formatting_prompts_func,
    data_collator,
    model_cfg: DictConfig,
    train_cfg: DictConfig,
    save_path: str,
) -> Callable[[str], FlowerClient]:  # pylint: disable=too-many-arguments
    """Generate the client function that creates the Flower Clients."""

    def client_fn(context: Context) -> FlowerClient:
        """Create a Flower client representing a single organization."""

        # Let's get the partition corresponding to the i-th client
        partition_id = int(context.node_config["partition-id"])
        client_trainset = fds.load_partition(partition_id, "train")
        client_trainset = client_trainset.remove_columns(["instruction"])
        client_trainset = client_trainset.rename_column("input", "instruction")
        client_trainset = client_trainset.rename_column("output", "response")
        return FlowerClient(
            model_cfg,
            train_cfg,
            client_trainset,
            tokenizer,
            formatting_prompts_func,
            data_collator,
            save_path,
        ).to_client()

    return client_fn


################################ server components #############################


# Get function that will be executed by the strategy's evaluate() method
# Here we use it to save global model checkpoints
def get_evaluate_fn(model_cfg, save_every_round, total_round, save_path):
    """Return an evaluation function for saving global model."""

    def evaluate(server_round: int, parameters, config):
        # Save model
        if server_round != 0 and (
            server_round == total_round or server_round % save_every_round == 0
        ):
            # Init model
            model = get_model(model_cfg)
            set_parameters(model, parameters)

            model.save_pretrained(f"{save_path}/peft_{server_round}")

        return 0.0, {}

    return evaluate


# Get a function that will be used to construct the config that the client's
# fit() method will receive
def get_on_fit_config():
    def fit_config_fn(server_round: int):
        fit_config = {"current_round": server_round}
        return fit_config

    return fit_config_fn


def fit_weighted_average(metrics):
    """Aggregation function for (federated) evaluation metrics."""
    # Multiply accuracy of each client by number of examples used
    losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"train_loss": sum(losses) / sum(examples)}





################################ Make bar plots ##########################
results_offline = {"7b/pretrained": 0.322, "7b/cen_full": 0.660, "7b/cen_10": 0.488, "7b/fl": 0.667}

def get_label(key: str, compact: bool = False) -> str:
    label_mapping = {"7b/pretrained": "Pre-trained" if compact else "Pre-trained model",
                    "7b/cen_10":"Finetuned Cen.\n(10% data)" if compact else "Finetuned model\n(centralized - 10% data)",
                    "7b/cen_full": "Finetuned Cen.\n(100% data)" if compact else "Finetuned model\n(centralized - 100% data)",
                    "7b/fl": "Finetuned FL\n(Flower)" if compact else "Finetuned model\n(Flower Federated)"}
    return label_mapping[key]


def make_plot(axs, data_keys, labels):
    for i, data_key in enumerate(data_keys):
        axs.bar([i], [results_offline[data_key]])

    axs.set_ylabel("Validation Accuracy")
    axs.set_xticks(range(len(labels)), labels)
    plt.show()


def visualize_results(results: List[str], compact: bool=False) -> None:
    _, axs = plt.subplots(figsize=(6, 4))
    make_plot(
        axs,
        results,
        [get_label(res, compact) for res in results]
    )

############################ Evaluation ########################
# Fixed seed
torch.manual_seed(2024)

INSTRUCTIONS = {
    'pubmedqa': {'task': 'mcq', 'partition': 'test', 'instructions': 'pubmedqa'},
}

pubmedqa_instruction = {
        "system": "As an expert doctor in clinical science and medical knowledge, can you tell me if the following statement is correct? Answer yes, no, or maybe.",
        "user": "The answer is:",
        "type": "task-oriented",
        "source": ""
    }

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def benchmark_factory(name):
    """
    Creates a benchmark object.

    :param name: str, with the benchmark name.
    return:
    """
    # Note: benchmark is instantiated *after* selection.
    factories = {
        "pubmedqa": ClosedPubMedQA,
    }
    if name not in factories:
        raise ValueError("Benchmark {} not found. \
                         Select one of the following: {}".format(name, list(factories.keys())))
    return factories[name](name)


class Benchmark:
    def __init__(self, name):
        """
        Class to implement a benchmark for evaluation.

        :param name: str, with the benchmark name.
        :param path: str (optional), the path to the benchmark data.
        :param splits: list of str, the splits of the data: train / test
        :param hub_name: str, the name of the HuggingFace hub dataset.
        :param dir_name: str, the name of the directory where the data is stored.
        :param train_data: HuggingFace Dataset, the train data.
        :param test_data: HuggingFace Dataset, the test data.
        :param generations: HuggingFace Dataset, the generations.
        :param subsets: list of str (optional), the subsets of the data to download from the HuggingFace hub.
        :param has_instruction: bool, whether the dataset already contains instructions.
        :param local_path: str (optional), the path to a directory holding train and test json local data files.
        """
        self.name = name
        self.path = None
        self.splits = None
        self.hub_name = None
        self.dir_name = None
        self.train_data = None
        self.test_data = None
        self.generations = None
        self.subsets = None
        self.has_instructions = False
        self.local_path = None

    def load_from_hub(self):
        """
        Downloads the benchmark data from the HuggingFace hub (for 1st time loading)
        This is specific to each benchmark and must be implemented in the extended class.
        """
        print(f'Downloading benchmark from HuggingFace hub ({self.hub_name}).')
        try:
            if self.subsets is None:
                load_dataset(self.hub_name,
                             cache_dir=os.path.join(ROOT_DIR, 'benchmarks', 'datasets'),
                             download_mode='force_redownload')
            else:
                for subset in self.subsets:
                    load_dataset(self.hub_name,
                                 subset,
                                 cache_dir=os.path.join(ROOT_DIR, 'benchmarks', 'datasets'),
                                 download_mode='force_redownload')
        except:
            raise ValueError("Default Huggingface loader failed for benchmark {}. \
                             Try implementing a custom load_from_hub function.".format(self.name))

    def load_data(self, partition='train'):
        """
        Loads benchmark data from a local directory, or from the HuggingFace hub if not yet downloaded.
        Based on the input partition type, instantiates the respective class attribute.

        :param path: str (optional), the path to the benchmark data.
        :param partition: str, the split of the data: train / test
        """
        print('='*50 + f'\nLoading data for benchmark {self.name}.\n')
        if partition not in self.splits:
            raise ValueError("Please provide a valid partition split: {}".format(self.splits))
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            self.load_from_hub()
        try:
            if self.subsets is None:
                if partition == 'train':
                    self.train_data = load_dataset(self.path, split=partition)
                elif partition in ['test', 'validation']:
                    self.test_data = load_dataset(self.path, split=partition)
            else:
                if partition == 'train':
                    self.train_data = aggregate_datasets(self.path, self.subsets, partition=partition)
                elif partition in ['test', 'validation']:
                    self.test_data = aggregate_datasets(self.path, self.subsets, partition=partition)

        except ValueError as e:
            print(e)
            raise ValueError("Couldn't load benchmark {} from local path.".format(self.name))

    def save_data(self, partition='train'):
        """
        Saves any preprocessing data partition.

        :param data: pd.DataFrame
        :param file_name: str
        """
        path = os.path.join('benchmarks', 'preprocessing', f"{self.name}_{partition}")
        print("Saving {} data to the following path: {}".format(self.name, path))
        if partition == 'train':
            pd.to_pickle(self.train_data, path)
        elif partition == 'test':
            pd.to_pickle(self.test_data, path)

    def preprocessing(self, partition='train'):
        """
        Applies a custom pre-processing over the partition.
        If instruction is provided, preprends it to the question
        Updates the train or test self attributes.

        :param _preprocess: function: dict -> dict, the preprocessing function to apply.
        :param partition: str, the split of the data: train / test
        """
        try:
            if partition == 'train':
                self.train_data = self.train_data.map(self.custom_preprocessing)
            elif partition in ['test', 'validation']:
                self.test_data = self.test_data.map(self.custom_preprocessing)
            else:
                raise ValueError("Please provide a valid partition split: train or test")
        except Exception as e:
            print(e)
            raise ValueError("Error when pre-processing {} {} data.".format(self.name, partition))

    def custom_preprocessing(self):
            """
            Wraps a pre-processing function (dict -> dict) specific to the benchmark.
            Needs to be overriden in the extended class.

            The return dictionary must contains keys 'prompt' & 'answer' for inference to work.
            """
            raise NotImplementedError('Implement custom_preprocessing() in a child class.')

    def add_instruction(self, instruction=None, cot_column=None, partition='train'):
        """
        Adds instructions to the data based on the input partition.

        :param instruction: dict, with the `system` and `user` instructions. If None, then it creates prompt with few shot
        :param cot_column: str, the column that has the CoT explanation behind the gold answer.
        :param partition: str, the split of the data: train / test
        """
        def _add_instruction(row):
            row['prompt'] = '{}\n{}\n{}\n'.format(
                instruction['system'],
                row['prompt'],
                instruction['user'])
            if cot_column:
                row['gold'] = '{}.\nThe answer is: {} ###'.format(row[cot_column], row['gold'])
            return row

        if partition == 'train':
            self.train_data = self.train_data.map(_add_instruction)
        elif partition == 'test' or partition == 'validation':
            self.test_data = self.test_data.map( _add_instruction)
        else:
            raise ValueError("Please provide a valid partition split: {}".format(self.splits))

    def add_generations(self, data):
        """
        Adds the generations to the respective class attribute as a HuggingFace Dataset.

        :param data: pd.DataFrame or HuggingFace Dataset
        """
        if isinstance(data, pd.DataFrame):
            self.generations = Dataset.from_pandas(data)
        elif isinstance(data, Dataset):
            self.generations = data

    def save_generations(self, benchmark_name, run_name):
        """
        Saves the generations in the respective directory.
        """
        path = os.path.join(ROOT_DIR, 'benchmarks', 'generations')
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        gen_path = os.path.join(path, f"{benchmark_name}-{run_name}.jsonl")

        self.generations.to_json(gen_path, orient="records")
        print("Stored {} generations to the following path: {}".format(self.name, gen_path))


    def load_generations(self, benchmark_name):
        """
        Loads the generations from the respective directory.
        """
        path = os.path.join(ROOT_DIR, 'benchmarks', 'generations', f"{self.name}_{benchmark_name}.json")
        if not os.path.exists(path):
            raise ValueError("No generations found for {} at path: {}. \
                             Please run inference first.".format(self.name, path))
        print("Loading {} generations from the following path: {}".format(self.name, path))
        self.generations = pd.read_json(path)


class ClosedPubMedQA(Benchmark):
    '''
    PubMedQA is a novel biomedical question answering (QA) dataset.
    Its task is to answer research biomedical questions with yes/no/maybe using PubMed abstracts.

    Huggingface card: https://huggingface.co/datasets/bigbio/pubmed_qa
    '''
    def __init__(self, name='pubmedqa') -> None:
        super().__init__(name)
        self.hub_name = "bigbio/pubmed_qa"
        self.dir_name = 'bigbio___pubmed_qa'
        self.path = os.path.join(ROOT_DIR, 'benchmarks', 'datasets', self.dir_name)
        self.splits = ['train', 'validation', 'test']
        self.subsets = ['pubmed_qa_labeled_fold0_source']
        self.num_options = 3

    @staticmethod
    def custom_preprocessing(row):
        context = '\n'.join(row['CONTEXTS'])
        row["prompt"] = f"{context}\n{row['QUESTION']}"
        row["gold"] = row['final_decision']
        row["long_answer"] = row["LONG_ANSWER"]
        return row


def aggregate_datasets(path, subsets, partition='train'):
    """
    Takes as input a Huggingface DatasetDict with subset name as key, and Dataset as value.
    Returns a pd.DataFrame with all subsets concatenated.

    :param subsets: list of str, the subsets of the data to download from the HuggingFace hub.
    :return: pd.DataFrame
    """
    dataframes = []
    for subset in subsets:
        subset_data = load_dataset(os.path.join(path, subset), split=partition)
        subset_df = pd.DataFrame(subset_data.map(lambda x: {'subset': subset, **x}))
        dataframes.append(subset_df)
    aggregate_df = pd.concat(dataframes, axis=0)
    aggregate = Dataset.from_pandas(aggregate_df)
    if '__index_level_0__' in aggregate.column_names:
        aggregate = aggregate.remove_columns('__index_level_0__')
    return aggregate


def tokenizer_param(tokenizer, target):
    """
    Determines the maximum number of tokens to generate for a given prompt and target.
    Also determines the stop sequence to use for generation.

    :param tokenizer: transformers.PreTrainedTokenizer, the tokenizer to use for inference
    :param target: str, the target to generate
    """
    stop_seq = ["###"]
    stop_seq.append(tokenizer.eos_token)
    max_new_tokens = len(tokenizer(target[0], add_special_tokens=False)['input_ids'])

    return max_new_tokens, stop_seq


def benchmark_infer(model, tokenizer, data, device):
    """
    Runs inference on a benchmark and stores generations in a pd.DataFrame.

    :param tokenizer: transformers.PreTrainedTokenizer, the tokenizer to use for inference
    :param data: HuggingFace Dataset, the dataset to run inference on

    return: pd.DataFrame, a DataFrame containing the scores for each answer
    """
    columns_to_save = ['prompt', 'gold']
    if 'subset' in data.features:
        columns_to_save.append('subset')
    predictions = pd.DataFrame(data, columns=data.features)[columns_to_save]
    predictions = predictions.assign(output="Null")
    temperature = 1.0

    inference_data = json.loads(predictions.to_json(orient='records'))
    data_loader = DataLoader(inference_data, batch_size=16, shuffle=False)

    batch_counter = 0
    for batch in tqdm(data_loader, total=len(data_loader), position=0, leave=True):
        prompts = [f"<|im_start|>question\n{prompt}<|im_end|>\n<|im_start|>answer\n" for prompt in batch["prompt"]]
        if batch_counter == 0:
            print(prompts[0])

        max_new_tokens, stop_seq = tokenizer_param(tokenizer, batch['gold'])

        outputs = []
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            output_ids = model.generate(inputs=input_ids, max_new_tokens=max_new_tokens, do_sample=False, top_p=1.0,
                                        temperature=temperature, pad_token_id=tokenizer.eos_token_id)
            output_ids = output_ids[0][len(input_ids[0]):]
            output = tokenizer.decode(output_ids, skip_special_tokens=True)
            outputs.append(output)

        for prompt, out in zip(batch["prompt"], outputs):
            predictions.loc[predictions['prompt'] == prompt, 'output'] = out
        batch_counter += 1

    return predictions


def benchmark_preparation(data_obj, partition):
    """
    Runs the benchmark preparation pipeline on a given benchmark.

    :param data_obj: benchmark.Benchmark, the benchmark to run the preparation pipeline on
    :param partition: str, the partition to run the preparation pipeline on
    """
    data_obj.load_data(partition=partition)
    data_obj.preprocessing(partition=partition)
    prompt_name = INSTRUCTIONS['pubmedqa']['instructions']

    instruction = pubmedqa_instruction
    print(f'Instruction used for evaluation: \n\t{instruction["system"]}\n\t{instruction["user"]}\n')

    data_obj.add_instruction(
        instruction=instruction,
        partition=partition)
    return prompt_name


def inference(base_model_name_path, peft_path=None, run_name='fl', quantization=4):
    # Load model and tokenizer
    if quantization == 4:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        torch_dtype = torch.float32
    elif quantization == 8:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        torch_dtype = torch.float16
    else:
        raise ValueError(
            f"Use 4-bit or 8-bit quantization. You passed: {quantization}/"
        )

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        quantization_config = None
        print(f"{quantization} bit quantization is chosen, but a GPU is not found, running on CPU without quantization.")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name_path,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype)
    if peft_path is not None:
        model = PeftModel.from_pretrained(model, peft_path, torch_dtype=torch_dtype).to(device)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name_path, use_fast=False)

    # Prepare data
    partition = INSTRUCTIONS['pubmedqa']['partition']
    data_obj = benchmark_factory('pubmedqa')
    benchmark_preparation(data_obj, partition)

    # Prediction
    predictions = benchmark_infer(model, tokenizer, data_obj.test_data, device)

    # Save results
    data_obj.add_generations(data=predictions)
    data_obj.save_generations(benchmark_name='pubmedqa', run_name=run_name)
    print(f'{len(predictions)} generations store.')


def load_jsonl(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Error decoding JSON for line: {line}")
    return data


def clean_double_answer(output):
    if "yesyes" in output:
        output = output.replace('yesyes', 'yes')
    elif "nono" in output:
        output = output.replace('nono', 'no')
    elif "yesno" in output:
        output = output.replace('yesno', 'yes')
    elif "noyes" in output:
        output = output.replace('noyes', 'no')
    output = clean_answer(output)
    return output


def clean_answer(output):
    output_clean = output.encode('ascii', 'ignore').decode('ascii')
    return output_clean


def eval(output_full, answer):
    output = output_full
    default = (2, output_full, answer)

    if "\n##" in output:
        try:
            output = output.split("\n##")[1].split("\n")[0].strip().lower()
        except Exception:
            return default
    if "###" in answer:
        try:
            answer = answer.split("answer is:")[1].split("###")[0].strip()
        except Exception:
            return default

    output = re.sub(r"[^a-zA-Z0-9]", " ", output).strip()
    output = re.sub(" +", " ", output)
    output = clean_double_answer(output)

    if output in ['a', 'b', 'c', 'd', 'e', 'yes', 'no']:
        return output == answer, output, answer
    else:
        return default


def accuracy_metric(data):
    acc, counter, error = 0, 0, 0
    preds, golds = [], []
    ignored_prompts = []
    for row in data:
        answer = row['gold'].lower()
        output = row['output'].lower()
        correct, pred, gold = eval(
            output, answer)

        preds.append(pred)
        golds.append(gold)

        if correct == 2:
            error += 1
            correct = 0
            ignored_prompts.append(row)
        else:
            acc += correct
            counter += 1

    accuracy = accuracy_score(preds, golds)

    return {
        "accuracy": accuracy,
        "correct": acc,
        "counted": counter,
        "ignored": ignored_prompts,
        "unable_to_find_answer": error,
        "total": len(data)
    }


def display(metric_dict, run_name):
    print("====================================")
    print(f'Report accuracy for {run_name}:')
    print(f'# Accuracy: {metric_dict["accuracy"]}')


def evaluate(gen_dir=f'{ROOT_DIR}/benchmarks/generations', run_name='fl'):
    # Load data
    path = f'{gen_dir}/pubmedqa-{run_name}.jsonl'
    run_name = path.split('/')[-1].split('.')[0]
    data = load_jsonl(path)

    # Run evaluation
    metrics = accuracy_metric(data)

    # Display
    display(metrics, run_name)