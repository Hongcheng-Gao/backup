""" Preprocess dataset for integer add logic task (minimal: only problem/answer) """
import argparse
import os
import random
from typing import List, Tuple, Set

from datasets import Dataset, concatenate_datasets
import pandas as pd


def int_to_base(n, base=10):
    if n == 0:
        return '0'
    digits = []
    while n > 0:
        digits.append(str(n % base))
        n = n // base
    return ''.join(reversed(digits))


def difficulty_range(digits, base=10):
    """
    Return [min_value, max_value] inclusive for a number with given digits in given base.
    """
    if digits <= 0:
        raise ValueError("digits must be positive")
    min_value = base ** (digits - 1)
    max_value = (base ** digits) - 1
    return min_value, max_value


def sample_one_pair(difficulty_list: List[List[int]], difficulty_weights: List[float], base: int, rng: random.Random):
    """
    Sample one (a, b) according to a sampled difficulty pair and base.
    """
    difficulty_choice = rng.choices(difficulty_list, weights=difficulty_weights)[0]
    if isinstance(difficulty_choice, (list, tuple)):
        digits_a, digits_b = difficulty_choice
    else:
        digits_a = digits_b = difficulty_choice

    range_start_a, range_end_a = difficulty_range(digits_a, base)
    range_start_b, range_end_b = difficulty_range(digits_b, base)
    a = rng.randint(range_start_a, range_end_a)
    b = rng.randint(range_start_b, range_end_b)
    return a, b, digits_a, digits_b


def make_problem_entry(a: int, b: int, base: int):
    str_a = int_to_base(a, base)
    str_b = int_to_base(b, base)
    str_result = int_to_base(a + b, base)

    prompt_text = (
        f"Add the two base-{base} numbers {str_a} and {str_b}. First, pad both numbers with leading zeros so they have the same number of digits. "
        f"Then, start from the rightmost (least significant) digit and perform the addition digit by digit from right to left. "
        f"For each digit position, add the corresponding digits of {str_a} and {str_b} (treating missing digits in the shorter number as 0). "
        f"Add any carry-over from the previous step. Record the sum digit (sum modulo {base}) and carry-over (sum divided by {base}) for the next position. "
        f"Continue this process until you have processed all digits of the longer number AND resolved any remaining carry-over (e.g., if adding 66 + 1 in base-7, you must account for the final carry to get 100 in base-7). "
        f"Present the final result as a base-{base} number, including all digits. Perform the calculation digit by digit in the {base}-base system without converting to base-10. "
        f"Put your final answer in \\boxed{{...}}.\n"
    )

    # 仅保留生成下游 map 所需的最小字段
    return {
        "messages": [
            {
                "role": "user",
                "content": prompt_text,
            },
        ],
        "base": base,
        "solution": f"{str_result}",
        "question_short": f"{str_a}+{str_b}",
    }


def generate_unique_problems(num_unique: int,
                             difficulty_list: List[List[int]],
                             difficulty_weights: List[float],
                             seed: int,
                             base: int,
                             existing_pairs: Set[Tuple[int, int]] = None) -> Dataset:
    """
    Generate 'num_unique' unique problems (by exact (a,b) unordered pair to avoid duplicates).
    If existing_pairs provided, ensure no overlap with them.
    """
    rng = random.Random(seed)
    problems = []
    seen = set() if existing_pairs is None else set(existing_pairs)

    # normalize weights
    if difficulty_weights is None:
        difficulty_weights = [1 / len(difficulty_list)] * len(difficulty_list)

    attempts = 0
    max_attempts = num_unique * 1000  # safety
    while len(problems) < num_unique and attempts < max_attempts:
        attempts += 1
        a, b, _, _ = sample_one_pair(difficulty_list, difficulty_weights, base, rng)
        key = (min(a, b), max(a, b))  # 无序对，避免 a+b 与 b+a 被视为不同
        if key in seen:
            continue
        seen.add(key)
        problems.append(make_problem_entry(a, b, base))

    if len(problems) < num_unique:
        raise RuntimeError(f"Could not generate {num_unique} unique problems after {attempts} attempts. "
                           f"Consider widening difficulty ranges.")

    df = pd.DataFrame.from_records(problems)
    return Dataset.from_pandas(df)


def repeat_dataset(ds: Dataset, times: int) -> Dataset:
    """
    Repeat a small dataset by concatenation 'times' times.
    """
    parts = [ds] * times
    return concatenate_datasets(parts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate addition problems with specified train/test scheme (only problem/answer)')
    # 训练集重复次数：最终训练条数 = 10 * repeat_times
    parser.add_argument('-n', '--repeat_times', type=int, default=100,
                        help='Repeat times for the 100 unique training problems (default: 100)')
    # test_size 固定 1000（保留参数不使用，以兼容旧命令）
    parser.add_argument('--test_size', type=int, default=500,
                        help='Number of unique test problems (will be forced to 500)')
    parser.add_argument('-d', '--digits_a', type=int, nargs='+', required=True,
                        help='List of digit counts for first operand (a)')
    parser.add_argument('-D', '--digits_b', type=int, nargs='+',
                        help='List of digit counts for second operand (b). If not provided, uses same as -d')
    parser.add_argument('-w', '--difficulty_weights', type=float, nargs='+', default=None,
                        help='Weights for each difficulty pair (must match length of digit lists)')
    parser.add_argument('-o', '--output_dir', type=str, default="/mnt/workspace/ghc/data/datasets/7bit",
                        help='Output directory (default: /mnt/workspace/ghc/data/datasets/7bit)')
    parser.add_argument('-s', '--seed', type=int, default=789,
                        help='Random seed for reproducible results (default: 789)')
    parser.add_argument('-b', '--base', type=int, default=7,
                        help='Base of the integer.')
    args = parser.parse_args()

    data_source = "integer_add_train"  # 仅用于内部标注，不会写入结果
    TRAIN_UNIQUE = 100
    TRAIN_REPEAT = max(1, args.repeat_times)
    TEST_UNIQUE = 500  # 固定 1000，覆盖 --test_size

    # Prepare difficulty list
    digits_a = args.digits_a
    digits_b = args.digits_b if args.digits_b is not None else digits_a
    if args.digits_b is not None and len(digits_a) != len(digits_b):
        parser.error("When both -d and -D are provided, they must have the same length")
    difficulty_list = [[a, b] for a, b in zip(digits_a, digits_b)]

    if args.difficulty_weights and len(args.difficulty_weights) != len(difficulty_list):
        parser.error("Number of weights must match number of digit pairs")

    # 1) 生成训练集的 10 道唯一题
    train_unique_ds = generate_unique_problems(
        num_unique=TRAIN_UNIQUE,
        difficulty_list=difficulty_list,
        difficulty_weights=args.difficulty_weights,
        seed=args.seed,  # 固定种子确保可复现
        base=args.base
    )

    # 记录训练基集的 (a,b) 键，以避免测试集重复
    train_pairs_str = set()
    for ex in train_unique_ds:
        train_pairs_str.add(ex["question_short"])          # "A+B"
        # 也加入 "B+A" 作为镜像，避免交换顺序导致重复
        A, B = ex["question_short"].split("+")
        train_pairs_str.add(f"{B}+{A}")

    # 2) 生成测试集 1000 道唯一题，与训练基集不重叠
    rng = random.Random(args.seed + 99991)  # 与训练不同的种子空间
    if args.difficulty_weights is None:
        difficulty_weights = [1 / len(difficulty_list)] * len(difficulty_list)
    else:
        difficulty_weights = args.difficulty_weights

    test_records = []
    seen_pairs_str = set(train_pairs_str)  # 初始化时包含训练对，确保不重叠
    attempts, max_attempts = 0, TEST_UNIQUE * 2000
    while len(test_records) < TEST_UNIQUE and attempts < max_attempts:
        attempts += 1
        a, b, _, _ = sample_one_pair(difficulty_list, difficulty_weights, args.base, rng)
        # 以字符串键在目标进制下判重
        str_a = int_to_base(a, args.base)
        str_b = int_to_base(b, args.base)
        key1 = f"{str_a}+{str_b}"
        key2 = f"{str_b}+{str_a}"
        if key1 in seen_pairs_str or key2 in seen_pairs_str:
            continue
        seen_pairs_str.add(key1)
        seen_pairs_str.add(key2)
        test_records.append(make_problem_entry(a, b, args.base))

    if len(test_records) < TEST_UNIQUE:
        raise RuntimeError(f"Could not generate {TEST_UNIQUE} unique test problems after {attempts} attempts. "
                           f"Consider widening difficulty ranges.")

    test_unique_ds = Dataset.from_pandas(pd.DataFrame.from_records(test_records))

    # 3) 训练集扩增：将 10 道唯一题重复 TRAIN_REPEAT 次
    train_dataset = repeat_dataset(train_unique_ds, TRAIN_REPEAT)

    # 打印检查
    print("Train unique size:", len(train_unique_ds))
    print("Train repeated size:", len(train_dataset))
    print("Test unique size:", len(test_unique_ds))
    print("Sample train example (raw):", train_dataset[0])
    print("Sample test example (raw):", test_unique_ds[0])

    # 只保留 problem 与 answer 两个字段
    def make_map_fn():
        def process_fn(example, idx):
            user_question = example["messages"][0]["content"]
            solution = example["solution"]
            return {
                "problem": user_question,
                "answer": solution,
            }
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn(), with_indices=True, remove_columns=train_dataset.column_names)
    test_dataset = test_unique_ds.map(function=make_map_fn(), with_indices=True, remove_columns=test_unique_ds.column_names)

    print("Mapped train example:", train_dataset[0])
    print("Mapped test example:", test_dataset[0])

    # 输出
    local_dir = args.output_dir
    os.makedirs(os.path.expanduser(local_dir), exist_ok=True)
    train_path = os.path.join(local_dir, "train.parquet")
    test_path = os.path.join(local_dir, "test.parquet")
    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)
    print(f"Saved train to: {train_path}")
    print(f"Saved test to: {test_path}")