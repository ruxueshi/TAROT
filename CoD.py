import openai
import utils
import json
import argparse

from utils import *
from utils_llm import *


openai_key = "YOUR API KEY"
openai.default_headers = {"x-foo": "true"}
os.environ["OPENAI_API_KEY"] = openai_key


def random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description='adapter')
    parser.add_argument('--data', type=str, default=f"heart")
    parser.add_argument('--shot', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1024)
    args = parser.parse_args()

    return args


args = parse_args()
path = f"./data/{args.data}.csv"

random_seed(args.seed)
df, _, X_train, X_test, _, y_train, y_test, target, label_list, is_cat = utils.get_dataset(args.data, args.shot, args.seed)
cat_cols, num_cols, bin_cols = load_single_data_all(path)
y_train_ = pd.DataFrame(y_train, columns=[target])
multiclass = True if len(label_list) > 2 else False

X_train, y_train = data_process(X_train, y_train, num_cols, cat_cols, bin_cols)
X_test, y_test = data_process(X_test, y_test, num_cols, cat_cols, bin_cols)

X_train = X_train.reset_index(drop=True)
y_train_ = y_train_.reset_index(drop=True)
data = pd.concat((X_train, y_train_), axis=1)
CATEGORICAL_FEATURES = cat_cols
columns_names = data.columns
NAME_COLS = ','.join(columns_names) + '\n'

prompt = ""
instruction = "You are an expert. Given the task description, task feature and the list of features and data examples, you are finding relations between task feature and feature and between features to solve the task.\n\n"
prompt += instruction
prompt += "Task:\n"
prompt += TASK_DICT[args.data] + "\n\n"

prompt += "Task Feature:\n"
prompt += target +"\n\n"

prompt += "Feature:\n"
meta_data_name = f"./data/{args.data}-metadata.json"
with open(meta_data_name, "r", encoding="utf8") as f:
    meta_data = json.load(f)

for columns_name in columns_names:
    for key in meta_data.keys():
        if columns_name == key.lower():
            prompt += key + ":"
            prompt += meta_data[key] + "\n"

prompt += "\nSamples:\n" + NAME_COLS

instruction = fr"""Step 1. Analyze all the causal relationship or tendency between features based on general knowledge and common sense within a short sentence to answer the task.
Step 2. Based on the above samples and Step 1’s results, generate a tensor with [{len(data.columns) - 1}, {len(data.columns) - 1}], which represents the relationships between features that contribute to solving the task when analyzed from different perspectives, one perspective is Feature-to-feature dependency (e.g., feature a ↔ feature b), the other is combined or derived effects (e.g., feature a + feature b ↔ task). If a relationship exists, it is assigned a value of 1; otherwise, it is assigned a value of 0.
Finally, output a piece of Python code that generates this tensor.
Python code example:
import numpy as np

feature_number = {len(data.columns)-1}
tensor = np.zeros((feature_number, feature_number), dtype=int)

# Head 1: Feature-to-feature dependency
dependencies = [...]
for i, j in dependencies:
    tensor[i, j] = tensor[j, i] = 1

# Head 2: Combined/derived effects
combined = [...]
for i, j in combined:
    tensor[i, j] = tensor[j, i] = 1

np.save(r"data\graph\{args.data}_{args.seed}_graph", tensor)
"""

# np.save(r"data\graph\{args.data}_{args.seed}_{args.shot}_graph", tensor)
prompt += instruction

text = query_gpt(prompt, openai_key, max_tokens=30, temperature=0, max_try_num=10, model="gpt-4o-mini")


finally_end = fr'np.save(r"data\graph\{args.data}_{args.seed}_graph", tensor)'
text = text.split('import numpy as np')[-1]
text = text.split(finally_end)[0]
text = "import numpy as np\n\n" + text + "\n" + finally_end

exec(text)



