<div align="center">
  <h2>CoDTab: Chain of Dependencies-based Tabular Graph for Web Data Mining</h2>
</div>

## About our Work
Tabular data is one of the most prevalent forms of information on the Web, yet its complex and unknown dependencies between heterogeneous features make knowledge mining challenging. 
To better capture and model these dependencies, recent work attempts to transform tabular data into graph structures, but existing automatic construction methods often produce spurious edges and fail to explicitly capture task-related and task-independent dependencies.
Large language models provide a new opportunity to infer such dependencies with semantic reasoning, though their effectiveness is still limited by the large search space and the risk of hallucination. 
To address these challenges, we propose CoDTab, a novel Chain of Dependencies-based Tabular Graph framework. CoDTab first introduce a Training-Free Tabular Node Encoder (TNE) that maps heterogeneous tabular data into node-level semantic representations without extra parameters, thereby reducing training overhead. Then, CoDTab employs a carefully designed Chain of Dependencies (CoD) process, which combines dataset descriptions, tabular examples, instruction-guided dependency inference, and template-based graph generation to explicitly capture both task-related and task-independent dependencies in a large search space and mitigate the risk of hallucination. Finally, a message passing mechanism is applied to model the constructed tabular graphs for robust tabular representations. Extensive experiments on nine datasets demonstrate that CoDTab consistently outperforms state-of-the-art baselines, highlighting its effectiveness in advancing web tabular data mining.

## How to Run
1. Install requirements.
```
conda create -n CoDTab python=3.9.19 
conda activate CoDTab
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install -r requirements.txt
```

2. First, CoDTab Use CoD to explicitly derive the tabular graph structure. Consider the following example:
```
python CoD.py --data [NAME_OF_DATASET] --shot [NUMBER_OF_FEW-SHOT_TABULAR_DATA_EXAMPLES]
```

3. Then, the running examples on small-scale tabular datasets and large-scale tabular datasets are as follows:
```
python main_CoDTab_Large-scale.py --data [NAME_OF_DATASET] --layer [LAYER_OF_GCN]
python main_CoDTab_Large-scale.py --data [NAME_OF_DATASET] --layer [LAYER_OF_GCN] --shot [NUMBER_OF_SAMPLES]
```