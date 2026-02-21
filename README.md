<div align="center">
  <h2>TAROT: Task-Adaptive Refinement of LLM-prior Graphs for Few-shot Tabular Learning</h2>
</div>

## About our Work
Few-shot tabular learning provides a cost-effective approach for real-world applications where annotation is costly and collecting sufficient samples for new tasks is difficult. Existing Traditional and LLM-based methods have demonstrated effectiveness in few-shot scenarios. However, traditional methods need additional training on unlabeled or generated data, which incur significant computational overhead. In addition, LLM-based methods that directly feed raw tabular data into LLMs raise privacy and compliance concerns. More importantly, both paradigms largely overlook the semantic relationships between features, which provide structural and semantic prior for constructing a semantic graph. Semantic graph is essential for modeling meaningful feature interactions in few-shot scenarios. In this paper, we propose TAROT, a GNN-based framework that encodes the structural and semantic prior by constructing and refining a task-adaptive semantic graph from this prior, thereby improving predictive performance in few-shot tabular learning. TAROT first encodes heterogeneous tabular data into unified node semantic representations via a Unified Semantic Tabular Node Encoder (USTNE). Then, it prompts LLMs to infer the semantic relationship between features based on the task description and feature names to construct a semantic graph. To mitigate structural noise introduced by the hallucination of LLMs, TAROT introduces Task-adaptive Semantic Graph Refinement that prunes spurious or task-unrelated edges and adds missing task-related ones, aligning the graph structure with the downstream objective. Finally, a GNN performs message passing over the refined graph to capture task-related semantic dependencies for prediction. Extensive experiments on various few-shot tabular learning benchmarks demonstrate the superior performance of TAROT, establishing it as a state-of-the-art approach in this domain. 

## How to Run
1. Install requirements.
```
conda create -n TAROT python=3.9.19 
conda activate TAROT
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install -r requirements.txt
```

2. First, TAROT Use CoD to explicitly derive the tabular graph structure. Consider the following example:
```
python CoD.py --data [NAME_OF_DATASET] --shot [NUMBER_OF_FEW-SHOT_TABULAR_DATA_EXAMPLES]
```

3. Then, the running examples on tabular datasets are as follows:
```
python main_TAROT.py --data [NAME_OF_DATASET] --layer [LAYER_OF_GCN] --shot [NUMBER_OF_SAMPLES]

```


