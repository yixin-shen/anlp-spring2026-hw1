# Min-Llama Assignment 1

<div align="center">
  <img src="mini_llama.jpeg" alt="ANLP mini-llama" width="200">
</div>

**Acknowledgement:** This assignment is based on the corresponding assignment from Fall 2024 and the previous version by Vijay Viswanathan (based on the [minbert-assignment](https://github.com/neubig/minbert-assignment)).

This is an exercise in developing a minimalist version of Llama2, part of Carnegie Mellon University's [CS11-711 Advanced NLP](https://cmu-l3.github.io/anlp-spring2026/). See the course website for the [assignment writeup](https://cmu-l3.github.io/anlp-spring2026/assignment1).

## Overview

In this assignment, you will implement important components of the Llama2 model to better understand its architecture. You will pre-train the Llama model to perform 4-digit addition. 

## Assignment Details

### Your Task

You are responsible for implementing core components of Llama2 in the following files:
- `llama.py` - Main model architecture
<!--- `classifier.py` - Classification head-->
- `optimizer.py` - AdamW optimizer  
- `rope.py` - Rotary position embeddings

You are also responsible for implementing additional modules in the following files: 
- `addition_data_generation.py`
- `addition_run.py`

### Llama2

You will work with `stories42M.pt`, an 8-layer, 42M parameter language model pretrained on the [TinyStories](https://arxiv.org/abs/2305.07759) dataset (machine-generated children's stories). This model is small enough to train without a GPU, though using Colab or a personal GPU is recommended for faster iteration.

### Testing Your Implementation

Once you have implemented the components, you will test your model in four settings:

1. **Text Generation**: Generate completions starting with: *"I have wanted to see this thriller for a while, and it didn't disappoint. Keanu Reeves, playing the hero John Wick, is"*. You should see coherent, grammatical English (though content may be absurd due to the children's stories training data).

2. **Zero-shot Prompting**: Perform prompt-based sentiment analysis on SST-5 and CFIMDB datasets. This will give poor results (roughly random performance).

3. **4-Digit Addition**: Pre-train a tiny Llama model to answer addition questions.

5. **Advanced Implementation (A+ requirement)**: Given a fixed model capacity, implement the tiniest Llama architecture that can solve the addition problem with 100% accuracy on the given test set. The top 5% students with the smallest model size will qualify for extra credits (A+). 

### Important Notes

- Follow `setup.sh` to properly set up the environment and install dependencies
- See [structure.md](./structure.md) for detailed code structure descriptions
- Use only libraries installed by `setup.sh` - no external libraries (e.g., `transformers`) allowed
- The `data/cfimdb-test.txt` file contains placeholder labels (-1), so test accuracies may appear low
- Ensure reproducibility using the provided commands
- Do not change existing command options or add new required parameters
- Refer to [checklist.md](./checklist.md) for assignment requirements

## Reference Commands and Expected Results

### Text Generation
```bash
python run_llama.py --option generate
```
You should see continuations of the sentence `I have wanted to see this thriller for a while, and it didn't disappoint. Keanu Reeves, playing the hero John Wick, is...`. We will generate two continuations - one with temperature 0.0 (which should have a reasonably coherent, if unusual, completion) and one with temperature 1.0 (which introduces randomness for more creative and diverse outputs, though potentially less coherent).


### Zero-Shot Prompting

**SST Dataset:**
```bash
python run_llama.py --option prompt --batch_size 10 --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt --label-names data/sst-label-mapping.json --dev_out sst-dev-prompting-output.txt --test_out sst-test-prompting-output.txt [--use_gpu]
```
- Dev Accuracy: 0.197 (0.000)
- Test Accuracy: 0.176 (0.000)

**CFIMDB Dataset:**
```bash
python run_llama.py --option prompt --batch_size 10 --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt --label-names data/cfimdb-label-mapping.json --dev_out cfimdb-dev-prompting-output.txt --test_out cfimdb-test-prompting-output.txt [--use_gpu]
```
- Dev Accuracy: 0.514 (0.000)
- Test Accuracy: 0.232 (0.000)

### Addition Llama
Before running the following commands, ensure that your model capacity does not exceed 5,808,844,800,000 (see train command below).
You must run at least 5 distinct ablation experiments, plus one final best model. You must have at least two abalations that achieve lower than 100%. Each ablation should be saved in a unique directory named:
- addition_models/ablation_1
- addition_models/ablation_2
- …
- addition_models/ablation_N


After identifying the smallest model that achieves 100% accuracy on the test set, run the final experiment and save it to:
- addition_models/best_model
  
In total, you must submit at least 6 (5 ablations + 1 best model) experiments.

**Train**
```bash
python addition_run.py train --use_gpu --capcaity 5808844800000 --dim 16 --n_layers 6 --n_heads 4 --n_kv_heads 4 --epochs 10 --save_dir addition_models/best_model
```
**Test**
```bash 
python addition_run.py test --use_gpu --checkpoint addition_models/best_model/best_model.pth
```


## Submission Requirements

### File Structure
Submit a zip file with the following structure (replace `ANDREWID` with your Andrew ID):

```
ANDREWID/
├── addition_models/
├── addition_data_generation.py
├── addition_lib.py
├── addition_run.py
├── run_llama.py
├── base_llama.py
├── llama.py
├── rope.py
├── config.py
├── optimizer.py
├── sanity_check.py
├── tokenizer.py
├── utils.py
├── README.md
├── structure.md
├── checklist.md
├── sanity_check.data
├── generated-sentence-temp-0.txt
├── generated-sentence-temp-1.txt
├── sst-dev-prompting-output.txt
├── sst-test-prompting-output.txt
├── cfimdb-dev-prompting-output.txt
├── cfimdb-test-prompting-output.txt
├── [OPTIONAL] feedback.txt
└── setup.sh
```

### Submission Guidelines

- **File size limit**: 15MB maximum
- **Model weights**: Host separately and provide links in report if needed
- **Validation**: Use `prepare_submit.py` to create and validate your submission
- **Format compliance**: Improper format results in 1/3 grade reduction

### Reports and Feedback

- **Feedback**: Optional `feedback.txt` to help improve future assignments

### Grading

* **A+**: (Tiniest Addition LLaMA) The A+ grade is reserved for the top 5% of students who explore the boundaries of training and demonstrate a strong understanding of model capacity through systematic experimentation. You must perform multiple ablation studies and use them to identify the tiniest LLaMA architecture that achieves 100% accuracy on the test set, while remaining within the specified model capacity constraint. You are required to submit at least 5 different ablations, as well as your best-performing (smallest) model, with each experiment saved in a separate directory as instructed. Ensure that all experiments are executable using the provided commands. You must plot the results of all experiments and save training logs for each run. Submissions that do not include the required ablations, plots, and logs will not be eligible for an A+.

* **A**: You correctly implement all required components in: `llama.py`, <!--`classifier.py`-->, `optimizer.py`, `rope.py`, `addition_data_generation.py` and `addition_run.py`. Your implementation passes the provided sanity checks and produces: Coherent, grammatical text generation Correct execution of zero-shot prompting on SST-5 and CFIMDB A working addition LLaMA that successfully trains, evaluates, and achieves 100% accuracy on the test set. 

* **A-**: All required components are implemented and executable, but one or more of the following issues occur: Generated text is not coherent or not grammatically well-formed Zero-shot prompting or addition training runs but produces clearly incorrect or unstable results. The model fails to meaningfully learn the addition task. The pipeline runs end-to-end, but outputs do not meet expected qualitative or quantitative behavior.

* **B+**: All missing pieces are implemented and pass tests in `sanity_check.py` (llama implementation) and `optimizer_test.py` (optimizer implementation).

* **B or below**: Some parts of the missing pieces are not implemented.

If your results can be confirmed through the submitted files, but there are problems with your code submitted through Canvas, such as not being properly formatted or not executing in the appropriate amount of time, you will be graded down by 1/3 grade (e.g. A+ → A or A- → B+).

All assignments must be done individually, and we will be running plagiarism detection on your code. If we confirm that any code was plagiarized from that of other students in the class, you will be subject to strict measures according to CMU's academic integrity policy. That being said, *you are free to use publicly available resources* (e.g. papers or open-source code), but you ***must provide proper attribution***.


### Acknowledgement

This code is based on llama2.c by Andrej Karpathy. Parts of the code are also from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).
