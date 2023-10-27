# ACT-SQL

This is the project containing the source code for the EMNLP2023 paper [*ACT-SQL: In-Context Learning for Text-to-SQL with Automatically-Generated Chain-of-Thought*](https://arxiv.org/abs/2310.17342) in **EMNLP 2023 findings**. If you find it useful, please cite our work.

    @misc{zhang2023actsql,
          title={ACT-SQL: In-Context Learning for Text-to-SQL with Automatically-Generated Chain-of-Thought}, 
          author={Hanchong Zhang and Ruisheng Cao and Lu Chen and Hongshen Xu and Kai Yu},
          year={2023},
          eprint={2310.17342},
          archivePrefix={arXiv},
          primaryClass={cs.CL}
    }

## Run ACT-SQL

1. Create the `data` directory and move the downloaded datasets into this directory.
2. Create the `plm` directory and move the downloaded pretrained sentence BERT models into this directory.
3. As for the multi-turn text-to-SQL dataset, run `multiturn.py` firstly to convert the dataset into the single-turn text-to-SQL dataset. Here is an example.

```
python multiturn.py --dataset sparc
```

4. Run `cot.py` to automatically generate the chain-of-thoughts for all examples in the train set. Here is an example.

```
python cot.py --dataset spider
```

5. Run `main.py` to run ACT-SQL on the dev set. Here is an example.

```
python main.py --dataset spider --cot
```
