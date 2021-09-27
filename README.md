# KLUE LEVEL2 NLP Team 5
# ㅇㄱㄹㅇ

## Arguments

### Basic Setting

```bash
--data_dir {data_directory}   # default: /opt/ml/dataset/
--model_dir {model_directory} # default: /opt/ml/saved/

--name {model_name}           # please set the model name

--dataset {dataset}           # dataset class name
--batch_size {batch_size}

--model {model_class}         # model class name -> first try to look up model/{model}.py, 
                              # then look up model/models.py

--optim {optim_type}          # optimizer type
--epochs {num_epochs}         # num epochs to be trained
--lr {learning_rate}          # learning rate (typically means the max lr)
```

### Additional Setting

If saved model directory is given to `--load_model`, then it will load the pretrained weights.

```bash
--seed {seed_value}              # default: 42
--verbose {y or n}               # y or n

--load_model {saved_model_path}  # if given, then the model will be loaded

--val_ratio {val_ratio}          # if 0.0 -> does not split
--val_batch_size {batch_size}    # default set to batch_size

--momentum {momentum}            # set momentum if momentum > 0.0
--log_every {log_every}          # loging & printing interval

--lr_type
--lr_gamma
--lr_decay_step
```

### WanDB Setting

```bash
--wandb_use {y or n}            # y if use wandb
--wandb_project {project_name}  # project_name
```