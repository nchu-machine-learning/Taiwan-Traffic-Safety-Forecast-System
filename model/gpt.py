import warnings
warnings.simplefilter(action='ignore')
from utility.data import transformer_slice
from transformers import GPT2Config, GPT2LMHeadModel, default_data_collator, \
    TrainingArguments, Trainer
from datasets import Dataset
import torch
import numpy as np
import transformers
import os
def find_highest_checkpoint(directory):
    """
    Finds the checkpoint file with the highest value at the end in the given directory.

    Args:
        directory: The directory containing the checkpoint files.

    Returns:
        The path to the checkpoint file with the highest value, or None if no checkpoints found.
    """

    checkpoints = []
    for file in os.listdir(directory):
        if file.startswith("checkpoint-"):
            checkpoint_num = int(file.split("-")[1])
            checkpoints.append((checkpoint_num, file))

    if checkpoints:
        checkpoints.sort(reverse=True)
        highest_checkpoint = checkpoints[0][1]
        return os.path.join(directory, highest_checkpoint)
    else:
        return None

def GPT_fit(
    train,
    params: dict = {"e_embed": 256, "n_layers": 2, "n_head": 2},
    checkpoint_dir: str = '../output/result',
    trained: bool = False,
    device: torch.device = torch.device('cuda')
) -> tuple['transformers.Trainer', 'transformers.GPT2LMHeadModel', 'list']:
    """
    Train a GPT-like language model on the given training data or load from an existing checkpoint.

    Parameters
    ----------
    train : array-like
        The input training data. Should be a sequence of integers representing tokenized text.
    params : dict, optional
        Configuration parameters for the GPT model. Includes:
            - e_embed: int
                Embedding size for the model.
            - n_layers: int
                Number of transformer layers in the model.
            - n_head: int
                Number of attention heads in each layer.
    checkpoint_dir : str, optional
        The directory where the model checkpoints are saved. If a checkpoint exists, the model will be loaded from it.
        Default is '../output/result'.
    trained : bool, optional
        If True, the model is loaded from the `checkpoint_dir` without retraining. Default is False.
    device : torch.device, optional
        Set the device used for the model. 
        Default is torch.device('cuda')

    Returns
    -------
    trainer : transformers.Trainer
        The Trainer object used to train the model.
    model : transformers.GPT2LMHeadModel
        The trained or loaded GPT model.
    final_segment : list
        The final segment of the training data used for prediction purposes.
    """
    # Ensure checkpoint directory exists
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    VOCAB = int(np.array(train).max()) + 10
    EPOCHS = 20

    train_x_g, train_y_g, final_segment = transformer_slice(train, lookback=150)

 

   
    train_x_gpt = [list(map(int, x)) for x in train_x_g]
    train_y_gpt = [list(map(int, y)) for y in train_y_g]

    train_dataset = Dataset.from_dict({'input_ids': train_x_gpt, 'labels': train_y_gpt})


    if trained and os.path.exists(checkpoint_dir) and find_highest_checkpoint(checkpoint_dir) != None:
        highest_chec = find_highest_checkpoint(checkpoint_dir)
        print(f"Loading model from checkpoint: {highest_chec}")
        model = GPT2LMHeadModel.from_pretrained(highest_chec).to(device)
        trainer = None

    else:
        print("No checkpoint found. Training a new model...")
        config = GPT2Config(
            vocab_size=VOCAB,
            **params
        )
        model = GPT2LMHeadModel(config)
        data_collector = default_data_collator
        training_args = TrainingArguments(
            output_dir=checkpoint_dir,
            overwrite_output_dir=True,
            num_train_epochs=EPOCHS,
            prediction_loss_only=True,
            learning_rate=0.001,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collector,
            train_dataset=train_dataset
        )
        trainer.train()
    model.eval()
    return trainer, model, final_segment


def GPT_predict(
    model,
    final_segment,
    max_length,
    num_return_sequences=1,
    device=torch.device('cuda')    
    ) -> torch.tensor:
    """
    Generate predictions using a trained GPT-like model.

    Parameters
    ----------
    model : transformers.GPT2LMHeadModel
        The trained GPT model.
    final_segment : list
        The input sequence used as a prompt for generating predictions.
    max_length : int
        The maximum length of the generated sequence.
    num_return_sequences : int, optional
        The number of sequences to generate. Default is 1.

    Returns
    -------
    outputs : torch.Tensor
        A tensor containing the generated sequences.

    Notes
    -----
    The function uses the Hugging Face `generate` method for autoregressive text generation.
    """
    with torch.no_grad():
        outputs = model.generate(
            input_ids = torch.tensor([final_segment], device=device, dtype=torch.long),
            max_length = max_length,
            do_sample=True,
            num_return_sequences = num_return_sequences
        )
    return outputs
def get_desired_sequence(
        pred,
        final_segment,
        test
    ) -> list:
    """
    Extract the desired portion of the generated sequence.

    Parameters
    ----------
    pred : torch.Tensor
        The generated sequences returned by the `GPT_predict` function.
    final_segment : list
        The input sequence used as the prompt for prediction.
    test : array-like
        The ground truth sequence for comparison.

    Returns
    -------
    numpy.ndarray
        The extracted portion of the generated sequence, matching the length of the `test` sequence.

    Notes
    -----
    This function is used to align generated predictions with the ground truth for evaluation.
    """
    return pred.cpu().detach().numpy()[0][len(final_segment):][:len(test)]

def get_desired_sequence_by_len(
        pred,
        final_segment,
        length
    ) -> list:
    """
    Extract the desired portion of the generated sequence.

    Parameters
    ----------
    pred : torch.Tensor
        The generated sequences returned by the `GPT_predict` function.
    final_segment : list
        The input sequence used as the prompt for prediction.
    length : int
        The length of the output

    Returns
    -------
    numpy.ndarray
        The extracted portion of the generated sequence, matching the length of the `test` sequence.

    Notes
    -----
    This function is used to align generated predictions with the ground truth for evaluation.
    """
    return pred.cpu().detach().numpy()[0][len(final_segment):][:length]