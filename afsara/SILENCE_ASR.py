#!/usr/bin/env/python3
"""
Recipe for "direct" (speech -> semantics) SLU.
We encode input waveforms into features using the wav2vec2/HuBert model,
then feed the features into a seq2seq model to map them to semantics.
(Adapted from the LibriSpeech seq2seq ASR recipe written by Ju-Chieh Chou, Mirco Ravanelli, Abdel Heba, and Peter Plantinga.)
Run using:
> python train_with_wav2vec2.py hparams/train_with_wav2vec2.yaml
Authors
 * Loren Lugosch 2020
 * Mirco Ravanelli 2020
 * Boumadane Abdelmoumene 2021
 * AbdelWahab Heba 2021
 * Yingzhi Wang 2021
For more wav2vec2/HuBERT results, please see https://arxiv.org/pdf/2111.02735.pdf
"""
import logging
import os

import whisper
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

process_id = os.getpid()
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO,
                        format=str(
                            process_id) + ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')
import random

import sys
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
import jsonlines
import ast
import pandas as pd

import torch.nn.functional as F
import torch.nn as nn

# for mask generator
from transformers.modeling_outputs import BaseModelOutput
from typing import Optional, Tuple, Union
from transformers.models.hubert.modeling_hubert import HubertModel
from transformers.models.hubert.configuration_hubert import HubertConfig
import torchaudio.transforms as T

torch.manual_seed(0)
max_input_dim = 100000
max_bs = 16

def str2df(str_):
    data_list = eval(str_)
    ent_df = pd.DataFrame(data_list)
    return ent_df


def get_parameter_number(nets):
    total_num = 0
    trainable_num = 0
    for net_id in nets:
        net = nets[net_id]
        total_num = total_num + sum(p.numel() for p in net.parameters())
        trainable_num = trainable_num + sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def print_model_parameter(models):
    for model in models:
        logging.info(model)
        model_params_total = get_parameter_number({model: models[model]})["Total"]
        model_params_trainable = get_parameter_number({model: models[model]})["Trainable"]
        logging.info(f"Number of {model} parameters: {model_params_total}, trainable: {model_params_trainable}")

def cut_dataset(data, percent, seed=1986):
    if seed is not None:
        torch.manual_seed(seed)

    if isinstance(data, dict):
        keys = list(data.keys())
        num_to_get = int(len(keys) * percent)

        # 使用随机排列的索引来选择键
        random_indices = torch.randperm(len(keys))[:num_to_get]
        random_keys = [keys[i] for i in random_indices]

        result = {key: data[key] for key in random_keys}
        return result
    elif isinstance(data, list):
        num_to_get = int(len(data) * percent)

        # 使用随机排列的索引来选择元素
        random_indices = torch.randperm(len(data))[:num_to_get]
        result = [data[i] for i in random_indices]
        return result
    else:
        print("Not supported data type")


import torch.nn as nn
import torch

def apply_mask_to_audio(audio_features, mask):
    # total_length = audio_features.shape[1]
    total_length = audio_features.shape[0]
    num_segments = mask.shape[1]
    segment_length = total_length // num_segments
    remainder = total_length % num_segments

    # 分割并应用mask
    audio_features = torch.tensor(audio_features).unsqueeze(0)
    processed_audio = torch.zeros_like(audio_features)
    for i in range(num_segments):
        start_idx = i * segment_length
        end_idx = start_idx + segment_length
        if i == num_segments - 1:  # 对于最后一个segment，包含所有剩余部分
            end_idx += remainder
        processed_audio[:, start_idx:end_idx] = audio_features[:, start_idx:end_idx] * mask[:, i]

    # 处理多余部分（如果有的话）
    if remainder > 0:
        processed_audio[:, -remainder:] = audio_features[:, -remainder:]

    return processed_audio

# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2SamePadLayer with Wav2Vec2->Hubert
class HubertSamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states):
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states

# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2PositionalConvEmbedding with Wav2Vec2->Hubert
class HubertPositionalConvEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
        )

        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        self.conv = weight_norm(self.conv, name="weight", dim=2)

        self.padding = HubertSamePadLayer(config.num_conv_pos_embeddings)
        from transformers.activations import ACT2FN
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)
        hidden_states = self.activation(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states

class ShallowHubertModel(HubertModel):
    def __init__(self, config: HubertConfig):
        super().__init__(config)
        self.pos_conv_embed = HubertPositionalConvEmbedding(config)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        x: Optional[int] = None  # 新添加的参数，用于指定encoder层数
    ) -> Union[Tuple, BaseModelOutput]:

        # 调用原始HubertModel的forward方法来获得特征提取器的输出
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None:
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)

        hidden_states = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices)

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # 如果x为None，则使用所有层，否则只使用指定数量的层
        if x is not None:
            # 确保x不超出encoder层数量的范围
            x = min(x, len(self.encoder.layers))

            # 初始化输出的hidden_states列表，如果需要输出所有hidden states
            all_hidden_states = () if output_hidden_states else None

            # 仅通过指定数量的encoder层
            for i in range(x):
                layer = self.encoder.layers[i]

                layer_outputs = layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )

                hidden_states = layer_outputs[0]

                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

            # 将最后一层的输出作为encoder输出
            encoder_outputs = (hidden_states, all_hidden_states) if output_hidden_states else (hidden_states,)
        else:
            # 使用所有encoder层
            encoder_outputs = self.encoder(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        hidden_states = encoder_outputs[0]

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

from dongqi.diffmask.distributions import RectifiedStreched, BinaryConcrete
class MLPMaxGate(torch.nn.Module):
    def __init__(self, input_size, hidden_size, max_activation=10, bias=True):
        super().__init__()
        self.f = torch.nn.Sequential(
            torch.nn.utils.weight_norm(torch.nn.Linear(input_size, hidden_size)),
            torch.nn.Tanh(),
            torch.nn.utils.weight_norm(torch.nn.Linear(hidden_size, 1, bias=bias)),
            torch.nn.Tanh(),
        )
        self.bias = torch.nn.Parameter(torch.tensor(5.0))
        self.max_activation = max_activation

    def forward(self, *args):
        return self.f(torch.cat(args, -1)) * self.max_activation + self.bias

# Define model for audio mask generation
class AudioMaskGenerator(nn.Module):

    def __init__(self, shallow_hubert_config, output_dim, freeze_shallow_hubert=True):
        super(AudioMaskGenerator, self).__init__()
        self.shallow_hubert = ShallowHubertModel(shallow_hubert_config)
        self.shallow_hubert.load_state_dict(pretrained_hubert.state_dict(), strict=False)
        # self.linear = nn.Linear(768, output_dim)  # 假设ShallowHubertModel的输出维度为768
        self.gate = MLPMaxGate(768, 128).to(device)

        # freeze shallow hubert
        if freeze_shallow_hubert:
            for param in self.shallow_hubert.parameters():
                param.requires_grad = False

        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=1600
        )

    def forward(self, audio_feature, x):
        # 通过ShallowHubertModel
        hidden_states = self.shallow_hubert(audio_feature, x=x)[0] # torch.Size([4, 280, 768])

        # hidden_states = self.mel_spectrogram(audio_feature).transpose(1, 2) dui ok了好像 额

        logits = self.gate(hidden_states) # torch.Size([4, 280, 1])

        dist = RectifiedStreched(
            BinaryConcrete(torch.full_like(logits, 0.2), logits), l=-0.2, r=1.0,
        )

        gates_full = dist.rsample().cumprod(-1) # torch.Size([4, 280, 1])
        expected_L0_full = dist.log_expected_L0().cumsum(-1) # torch.Size([4, 280, 1])

        fold = 1

        # 从 gates_full 中选择 num_selected 个元素，按照等差的方式选择。
        gates_full_fold = select_elements_with_stride(gates_full, fold=fold) # torch.Size([4, 56, 1])
        expected_L0_full_fold = select_elements_with_stride(expected_L0_full, fold=fold) # torch.Size([4, 56, 1])

        print("gates_full_fold:", gates_full_fold.shape, "fold", fold)

        return gates_full_fold, expected_L0_full_fold

def select_elements_with_stride(gates_full, fold=2):
    """
    从 gates_full 中选择 num_selected 个元素，按照等差的方式选择。

    参数：
    - gates_full: 输入的张量，形状为 torch.Size([batch_size, xx, 1])
    - flod: 要折叠的数量

    返回：
    - gates_selected: 选择后的张量，形状为 torch.Size([batch_size, xx // fold, 1])
    """
    # 获取第二个维度的大小
    dim_size = gates_full.size(1)
    num_selected = dim_size // fold

    # 选择等差的索引
    indices = np.linspace(0, dim_size - 1, num_selected, dtype=int)

    # 在第二个维度上选择这些索引
    gates_selected = gates_full[:, indices, :]

    return gates_selected

import torch

def discretize_tensor(tensor, threshold):
    """
    离散化张量中的值。高于阈值的元素变为 1, 低于阈值的元素变为 0。

    参数:
    tensor -- 要离散化的 PyTorch 张量
    threshold -- 用于离散化的阈值

    返回:
    离散化后的张量
    """
    tensor = tensor.to(torch.device("cpu"))
    return torch.where(tensor > threshold, torch.tensor(1.0), torch.tensor(0.0))

def padding_transcript_tokens(transcript_tokens):
    tmp = tuple([transcript_tokens[0], padding(transcript_tokens[1])])
    transcript_tokens = 0
    transcript_tokens = tmp
    return transcript_tokens

def padding(tokens):
    max_len = 0
    for token in tokens:
        if len(token) > max_len:
            max_len = len(token)
    for token in tokens:
        if len(token) < max_len:
            token += [0]*(max_len-len(token))
    return tokens
# 示例用法
# tensor = torch.tensor([...])  # 用你的张量替换这里
# thresholded_tensor = discretize_tensor(tensor, 0.2)
# print(thresholded_tensor)


import numpy as np
class SLU(sb.Brain):
    def compute_forward(audio_path):
        """Forward computations from the waveform batches to the output probabilities."""
        audio = whisper.load_audio(audio_path)
        wavs, wav_lens = audio, len(audio)

        diffmask = True
        if diffmask:
            encoder_layers = 3
            gates_full, expected_L0_full = mask_generator(wavs, encoder_layers)

            gates_full = discretize_tensor(gates_full, 0.2).to(device)

            # print zero ratio in each gate dimension
            gate_ratio = np.array([])
            for i in range(gates_full.shape[0]):
                gate_ratio = np.append(gate_ratio, (gates_full[i, :] <0.5).sum().item() / gates_full[i, :].numel())
            print("gate_ratio:", gate_ratio)

            wavs_gated = apply_mask_to_audio(wavs, gates_full) # masked batch

            wavs = wavs_gated
            return wavs # aben
            # /data/cdq/current_project/speechbrain/speechbrain/speechbrain/lobes/models/huggingface_wav2vec.py line 311 -> 328
            # /home/cdq/.conda/envs/speechbrain/lib/python3.9/site-packages/transformers/models/hubert/modeling_hubert.py line 1100
            # FeatureEncoder: line 341
        else:
            # Add augmentation if specified
            if stage == sb.Stage.TRAIN:
                if hasattr(self.hparams, "augmentation"):
                    wavs = self.hparams.augmentation(wavs, wav_lens)

        #  encoder forward pass
        # wav2vec2_out = self.modules.wav2vec2(wavs)

        # if stage == sb.Stage.TEST:
        transcript_tokens = self.hparams.asr_model.transcribe_batch(
            wavs, wav_lens
            )
        transcript_tokens = padding_transcript_tokens(transcript_tokens)
        transcript_tokens = torch.LongTensor(transcript_tokens[1])
        transcript_tokens = transcript_tokens.to(self.device)

        embedded_transcripts = self.hparams.input_emb(transcript_tokens)
        encoder_out = self.hparams.slu_enc(embedded_transcripts)
        # SLU forward pass
        e_in = self.hparams.output_emb(tokens_bos)
        h, _ = self.hparams.dec(e_in, encoder_out, wav_lens)

        # print(e_in.shape)
        # print(wav2vec2_out.shape)
        # print(wav_lens.shape)

        # Output layer for seq2seq log-probabilities
        logits = self.hparams.seq_lin(h)
        p_seq = self.hparams.log_softmax(logits)


        # Compute outputs
        if (
            stage == sb.Stage.TRAIN
            and self.batch_count % show_results_every != 0
        ):
            return p_seq, wav_lens
        else:
            p_tokens, scores = self.hparams.beam_searcher(
                encoder_out, wav_lens
            )
            return p_seq, wav_lens, p_tokens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (NLL) given predictions and targets."""

        if (
            stage == sb.Stage.TRAIN
            and self.batch_count % show_results_every != 0
        ):
            p_seq, wav_lens = predictions
        else:
            p_seq, wav_lens, predicted_tokens = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos

        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        )

        print("p_seq.shape", p_seq.shape)
        print("tokens_eos.shape", tokens_eos.shape)
        print("tokens_eos_lens.shape", tokens_eos_lens.shape)
        loss = loss_seq

        if (stage != sb.Stage.TRAIN) or (
            self.batch_count % show_results_every == 0
        ):
            # Decode token terms to words
            predicted_semantics = [
                tokenizer.decode_ids(utt_seq).split(" ")
                for utt_seq in predicted_tokens
            ]

            target_semantics = [wrd.split(" ") for wrd in batch.semantics]

            self.log_outputs(predicted_semantics, target_semantics)

            if stage != sb.Stage.TRAIN:
                self.wer_metric.append(
                    ids, predicted_semantics, target_semantics
                )
                self.cer_metric.append(
                    ids, predicted_semantics, target_semantics
                )

            if stage == sb.Stage.TEST:
                # write to "predictions.jsonl"
                predictions_path = hparams["output_folder"] + "/test"
                epoch = hparams["number_of_epochs"]
                if not os.path.exists(predictions_path):
                    os.makedirs(predictions_path)
                with jsonlines.open(
                    hparams["output_folder"] + f"predictions_{global_epoch}.jsonl", mode="a"
                ) as writer:
                    print(f"save predictions.json to {hparams['output_folder']}  predictions_{global_epoch}.jsonl")
                    for i in range(len(predicted_semantics)):
                        try:
                            _dict = ast.literal_eval(
                                " ".join(predicted_semantics[i]).replace(
                                    "|", ","
                                )
                            )
                            if not isinstance(_dict, dict):
                                _dict = {
                                    "scenario": "none",
                                    "action": "none",
                                    "entities": [],
                                }
                        except SyntaxError:  # need this if the output is not a valid dictionary
                            _dict = {
                                "scenario": "none",
                                "action": "none",
                                "entities": [],
                            }
                        _dict["file"] = id_to_file[ids[i]]
                        writer.write(_dict)

        return loss

    def log_outputs(self, predicted_semantics, target_semantics):
        """ TODO: log these to a file instead of stdout """
        for i in range(len(target_semantics)):
            logging.info(" ".join(predicted_semantics[i]).replace("|", ","))
            logging.info(" ".join(target_semantics[i]).replace("|", ","))
            logging.info("")

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.wav2vec2_optimizer.step()
            self.optimizer.step()
        self.wav2vec2_optimizer.zero_grad()
        self.optimizer.zero_grad()
        self.batch_count += 1
        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        self.batch_count = 0

        if stage != sb.Stage.TRAIN:

            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["WER"])
            (
                old_lr_wav2vec2,
                new_lr_wav2vec2,
            ) = self.hparams.lr_annealing_wav2vec2(stage_stats["WER"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            sb.nnet.schedulers.update_learning_rate(
                self.wav2vec2_optimizer, new_lr_wav2vec2
            )
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr": old_lr,
                    "wave2vec_lr": old_lr_wav2vec2,
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            if self.checkpointer is not None:
                self.checkpointer.save_and_keep_only(
                    meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
                )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)

    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"
        self.wav2vec2_optimizer = self.hparams.wav2vec2_opt_class(
            self.modules.wav2vec2.parameters()
        )
        self.optimizer = self.hparams.opt_class(self.hparams.model.parameters())

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "wav2vec2_opt", self.wav2vec2_optimizer
            )
            self.checkpointer.add_recoverable("optimizer", self.optimizer)

    def zero_grad(self, set_to_none=False):
        self.wav2vec2_optimizer.zero_grad(set_to_none)
        self.optimizer.zero_grad(set_to_none)


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["csv_train"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["csv_valid"], replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["csv_test"], replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]

    tokenizer = hparams["tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("semantics")
    @sb.utils.data_pipeline.provides(
        "semantics", "token_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(semantics):
        yield semantics
        tokens_list = tokenizer.encode_as_ids(semantics)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "semantics", "tokens_bos", "tokens_eos", "tokens"],
    )
    return train_data, valid_data, test_data, tokenizer

import  jiwer
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    from transformers import HubertModel, HubertConfig

    # Initializing a Hubert facebook/hubert-base-ls960 style configuration
    configuration = HubertConfig()

    # Initializing a model from the facebook/hubert-base-ls960 style configuration
    pretrained_hubert = HubertModel(configuration)

    # Accessing the model configuration
    configuration = pretrained_hubert.config
    output_dim = 100
    mask_generator = AudioMaskGenerator(configuration, output_dim, freeze_shallow_hubert=False).to(device)
    mask_model = torch.load('/Users/afsarabenazir/Downloads/speech_projects/whisper-timestamped-master/dongqi/mask_generator_bs16_whisper_timeframe-feature-extractor-slu-freeze_diffmask_fold_k3_watch_alpha--1.4440691471099854.pt', map_location='cpu')

    mask_generator.load_state_dict(mask_model)
    for param in mask_generator.parameters():
        param.requires_grad = False

    whisper_model = whisper.load_model('/Users/afsarabenazir/Downloads/speech_projects/whisper-timestamped-master/models/large-v3.pt').float()  # vanilla model
    audio_path = '/Users/afsarabenazir/Downloads/speech_projects/whisper-timestamped-master/tests/data/meeting_tmrw_morn_at_10.wav'
    wavs = SLU.compute_forward(audio_path)
    transcript = whisper.transcribe(whisper_model, wavs.squeeze(0), language='en')
    cloud_asr = transcript['text']
    # print('REF ', reference)
    print('HYP ', cloud_asr)

    #     wer = jiwer.wer(reference.lower(), cloud_asr.lower())
    #     if wer > 0.99:
    #         continue
    #     total_wer += wer
    #     num_entries += 1
    #     print(f'wer {wer} ------ running wer {total_wer/num_entries}')
    #     # PRIVACY EVAL
    #     for _, entity in str2df(row.entities).iterrows():
    #         if entity['entity'].lower() in cloud_asr.lower():  # entity found in cloud asr
    #             total_fn += 1
    #             print(f"FN {entity['entity']} is present")
    #         else:  # entity not present in cloud asr
    #             total_tp += 1
    #             print(f"TP {entity['entity']} is not present")
    #         total_ent += 1
    #         print('running true positive ', total_tp/total_ent)
    #         print('running false negative ', total_fn/total_ent)
    #         print()
    #
    # root = "/Users/afsarabenazir/Downloads/speech_projects/whisper-timestamped-master/"
    # audio_root_path = '/Users/afsarabenazir/Downloads/speech_datasets/slurp-wav'
    # df = pd.read_csv(os.path.join(root, 'whisper_timestamped/test-set-ent-only-headset.csv'))
    # df = deepcopy(df)
    # whisper_model = whisper.load_model('/Users/afsarabenazir/Downloads/speech_projects/whisper-timestamped-master/models/large-v3.pt').float()  # vanilla model
    # total_wer, num_entries = 0, 0
    # total_fn, total_tp, total_ent = 0, 0,0
    # for _, row in df.iterrows():
    #     if row['entities'] == "[]":  # no entity
    #         continue
    #     audio_path = os.path.join(audio_root_path, row['path'].replace('flac', 'wav'))
    #     reference = row['text']
    #     # audio_path = '/Users/afsarabenazir/Downloads/speech_projects/whisper-timestamped-master/tests/data/have_any_emails_arrived.wav'
    #     # audio_path = '/Users/afsarabenazir/Downloads/speech_projects/whisper-timestamped-master/tests/data/meeting_tmrw_morn_at_10.wav'
    #     wavs = SLU.compute_forward(audio_path)
    #     transcript = whisper.transcribe(whisper_model, wavs.squeeze(0), language='en')
    #     cloud_asr = transcript['text']
    #     print('REF ', reference)
    #     print('HYP ', cloud_asr)
    #
    #     wer = jiwer.wer(reference.lower(), cloud_asr.lower())
    #     if wer > 0.99:
    #         continue
    #     total_wer += wer
    #     num_entries += 1
    #     print(f'wer {wer} ------ running wer {total_wer/num_entries}')
    #     # PRIVACY EVAL
    #     for _, entity in str2df(row.entities).iterrows():
    #         if entity['entity'].lower() in cloud_asr.lower():  # entity found in cloud asr
    #             total_fn += 1
    #             print(f"FN {entity['entity']} is present")
    #         else:  # entity not present in cloud asr
    #             total_tp += 1
    #             print(f"TP {entity['entity']} is not present")
    #         total_ent += 1
    #         print('running true positive ', total_tp/total_ent)
    #         print('running false negative ', total_fn/total_ent)
    #         print()
    #

