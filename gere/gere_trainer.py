import torch
from transformers.trainer import *
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.trainer import logger
import transformers
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_pt_utils import LengthGroupedSampler
from dataclasses import dataclass
from transformers.data.data_collator import *
import json
import os
from typing import Sized, Iterator, Callable, List
from tqdm import tqdm
from datasets import load_dataset

# import ipdb

# import logging
# logging.basicConfig(
#     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#     datefmt="%m/%d/%Y %H:%M:%S",
#     level=logging.INFO,  # if training_args.local_rank in [-1, 0] else logging.WARN,
#     handlers=[logging.StreamHandler(sys.stdout)],
#     )
# transformers.utils.logging.set_verbosity(log_level)
# transformers.utils.logging.enable_default_handler()
# transformers.utils.logging.enable_explicit_format()

transformers.utils.logging.set_verbosity(transformers.logging.INFO)
torch.set_printoptions(edgeitems=6, linewidth=700, precision=5, sci_mode=True)


def save_json(obj, json_file, indent=2, verbose=True):
    json.dump(obj, open(json_file, 'w', encoding='U8'),
              ensure_ascii=False, indent=indent)
    if verbose: print(f'save json file ok! {json_file}')


def load_json(json_file):
    with open(json_file, 'r', encoding='U8') as f:
        json_data = json.load(f)
    return json_data


def save_jsonl(obj_lst, jsonl_file, verbose=True):
    """write data by line with json"""
    with open(jsonl_file, 'w', encoding='U8') as f:
        for obj in obj_lst:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')
    if verbose: print(f'save jsonl file ok! {jsonl_file}, length: {len(obj_lst)}')


def load_jsonl(jsonl_file):
    """read data by line with json"""
    with open(jsonl_file, 'r', encoding='U8') as f:
        return [json.loads(line.strip()) for line in f]


def check_tensor(loss: torch.Tensor, print_name='loss'):
    if loss.isnan().any().item():
        print(f'Detected! {print_name} has a NaN value:', loss)
    elif (loss == 0).all().item():
        print(f'Detected! All values of {print_name} are 0: ', loss)


def pad_sequence(sequences, padding_value=0.0, direction='right', force_pad_to_max_specific_len=None):
    if direction == 'right':
        padded_sequences = torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=padding_value)
    elif direction == 'left':
        # 先反转再pad再反转
        padded_sequences = torch.nn.utils.rnn.pad_sequence(
            [seq.flip(0) for seq in sequences], batch_first=True, padding_value=padding_value).flip(1)
    if force_pad_to_max_specific_len is not None:
        pad_len = force_pad_to_max_specific_len - padded_sequences.shape[1]  # 默认是[bsz,len,*] 长度在第二维
        if pad_len > 0:  # 如果是长度不够指定长度才开始补充
            if padded_sequences.dim() >= 3:  # 补hidden_states [bsz,len,hid]
                if direction == 'right':
                    pad = (0, 0, 0, pad_len)  # (最后一维左 最后一维右 倒数第二维左 倒数第二维右) 最后一维是hidden
                if direction == 'left':
                    pad = (0, 0, pad_len, 0)
            elif padded_sequences.dim() == 2:  # 补input_ids/labels [bsz,len]
                if direction == 'right':
                    pad = (0, pad_len)
                if direction == 'left':
                    pad = (pad_len, 0)
            padded_sequences = torch.nn.functional.pad(padded_sequences, pad, mode='constant', value=padding_value)
    return padded_sequences


def sequence_mask(lengths, maxlen=None, dtype=torch.bool, direction='right'):
    if not isinstance(lengths, torch.Tensor):
        lengths = torch.tensor(lengths)
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix
    mask.type(dtype)
    if direction == 'left':
        mask = mask.flip(1)
    return mask


@dataclass
class GeReDataCollator:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    ori_data_collator: Callable = None
    gere_hidden_state_saving_dir: Optional[str] = None  # dir saving gere hidden state
    do_load_gere_hidden_states: bool = True,  # whether to load gere hidden state in iteration
    pad_direction: str = 'right'
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    print_first_data: bool = True
    print_first_load_hidden_file: bool = True
    force_pad_to_max_specific_len: Optional[int] = None

    def __post_init__(self):
        print(f'GeReDataCollator.gere_hidden_state_saving_dir: {self.gere_hidden_state_saving_dir}')

    def __call__(self, batch, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        model_inputs = self.decoder_call(batch, return_tensors)
        # ipdb.set_trace()
        if self.print_first_data:
            # print data after collator. (decoded mode)

            if model_inputs['gere_model_inputs']:
                input_ids = model_inputs['gere_model_inputs']['gere_input_ids']
                labels = model_inputs['gere_model_inputs']['gere_labels']
            else:
                input_ids = model_inputs['ori_model_inputs']['input_ids']
                labels = model_inputs['ori_model_inputs']['labels']

            input_ids = input_ids[0]
            input_ids_str = self.tokenizer.decode(input_ids, skip_special_tokens=False)
            labels = labels[0]
            labels_ = labels.masked_fill(labels == self.label_pad_token_id, self.tokenizer.pad_token_id)  # replace the -100 with pad for decode
            labels_str = self.tokenizer.decode(labels_, skip_special_tokens=False)
            logger.info(f"\n>>  Inspect data after Collator (only one-time):\n"
                        f">>  ----input_ids(deocded)----:\n{input_ids_str}\n"
                        f">>  ----labels(deocded)----: (replaced -100 with pad_token)\n{labels_str}\n"
                        f">>  ----input_ids(ids)----: \n{input_ids.tolist()}\n"
                        f">>  ----labels(ids)----: \n{labels.tolist()}\n"
                        )
            # print(len(input_ids.tolist()))
            # print(len(labels.tolist()))
            self.print_first_data = False

        return model_inputs

    def decoder_call(self, batch, return_tensors):
        self.tokenizer.padding_side = 'left'
        LABEL_PAD_ID = self.label_pad_token_id  # -100
        PAD_ID = self.tokenizer.pad_token_id

        gere_batch = [e['GeRe'] for e in batch if e['GeRe'] is not None]
        ori_batch = [e for e in batch if e['GeRe'] is None]

        for e in ori_batch:
            e.pop('GeRe')

        model_inputs = {
            'ori_model_inputs': None,
            'gere_model_inputs': None
        }

        if ori_batch:
            if self.ori_data_collator is not None:
                model_inputs['ori_model_inputs'] = self.ori_data_collator(ori_batch)

        gere_input_ids = []
        gere_labels = []
        gere_ids = []  # for find and load gere hidden_state to return
        gere_hidden_states = []

        for e in gere_batch:
            max_gere_length = 512 + 50 + 1  # 512+50+1=563 兼容之前加了1
            text = e['text']
            tokenized_text = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            tokenized_text = tokenized_text[:max_gere_length]
            gere_input_ids.append(tokenized_text)
            gere_labels.append(tokenized_text)  # 正常训练抗遗忘的label

            gere_ids.append(e['id'])

        # pad_direction = 'right'
        pad_direction = self.pad_direction
        # ipdb.set_trace()

        if gere_input_ids:
            gere_seq_lens = [len(e) for e in gere_input_ids]
            gere_input_ids = pad_sequence([torch.tensor(e) for e in gere_input_ids],
                                          padding_value=PAD_ID, direction=pad_direction,
                                          force_pad_to_max_specific_len=self.force_pad_to_max_specific_len)
            gere_labels = pad_sequence([torch.tensor(e) for e in gere_labels],
                                       padding_value=LABEL_PAD_ID, direction=pad_direction,
                                       force_pad_to_max_specific_len=self.force_pad_to_max_specific_len)
            gere_attention_mask = sequence_mask(gere_seq_lens, direction=pad_direction,
                                                maxlen=self.force_pad_to_max_specific_len).long()

            gere_max_len = gere_input_ids.shape[1]
            # target_mask = (labels != LABEL_PAD_ID).long()

        if self.do_load_gere_hidden_states and gere_ids:  # find and load gere hidden_state to return
            for gere_id in gere_ids:
                # PT_SlimRepajama_train_1.pt
                gere_hidden_states_presaved_file = f'{self.gere_hidden_state_saving_dir}/{gere_batch[0]["dataset"]}_train_{gere_id}.pt'
                if os.path.exists(gere_hidden_states_presaved_file):
                    gere_hidden_states.append(torch.load(gere_hidden_states_presaved_file)['hidden_states'])
                    if self.print_first_load_hidden_file:
                        print(f'load hidden_states pt file success! First file is: {gere_hidden_states_presaved_file}')
                        self.print_first_load_hidden_file = False

        if gere_hidden_states:
            gere_hidden_states = pad_sequence(gere_hidden_states, padding_value=0., direction=pad_direction,
                                              force_pad_to_max_specific_len=gere_max_len)

        if gere_ids:
            model_inputs['gere_model_inputs'] = {
                'gere_input_ids': gere_input_ids,
                'gere_labels': gere_labels,
                'gere_attention_mask': gere_attention_mask,
                'gere_seq_lens': gere_seq_lens,
                # 'target_mask': target_mask,
                'gere_batch': gere_batch,
                'gere_hidden_states': gere_hidden_states,
            }

        return model_inputs


class GeRePresaveCallback(TrainerCallback):
    def __init__(self, trainer, gere_dataset, gere_dataset_name, gere_data_collator, gere_hidden_state_saving_dir):
        self.trainer = trainer
        self.gere_dataset = gere_dataset
        self.gere_dataset_name = gere_dataset_name
        self.gere_data_collator = gere_data_collator
        self.gere_hidden_state_saving_dir = gere_hidden_state_saving_dir

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of training.
        """
        model = kwargs['model']

        # tokenizer = kwargs.get('tokenizer', None)
        # tokenzier = self.trainer.tokenizer

        stats_file = f'{self.gere_hidden_state_saving_dir}/{self.gere_dataset_name}_mean_var.pt'
        if self.trainer.reuse_gere_hidden_state and os.path.exists(stats_file):
            print(f"Rank {args.local_rank} {stats_file} exists!")

        else:

            dataloader = DataLoader(self.gere_dataset, batch_size=1, shuffle=False, collate_fn=self.gere_data_collator)  # batch_size=1 to avoid padding

            print(f"Rank {args.local_rank} starting presave hidden states and statistics...")
            os.makedirs(self.gere_hidden_state_saving_dir, exist_ok=True)
            for batch in tqdm(dataloader, desc='iterating for generating gere hidden states'):
                batch = batch['gere_model_inputs']
                if batch is None:
                    continue
                input_ids = batch['gere_input_ids'].to(model.device)
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, output_hidden_states=True)

                # ipdb.set_trace()
                logits_lst = [t.cpu() for t in torch.unbind(outputs.logits, dim=0)]  # batch lst of [len,logit]
                hidden_states_lst = [t.float().cpu() for t in torch.unbind(outputs.hidden_states[-1], dim=0)]  # batch lst of [len,hid]
                # lm_head_weight = model.lm_head.weight.data.detach().cpu().float()  # [voc, hid]

                # assert len(logits_lst) == len(hidden_states_lst)
                for bdx, (l, h) in enumerate(zip(logits_lst, hidden_states_lst)):
                    task = batch['gere_batch'][bdx]['task']  # GeRe
                    dataset_name = batch['gere_batch'][bdx]['dataset']  # SlimRedpajama
                    idx = batch['gere_batch'][bdx]['id']
                    ret = {
                        # 'logits':l,
                        'hidden_states': h,
                    }
                    save_pt_file = f'{self.gere_hidden_state_saving_dir}/{dataset_name}_train_{idx}.pt'
                    if not os.path.exists(save_pt_file):
                        torch.save(ret, save_pt_file)
                        print(f'[Rank {args.local_rank}] Saving hidden state pt file: {save_pt_file}')

            if torch.distributed.is_initialized():
                torch.distributed.barrier()  # statistics after synchronize all processes

            if args.local_rank == 0:
                print(f"Rank {args.local_rank} starting statistics")
                total_sum = None
                total_squared_sum = None
                count = 0
                for idx in tqdm(range(len(self.gere_dataset))):
                    # print(f'processing {file_id}')
                    pt = torch.load(f'{self.gere_hidden_state_saving_dir}/{dataset_name}_train_{idx}.pt', map_location=torch.device('cpu'))
                    hidden_states = pt['hidden_states']  # [2048, 5120]
                    if idx == 0 and total_sum is None and total_squared_sum is None:
                        dim = hidden_states.shape[-1]
                        total_sum = torch.zeros(dim)
                        total_squared_sum = torch.zeros(dim)

                    count += hidden_states.shape[0]
                    sum_ = hidden_states.sum(0)  # 5120
                    squared_sum_ = (hidden_states ** 2).sum(0)  # 5120

                    total_sum += sum_
                    total_squared_sum += squared_sum_

                    cur_mean = total_sum / count
                    cur_var = total_squared_sum / count - cur_mean ** 2
                    # print(cur_mean, cur_var)

                dct = {
                    'mean': cur_mean,
                    'var': cur_var,
                    'std': cur_var ** 0.5,
                    'total_sum': total_sum,
                    'total_squared_sum': total_squared_sum,
                    'count': count
                }
                torch.save(dct, stats_file)
                print('saved statistics success! ', stats_file)

            if torch.distributed.is_initialized():
                torch.distributed.barrier()  # Synchronize again to ensure finish statistics

        print(f"Rank {args.local_rank} completed on_train_begin")
        self.trainer.init_margin_loss_thres()


class GeReTrainer(Seq2SeqTrainer):
    def __init__(self, *args, gere_hidden_state_saving_dir='./gere_saving', reuse_gere_hidden_state=True,
                 num_interpolate_per_batch=None, w_strategy='100', gere_data_file=None, **kwargs):
        CURRENT_DIR = os.path.dirname(__file__)
        self.num_interpolate_per_batch = num_interpolate_per_batch
        self.gere_hidden_state_saving_dir = gere_hidden_state_saving_dir
        self.w_strategy = w_strategy
        self.reuse_gere_hidden_state = reuse_gere_hidden_state
        self.args = kwargs['args']
        self.gere_data_file = gere_data_file
        if self.gere_data_file is None:
            self.gere_data_file = os.path.join(CURRENT_DIR, "slimpajama/slimpajama_6B_chunk0_head1k.jsonl")

        self.gere_dataset = load_dataset(
            os.path.join(CURRENT_DIR, "gere_dataset.py"),
            data_file=self.gere_data_file,
            gere_dataset_name='SlimRedpajama',
            num_gere_samples=1000,
            split='train',
            trust_remote_code=True,
        )
        # For compatibility
        if 'processing_class' in kwargs:
            tokenizer = kwargs['processing_class']
        else:
            tokenizer = kwargs['tokenizer']

        self.args.remove_unused_columns = False  # avoid removing 'GeRe' data

        # self.gere_dataset = self.gere_dataset.select(range(100))  # for test
        self.num_gere_dataset = len(self.gere_dataset)

        if self.num_interpolate_per_batch:
            # Using interpolation batch - Note: transformers v4.44+ with accelerator
            # automatically sets use_seedable_sampler=True by default, which prevents
            # our custom interpolation sampler from being used.
            if hasattr(self.args, 'accelerator_config'):  # hasattr For backward compatibility with other transformers versions
                self.args.accelerator_config.use_seedable_sampler = False

        self.gere_data_collator_for_only_saving = GeReDataCollator(
            tokenizer=tokenizer,
            model=kwargs['model'],
            do_load_gere_hidden_states=False,
            gere_hidden_state_saving_dir=self.gere_hidden_state_saving_dir,
        )

        self.gere_data_collator = GeReDataCollator(
            tokenizer=tokenizer,
            model=kwargs['model'],
            ori_data_collator=kwargs['data_collator'],
            gere_hidden_state_saving_dir=self.gere_hidden_state_saving_dir,
        )
        kwargs['data_collator'] = self.gere_data_collator

        self.gere_presaving_callback = GeRePresaveCallback(
            trainer=self,
            gere_dataset=self.gere_dataset,
            gere_dataset_name='SlimRedpajama',
            gere_data_collator=self.gere_data_collator_for_only_saving,
            gere_hidden_state_saving_dir=self.gere_hidden_state_saving_dir
        )

        if 'callbacks' not in kwargs or not kwargs['callbacks']:
            kwargs['callbacks'] = [self.gere_presaving_callback]
        else:
            kwargs['callbacks'].append(self.gere_presaving_callback)

        kwargs['train_dataset'] = datasets.concatenate_datasets([self.gere_dataset, kwargs['train_dataset']])
        # kwargs['train_dataset'].set_format(type="torch")

        # ipdb.set_trace()
        super().__init__(*args, **kwargs)
        self.ori_loss = None
        self.total_loss = None
        self.printed_first_margin_loss = False

        self.actual_batch_size = self._train_batch_size * self.args.gradient_accumulation_steps * self.args.world_size
        if self.num_interpolate_per_batch:
            assert self.num_interpolate_per_batch < self.actual_batch_size, \
                f'num_interpolate_per_batch should be smaller to actual batch size({self.actual_batch_size})'

    def init_margin_loss_thres(self):
        self.thr = {}
        for f in Path(self.gere_hidden_state_saving_dir).glob('*_mean_var.pt'):
            datasetname = f.stem.split('_')[0]
            logger.info(f"load mean/var/std file: {f} for dataset {datasetname}")
            self.thr[datasetname] = {}
            self.thr[datasetname]['stats'] = torch.load(f)
            logger.info(f"mean/var/std:  {self.thr[datasetname]['stats']}")
            self.thr[datasetname]['pos_thr'] = self.thr[datasetname]['stats']['mean'] + 1 * self.thr[datasetname]['stats']['std']  # [hid]
            self.thr[datasetname]['neg_thr'] = self.thr[datasetname]['stats']['mean'] - 1 * self.thr[datasetname]['stats']['std']  # [hid]

    def get_thres_by_datasetname(self, dsnames: List[str]):
        # Supports multiple types of GeRe replay data in future
        batch_pos_thr = []
        batch_neg_thr = []
        for dsname in dsnames:
            batch_pos_thr.append(self.thr[dsname]['pos_thr'])
            batch_neg_thr.append(self.thr[dsname]['neg_thr'])
        batch_pos_thr = torch.stack(batch_pos_thr, dim=0)[:, None, :]  # bsz,1,hid
        batch_neg_thr = torch.stack(batch_neg_thr, dim=0)[:, None, :]  # bsz,1,hid
        return batch_pos_thr, batch_neg_thr

    def check_tensor(self, *args, **kwargs):
        if self.model.training:
            check_tensor(*args, **kwargs)

    def calc_kl(self, pred_logit, label_logit, temp=2):
        pred_ = torch.nn.functional.log_softmax(pred_logit / temp, dim=-1)  # [bsz, len, vocab]

        label_ = torch.nn.functional.log_softmax(label_logit / temp, dim=-1)  # [bsz, len, vocab]

        kl_loss = torch.nn.functional.kl_div(pred_, label_, reduction='none', log_target=True)  # [bsz, len, vocab]
        kl_loss = torch.sum(kl_loss, -1)  # kl definition [bsz, len]
        kl_loss = torch.mean(kl_loss)  # kl token-level means
        self.check_tensor(kl_loss, print_name='kl_loss')
        return kl_loss

    def compute_gere_loss(self, model, gere_inputs, return_outputs=False):
        # gere_inputs = {
        #     'gere_input_ids': gere_input_ids,
        #     'gere_labels': gere_labels,
        #     'gere_attention_mask': gere_attention_mask,
        #     'gere_seq_lens': gere_seq_lens,
        #     'gere_batch': gere_batch,
        #     'gere_hidden_states': gere_hidden_states,
        # }
        inputs = gere_inputs
        input_ids = gere_inputs['gere_input_ids']
        labels = gere_inputs['gere_labels']
        attention_mask = gere_inputs['gere_attention_mask']

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # Custom loss calculation  ==========================================================================
        # Original↓↓↓
        #  Shift so that tokens < n predict n
        shift_logits = outputs['logits'][..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')  # default ignore_index=-100 → these labels will yield 0 loss
        shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        self.check_tensor(loss)
        # ipdb.set_trace()

        # We add below↓↓↓
        # manual and proper loss reduction accounting for padding (-100 labels)
        loss = loss.sum() / ((shift_labels != -100).sum().item() + 1e-8)  # Prevent division by zero
        outputs['loss'] = loss
        labels = None  # restore the scene
        # Custom loss calculation  ==========================================================================

        if labels is not None:
            # unwrapped_model = unwrap_model(model)
            unwrapped_model = self.accelerator.unwrap_model(model)  # 新版transformers 4.41
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        gere_ce_loss = loss

        gere_hidden_states = gere_inputs['gere_hidden_states'].to(attention_mask.device)
        hid_len, hid_dim = gere_hidden_states.shape[-2:]
        # 方法1.原来预测值
        pred_hidden = outputs.hidden_states[-1]  # [bsz, len, hidden] 原来用的pred_hidden

        batch_pos_thr, batch_neg_thr = self.get_thres_by_datasetname([e['dataset'] for e in gere_inputs['gere_batch']])

        pos_thr = batch_pos_thr.to(attention_mask.device)
        neg_thr = batch_neg_thr.to(attention_mask.device)

        label_neg = (gere_hidden_states <= neg_thr)  # [bsz, 2048, hidden]
        label_neu = (gere_hidden_states > neg_thr) & (gere_hidden_states < pos_thr)  # [bsz, 2048, hidden]
        label_pos = (gere_hidden_states >= pos_thr)  # [bsz, 2048, hidden]

        neg_loss = torch.relu(pred_hidden - neg_thr)  # [bsz, 2048, hidden]
        neu_loss = torch.relu(pred_hidden - pos_thr) + torch.relu(neg_thr - pred_hidden)  # [bsz, 2048, hidden]
        pos_loss = torch.relu(pos_thr - pred_hidden)  # [bsz, 2048, hidden]

        margin_loss = neg_loss * label_neg + label_neu * neu_loss + label_pos * pos_loss  # [bsz, 2048, hidden]

        hid_mask = attention_mask.unsqueeze(-1)
        margin_loss = margin_loss * hid_mask

        margin_loss = margin_loss.sum() / ((margin_loss != 0).sum() + 1e-8)
        if not self.printed_first_margin_loss:
            logger.info(f'======\nmargin_loss at the first step (should be 0.): {margin_loss.item()}\n')
            self.printed_first_margin_loss = True
        # self.check_tensor(margin_loss, print_name='margin_loss')

        loss_dct = {
            'gere_ce_loss': gere_ce_loss,
            'gere_margin_loss': margin_loss
        }

        return (loss_dct, outputs) if return_outputs else loss_dct

    def compute_loss(self, model, inputs, *args, return_outputs=False, **kwargs):
        device = model.device
        self.total_ce_loss = torch.tensor(0., requires_grad=True).to(device)
        self.total_loss = torch.tensor(0., requires_grad=True).to(device)

        outputs = None
        gere_model_inputs = inputs['gere_model_inputs']
        self.gere_margin_loss, self.gere_ce_loss = None, None
        if gere_model_inputs is not None:
            # print(gere_model_inputs['gere_batch'])
            loss_dct = self.compute_gere_loss(model, gere_model_inputs, return_outputs=return_outputs)  # self.margin_loss have been assigned
            if return_outputs:
                loss_dct, outputs = loss_dct
            self.gere_margin_loss = loss_dct['gere_margin_loss']
            self.gere_ce_loss = loss_dct['gere_ce_loss']
            self.total_ce_loss += self.gere_ce_loss
            # self.total_loss += (self.gere_margin_loss * 100)

        ori_model_inputs = inputs['ori_model_inputs']
        self.ori_loss = None
        if ori_model_inputs is not None:
            # print(ori_model_inputs['labels'])
            ori_loss = super().compute_loss(model, ori_model_inputs, return_outputs=return_outputs)  # Using parent method
            if return_outputs:
                ori_loss, outputs = ori_loss
            self.ori_loss = ori_loss
            self.total_ce_loss += ori_loss

        if gere_model_inputs is not None and ori_model_inputs is not None:
            self.total_ce_loss /= 2

        if gere_model_inputs:
            if self.w_strategy == 'dy':
                w_margin = self.total_ce_loss.item() / (self.gere_margin_loss.item() + 1e-8)
            else:
                w_margin = float(self.w_strategy)
            self.total_loss = self.total_ce_loss + w_margin * self.gere_margin_loss
        else:
            self.total_loss = self.total_ce_loss

        return (self.total_loss, outputs) if return_outputs else self.total_loss

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen

        for loss_name in [
            'ori_loss', 'total_loss',
            'gere_ce_loss', 'gere_margin_loss', 'total_ce_loss'
        ]:
            l = getattr(self, loss_name, None)
            if l is not None:
                if isinstance(l, torch.Tensor):
                    l = l.item()
                logs[loss_name] = l

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def _get_train_sampler(self, *args, **kwargs) -> Optional[torch.utils.data.Sampler]:
        # from torch.utils.data import SequentialSampler
        # return SequentialSampler(self.train_dataset)
        if self.num_interpolate_per_batch in [0, None] or self.num_gere_dataset in [0, None]:
            return super()._get_train_sampler(*args, **kwargs)  # Using parent method
        else:
            if self.train_dataset is None or not has_length(self.train_dataset):
                return None

            # Build the sampler.
            if self.args.group_by_length:
                if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                    lengths = (
                        self.train_dataset[self.args.length_column_name]
                        if self.args.length_column_name in self.train_dataset.column_names
                        else None
                    )
                else:
                    lengths = None
                model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
                return LengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=self.train_dataset,
                    lengths=lengths,
                    model_input_name=model_input_name,
                )

            else:
                return InterpolateRandomSampler(self.train_dataset,
                                                num_base_examples=len(self.train_dataset) - self.num_gere_dataset,
                                                num_extra_examples=self.num_gere_dataset,
                                                num_interpolate_per_batch=self.num_interpolate_per_batch,
                                                # train_batch_size=self._train_batch_size * self.args.gradient_accumulation_steps,
                                                train_batch_size=self._train_batch_size * self.args.gradient_accumulation_steps * self.args.world_size,
                                                )


class InterpolateRandomSampler(RandomSampler):
    # Note: The dataset must be organized with extra samples (GeRe data) in the front portion
    num_base_examples: int
    num_extra_examples: int
    num_interpolate_per_batch: float
    train_batch_size: int

    def __init__(self, data_source: Sized, replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None,
                 # custom configurations:↓↓↓
                 num_base_examples: int = None,
                 num_extra_examples: int = None,
                 num_interpolate_per_batch: int = 4,
                 train_batch_size: int = None) -> None:
        super().__init__(data_source,
                         replacement=replacement,
                         num_samples=num_samples,
                         generator=generator)
        logger.info(f'============\nInitialize InterpolateRandomSampler\nnum_base_examples:{num_base_examples}\nnum_extra_examples:{num_extra_examples}\n'
                    f'num_interpolate_per_batch:{num_interpolate_per_batch}\ntrain_batch_size:{train_batch_size}\n============')
        self.num_base_examples = num_base_examples
        self.num_extra_examples = num_extra_examples
        self.num_interpolate_per_batch = num_interpolate_per_batch
        self.train_batch_size = train_batch_size

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator
        # ipdb.set_trace()
        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()
        else:
            # for _ in range(self.num_samples // n):
            #     yield from torch.randperm(n, generator=generator).tolist()
            # yield from torch.randperm(n, generator=generator).tolist()[:self.num_samples % n]
            # original RandomSampler code↑↑↑
            interpolate_indices = self.interpolate(self.train_batch_size,
                                                   self.num_interpolate_per_batch,
                                                   self.num_base_examples,
                                                   self.num_extra_examples,
                                                   generator,
                                                   )
            yield from interpolate_indices

    def interpolate(self, train_batch_size, num_interpolate_per_batch, num_base_examples, num_extra_examples, generator):
        """ Generate interpolated indices with extra samples inserted randomly into base sample batches.
            Note: Dataset must be organized with extra samples first, followed by base samples.
        """
        logger.info('== Generating interpolate indices in InterpolateRandomSampler...')
        num_base_per_batch = train_batch_size - num_interpolate_per_batch  # thr remaining base samples per batch  (total_bsz - extra_bsz)
        # datasource has been processed with the extra data placed before the base data.
        base_rnd_indices = (torch.randperm(num_base_examples, generator=generator) + num_extra_examples).tolist()  # offset since extra samples at first
        extra_rnd_indices = torch.randperm(num_extra_examples, generator=generator).tolist()  # Generate randomized indices for extra samples
        # base_rnd_indices = [1] * num_base_examples  # test
        # extra_rnd_indices =[0] * num_extra_examples  # test
        # Split base indices into groups (each representing one batch)
        grouped_base_rnd_indices = [base_rnd_indices[i:i + num_base_per_batch] for i in range(0, len(base_rnd_indices), num_base_per_batch)]
        extra_idx = 0
        added_extra_num = 0

        # Insert extra samples randomly into each base sample group
        for group in grouped_base_rnd_indices:
            # Generate random positions within current group to insert extra samples
            rnd_insert_position = torch.randint(0, len(group) - 1, (num_interpolate_per_batch,), generator=generator)
            # Insert extra samples (sorted descending to maintain correct indices when inserting)
            for pos in rnd_insert_position.sort(descending=True).values:
                group.insert(pos, extra_rnd_indices[extra_idx])  # insert from behind
                extra_idx += 1
                added_extra_num += 1
                if extra_idx >= num_extra_examples:
                    extra_rnd_indices = torch.randperm(num_extra_examples, generator=generator).tolist()
                    extra_idx = 0
        logger.info("Checking interpolated batch indices (including the inserted ids)")
        # print(*grouped_base_rnd_indices[:2], sep='\n')  # test
        logger.info(grouped_base_rnd_indices[:1])  # test
        logger.info(grouped_base_rnd_indices[1:2])  # test

        # Flatten
        flatted_indices = []
        for g in grouped_base_rnd_indices:
            flatted_indices.extend(g)
        # flatted_indices = sum(grouped_base_rnd_indices, [])  # this approach will be very slow
        logger.info('== Successfully generated interpolate indices in InterpolateRandomSampler')
        logger.info(f'== Total added extra number is {added_extra_num}')
        return flatted_indices
