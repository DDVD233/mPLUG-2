import argparse
import os
from collections import defaultdict

try:
    import ruamel_yaml as yaml
except:
    import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_videoqa_mplug import MPLUG2
from models.vit import interpolate_pos_embed, resize_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset.utils import save_result
from dataset import create_dataset, create_sampler, create_loader, vqa_collate_fn, vqa_collate_fn_no_sgg
from dataset.utils import load_jsonl

from scheduler import create_scheduler
from optim import create_optimizer, create_two_optimizer

import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine
import urllib.request
from tqdm import tqdm
import Levenshtein

# Path to the GloVe file
glove_file = "glove.6B.100d.txt"

class TqdmUpTo(tqdm):
    """Provide `tqdm` progress bar for urllib."""
    def update_to(self, blocks=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(blocks * bsize - self.n)

def download_with_progress(url, filename):
    with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename, reporthook=t.update_to)

# Check if the file exists, if not download it
if not os.path.exists(glove_file):
    print(f"{glove_file} not found. Downloading...")
    url = "http://nlp.stanford.edu/data/glove.6B.zip"
    zip_file = "glove.6B.zip"
    download_with_progress(url, zip_file)

    # Extract the desired file from the zip archive
    import zipfile
    with zipfile.ZipFile(zip_file, 'r') as z:
        z.extract(glove_file)
    print(f"Downloaded and extracted {glove_file}")

# Load GloVe vectors using Gensim
glove = KeyedVectors.load_word2vec_format(glove_file, binary=False, no_header=True)
vocab = list(glove.index_to_key)

def phrase_embedding(phrase, model):
    """Compute the average embedding of the phrase."""
    words = phrase.split()
    vectors = [model[word] for word in words if word in vocab]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

def compute_distance(phrase1, phrase2, model):
    """Compute cosine distance between two phrases' embeddings."""
    vec1 = phrase_embedding(phrase1, model)
    vec2 = phrase_embedding(phrase2, model)
    return cosine(vec1, vec2)


def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config, do_amp=False,
          do_two_optim=False, do_accum=True, accum_steps=1):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if do_two_optim:
        metric_logger.add_meter('lr1', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
        metric_logger.add_meter('lr2', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    else:
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 1
    warmup_iterations = warmup_steps * step_size

    len_batch = len(data_loader)
    print("Total Batch {}".format(len_batch))

    for i, (video, question, answer, weights, n) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        video, weights = video.to(device, non_blocking=True), weights.to(device, non_blocking=True)
        question_input = tokenizer(question, padding='longest', truncation=True, max_length=args.max_input_length if config["add_ocr"] else 25, return_tensors="pt").to(
            device)
        if i == 0:
            print ("question: ", question)
        answer_input = tokenizer(answer, padding='longest', return_tensors="pt").to(device)

        if epoch > 0:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1, i / len(data_loader))

        loss = model(video, question_input, answer_input, train=True, alpha=alpha, k=n, weights=weights)
        if accum_steps > 1:
           loss = loss / accum_steps

        if do_amp:
           from apex import amp
           with amp.scale_loss(loss, optimizer) as scaled_loss:
               # logger.info('scaled loss: {}'.format(str(scaled_loss)))
               scaled_loss.backward()
        else:
           loss.backward()
        if (i + 1) % accum_steps == 0:
           optimizer.step()
           optimizer.zero_grad()

        # model.backward(loss)
        # model.step()
        metric_logger.update(loss=loss.item())

        if do_two_optim:
            metric_logger.update(lr1=optimizer.param_groups[0]["lr"])
            metric_logger.update(lr2=optimizer.param_groups[2]["lr"])
        else:
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)
        elif scheduler.step_mode:
            scheduler.step(epoch * len_batch + i)
        del video, weights, question_input,answer_input, loss
            # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    raise NotImplementedError
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VideoQA test result:'
    print_freq = 50

    result = []

    answer_list = [answer + config['eos'] for answer in data_loader.dataset.answer_list]
    answer_input = tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)

    for n, (video, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        video = video.to(device, non_blocking=True)
        question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)

        topk_ids, topk_probs = model(video, question_input, answer_input, train=False, k=config['k_test'])

        for ques_id, topk_id, topk_prob in zip(question_id, topk_ids, topk_probs):
            ques_id = int(ques_id.item())          
            ans = tokenizer.decode(topk_id[0]).replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").strip()
            result.append({"question_id":ques_id, "answer":ans})   

    return result


@torch.no_grad()
def evaluate(model, data_loader, dataset, tokenizer, device, answer_list, rerank, config):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 50
    
    if answer_list.split('.')[-1] == 'json':
        answer_list = list(json.load(open(answer_list, 'r')).keys())
    else:
        answer_list = list(set([x['answer'] for x in load_jsonl(answer_list)]))
    
    answer_list_ = [answer+config['eos'] for answer in answer_list]
    answer_input = tokenizer(answer_list_, padding='longest', return_tensors='pt').to(device)    
    for n, (video, question, question_id, ann) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if 'sgg_question' in ann and config['sgg']:
            question = ann['sgg_question']
        video = video.to(device,non_blocking=True)
        question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)

        topk_ids, topk_probs = model(video, question_input, answer_input, train=False, k=config['k_test'], rerank=rerank)      
        result = []
        
        for ques_id, topk_id, topk_prob in zip(question_id, topk_ids, topk_probs):
            ques_id = int(ques_id.item())          
            if not rerank:
                ans = tokenizer.decode(topk_id[0]).replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").strip()
                result.append({"question_id":ques_id, "answer":ans})   
            else:
                _, pred = topk_prob.max(dim=0)
                result.append({"question_id": ques_id, "answer": answer_list[topk_id[pred]]})
        metrics = cal_metric(result, dataset)
        metric_logger.update(**metrics)

    # gather the stats from all processes
    torch.cuda.empty_cache()
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def cal_metric(vqa_result, val_file):
    data_list = load_jsonl(val_file)
    id2datum = {}
    for idx, each in enumerate(data_list):
        question_id = idx
        id2datum[question_id] = {
            'question': each['question'],
            'video_id': each['video_id'],
            'answer': each['answer'],
        }
    result = defaultdict(float)
    result['acc'] = 0
    for each in vqa_result:
        quesid = each["question_id"]
        ans = each["answer"]
        label = id2datum[quesid]['answer']
        semantic_distance = compute_distance(ans, label, glove)
        result['semantic_distance'] += semantic_distance
        result['edit_distance'] += Levenshtein.distance(ans, label)
        if label == ans:
            result['acc'] += 1
    result = {k: v / len(vqa_result) for k, v in result.items()}
    return result

def main(args, config):
    print('master addr: ', os.environ.get('MASTER_ADDR', 'localhost'))
    print('master port: ', os.environ.get('MASTER_PORT', 0))

    utils.init_distributed_mode(args)    
    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model ####
    print("Creating model")
    model = MPLUG2(config=config, tokenizer=tokenizer)
    model = model.to(device)
    if not args.do_two_optim:
        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_optimizer(arg_opt, model)
    else:
        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_two_optimizer(arg_opt, model)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        try:
            state_dict = checkpoint['model']
        except:
            state_dict = checkpoint['module']

        # reshape positional embedding to accomodate for image resolution change

        if not args.evaluate:
            if config["clip_name"] == "ViT-B-16":
                num_patches = int(config["image_res"] * config["image_res"]/(16*16))
            elif config["clip_name"] == "ViT-L-14":
                num_patches = int(config["image_res"] * config["image_res"]/(14*14))
            pos_embed = nn.Parameter(torch.zeros(num_patches + 1, 768).float())

            pos_embed = resize_pos_embed(state_dict['visual_encoder.visual.positional_embedding'].unsqueeze(0),
                                                   pos_embed.unsqueeze(0))
            state_dict['visual_encoder.visual.positional_embedding'] = pos_embed
            if config['distill']:
                if config["clip_name"] == "ViT-B-16":
                    num_patches = int(config["image_res"] * config["image_res"]/(16*16))
                elif config["clip_name"] == "ViT-L-14":
                    num_patches = int(config["image_res"] * config["image_res"]/(14*14))
                pos_embed = nn.Parameter(torch.zeros(num_patches + 1, 768).float())

                pos_embed = resize_pos_embed(state_dict['visual_encoder_m.visual.positional_embedding'].unsqueeze(0),
                                             pos_embed.unsqueeze(0))
                state_dict['visual_encoder_m.visual.positional_embedding'] = pos_embed

            for key in list(state_dict.keys()):
                if ('fusion' in key or 'bert' in key) and 'decode' not in key:
                    encoder_key = key.replace('fusion.', '').replace('bert.', '')
                    state_dict[encoder_key] = state_dict[key]
                    del state_dict[key]

        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)

    model_without_ddp = model
    if args.distributed:
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        import apex
        model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)
        model_without_ddp = model.module    
        
    if args.do_amp:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    #### Dataset ####
    print("Creating video qa datasets")
    if args.no_randaug:
        datasets = create_dataset('video_qa_no_randaug', config)
    else:
        datasets = create_dataset('video_qa', config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [True, True], num_tasks, global_rank)
    else:
        samplers = [None, None]

    collate_fn = vqa_collate_fn if config['sgg'] else vqa_collate_fn_no_sgg

    train_loader, val_loader = create_loader(datasets,samplers,
                                            batch_size=[config['batch_size_train'],config['batch_size_test']],
                                            num_workers=[16, 16],is_trains=[True, False],
                                            collate_fns=[collate_fn,None])

    arg_sche = utils.AttrDict(config['schedular'])
    train_step_per_epoch = len(train_loader)
    print("train_step_per_epoch: {}".format(train_step_per_epoch))
    arg_sche["num_iterations"] = max_epoch * train_step_per_epoch - arg_sche['warmup_epochs']
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    best_epoch = -1
    best_acc = 0
    print("Start training")
    start_time = time.time()

    # val_stats = evaluate(model, val_loader, config["label_file"], tokenizer, device, config['answer_list'], False, config)
    # val_stats_rerank = evaluate(model, val_loader, config["label_file"], tokenizer, device, config['answer_list'], True, config)
    # val_stats_rerank_vocab = evaluate(model, val_loader, config["label_file"], tokenizer, device, config['answer_list_vocab'], True, config)
    # val_stats_rerank_vocab_1000 = evaluate(model, val_loader, config["label_file"], tokenizer, device, config['answer_list_vocab_1000'], True, config)
        
    # if utils.is_main_process():
    #     log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
    #                  # **{f'val_rerank_{k}': v for k, v in val_stats_rerank.items()},
    #                  # **{f'val_rerank_vocab_{k}': v for k, v in val_stats_rerank_vocab.items()},
    #                  # **{f'val_rerank_vocab_1000_{k}': v for k, v in val_stats_rerank_vocab_1000.items()},
    #                  'epoch': -1,
    #                  }
    #     with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
    #         f.write(json.dumps(log_stats) + "\n")
    #     best_acc = float(val_stats['acc'])

    for epoch in range(start_epoch, max_epoch):
        # if epoch > 0:
        #     lr_scheduler.step(epoch + warmup_steps)

        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler,
                                config, do_amp=args.do_amp, do_two_optim=args.do_two_optim, accum_steps=args.accum_steps)

        val_stats = evaluate(model, val_loader, config["label_file"], tokenizer, device, config['answer_list'], False, config)
        # val_stats_rerank = evaluate(model, val_loader, config["label_file"], tokenizer, device, config['answer_list'], True, config)
        # val_stats_rerank_vocab = evaluate(model, val_loader, config["label_file"], tokenizer, device, config['answer_list_vocab'], True, config)
        # val_stats_rerank_vocab_1000 = evaluate(model, val_loader, config["label_file"], tokenizer, device, config['answer_list_vocab_1000'], True, config)
        

        if args.evaluate:
            break
        
        if utils.is_main_process():               
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in val_stats.items()},
                         # **{f'val_rerank_{k}': v for k, v in val_stats_rerank.items()},
                         # **{f'val_rerank_vocab_{k}': v for k, v in val_stats_rerank_vocab.items()},
                         # **{f'val_rerank_vocab_1000_{k}': v for k, v in val_stats_rerank_vocab_1000.items()},
                         'epoch': epoch,
                        }                
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")                        
                         
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))

            if float(val_stats['acc']) >= best_acc:
                best_epoch = epoch
                best_acc = float(val_stats['acc'])
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
            
        if not lr_scheduler.step_mode:
            lr_scheduler.step(epoch + warmup_steps + 1)
        dist.barrier()
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if utils.is_main_process() and not args.evaluate:
        with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
            f.write("best epoch: %d"%best_epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/VQA.yaml')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--output_dir', default='output/vqa')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--text_decoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--min_length', default=1, type=int)
    parser.add_argument('--max_length', default=10, type=int)
    parser.add_argument('--max_input_length', default=50, type=int)
    parser.add_argument('--beam_size', default=5, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--do_two_optim', action='store_true')
    parser.add_argument('--do_amp', action='store_true')
    parser.add_argument('--do_accum', action='store_true')
    parser.add_argument('--no_randaug', action='store_true')
    parser.add_argument('--accum_steps', default=1, type=int)

     # Model architecture
    parser.add_argument('--temporal_stride', default=2, type=int)
    parser.add_argument('--temporal_downsampling', action='store_true')
    # parser.add_argument('--use_st', action='store_true')
    # parser.add_argument('--double_lmhra', action='store_true')

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    config["min_length"] = args.min_length
    config["max_length"] = args.max_length
    config["beam_size"] = args.beam_size
    config['text_encoder'] = args.text_encoder
    config['text_decoder'] = args.text_decoder

    config['temporal_stride'] = args.temporal_stride
    config['temporal_downsampling'] = args.temporal_downsampling
    # config['double_lmhra'] = args.double_lmhra
    # config['use_st'] = args.double_lmhra
    config['accum_steps'] = args.accum_steps
    config['no_randaug'] = args.no_randaug

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)
