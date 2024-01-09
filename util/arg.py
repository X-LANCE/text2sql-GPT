import argparse
import os


def main_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--dataset', default='spider', type=str, help='dataset name')
    arg_parser.add_argument('--gpt', default='gpt-3.5-turbo', type=str, help='GPT model')
    arg_parser.add_argument('--seed', default=42, type=int, help='random seed')
    arg_parser.add_argument('--api_doc', action='store_true', help='write schema according to api doc')
    arg_parser.add_argument('--pf', default='eot', type=str, choices=['no', 'eoc', 'eot'], help='format of primary and foreign keys')
    arg_parser.add_argument('--content', default=3, type=int, help='number of database records')
    arg_parser.add_argument('--zero_shot', action='store_true', help='zero shot')
    arg_parser.add_argument('--labeled_shot', action='store_true', help='use labeled shots')
    arg_parser.add_argument('--plm', default='text2vec-base-chinese', type=str, help='plm for preprocessing')
    arg_parser.add_argument('--batch_size', default=64, type=int, help='batch size for preprocessing')
    arg_parser.add_argument('--device', default=-1, type=int, help='gpu id (-1 represents cpu)')
    arg_parser.add_argument('--cluster_method', default='random', type=str, choices=['kmeans', 'agglomerative', 'random'], help='clustering method')
    arg_parser.add_argument('--cluster_num', default=2, type=int, help='number of clusters')
    arg_parser.add_argument('--dynamic_num', default=2, type=int, help='number of dynamic shots')
    arg_parser.add_argument('--encoding', default='question', type=str, choices=['question', 'query'], help='according to question or query encoding')
    arg_parser.add_argument('--cot', action='store_true', help='use chain of thought')
    arg_parser.add_argument('--tot', action='store_true', help='use tree of thought')
    arg_parser.add_argument('--tot_k', default=3, type=int, help='k for tree of thought')
    arg_parser.add_argument('--tot_b', default=1, type=int, help='b for tree of thought')
    arg_parser.add_argument('--tot_t', default=1.5, type=float, help='temperature for tree of thought')
    arg_parser.add_argument('--reflection', action='store_true', help='use self-reflection')
    arg_parser.add_argument('--ref_shot', action='store_true', help='few-shot for self-reflection')
    arg_parser.add_argument('--oracle', action='store_true', help='given queries in the dev dataset')
    arg_parser.add_argument('--two_phase', action='store_true', help='use two phase method')
    arg_parser.add_argument('--hard_and_extra', action='store_true', help='only test hard and extra hard examples')
    args = arg_parser.parse_args()
    assert not (args.zero_shot and args.labeled_shot)
    assert not (args.cot and args.tot)
    args.device = 'cpu' if args.device < 0 else f'cuda:{args.device}'
    args.log_path = args.gpt
    args.log_path += '__seed_' + str(args.seed)
    if args.api_doc:
        args.log_path += '__api_doc'
    args.log_path += '__' + args.pf + '_pf'
    args.log_path += '__content_' + str(args.content)
    args.log_path += '__' + ('zero' if args.zero_shot else 'few') + '_shot'
    if args.labeled_shot:
        args.log_path += '__labeled_shot'
    elif not args.zero_shot:
        args.log_path += '__' + args.cluster_method + '__cluster_' + str(args.cluster_num)
        args.log_path += '__dynamic_' + str(args.dynamic_num)
        args.log_path += '__encoding_' + args.encoding
    if args.cot:
        args.log_path += '__cot'
    elif args.tot:
        args.log_path += '__tot_k_' + str(args.tot_k)
        args.log_path += '__tot_b_' + str(args.tot_b)
        args.log_path += '__tot_t_' + str(args.tot_t)
    if args.reflection:
        args.log_path += '__reflection'
        if args.ref_shot:
            args.log_path += '__ref_shot'
    if args.oracle:
        args.log_path += '__oracle'
    if args.two_phase:
        args.log_path += '__two_phase'
    if args.hard_and_extra:
        args.log_path += '__hard_and_extra'
    args.log_path = os.path.join('log', args.dataset, args.log_path)
    return args


def cluster_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--dataset', default='spider', type=str, help='dataset name')
    arg_parser.add_argument('--plm', default='text2vec-base-chinese', type=str, help='plm for preprocessing')
    arg_parser.add_argument('--batch_size', default=64, type=int, help='batch size for preprocessing')
    arg_parser.add_argument('--device', default=0, type=int, help='gpu id (-1 represents cpu)')
    arg_parser.add_argument('--method', default='kmeans', type=str, choices=['kmeans', 'agglomerative'], help='clustering method')
    arg_parser.add_argument('--cluster', default=3, type=int, help='number of clusters')
    arg_parser.add_argument('--encoding', default='question', type=str, choices=['question', 'query'], help='according to question or query encoding')
    args = arg_parser.parse_args()
    args.device = 'cpu' if args.device < 0 else f'cuda:{args.device}'
    return args


def cot_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--dataset', default='spider', type=str, help='dataset name')
    arg_parser.add_argument('--plm', default='text2vec-base-chinese', type=str, help='plm for preprocessing')
    arg_parser.add_argument('--batch_size', default=64, type=int, help='batch size for preprocessing')
    arg_parser.add_argument('--device', default=0, type=int, help='gpu id (-1 represents cpu)')
    args = arg_parser.parse_args()
    args.device = 'cpu' if args.device < 0 else f'cuda:{args.device}'
    return args


def tot_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--dataset', default='spider', type=str, help='dataset name')
    args = arg_parser.parse_args()
    return args

def multiturn_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--dataset', default='sparc', type=str, help='dataset name')
    arg_parser.add_argument('--gpt', default='gpt-3.5-turbo', type=str, help='GPT model')
    args = arg_parser.parse_args()
    return args
