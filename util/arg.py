import argparse
import os


def main_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--dataset', default='spider', type=str, help='dataset name')
    arg_parser.add_argument('--gpt', default='gpt-3.5-turbo', type=str, help='GPT model')
    arg_parser.add_argument('--seed', default=42, type=int, help='random seed')
    arg_parser.add_argument('--api_doc', action='store_true', help='write schema according to api doc')
    arg_parser.add_argument('--pf', default='eoc', type=str, choices=['no', 'eoc', 'eot'], help='format of primary and foreign keys')
    arg_parser.add_argument('--content', default=3, type=int, help='number of database records')
    arg_parser.add_argument('--zero_shot', action='store_true', help='zero shot')
    arg_parser.add_argument('--plm', default='text2vec-base-chinese', type=str, help='plm for preprocessing')
    arg_parser.add_argument('--batch_size', default=64, type=int, help='batch size for preprocessing')
    arg_parser.add_argument('--device', default=0, type=int, help='gpu id (-1 represents cpu)')
    arg_parser.add_argument('--cluster_method', default='random', type=str, choices=['kmeans', 'agglomerative', 'random'], help='clustering method')
    arg_parser.add_argument('--cluster_num', default=2, type=int, help='number of clusters')
    arg_parser.add_argument('--dynamic_num', default=2, type=int, help='number of dynamic shots')
    arg_parser.add_argument('--encoding', default='question', type=str, choices=['question', 'query'], help='according to question or query encoding')
    arg_parser.add_argument('--oracle', action='store_true', help='given queries in the dev dataset')
    arg_parser.add_argument('--two_phase', action='store_true', help='use two phase method')
    arg_parser.add_argument('--hard_and_extra', action='store_true', help='only test hard and extra hard examples')
    arg_parser.add_argument('--speech_api', action='store_true', help='use speech api')
    args = arg_parser.parse_args()
    assert (not args.api_doc) or args.pf == 'no'
    assert args.encoding == 'query' or (not args.oracle)
    assert args.zero_shot or (not args.two_phase)
    args.log_path = args.gpt
    args.log_path += '__seed_' + str(args.seed)
    args.log_path += '__' + ('api_doc' if args.api_doc else (args.pf + '_pf'))
    args.log_path += '__content_' + str(args.content)
    args.log_path += '__' + ('zero' if args.zero_shot else 'few') + '_shot'
    if not args.zero_shot:
        args.log_path += '__' + args.cluster_method + '__cluster_' + str(args.cluster_num)
        args.log_path += '__dynamic_' + str(args.dynamic_num)
        args.log_path += '__encoding_' + args.encoding
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
