"""NLP Parser

This script allows the controller to get parsed results of texts through the StanfordCoreNLP server.
Annotators include tokenization, part-of-speech tagging, coreference resolution, and dependency parsing.

This script requires that `stanfordcorenlp` be installed within the Python
environment you are running this script in, as well as the 'nlp_server.sh' script running in port 9000.

This file can also be imported as a module and contains the following
functions:

    * parse - annotate a chunk of text
    * resolve_coref - resolve coreference of the annotation
    * combine_resolved_coref - reformat the annotation with resolved coreference
    * resolved_to_triples - get triples from the original annotation
    * get_words - get word information from the original annotation
"""
import copy
import string
import glob
import json
import os
import re
import unidecode
import pandas as pd
import itertools as itert
import numpy as np
import yaml
from stanfordcorenlp import StanfordCoreNLP

from py2graphdb.utils.misc_lib import *

import inspect
import os

BASE_DIR = os.path.dirname(__file__) + "/nlp_parser/"


print("Connecting to nlp_server at port=9000 ..... ")
try:
    nlp = StanfordCoreNLP('http://host.docker.internal', 9000)
except:
    nlp = StanfordCoreNLP('http://localhost', 9000)
print("Done connecting to nlp_server.")


import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

STOPWORDS = stopwords.words('english')


def parse(chunk, requests=['sentence', 'word', 'pos', 'coref', 'dep']):
    """Annotate the input chunk of text to get pos and coreference information.

    :param chunk: any kind of chunk of string to be annotated
    :return: default formatted dictionary of annotated information
    """
    # print(chunk)
    annotators = []
    # if 'sentence' in requests: annotators.append('ssplit')
    if 'pos' in requests: annotators.append('pos')
    if 'coref' in requests: annotators.append('dcoref')

    properties={
        'outputFormat': 'json',
        'ner.useSUTime': 'false'
    }
    if len(annotators)>0:  properties['annotators'] = ','.join(annotators)
    ann = json.loads(
        unidecode.unidecode(
            nlp.annotate(
                unidecode.unidecode(chunk, 'utf-8'),
                properties = properties
            )
        ))
    rename = {}
    if 'pos' in requests and 'Word' in requests: rename["Words and POS"] = 'Word and Pos'
    elif 'pos' in requests: rename['Pos'] = 'Pos'
    elif 'word' in requests: rename['Words'] = 'Word'

    # if 'dep' in requests: rename['Dep'] = "Dep", "CoRef")
    # if 'coref' in requests: rename[''] = "Dep", "CoRef")

    for k_old,k_new in rename.items():
        if k_old in ann.keys():
            ann[k_new] = ann.pop(k_old)

    return ann


def resolve_coref(corenlp_output):
    """Transfer the word form of the antecedent to its associated pronominal anaphor(s)

    :param corenlp_output: dictionary output of the annotated information
    """
    for coref in corenlp_output['corefs']:
        mentions = corenlp_output['corefs'][coref]
        antecedent = mentions[0]  # the antecedent is the first mention in the coreference chain
        for j in range(1, len(mentions)):
            mention = mentions[j]
            if mention['type'] == 'PRONOMINAL':
                # get the attributes of the target mention in the corresponding sentence
                target_sentence = mention['sentNum']
                target_token = mention['startIndex'] - 1
                # transfer the antecedent's word form to the appropriate token in the sentence
                corenlp_output['sentences'][target_sentence - 1]['tokens'][target_token]['word'] = antecedent['text']


def combine_resolved_coref(corenlp_output):
    """Create a list of token information with the resolved coreference.

    :param corenlp_output: dictionary output of the annotated information
    :return: a nested list
    """
    res = []
    possessives = ['hers', 'his', 'their', 'theirs', 'he', 'her', ]
    for sentence in corenlp_output['sentences']:
        for ti, token in enumerate(sentence['tokens']):
            output_word = token['word']
            output_orgword = token['originalText']
            coref = 1 if token['word'] != token['originalText'] else 0
            regexPattern = '|'.join(map(re.escape, [',', ' ', ';', 'and', 'or']))
            sub_tokens = [t for t in re.split(regexPattern, token['word']) if len(t) > 0]

            # check lemmas as well as tags for possessive pronouns in case of tagging errors
            if token['lemma'] in possessives or token['pos'] == 'PRP$':
                output_word += "'s"  # add the possessive morpheme
            output_word += token['after']
            output_orgword += token['after']
            res.append([ti + 1, output_word, coref, output_orgword, token['originalText'], sub_tokens])
    return res


def resolved_to_triples(corenlp_output):
    """Make triples from the original annotation.

    :param corenlp_output: dictionary output of the annotated information
    :return: list of triples
    """
    parsers = {
        "basic": "basicDependencies",
        "enhanced": "enhancedDependencies",
        "enhancedpp": "enhancedPlusPlusDependencies"}
    triples = []
    prev_i_max = -1
    i_offset = 0
    for sent in corenlp_output['sentences']:
        deps = sent[parsers["basic"]]
        tokens = sent['tokens']
        for v in deps:
            w1 = v['governorGloss']
            w2 = v['dependentGloss']
            i1 = v['governor'] + i_offset
            i2 = v['dependent'] + i_offset
            prev_i_max = np.max([prev_i_max, i1, i2])
            pos1 = next((x['pos'] for x in tokens if x['index'] == (i1 - i_offset)), 'NONE')
            pos2 = next((x['pos'] for x in tokens if x['index'] == (i2 - i_offset)), 'NONE')
            triples.append([["%s-%s" % (w1, i1), pos1], v['dep'], ["%s-%s" % (w2, i2), pos2]])
        i_offset = prev_i_max + 1
    return triples


def resolve_triples(triples, res, which='left'):
    corefs = copy.deepcopy([c for c in res if c[2] == 1])
    out_triples = []
    for coref in copy.deepcopy(corefs):

        tokens = set(coref[5])
        ii, jj = (0, 2) if which == 'left' else (2, 0)
        # change subject in s-p-o
        to_change = copy.deepcopy([t for t in triples if t[1][ii][0].split('-')[0] == coref[4]])

        for new_trp1, token in itert.product(to_change, tokens):
            new_trp = copy.deepcopy(new_trp1)
            from_change = [t[1][ii] for t in triples if t[1][ii][0].split('-')[0] == token]
            from_change += [t[1][jj] for t in triples if t[1][jj][0].split('-')[0] == token]
            for from_token in unique_all(from_change):
                new_trp[1][ii] = from_token
                out_triples.append(new_trp)

    return unique_all(out_triples)


def replace_resolved_triples(triples, replaced):
    # replace triple corefs with newly built ones
    to_replace = set([t[0] for t in replaced])
    replaced_corefs = []
    out_triples = []
    for t1 in triples:
        if t1[0] not in to_replace:
            out_triples.append(t1)
        else:
            t2 = [tt1 for tt1 in replaced if t1[0] == tt1[0]]
            out_triples += t2
            replaced_corefs += [(tt, t1[1][2]) for _, [_, _, tt] in t2]

    return {'triples': out_triples, 'replaced': replaced_corefs}

def generate_coref(corenlp_output):
    res = combine_resolved_coref(corenlp_output)
    triples = list(enumerate(resolved_to_triples(corenlp_output)))
    
    triples1 = resolve_triples(triples,res,which='left')
    tmp1 = replace_resolved_triples(triples,triples1)
    triples_1 = tmp1['triples']

    triples2 = resolve_triples(triples_1,res,which='right')
    tmp2 = replace_resolved_triples(triples_1,triples2)
    replaced_triples = tmp2['replaced']
    # remove POS and return just the terms e.g. [('Bob-1', 'he-10'), ('Mary-2', 'she-10')]
    return [(t[0][0], t[1][0]) for t in replaced_triples]

def generate_spos(id1, id2, corenlp_output):
        spo_file_paths = r_t_run_test(id1=id1,id2=id2)
        for filepath in spo_file_paths:
            _ = format_results(filepath)

        return gen_spo_pos_df(id1=id1,id2=id2)


def get_words(corenlp_output):
    """ Reformat the original annotation into a list of word information

    :param corenlp_output: dictionary output of the annotated information
    :return: list of dictionary with word information
    """
    
    # get the deps with labeled content (e..g Word-1)
    deps = resolved_to_triples(corenlp_output)
    terms = []
    for d in deps:
        if d[1] != 'ROOT':
            terms.append(d[0])
            terms.append(d[2])
    tmp = []
    for term, pos in terms:
        content, content_label = re.findall(r'(.+)\-([0-9]+)', term)[:2][0]
        tmp.append([int(content_label), content, term, pos])
    labeled_words = [t for t in sorted(dedup(tmp))]

    # TODO: 'Ltd.' in tokens parsed as "Ltd", but deps parsed as "Ltd."
    words = []
    for sentence in corenlp_output['sentences']:
        for token in sentence['tokens']:
            details = {
                'sindex'        : sentence['index'],
                "content"   : token['originalText'],
                'start'     : token['characterOffsetBegin'],
                'end'       : token['characterOffsetEnd'],
                'pos'       : token['pos']
            }
            words.append(details)

    # match labeled_words with words, to get 'content_label' value into words
    i2 = 0
    final_words = []
    for i,content, term, pos in labeled_words:
        matches = [aw for aw in words[i2:] if aw['content']==content and aw['pos']==pos]
        if len(matches)>0:
            match = matches[0]
            match['content_label'] = term
            final_words.append(match)
            i2+=1
    return final_words


def build_data_from_output(id1, id2, text, corenlp_output):
    deps = resolved_to_triples(corenlp_output)
    dep_triples = list(enumerate(deps))
    resolve_coref(corenlp_output=corenlp_output)
    resolved_corefs = combine_resolved_coref(corenlp_output)

    to_dir = f"{BASE_DIR}tmp/scroll/data/{id1}/"
    _ = os.makedirs(to_dir, exist_ok=True)

    indexed_sentences = pd.DataFrame(columns=['idx', 'indexed_text'])
    indexed_text = generate_sentence_with_term_indexes(dep_triples)
    indexed_sentences = indexed_sentences.append(
        pd.Series([id2, indexed_text], index=['idx', 'indexed_text']), ignore_index=True)
    indexed_sentences.to_csv(to_dir+f"{id2}_indexed_sentences.csv")

    triples_1 = resolve_triples(dep_triples,
                                resolved_corefs, which='left')
    tmp1 = replace_resolved_triples(dep_triples, triples_1)
    triples_1 = tmp1['triples']

    triples_2 = resolve_triples(triples_1,
                                resolved_corefs, which='right')
    tmp2 = replace_resolved_triples(triples_1, triples_2)
    spo_triples = tmp2['triples']
    replaced_resolved_corefs = tmp2['replaced']
    return dep_triples, replaced_resolved_corefs

def build_prolog_from_output(id1,id2, text, corenlp_output):
    dep_triples, replaced_resolved_corefs = build_data_from_output(id1=id1,id2=id2, text=text, corenlp_output=corenlp_output)
    build_prolog(id1=id1,id2=id2, text=text, triples=dep_triples, replaced_triples=replaced_resolved_corefs)

def build_prolog(id1,id2, text, triples, replaced_triples):
    to_dir = f"{BASE_DIR}tmp/scroll/"
    prolog_dir = f"{BASE_DIR}../scroll/"

    _ = os.makedirs(to_dir+f"data/{id1}", exist_ok=True)

    # print('---------------------------------')
    # print(text)
    # print(triples)
    # print(os.getcwd())
    res_content = ":- style_check(-discontiguous).\n"
    res_content += f":- ensure_loaded(\"{prolog_dir}prolog/parsing\").\n"
    res_content += ":- dynamic coref/1.\n"
    res_content += f"text(\"{text}\").\n"
    res_content += ("\n".join(
        ["gram(%s,\"%s\",%s,%s)." % (i, t, [l1, l2], [r1, r2]) for i, [[l1, l2], t, [r1, r2]] in triples])).replace(
        "['", "[\"").replace("']", "\"]").replace("',", "\",").replace(", '", ", \"")
    res_content += "\n"
    res_content += (
        "\n".join(["coref([%s,%s])." % ([l1, l2], [r1, r2]) for ([l1, l2], [r1, r2]) in replaced_triples]).replace("['",
                                                                                                                   "[\"").replace(
            "']", "\"]").replace("',", "\",").replace(", '", ", \""))
    # print(res_content)
    try:
        filename = to_dir+f"data/{id1}/{id2}.pl"
        file = open(filename, "w")
        file.write(res_content)
        file.close()
        return filename
    except FileNotFoundError:
        print("filenotfound build prolog")
        return False


def to_phrase_run_test(id1,id2):
    to_dir = f"{BASE_DIR}tmp/scroll/res/{id1}/"

    _ = os.makedirs(to_dir, exist_ok=True)

    test_configs = [
        {'file': to_dir+f"{id2}_phrase_results.txt",
        'cmd': "findall([W,D],(phrase_([_,D],W)),L),remove_duplicates(L,L2),L2\=[],text(Text),writeln(['%s',Text]),findall(_,(member(T,L2),writeln(['%s',T])),_)",
        'ids_n': 2},
        {'file': to_dir+f"{id2}_phrase_pos_results.txt",
        'cmd': "findall([W,D],(phrase_pos([_,D],W)),L),remove_duplicates(L,L2),L2\=[],text(Text),writeln(['%s',Text]),findall(_,(member(T,L2),writeln(['%s',T])),_)",
        'ids_n': 2},
        {'file' : to_dir+f"{id2}_spo_results.txt",
        'cmd'  : "findall([W,D],(spo(D,W)),L),remove_duplicates(L,L2),L2\=[],text(Text),writeln(['%s',Text]),findall(_,(member(T,L2),writeln(['%s',T])),_)",
        'ids_n'  : 2},
    ]
    output_files = run_test_configs(test_configs, id1=id1,id2=id2)
    return output_files


def r_t_run_test(id1,id2):
    from_dir = f"{BASE_DIR}tmp/scroll/res/{id1}/"
    _ = os.makedirs(from_dir, exist_ok=True)

    test_configs = [
        {'file': from_dir+f"{id2}_spo_qualifier_results.txt",
        # 'cmd': "findall([W,D],(spo_qualifier(D,W)),L),remove_duplicates(L,L2),L2\=[],text(Text),writeln(['%s',Text]),findall(_,(member(T,L2),writeln(['%s',T])),_)",
        'cmd':None,
        'ids_n': 2},
        {'file': from_dir+f"{id2}_spo_results.txt",
        'cmd': "findall([W,D],(spo(D,W)),L),remove_duplicates(L,L2),L2\=[],text(Text),writeln(['%s',Text]),findall(_,(member(T,L2),writeln(['%s',T])),_)",
        'ids_n': 2},
    ]
    output_files = run_test_configs(test_configs, id1=id1,id2=id2)
    return output_files

def run_test_configs(test_configs, id1,id2):
    output_files = []
    for test_config in test_configs:
        output_files.append(run_test_config(test_config, id1=id1,id2=id2))
    return output_files


def run_test_config(test_config, id1,id2):
    from_dir = f"{BASE_DIR}tmp/scroll/data/{id1}/"

    service_file = test_config['file']
    search_cmd = test_config['cmd']
    ids_n = test_config['ids_n']
    _ = os.system("rm -f %s" % (service_file))
    _ = os.system("touch %s" % (service_file))
    filename = from_dir+f"{id2}.pl"
    i = str(id2)
    if search_cmd is not None and len(search_cmd)>0:
        cmd = "swipl -s %s -g \"" % (filename) + search_cmd % tuple([i] * ids_n) + \
                ".\" -g halt >> %s" % (service_file)
        print("\n>>>>>>>>>", i)
        _ = os.system('whoami')
        _ = os.system('swipl -g halt')
        print(cmd)
        _ = os.system(cmd)
        print("\n- - - - -", i)
    return service_file


def r_e_run_test(id1,id2):
    # from_dir = 'pyscript/app/scripts/'
    from_dir = f"{BASE_DIR}tmp/scroll/"

    _ = os.makedirs(from_dir+f"data/{id1}/", exist_ok=True)

    test_config = {'file': from_dir+f"res/{id1}/{id2}_mod_results.txt",
     'cmd': "findall([W,D],(mod_(D,W)),L),remove_duplicates(L,L2),L2\=[],text(Text),writeln(['%s',Text]),findall(_,(member(T,L2),writeln(['%s',T])),_)",
     'ids_n': 2}

    output_file = run_test_config(test_config, id1,id2)
    return output_file

def format_results(ori_file_path):
    # "prolog_to_csv" in scroll
    parse_re = re.compile('([^\[,]+,)|(\[[^\]]+\])|([\]],[^,]+,)|([\]],[^,]+)|([^\],]+)$|(^[^\[,]+)')

    with open(ori_file_path, "r") as f:
        rows = f.readlines()
    res_content = ''
    prev_idx = ''
    for row in rows:
        matches = re.search(r"\[([0-9]+),(.*)\]", row.strip())
        if matches is not None:
            idx = matches[1]
            data = matches[2]
            if idx != prev_idx:
                # assuming first record is the sentence itself
                res_content += "%s\t\"TEXT\"\t\"%s\"\n" % (idx, data)
            else:
                # parsing componentns that may be single items, or list of itmes grouped by "[x]" or "(x)".
                # all seperated by commas ","
                data = data[1:-1]
                fields = [[rr.strip('[](),') for rr in r if rr != ''][0] for r in re.findall(parse_re, data)]
                res_content += idx + "\t\"" + ("\"\t\"".join(fields)) + "\"\n"
            prev_idx = idx

    filename2 = ori_file_path.replace('.txt', '_format.txt')
    file = open(filename2, "w")
    _ = file.write(res_content)
    file.close()
    return True


def res_file_to_df(filename, remove_text=True):
    with open(filename, 'r') as temp_f:
        # get No of columns in each line
        col_count = [len(l.split("\t")) for l in temp_f.readlines()]

    if len(col_count) == 0:
        return pd.DataFrame()
    # Generate column names  (names will be 0, 1, 2, ..., maximum columns - 2)
    # Assuming the first colum will be a rankign of resutls: see parsing_ranking.yml
    # print('==================')
    # print(filename)
    # print(col_count)
    column_names = ['idx', 'ranking_ids'] + [i for i in range(0, max(col_count) - 2)]

    # Read csv
    df = pd.read_csv(filename, header=None, delimiter="\t", names=column_names)
    if remove_text:
        df = df[df['ranking_ids'] != 'TEXT'].reset_index(drop=True)

    # get parse ranking for each record
    df['parse_ranking'] = df['ranking_ids'].apply(calculate_parse_ranking)
    # remove duplicates, keeping the highest ranked record
    sort_cols = df.columns.tolist()
    sort_cols = [s for s in df.columns if s not in ['parse_ranking', 'ranking_ids']]
    df = df.loc[df.groupby(sort_cols, dropna=False)['parse_ranking'].idxmax()].sort_index().reset_index(drop=True)

    # normalize ranking
    df['parse_ranking'] = df['parse_ranking'] / df['parse_ranking'].max()

    return df


def gen_spo_pos_df(id1,id2):
    from_dir = f"{BASE_DIR}tmp/scroll/res/"

    filename = from_dir+f"{id1}/{id2}_spo_results_format.txt"
    spo_df = res_file_to_df(filename).rename(columns={0: 's', 1: 'p', 2: 'o'}).reset_index(drop=True)
    res_spo = []
    if not spo_df.empty:
        val_cols = ['s', 'p', 'o']
        idx_min = 0

        # id_min = vs.name
        for spo_id, vs in spo_df.loc[idx_min:][['idx'] + val_cols].iterrows():
            idx = vs['idx']
            for c, v in vs[val_cols].items():
                if v == v:
                    match = re.search(r'(.*[\-[0-9]+),(.+)', v)
                    if match:
                        try:
                            token = match[1]
                            pos = match[2]
                            word = re.search(r'^(.+)\-[0-9]+$', token)[1]
                            res_spo.append([spo_id, idx, c, token, pos, word])
                        except TypeError:
                            pass
    res_spo_df = pd.DataFrame(res_spo, columns=['spo_id', 'idx', 'slot', 'token', 'pos', 'word'])
    return res_spo_df


def gen_phrase_pos_rank(id1,id2):
    phrase_file_paths = to_phrase_run_test(id1=id1,id2=id2)
    for filepath in phrase_file_paths:
        _ = format_results(filepath)
    from_dir = f"{BASE_DIR}tmp/scroll/res/{id1}/"
    filename = from_dir+f"{id2}_phrase_pos_results_format.txt"

    df = res_file_to_df(filename)
    val_cols = [c for c in df.columns if type(c) is int]
    res = []
    idx_min = 0
    if not df.empty:
        for phrase_id,vs in df.loc[idx_min:][['idx']+val_cols].iterrows():
            idx=vs['idx']
            for v in vs[val_cols].values:
                if v==v:
                    match = re.search(r'(.*[\-[0-9]+),(.+)',v)
                    if match:
                        try:
                            token = match[1]
                            pos = match[2]
                            word = re.search(r'^(.+)\-[0-9]+$',token)[1]
                            res.append([phrase_id,idx,token,pos,word])
                        except TypeError:
                            pass
    res_df = pd.DataFrame(res, columns = ['phrase_id','idx','token','pos','word'])
    fileout = from_dir+f"{id2}_phrase_pos_results_words.csv"
    res_df.to_csv(fileout,index=False)
    return res_df


def calculate_parse_ranking(ranking_ids):
    # from_dir = 'pyscript/app/scripts/'
    from_dir = f"{BASE_DIR}../scroll/"
    if isinstance(ranking_ids, str):
        ranking_ids = ranking_ids.replace('|',',')
        ids = [float(r) for r in ranking_ids.split(',')]
        parser_ranking = yaml.safe_load(open(from_dir+'parsing_ranking.yml', 'r'))
        return np.sum([parser_ranking[i]*w for w,i in zip(np.arange(1,0, -1/len(ids)), ids)])
    else:
        return 0


def score_cols_and_mapping():
    score_cols = ['need_satisfier_in_o_1',         'need_satisfier_in_o_2',         'need_satisfier_in_o_3',    'need_in_o_8',         'need_in_o_9',   'need_satisfier_in_o_11',         'need_satisfier_in_s_12',         'need_satisfier_in_o_13', 'current_state_in_s_14',         'service_description_in_o_15',         'need_satisfier_in_o_16',         'service_in_o_17',         'need_satisfier_description_in_o_18',         'service_description_in_o_19',         'desired_state_in_o_20',    'need_satisfier_in_o_22',         'service_description_in_o_23',         'service_description_in_o_24',         'need_satisfier_in_o_25',    'required_for_in_s_27',         'required_criteria_in_o_28',         'eligibile_criteria_in_s_29',         'eligibile_for_in_o_30',         'need_satisfier_in_o_31',         'need_satisfier_description_in_o_32',         'service_description_in_o_33',         'client_description_in_p_34',         'program_in_s_35',         'need_satisfier_in_o_36',   'service_description_in_s_41',         'client_description_in_o_42',         'service_description_in_s_43', 'client_description_in_o_43', 'program_in_s_44',         'need_satisfier_in_o_44',  'program_in_s_45', 'service_description_in_o_45',      'need_satisfier_in_s_46',         'client_in_o_47']

    sheets = ['program name','service_description', 'required_criteria','need_satisfier','need_satisfier_description','client demographic','client_description',
            'desired_state (outcome)','need']
    org_sheets = ['program','service_description','required_criteria','need_satisfier','need_satisfier_description', 'client', 'client_description','desired_state','need']
    mapping = {}
    for i, j in zip(sheets, org_sheets):
        mapping.setdefault(i, []).append(j)
    return score_cols,mapping, sheets, org_sheets


def rank_by_length(series,df):
    tmp = pd.DataFrame(series.apply(lambda row: row.split(',')).tolist())
    # tmp = pd.DataFrame(series.apply(lambda row: sorted(row.split(','))))
    max_length = tmp.columns.shape[0]
    tmp[['idx','parse_ranking']] = df[['idx','parse_ranking']]
    tmp['text'] = series
    tmp['score'] = max_length - tmp.T.isna().sum()
    tmp['score'] = tmp['score']/tmp['score'].max()
    return tmp


def apply_parse_ranking(df, group_by_idx=True, cols=['score'], rank_cols=['parse_ranking']):
    df = df.copy()
    if len(cols)>1 and len(rank_cols)>1 and len(rank_cols) != len(cols):
        raise ValueError("col count %s not compatible with rank_col count %s"%(len(cols), len(rank_cols)))
    if len(rank_cols) == 1:
        rank_cols = rank_cols*len(cols)
    for col,rcol in zip(cols,rank_cols):
        df[col] = df[col]*df[rcol]
        if group_by_idx and 'idx' in df.columns:
            for idx,grp in df.groupby('idx'):
                df.loc[grp.index,col] = grp[col]/grp[col].max()
        else:
            df[col] = df[col]/df[col].max()
    return df


def generate_sentence_with_term_indexes(triples):
    re_compiled = re.compile(r'(.*)\-([0-9]+)')
    def assign_term(res,term):
        if term[1] != 'NONE':
            match = re.search(re_compiled,term[0])
            if match:
                res[int(match[2])] = match[0]
        return res
    res = [None]*len(triples)*2
    for _,[s,p,o] in triples:
        assign_term(res,s)
        # assign_term(res,p)
        assign_term(res,o)
    return '#DEL#'.join([s for s in res if s is not None])


def from_mods(text, id1,id2):
    text_df = pd.DataFrame({"idx": 0, "Text": text}, index=[0])
    from_dir = f"{BASE_DIR}tmp/scroll/"
    stats_directory = from_dir+f"stats/{id1}/"
    _ = os.makedirs(stats_directory, exist_ok=True)
    annotation_out_filename = stats_directory + f"{id2}_annot_obj.csv"

    res_spo_df = gen_spo_pos_df(id1=id1,id2=id2)

    filename = from_dir+f"res/{id1}/{id2}_spo_results_format.txt"
    spo_df = res_file_to_df(filename).rename(columns={0: 's', 1: 'p', 2: 'o'}).reset_index(drop=True)
    if spo_df.empty:
        spo_df = pd.DataFrame(columns = ['s','p','o','idx', 'token_s', 'parse_ranking'])
    spo_df.drop(columns=[c for c in spo_df.columns if type(c) == int], inplace=True)

    filename = from_dir+f"res/{id1}/{id2}_spo_qualifier_results_format.txt"
    spoq_df = res_file_to_df(filename).rename(columns={0: 's', 1: 'p', 2: 'o', 3: 'pq', 4: 'oq'}).reset_index(drop=True)
    spoq_df.drop(columns=[c for c in spoq_df.columns if type(c) == int], inplace=True)
    re_pattern = re.compile("(.*-[0-9]+)")

    if spoq_df.empty:
        spoq_df = pd.DataFrame(columns = ['s','p','o','pq','oq','idx', 'token_s', 'parse_ranking'])
    spoq_df['token_s'] = spoq_df['s'].apply(lambda row: re.match(re_pattern, row)[1])
    spoq_df['token_p'] = spoq_df['p'].apply(lambda row: re.match(re_pattern, row)[1])
    spoq_df['token_o'] = spoq_df['o'].apply(lambda row: re.match(re_pattern, row)[1])
    spoq_df['token_pq'] = spoq_df['pq'].apply(lambda row: re.match(re_pattern, row)[1])
    spoq_df['token_oq'] = spoq_df['oq'].apply(lambda row: re.match(re_pattern, row)[1])
    spoq_df.merge(text_df, on='idx', how='left').to_csv(
        from_dir+f"res/{id1}/{id2}_spoq_to_check.csv", index=False)

    fileout = from_dir+f"res/{id1}/{id2}_phrase_pos_results_words.csv"
    res_df = pd.read_csv(fileout)

    found_phrases = []
    for ix, grp in res_df.groupby(['idx', 'phrase_id'])[['word','token']]:
        tokens = ['#DEL#'.join([str(xx).strip() for xx in grp['token'].values])]
        # words = [' '.join([str(xx).strip() for xx in grp['word'].values])]
        row_data = list(ix) + tokens
        found_phrases.append(row_data)
    found_phrases = pd.DataFrame(found_phrases, columns=['idx', 'phrase_id', 'parsed_phrase'])

    filename = from_dir+f"res/{id1}/{id2}_phrase_results_format.txt"
    phrase_df = res_file_to_df(filename).rename(columns={0: 'phrase'}).reset_index(drop=True)
    if not phrase_df.empty:
        phrase_df = rank_by_length(series=phrase_df['phrase'], df=phrase_df)
        phrase_df = apply_parse_ranking(df=phrase_df)[['idx', 'score']]

    mod_filepath = r_e_run_test(id1=id1,id2=id2)
    _ = format_results(mod_filepath)

    filename = from_dir+f"res/{id1}/{id2}_mod_results_format.txt"
    mod_df = res_file_to_df(filename).rename(columns={0: 'term', 1: 'mod'}).reset_index(drop=True)
    val_cols = ['mod', 'term']
    res_mod = []
    if mod_df.empty:
        mod_df = pd.DataFrame(columns=['idx', 'mod', 'term'])
    else:
        idx_min = 0
        # id_min = vs.name
        for mod_idx, vs in mod_df.loc[idx_min:][['idx'] + val_cols].iterrows():
            idx = vs['idx']
            res_tmp = []
            for c, v in vs[val_cols].items():
                if v == v:
                    match = re.search(r'(.*[\-[0-9]+),(.+)', v)
                    if match:
                        try:
                            token = match[1]
                            pos = match[2]
                            # word = re.search(r'^(.+)\-[0-9]+$',token)[1]
                            res_tmp.append([token, pos])
                        except TypeError:
                            res_tmp.append(['', ''])
            row = [mod_idx, idx] + flatten(res_tmp)
            res_mod.append(row)

    res_mod_df = pd.DataFrame(res_mod, columns=['mod_id', 'idx', 'mod_token', 'mod_pos', 'term_token', 'term_pos'])

    mod_pos = '|'.join(['JJ', 'VB'])
    df_pos = res_mod_df[res_mod_df.mod_pos.str.contains(mod_pos)]
    tmp0 = res_df.merge(df_pos, left_on=['idx', 'token', 'pos'], right_on=['idx', 'term_token', 'term_pos'],
                        how='left'). \
        append(res_df.merge(df_pos, left_on=['idx', 'token', 'pos'], right_on=['idx', 'mod_token', 'mod_pos'])). \
        drop(columns=['word']).rename(columns={'phrase_id': 'phrase_id_mod'}). \
        drop_duplicates()

    tmp0o = res_spo_df[res_spo_df.slot == 'o'].merge(tmp0, on=['idx', 'token', 'pos'], how='inner').drop(
        columns=['slot']).rename(columns={'phrase_id_mod': 'phrase_id_o', 'word': 'word_o'})
    tmp0s = res_spo_df[res_spo_df.slot == 's'].merge(tmp0, on=['idx', 'token', 'pos'], how='inner').drop(
        columns=['slot']).rename(columns={'phrase_id_mod': 'phrase_id_s', 'word': 'word_s'})
    tmp = tmp0s.merge(tmp0o, on=['idx', 'spo_id'], suffixes=['_s', '_o'], how='inner')
    tmp = res_spo_df[res_spo_df.slot == 'p'].drop(columns=['slot']).merge(tmp, on=['spo_id', 'idx']).rename(
        columns={'token': 'token_p', 'pos': 'pos_p', 'word': 'word_p'})

    tmp2 = tmp.merge(found_phrases, left_on=['idx', 'phrase_id_o'], right_on=['idx', 'phrase_id']).drop(
        columns=['phrase_id'])
    tmp2 = tmp2.merge(found_phrases, left_on=['idx', 'phrase_id_s'], right_on=['idx', 'phrase_id'],
                      suffixes=['_o', '_s']).drop(columns=['phrase_id'])

    if tmp2.empty:
        return
    val1 = tmp2.apply(lambda row: '' if type(row['mod_token_o']) == float else re.sub(r'\-[0-9]+', '',row['mod_token_o']) + ' ' + re.sub(r'\-[0-9]+', '', row['term_token_o']), axis=1)
    tmp2['mod_term_o'] = val1 if not val1.empty else np.nan
    val2 = tmp2.apply(lambda row: '' if type(row['mod_token_s']) == float else re.sub(r'\-[0-9]+', '',row['mod_token_s']) + ' ' + re.sub(r'\-[0-9]+', '', row['term_token_s']), axis=1)
    tmp2['mod_term_s'] = val2 if not val2.empty else np.nan
        

    tmp3 = tmp2.drop_duplicates(
        ['idx', 'parsed_phrase_s', 'parsed_phrase_o', 'word_p', 'mod_term_s', 'mod_term_o']).reset_index(drop=True)
    score_cols = ['need_satisfier_in_o_1', 'need_satisfier_in_o_2', 'need_satisfier_in_o_3', 'client_state_in_o_6',
                  'need_in_o_8', 'need_in_o_9', 'need_satisfier_in_o_11', 'need_satisfier_in_s_12',
                  'need_satisfier_in_o_13', 'current_state_in_s_14', 'service_description_in_o_15',
                  'need_satisfier_in_o_16', 'service_in_o_17', 'need_satisfier_description_in_o_18',
                  'service_description_in_o_19', 'desired_state_in_o_20', 'need_satisfier_in_o_22',
                  'service_description_in_o_23', 'service_description_in_o_24', 'need_satisfier_in_o_25',
                  'required_for_in_s_27', 'required_criteria_in_o_28', 'eligibile_criteria_in_s_29',
                  'eligibile_for_in_o_30', 'need_satisfier_in_o_31', 'need_satisfier_description_in_o_32',
                  'service_description_in_o_33', 'client_description_in_p_34', 'program_in_s_35',
                  'need_satisfier_in_o_36', 'service_description_in_s_41', 'client_description_in_o_42',
                  'service_description_in_s_43', 'client_description_in_o_43', 'program_in_s_44',
                  'need_satisfier_in_o_44', 'program_in_s_45', 'service_description_in_o_45',
                  'need_satisfier_in_s_46', 'client_in_o_47']

    tmp3[score_cols] = 0.0
    tmp3.to_csv(stats_directory + f"{id2}_tmp3_pre_assignment.csv", index=False)

    tmp3 = assign_rule_score_obj(tmp3)

    # add parse ranking for SPOs
    tmp3 = tmp3.merge(spo_df[['parse_ranking']], left_on=['spo_id'], right_index=True, how='left').rename(
        columns={'parse_ranking': 'spo_parse_ranking'})
    # add parse ranking for SPOQs
    tmp3s = tmp3.merge(spoq_df[['idx', 'token_s', 'parse_ranking']], left_on=['idx', 'token_s'],
                       right_on=['idx', 'token_s'], how='left'). \
        rename(columns={'parse_ranking': 'spoq_parse_ranking_left'}). \
        fillna({'spoq_parse_ranking_left': 0.0})
    tmp3 = tmp3.merge(
        tmp3s.groupby(['idx', 'parsed_phrase_s', 'parsed_phrase_o', 'word_p', 'mod_term_s', 'mod_term_o'])[
            'spoq_parse_ranking_left'].mean().reset_index(drop=False),
        on=['idx', 'parsed_phrase_s', 'parsed_phrase_o', 'word_p', 'mod_term_s', 'mod_term_o'], how='left')
    tmp3o = tmp3.merge(spoq_df[['idx', 'token_o', 'parse_ranking']], left_on=['idx', 'token_s'],
                       right_on=['idx', 'token_o'], how='left', suffixes=['', '_to_delete']). \
        rename(columns={'parse_ranking': 'spoq_parse_ranking_right'}). \
        drop(columns={'token_o_to_delete'}). \
        fillna({'spoq_parse_ranking_right': 0.0})
    tmp3 = tmp3.merge(
        tmp3o.groupby(['idx', 'parsed_phrase_s', 'parsed_phrase_o', 'word_p', 'mod_term_s', 'mod_term_o'])[
            'spoq_parse_ranking_right'].mean().reset_index(drop=False),
        on=['idx', 'parsed_phrase_s', 'parsed_phrase_o', 'word_p', 'mod_term_s', 'mod_term_o'], how='left')
    # add parse ranking for subject phrase
    tmp3 = tmp3.merge(phrase_df[['score']], left_on=['phrase_id_s'], right_index=True, how='left').rename(
        columns={'score': 'phrase_parse_ranking_s'})
    # add parse ranking for object phrase
    tmp3 = tmp3.merge(phrase_df[['score']], left_on=['phrase_id_o'], right_index=True, how='left').rename(
        columns={'score': 'phrase_parse_ranking_o'})

    tmp3.to_csv(annotation_out_filename, index=False)


def extract_entities(id1,id2,annotations, rule_weights, expand_low_entities=False):
    to_dir = f"{BASE_DIR}tmp/scroll/models/"
    _ = os.makedirs(to_dir+f"r_e/{id1}", exist_ok=True)

    res_df = rule_weights.copy()
    df = annotations.copy()
    df['N'] = 1
    score_cols, mapping, _, _ = score_cols_and_mapping()

    unique_cat = list(mapping.keys())
    unique_cat.sort()
    # colors = {'correct': 'g', 'incorrect': 'r'}
    col = 'mcc'
    # fig,ax = plt.subplots(3,3, figsize=(8,8))
    # fig.suptitle("NER Hypothesis Evaluation")
    # xy = np.resize(unique_cat, ax.shape)
    # [axn.set_ylabel('N',rotation=0) for axn in ax[:,0]]
    # [axn.set_xlabel('Evaluation') for axn in ax[-1]]
    entity_cols = {}
    label_mapping = {}
    for k in unique_cat:
        entity_cols[k] = [[k, k + '_phrase']]
        cat_title = k
        if cat_title == 'client demographic':
            cat_title = 'client characteristic'
        cat_title = cat_title.replace('_', ' ').title()
        label_mapping[k] = cat_title
        descs = mapping[k]
        grp = df  # df[df.ranked_cat == k].copy()

        cat_cols = []
        for desc in descs:
            cat_cols.append([(re.sub(r'.*_([spo]_[0-9]+)$', r'\1', c), c) for c in score_cols if desc + '_in' in c])
        cat_cols = dict(flatten(cat_cols))

        cc = res_df[(res_df.rule.isin(cat_cols.keys()))].copy()
        cat = cc.iloc[0]['cat']
        aggr = res_df[(res_df['cat'] == cat) & (res_df['rule'] == 'Aggregate')].iloc[0].copy()
        ignore_score = expand_low_entities and aggr[col] <= 0
        if not ignore_score:
            cc = res_df[(res_df['mcc'] > 0.0) & (res_df.rule.isin(cat_cols.keys()))].copy()
            # ccn = res_df[(res_df['mcc']<0.0)&(res_df.rule.isin(cat_cols.keys()))]

        cs = res_df[(res_df.rule.isin(cat_cols.keys()))].copy()
        for c, mcc in cs[['rule', 'mcc']].values:
            grp[c] = grp[cat_cols[c]] * mcc

        cc['slot'] = cc.rule.apply(lambda r: r.split('_')[0])

        grp[col] = 0

        for slot, rules in cc.groupby('slot'):
            if not ignore_score:
                correct = grp[grp[rules.rule].gt(0).any(axis=1)].index
            else:
                correct = grp[grp[cat_cols.values()].gt(0).any(axis=1)].index
            grp.loc[correct, col] = 1
            grp.loc[correct, k] = grp.loc[correct]['token_%s' % (slot)]
            if slot == 'p':
                grp.loc[correct, k + '_phrase'] = grp.loc[correct]['word_%s' % (slot)]
            else:
                grp.loc[correct, k + '_phrase'] = grp.loc[correct]['parsed_phrase_%s' % (slot)]

    df.to_csv(to_dir+f"/r_e/{id1}/{id2}_extracted_entities.csv", index=False)
    extracted_entities = df.copy()
    return extracted_entities, label_mapping


def assign_rule_score_obj(tmp3_in):
    tmp3 = tmp3_in.copy()
    own_p = ['PRP', 'PRP$']
    adj_pos = ['JJ']
    eq_adj_pos = ['JJS', 'JJR']
    self_word = ['our', 'we', 'us']
    them_word = ['you', 'your', 'them', 'they', 'their', 'he', 'she'] + ['person', 'people', 'client', 'clients']

    # past
    t = tmp3[tmp3.pos_p == 'VBP']
    # service in s
    # need_satisfiers ranking
    idx = t[(t.pos_s.isin(own_p)) & (t.word_s.str.lower().isin(self_word)) & (~t.pos_o.isin(['CD', 'DT'])) & (
        t.mod_pos_o.isin(eq_adj_pos))].index
    tmp3.loc[idx, 'need_satisfier_in_o_1'] += 1

    # need satisfiers
    # TODO: check p for direction
    idx = t[(t.pos_s.isin(own_p)) & (t.word_s.str.lower().isin(self_word)) & (
        ~t.pos_o.isin(['CD', 'DT', 'JJ', 'PRP', 'NNP', 'NNPS', 'NNS'])) & (t.mod_pos_o.isin(adj_pos))].index
    tmp3.loc[idx, 'need_satisfier_in_o_2'] += 1

    idx = t[(t.pos_s.isin(own_p)) & (t.word_s.str.lower().isin(self_word)) & (
        ~t.pos_o.isin(['CD', 'DT', 'NN', 'PRP'])) & (t.mod_pos_o.isin(adj_pos))].index
    tmp3.loc[idx, 'need_satisfier_in_o_3'] += 1


    idx = t[(t.pos_s.isin(own_p)) & (t.word_s.str.lower().isin(them_word)) & (
        ~t.mod_pos_o.isin(adj_pos + eq_adj_pos))].index
    tmp3.loc[idx, 'need_in_o_8'] += 1
    # Todo: check direciton of p
    idx = t[(t.pos_s.isin(own_p)) & (t.word_s.str.lower().isin(them_word)) & (t.mod_pos_o.isin(adj_pos))].index
    tmp3.loc[idx, 'need_in_o_9'] += 1

    # past - participle
    t = tmp3[tmp3.pos_p == 'VBN']
    # service in s

    idx = t[(t.word_s.str.lower().isin(them_word)) & (~t.mod_pos_o.isin(["VB", "VBZ"]))].index
    tmp3.loc[idx, 'need_satisfier_in_o_11'] += 1

    # client in s
    idx = t[(t.word_o.str.lower().isin(them_word)) & (~t.mod_pos_s.isin(["VB", "VBZ"]))].index
    tmp3.loc[idx, 'need_satisfier_in_s_12'] += 1

    # present
    t = tmp3[tmp3.pos_p == 'VBG']
    # client in s
    idx = t[(t.pos_s.isin(own_p)) & (t.word_s.str.lower().isin(them_word))].index
    tmp3.loc[idx, 'current_state_in_s_14'] += 1

    # need in o
    idx = t[(t.pos_o.isin(['NN', 'NNS', 'NNP', 'NNPS']))].index
    tmp3.loc[idx, 'need_satisfier_in_o_13'] += 1

    # present
    t = tmp3[tmp3.pos_p == 'VBP']
    # service in s
    # n/a

    idx = t[(t.word_s.str.lower().isin(self_word)) & (t.mod_pos_o == 'VBG') & (
        ~t.term_pos_o.isin(['DT', 'JJ', 'NNS', 'NNP', 'NNPS']))].index
    tmp3.loc[idx, 'service_description_in_o_15'] += 1

    # TODO: check as some o are not corrrect (15/20 are good)
    idx = t[(t.word_s.str.lower().isin(self_word)) & (t.mod_pos_o == 'VBN') & (
        ~t.term_pos_o.isin(['DT', 'NNPS', 'NNP']))].index
    tmp3.loc[idx, 'need_satisfier_in_o_16'] += 1

    idx = t[(t.word_s.str.lower().isin(self_word)) & (t.mod_pos_o == 'VBN') & (
        t.term_pos_o.isin(['NNP', 'NNPS']))].index
    tmp3.loc[idx, 'service_in_o_17'] += 1

    idx = t[(t.word_s.str.lower().isin(self_word)) & (t.mod_pos_o == 'VBZ') & (
        ~t.term_pos_o.isin(['CD', 'NNP', 'NNS']))].index
    tmp3.loc[idx, 'need_satisfier_description_in_o_18'] += 1

    idx = t[(t.word_s.str.lower().isin(self_word)) & (t.mod_pos_o == 'VBP') & (
        ~t.term_pos_o.isin(['DT', 'NNPS', 'NNP']))].index
    tmp3.loc[idx, 'service_description_in_o_19'] += 1

    # service in o
    # todo: combine with p + o to get full picture
    # todo: get p direction
    idx = t[(t.word_s.str.lower().isin(them_word)) & (~t.pos_o.isin(['NNP', 'NNPS', 'PRP', 'PRP$'])) & (
        ~t.mod_pos_o.isin(['JJ'])) & (~t.term_pos_o.isin(['DT'])) & (~t.pos_o.isin(['DT', 'WP', 'WDT']))].index
    tmp3.loc[idx, 'desired_state_in_o_20'] += 1


    # todo: check is s=client
    idx = t[(t.word_s.str.lower().isin(them_word)) & (~t.mod_pos_o.isin(['JJ', 'JJS', 'JJR'])) & (
        ~t.term_pos_o.isin(['DT']))].index
    tmp3.loc[idx, 'need_satisfier_in_o_22'] += 1

    # present
    t = tmp3[tmp3.pos_p == 'VBZ']
    # service in s
    idx = t[(t.pos_s.isin(own_p)) & (t.word_s.str.lower().isin(self_word)) & (
        t.mod_pos_o.isin(['JJ', 'JJS', 'JJR'])) & (~t.term_pos_o.isin(['DT']))].index
    tmp3.loc[idx, 'service_description_in_o_23'] += 1
    idx = t[(t.pos_s.isin(own_p)) & (t.word_s.str.lower().isin(self_word)) & (
        ~t.mod_pos_o.isin(['JJ', 'JJS', 'JJR'])) & (t.term_pos_o.isin(['NNP']))].index
    tmp3.loc[idx, 'service_description_in_o_24'] += 1

    # client in s
    idx = t[(t.pos_s.isin(own_p)) & (t.word_s.str.lower().isin(them_word)) & (t.pos_o == 'NN')].index
    tmp3.loc[idx, 'need_satisfier_in_o_25'] += 1

    # JJ
    t = tmp3[tmp3.pos_p.isin(['JJ'])]

    # todo: provide context, esp on s since s can be gathered form other service "description"
    # s=service, o=client
    idx = t[
        (t.mod_pos_o == 'JJ') & (t.pos_s.isin(['NNS'])) & (t.pos_o.isin(['NNS'])) & (t.word_p == 'available')].index
    tmp3.loc[idx, 'required_for_in_s_27'] += 1
    tmp3.loc[idx, 'required_criteria_in_o_28'] += 1

    # s=client, o=servicec
    idx = t[(t.mod_pos_o == 'JJ') & (t.pos_s.isin(['NNS'])) & (t.pos_o.isin(['NN', 'NNS'])) & (
                t.word_p == 'eligible')].index
    tmp3.loc[idx, 'eligibile_criteria_in_s_29'] += 1
    tmp3.loc[idx, 'eligibile_for_in_o_30'] += 1

    # NN
    t = tmp3[tmp3.pos_p.str.contains('NN')]

    idx = t[
        (t.pos_o.isin(['NN', 'NNS', 'NNP', 'NNPS'])) & (t.mod_pos_o == 'JJ') & (t.pos_s.isin(['NN', 'NNS']))].index
    tmp3.loc[idx, 'need_satisfier_in_o_31'] += 1

    idx = t[(t.pos_o.isin(['NN', 'NNS', 'NNP', 'NNPS'])) & (t.mod_pos_o.isin(['VBN'])) & (
        t.pos_s.isin(['NNP', 'NNPS']))].index
    tmp3.loc[idx, 'need_satisfier_description_in_o_32'] += 1

    # NNS
    t = tmp3[tmp3.pos_p.str.contains('NNS')]

    # p=client, o=client_state
    idx = t[(t.pos_s == 'WP') & (t.mod_pos_o == 'JJ')].index
    tmp3.loc[idx, 'service_description_in_o_33'] += 1
    tmp3.loc[idx, 'client_description_in_p_34'] += 1

    # program as s=NNP(S)
    t = tmp3[tmp3.pos_s.isin(['NNP', 'NNPS'])]

    offers_sym = ["provides", "offers", "offer", "provide", "provided", "offered", "offering", 'include']

    idx = t[(t.word_p.isin(offers_sym))].index
    tmp3.loc[idx, 'program_in_s_35'] += 1

    idx = t[(t.word_p.isin(offers_sym)) & (t.pos_o.isin(['NN', 'NNS']))].index
    tmp3.loc[idx, 'need_satisfier_in_o_36'] += 1

    # build spoq
    spoq_df = tmp3.reset_index(drop=False).merge(tmp3.reset_index(drop=False), left_on=['idx', 'token_o'],
                                                 right_on=['idx', 'token_s'], how='inner', suffixes=['', 'q'])

    # save roles for program, service, client relationships
    # save as program [offers] client [with] service
    predicates = ['provides-', 'offer-', 'provide-', 'provided-', 'offered-', 'offering-', 'include-']
    spoq1 = spoq_df[(spoq_df['pos_pq'] == 'IN') &
                    (spoq_df['token_pq'].str.contains('with-')) &
                    (spoq_df['token_p'].str.contains('|'.join(predicates)))
                    ]
    spoq1[list(spoq1.columns[:11]) + list([c + 'q' for c in spoq1.columns[4:11]])]

    idx = tmp3[(~tmp3.pos_s.isin(['PRP'])) & (tmp3.pos_o.isin(['NN', 'NNS', 'JJ', 'VBG', 'FW']))].merge(
        spoq1[['indexq', 'idx', 'spo_idq', 'phrase_id_sq', 'phrase_id_oq']],
        left_on=['idx', 'spo_id', 'phrase_id_s', 'phrase_id_o'],
        right_on=['idx', 'spo_idq', 'phrase_id_sq', 'phrase_id_oq'], how='inner')['indexq']
    tmp3.loc[idx, 'service_description_in_s_41'] += 1
    tmp3.loc[idx, 'client_description_in_o_42'] += 1

    idx = tmp3[tmp3.pos_o.isin(['NN', 'NNS', 'JJ'])].merge(
        spoq1[['indexq', 'idx', 'spo_idq', 'phrase_id_sq', 'phrase_id_oq']],
        left_on=['idx', 'spo_id', 'phrase_id_s', 'phrase_id_o'],
        right_on=['idx', 'spo_idq', 'phrase_id_sq', 'phrase_id_oq'], how='inner')['indexq']
    tmp3.loc[idx, 'service_description_in_s_43'] += 1
    tmp3.loc[idx, 'client_description_in_o_43'] += 1

    # save as program [offers] service [to,for] client
    spoq2 = spoq_df[(spoq_df['pos_pq'] == 'IN') &
                    (spoq_df['token_pq'].str.contains('to-|for-')) &
                    (spoq_df['token_p'].str.contains('|'.join(predicates)))
                    ]
    spoq2[list(spoq2.columns[:11]) + list([c + 'q' for c in spoq2.columns[4:11]])]

    # find SPO but don't flag Program as s when s"we" with pos_s=='PRP'
    idx = \
    tmp3[tmp3.pos_s.isin(['NNP', 'NNPS'])].merge(spoq2[['index', 'idx', 'spo_id', 'phrase_id_s', 'phrase_id_o']],
                                                 on=['idx', 'spo_id', 'phrase_id_s', 'phrase_id_o'], how='inner')[
        'index']
    tmp3.loc[idx, 'program_in_s_44'] += 1
    idx = tmp3.merge(spoq2[['index', 'idx', 'spo_id', 'phrase_id_s', 'phrase_id_o']],
                     on=['idx', 'spo_id', 'phrase_id_s', 'phrase_id_o'], how='inner')['index']
    tmp3.loc[idx, 'need_satisfier_in_o_44'] += 1

    idx = tmp3[tmp3.pos_s.isin(['NN', 'NNS', 'JJ', 'JJ', 'VBG', 'FW'])].merge(
        spoq2[['index', 'idx', 'spo_id', 'phrase_id_s', 'phrase_id_o']],
        on=['idx', 'spo_id', 'phrase_id_s', 'phrase_id_o'], how='inner')['index']
    tmp3.loc[idx, 'program_in_s_45'] += 1
    tmp3.loc[idx, 'service_description_in_o_45'] += 1

    idx = tmp3.merge(spoq2[['indexq', 'idx', 'spo_idq', 'phrase_id_sq', 'phrase_id_oq']],
                     left_on=['idx', 'spo_id', 'phrase_id_s', 'phrase_id_o'],
                     right_on=['idx', 'spo_idq', 'phrase_id_sq', 'phrase_id_oq'], how='inner')['indexq']
    tmp3.loc[idx, 'need_satisfier_in_s_46'] += 1
    tmp3.loc[idx, 'client_in_o_47'] += 1

    return tmp3


def generate_annotation_file(extracted_entities, label_mapping, id1,id2):
    to_dir = f"{BASE_DIR}tmp/scroll/models/{id1}"
    _ = os.makedirs(to_dir, exist_ok=True)

    from_dir = f"{BASE_DIR}tmp/scroll/"
    df = extracted_entities.copy()
    indexed_sentences = pd.read_csv(from_dir+f"data/{id1}/{id2}_indexed_sentences.csv")
    _,mapping,_,_ = score_cols_and_mapping()
    unique_cat = list(mapping.keys())
    unique_cat.sort()
    res = pd.DataFrame(columns=['idx'])
    for k in unique_cat:
        if k not in df.columns:
            continue
        grp = df[~df[k].isnull()].copy()
        if grp.shape[0]==0:
            continue
        tmp = grp.groupby(['idx',k], as_index = False).agg({k+"_phrase":set})
        res = res.merge(tmp,on=['idx'],how='outer')

    annotations_dict = {}
    for ridx,r in res.iterrows():
        text = indexed_sentences[indexed_sentences.idx==r.idx].iloc[0].indexed_text
        terms = text.split('#DEL#')
        rris = {}
        for k in unique_cat:
            if k not in r.index or r[k] != r[k]:
                continue
            if k not in rris.keys():
                rris[k] = []
            for rr in r[k].split(','):
                rris[k].append(terms.index(rr))

        re_compiled = re.compile(r'(.+)(\-[0-9]+)')
        text = ''
        labels = []
        for i,term in enumerate(terms):
            match = re.search(re_compiled,term)
            hit, hi = match[1],match[2]
            for k,rr in rris.items():
                if i in rr:
                    tstart = len(text)
                    tend = tstart + len(hit)
                    labels.append([tstart,tend,k,hit,list(r[k+"_phrase"])])
            text += hit + ' '

        if r.idx not in annotations_dict.keys():
            annotations_dict[r.idx] = {'text':text,'label':[]}
        annotations_dict[r.idx]['label'].append(labels)

    annotations = []
    for idx,v in annotations_dict.items():
        labels = flatten(v['label'])
        spans = []
        for label in labels:
            tmp = []
            k = label[2]
            cat_title = label_mapping[k]
            for phrase in label[4]:
                combs = list(itert.permutations(phrase.split(' ')))
                for comb in combs:
                    regex = ("(%s)"%('(( |\\|\~|\-|\n|\+|&){1,3}(%s)*){1,2}'%('|'.join(STOPWORDS))).join(comb))

                    spans_tmp = [g.span() for g in list(re.finditer(regex, v['text'], re.IGNORECASE))]
                    spans_tmp = mergeIntervals(spans_tmp)
                    tmp.append(spans_tmp)
            tmp = flatten(tmp)
            tmp = mergeIntervals(tmp)
            spans.append([t+[cat_title] for t in tmp])
        spans = dedup(flatten(spans))


        annotations.append({"idx":idx,"text":v['text'], "label":spans})

    with open(to_dir+f"/{id2}_annotations.json", 'w') as file:
        for ann in annotations:
            json.dump(ann, file)
            _=file.write('\n')

    print("Saved to %s"%(to_dir+f"/{id2}_annotations.json"))
    return annotations

def generate_annotations(id1,id2, text, rule_weights):
    import os
    to_dir = f"{BASE_DIR}tmp/scroll/"
    annotation_file = to_dir+f"stats/{id1}/{id2}_annot_obj.csv"
    if os.path.exists(annotation_file): os.remove(annotation_file)
    from_mods(text=text, id1=id1,id2=id2)
    try:
        df_annotation_objects = pd.read_csv(annotation_file)

        extracted_entities, label_mapping = extract_entities(id1=id1,id2=id2, annotations=df_annotation_objects, rule_weights=rule_weights)
        match_dict = {}
        for col in [c for c in extracted_entities.columns if c.endswith('_phrase')]:
            # [c for c in extracted_entities.columns if c.endswith('_phrase')]
            match_dict[col] = extracted_entities[col].dropna().drop_duplicates().values

        # print(extracted_entities, label_mapping)
        return generate_annotation_file(extracted_entities, label_mapping, id1=id1,id2=id2)
    except FileNotFoundError:
        return []


def collect_ner_tokens(id1,id2, text, rule_weights):
    import os
    to_dir = f"{BASE_DIR}tmp/scroll/"
    annotation_file = to_dir+f"stats/{id1}/{id2}_annot_obj.csv"
    if os.path.exists(annotation_file): os.remove(annotation_file)
    from_mods(text=text, id1=id1,id2=id2)
    try:
        df_annotation_objects = pd.read_csv(annotation_file)

        extracted_entities, label_mapping = extract_entities(id1=id1,id2=id2, annotations=df_annotation_objects, rule_weights=rule_weights)
        match_dict = {}
        for col in [c for c in extracted_entities.columns if c.endswith('_phrase')]:
            # [c for c in extracted_entities.columns if c.endswith('_phrase')]
            mapped_col = label_mapping[re.sub(r"_phrase$", '', col)]
            match_dict[mapped_col] = extracted_entities[col].dropna().drop_duplicates().values
            match_dict[mapped_col] = [matches.split("#DEL#") for matches in match_dict[mapped_col]]
        return match_dict
    except FileNotFoundError:
        return {}

