import re, os, pandas as pd, tqdm
from owlready2 import default_world, ObjectProperty, DataProperty, rdfs, Thing 
from py2graphdb.config import config as CONFIG
smile = default_world.get_ontology(CONFIG.NM)
with smile:
    from py2graphdb.Models.graph_node import GraphNode, SPARQLDict, _resolve_nm
    from py2graphdb.utils.db_utils import resolve_nm_for_dict, PropertyList

    from smile_ks_parsetokenize.libs import nlp_parser
    from smile_base.Model.knowledge_source.knowledge_source import KnowledgeSource    
    from smile_base.Model.data_level.hypothesis import Hypothesis
    from smile_base.Model.data_level.word      import Word
    from smile_base.Model.data_level.pos       import Pos
    from smile_base.Model.data_level.coref     import CoRef
    from smile_base.Model.data_level.dep       import Dep
    from smile_base.Model.data_level.text      import Text
    from smile_base.Model.data_level.spo       import Spo
    from smile_base.Model.data_level.phrase    import Phrase
    from smile_base.Model.data_level.ner       import Ner
    from smile_base.Model.controller.ks        import Ks
    from smile_base.Model.controller.ks_ar     import KSAR
    from smile_base.Model.controller.trace     import Trace


import time


class ParseTokenize(KnowledgeSource):
    """
    A knowledge source class that processes QA1 Ner

    Attributes
    ----------
    description: str
        String of description to be parsed
    annotation: Dict
        Formatted annotation for each task
    corenlp_output: Dict
        Annotated output of StanfordCoreNLP parser
    """

    RULE_WEIGHTS = pd.read_csv(os.path.dirname(__file__)+"/libs/nlp_parser/rule_ranking_merged_on_mcc.csv")

    MAPPING = {
        'PERSON':'need_satisfier', 
        'LOCATION':'catchment_area', 
        'DATE':'date',
        'Client Characteristic':'client',
        'Client Description':'client_desc',
        'Desired State (Outcome)': 'service_output',
        'Need':'need',
        'Need Satisfier':'need_satisfier',
        'Need Satisfier Description':'need_satisfier_desc',
        'Program Name':'program_name',
        'Required Criteria':'criteria',
        'Service Description':'service_desc',
    }

    def __init__(self, hypothesis_ids, ks_ar, trace):
        fields = [v for v in Ks.ALL_KS_FORMATS.values() if v[0] == self.__class__.__name__][0]
        super().__init__(fields[1], fields[2], fields[3], trace, hypothesis_ids, ks_ar)


        self.description        = None
        self.annotation         = {"Word and Pos": None, "Dep": None, "CoRef": None}
        self.corenlp_output     = None
        self.df_phrases         = None
        self.df_spos            = None
        self.annotation_objects = None
        self.store_hypotheses = []

    @classmethod
    def process_ks_ars(cls, loop=True):
        """
        A class method that processes all the ks_ars with py_name='ParseTokenize' and status=0.

        :param cls: The class itself (implicit parameter).
        :type cls: type
        :return: None
        """
        while True:
            
            ks = Ks.search(props={smile.hasPyName:'ParseTokenize'}, how='first')
            if len(ks) >0:
                ks = ks[0]
            else:
                continue
            ks_ar = KSAR.search(props={smile.hasKS:ks.id, smile.hasKSARStatus:0}, how='first')
            if len(ks_ar) > 0:
                ks_ar = ks_ar[0]
                cls.logger(trace_id=ks_ar.trace, text=f"Processing ks_ar with id: {ks_ar.id}")

                # Get the hypothesis ids from the ks_ar
                in_hypo_ids = ks_ar.input_hypotheses
                if len(in_hypo_ids) != 1:
                    raise(Exception(f"Bad Input Hypothesis Count {len(in_hypo_ids)}"))

                in_hypo = Hypothesis(inst_id=in_hypo_ids[0])
                in_hypo.cast_to_graph_type()
                if not isinstance(in_hypo, (smile.Text, smile.Sentence)): #check if Phras
                    raise(Exception(f"Bad Input Hypothesis Type {type(in_hypo)}"))

                # Get the trace from the ks_ar
                trace = Trace(inst_id=ks_ar.trace)
                
                # Construct an instance of the ks_object
                ks_object = cls(hypothesis_ids=in_hypo_ids, ks_ar=ks_ar, trace=trace)
                
                # Call ks_object.set_input() with the necessary parameters
                ks_ar.ks_status = 1
                ks_object.set_input(description=in_hypo.content)
                
                ks_ar.ks_status = 2               
                hypotheses = ks_object.get_outputs()
                ks_ar.keep_db_in_synch = False
                trace.keep_db_in_synch = False
                for hypo in hypotheses:
                    ks_ar.hypotheses = hypo.id 
                    trace.hypotheses = hypo.id

                ks_ar.save()
                trace.save()
                ks_ar.keep_db_in_synch = True
                trace.keep_db_in_synch = True
                # log output
                LOG_FILE_TEMPLATE = CONFIG.LOG_DIR+'smile_trace_log.txt'
                filename = LOG_FILE_TEMPLATE.replace('.txt', f"_{trace.id}.txt")
                ks_ar.summary(filename=filename)

                ks_ar.ks_status = 3  

                if not loop:
                    return ks_ar                              

            time.sleep(1)        

    def set_input(self, description):
        import shutil, os
        BASE_DIR = os.path.dirname(__file__)

        if os.path.exists(BASE_DIR+'/libs/nlp_parser/scroll/data/'): shutil.rmtree(BASE_DIR+'/libs/nlp_parser/scroll/data/')
        if os.path.exists(BASE_DIR+'/libs/nlp_parser/scroll/models/'): shutil.rmtree(BASE_DIR+'/libs/nlp_parser/scroll/models/')
        if os.path.exists(BASE_DIR+'/libs/nlp_parser/scroll/res/'): shutil.rmtree(BASE_DIR+'/libs/nlp_parser/scroll/res/')
        if os.path.exists(BASE_DIR+'/libs/nlp_parser/scroll/stats/'): shutil.rmtree(BASE_DIR+'/libs/nlp_parser/scroll/stats/')

        self.store_hypotheses = []
        self.set_basics(description=description)
        self.set_ners()
        self.set_spos()
        self.set_phrases()
        self.set_r_t()
        self.set_r_e()

    def get_outputs(self):
        self.get_basics()
        self.get_ners()
        self.get_spos()
        self.get_phrases()
        self.get_r_t()
        self.get_r_e()

        return self.store_hypotheses

    def set_basics(self, description):
        data_levels=("Word", "Pos", "Dep", "CoRef")

        """Run corenlp parsing functions of the requested output data levels.

        :param data_levels: output data levels that the contoller requests.
                            default is set to all possible output levels.
                            it could be both a string or a list.
        :return: updated corenlp output
        """
        self.description = description
        self.corenlp_output = nlp_parser.parse(self.description)

        _ = nlp_parser.build_prolog_from_output(id1=self.trace.id, id2=self.ks_ar.id, text=self.description, corenlp_output=self.corenlp_output)


        if "Dep" in data_levels:  # Get triples
            if self.annotation["Dep"] is None:
                self.annotation["Dep"] = nlp_parser.resolved_to_triples(self.corenlp_output)

        if ("Word" in data_levels) or ("Pos" in data_levels):  # get a list of word information
            if self.annotation["Word and Pos"] is None:
                self.annotation["Word and Pos"] = nlp_parser.get_words(self.corenlp_output)


        if "CoRef" in data_levels:  # Get a nested list of token information with resolved coreference.
            if self.annotation["CoRef"] is None:
                self.annotation["CoRef"] = nlp_parser.generate_coref(self.corenlp_output)

        return self.annotation


    def set_ners(self):
        if self.corenlp_output is None:
            if self.description is None:
                raise(Exception(f"Missing description in set_ners"))
            else:
                self.corenlp_output = nlp_parser.parse(self.description)
        self.ners_list = []
        certainty = 1.0
        for sent in self.corenlp_output['sentences']:
            for entity_row in sent['entitymentions']:
                start = entity_row['characterOffsetBegin']
                end = entity_row['characterOffsetEnd']
                text = entity_row['text']
                if 'nerConfidences' in entity_row.keys():
                    for entity_guess, certainty in entity_row['nerConfidences'].items():
                        if entity_guess in self.MAPPING.keys():
                            entity = self.MAPPING[entity_guess]
                            self.ners_list.append({
                                'entity':entity,
                                'certainty': certainty,
                                'text':text,
                                'start':start,
                                'end':end,
                            })
                elif 'ner' in entity_row.keys():
                    if entity_row['ner'] in self.MAPPING.keys():
                        entity_guess = entity_row['ner']
                        entity = self.MAPPING[entity_guess]
                        self.ners_list.append({
                            'entity':entity,
                            'certainty': certainty,
                            'text':text,
                            'start':start,
                            'end':end,
                        })

    def get_ners(self):
        for row in self.ners_list:
            phrase = Phrase.find_generate(
                content=row['text'], start=row['start'], end=row['end'],trace_id=self.trace.id, certainty=row['certainty'])
            phrase.from_ks_ars = self.ks_ar.id
            self.store_hypotheses.append(phrase)

            ner = Ner.generate(phrase_id=phrase.id, entity=row['entity'],trace_id=self.trace.id, certainty=row['certainty'])
            ner.from_ks_ars = self.ks_ar.id
            self.store_hypotheses.append(ner)


    def get_basics(self):
        rel_word_queries = {}
        # self.ks_ar
        if self.annotation["Word and Pos"] is not None:
            for token in tqdm.tqdm(self.annotation["Word and Pos"], total=len(self.annotation["Word and Pos"]), desc='Word and Pos'):

                certainty = 1
                word = Word.find_generate(
                    trace_id=self.trace.id,
                    content=token["content"],
                    content_label = token["content_label"],
                    start=token["start"],
                    end=token["end"],
                    certainty=certainty)
                word.from_ks_ars = self.ks_ar.id
                self.store_hypotheses.append(word)
                rel_word_queries[token["content_label"]] = word

                certainty = 1
                pos = Pos.find_generate(word_ids=[word.id], pos_tag=token["pos"],trace_id=self.trace.id, certainty=certainty)
                pos.from_ks_ars = self.ks_ar.id
                word.pos = pos.inst_id
                self.store_hypotheses.append(pos)

        if self.annotation["Dep"] is not None:
            for v in tqdm.tqdm(self.annotation["Dep"], total=len(self.annotation["Dep"]), desc='Dep'):
                dep = v[1]
                subject_content_label = v[0][0]
                object_content_label = v[2][0]
                subject_w, subject_i = re.findall(r'(.+)\-([0-9]+)', subject_content_label)[:2][0]

                if subject_content_label not in rel_word_queries.keys():
                    continue
                subject_word_id = rel_word_queries[subject_content_label].id
                
                object_w, object_i = re.findall(r'(.+)\-([0-9]+)', object_content_label)[:2][0]
                if object_content_label not in rel_word_queries.keys():
                    continue
                object_word_id = rel_word_queries[object_content_label].id

                certainty = 1
                dep = Dep.find_generate(dep=dep, subject_id=subject_word_id, object_id=object_word_id,trace_id=self.trace.id, certainty=certainty)
                dep.from_ks_ars = self.ks_ar.id
                self.store_hypotheses.append(dep)


        if self.annotation["CoRef"] is not None:
            for v in tqdm.tqdm(self.annotation["CoRef"], total=len(self.annotation["CoRef"]), desc='CoRef'):
                coref_content_label, ref_content_label = v
                
                if coref_content_label in rel_word_queries.keys() and ref_content_label in rel_word_queries.keys():
                    coref_word = rel_word_queries[coref_content_label] 
                    ref_word = rel_word_queries[ref_content_label]

                    certainty = 1
                    coref = CoRef.find_generate(coref_word_id=coref_word.id, ref_word_id=ref_word.id,trace_id=self.trace.id, certainty=certainty)
                    coref.from_ks_ars = self.ks_ar.id
                    self.store_hypotheses.append(coref)

        return self.store_hypotheses


    def set_spos(self):
        self.df_spos = nlp_parser.generate_spos(id1=self.trace.id, id2=self.ks_ar.id, corenlp_output=self.corenlp_output)

    def get_spos(self):
        for spo_id in self.df_spos["spo_id"].unique():
            this_spo_df = self.df_spos[self.df_spos["spo_id"] == spo_id]
            this_spo = {"s": None, "p": None, "o": None}
            for i, row in this_spo_df.iterrows():
                content_label = row.token
                words = Word.search(props={smile.hasContentLabel:content_label, smile.hasTraceID:self.trace.id}, how='first')
                if len(words)>0:
                    word = words[0]
                    this_spo[row["slot"]] = word.id

            spo = Spo.find_generate(subject_id=this_spo["s"], predicate_id=this_spo["p"], object_id=this_spo["o"], trace_id=self.trace.id)
            spo.from_ks_ars = self.ks_ar.id
            self.store_hypotheses.append(spo)

    def set_r_t(self):
        if self.df_spos is None:
            self.set_spos()

    def get_r_t(self):
        for spo_id in self.df_spos["spo_id"].unique():
            this_spo_df = self.df_spos[self.df_spos["spo_id"] == spo_id]
            this_spo = {"s": None, "p": None, "o": None}
            for i, row in this_spo_df.iterrows():
                content_label = row.token
                words = Word.search(props={smile.hasContentLabel:content_label,smile.hasTraceID:self.trace.id}, how='first')
                if len(words)>0:
                    this_spo[row["slot"]] = words[0].id
            # TODO: check this call params
            spo = Spo.find_generate(subject_id=this_spo["s"], predicate_id=this_spo["p"], object_id=this_spo["o"], trace_id=self.trace.id)
            spo.from_ks_ars = self.ks_ar.id
            self.store_hypotheses.append(spo)


    def set_r_e(self):
        if self.df_phrases is None:
            self.df_phrases = nlp_parser.gen_phrase_pos_rank(id1=self.trace.id, id2=self.ks_ar.id)

        # self.annotation_objects = nlp_parser.generate_annotations(id1=self.trace.id, id2=self.ks_ar.id, text=self.description, rule_weights=self.RULE_WEIGHTS)
        self.ner_objects = nlp_parser.collect_ner_tokens(id1=self.trace.id, id2=self.ks_ar.id, text=self.description, rule_weights=self.RULE_WEIGHTS)


    def get_r_e(self):

        for entity_type, matches in self.ner_objects.items():
            for tokens in matches:
                words = Word.search(props={smile.hasTraceID:self.trace_id, has(smile.hasContent):(tokens)}, how='all')
                entity_text = ' '.join([w.content for w in words])
                start = min([w.start for w in words])
                end = min([w.end for w in words])
                
                ner_certainty = float(self.RULE_WEIGHTS[(self.RULE_WEIGHTS["rule"] == "Aggregate") &
                                                    (self.RULE_WEIGHTS["cat"] == entity_type)]["accuracy"])

                assoc_phrase = Phrase.find_generate(content=entity_text, trace_id=self.trace.id, start=start,end=end)
                assoc_phrase.from_ks_ar_id = self.ks_ar.id
                # Add words one by one if they don't exist in assoc_phrase.words
                for word in words:
                    if word.id not in assoc_phrase.words:
                        assoc_phrase.words = word.id
                # assoc_phrase.words += words
                
                ner = Ner.generate(phrase_id=assoc_phrase.id,entity=self.MAPPING[entity_type],trace_id=self.trace.id, certainty=ner_certainty)
                ner.from_ks_ars = self.ks_ar.id
                self.store_hypotheses.append(ner)
        return self.store_hypotheses



    def set_phrases(self):
        self.df_phrases = nlp_parser.gen_phrase_pos_rank(id1=self.trace.id, id2=self.ks_ar.id)

    def get_phrases(self):
        for phrase_id in self.df_phrases["phrase_id"].unique():
            this_phrase_df = self.df_phrases[self.df_phrases["phrase_id"] == phrase_id]
            this_phrase_words = []
            this_phrase_texts = []
            this_phrase_starts = []
            this_phrase_ends = []
            for i, row in this_phrase_df.iterrows():
                content_label = row.token
                words = Word.search({smile.hasContentLabel:content_label,smile.hasTraceID:self.trace.id}, how='first')

                if len(words) > 0:
                    # word = word_db.Word.find_generate(content_label=content_label,trace_id=self.trace.id).first()
                    word = words[0]
                    word.from_ks_ars = self.ks_ar.id

                    this_phrase_words.append(word)
                    this_phrase_texts.append(word.content)
                    this_phrase_starts.append(word.start)
                    this_phrase_ends.append(word.end)

            if len(this_phrase_texts) >0:
                phrase_text = " ".join(this_phrase_texts)
                phrase_start = min(this_phrase_starts)
                phrase_end = max(this_phrase_ends)
                phrase = Phrase.find_generate(content=phrase_text, start=phrase_start, end=phrase_end,trace_id=self.trace.id)
                phrase.from_ks_ars = self.ks_ar.id
                phrase.words = this_phrase_words
                self.store_hypotheses.append(phrase)

        return self.store_hypotheses


if __name__ == '__main__':
    print('ParseTokenize started')
    pt_ks = Ks()
    pt_ks.ALL_KS_FORMATS['Parse/Query'] = ['ParseQuery', False, ['Query'], ['Text']]
    pt_ks.initialize_ks(ks_name='Parse/Query', field=pt_ks.ALL_KS_FORMATS['Parse/Query'])

    with smile:
        ParseTokenize.process_ks_ars()

