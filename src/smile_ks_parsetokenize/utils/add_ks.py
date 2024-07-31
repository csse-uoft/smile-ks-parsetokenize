from owlready2 import default_world, onto_path, ObjectProperty, DataProperty, rdfs, Thing 
onto_path.append('./ontology_cache/')
from py2graphdb.config import config as CONFIG
smile = default_world.get_ontology(CONFIG.NM)
with smile:
    from smile_base.Model.controller.ks import Ks

def add_ks(reload_db=True):
    if reload_db:
        kss = Ks.search(props={smile.hasPyName:'ParseTokenize'}, how='all')
        for ks in kss:
            ks.delete(refs=False)

    ALL_KS_FORMATS = {
        'Parse/Tokenize': ['ParseTokenize', False, ["Text"], ["Word", "Pos", "Dep", "CoRef"]]
    }

    for ks_name, fields in ALL_KS_FORMATS.items():
        Ks.ALL_KS_FORMATS[ks_name] = fields
    if reload_db:
        for ks_name in ALL_KS_FORMATS.keys():
            Ks.initialize_ks(ks_name)

