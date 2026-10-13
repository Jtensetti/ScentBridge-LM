import json,sys,tempfile,unittest
from pathlib import Path
from decimal import Decimal
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from scentbridge.contracts import validate_percent,scale,parse_answer,grounded,reference
from scentbridge.runtime import Catalog,Assistant

class FakeModel:
    def __init__(self,mode):self.mode=mode;self.calls=0
    def generate(self,chat):
        self.calls+=1;r=json.loads(chat[-1]['content'])['katalogpost']
        if self.mode=='wrong_formula':r['components_percent_ww']={'a':'50','b':'50'}
        if self.mode=='hallucinated_description':r['description_sv']='Pungent leather with marine notes.'
        return ('broken JSON' if self.mode=='broken' else json.dumps(r)),{'seconds':0,'generated_tokens':1}

class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.row={'prior_id':'amber','formula_id':'f1','name_sv':'Amber','descriptions_sv':['Varm amber.'],
            'components':[{'ingredient_id':'a','name_sv':'A','weight_percent':'20'},
            {'ingredient_id':'b','name_sv':'B','weight_percent':'80'}]}
        path=Path(self.tmp.name)/'catalog.json';path.write_text(json.dumps([self.row]),encoding='utf-8');self.catalog=Catalog(path)

    def test_exact_scaling_preserves_total(self):
        r=scale({'a':'33.333','b':'66.667'},'0.0001')
        self.assertEqual(sum(map(Decimal,r['material_equivalent_masses_g'].values())),Decimal('.0001'))

    def test_nonfinite_negative_and_wrong_sum_rejected(self):
        for p in [{'a':'20%'},{'a':'not a number'},{'a':'NaN'},{'a':'Infinity'},{'a':'-1','b':'101'},{'a':'99'},{}]:
            with self.assertRaises((ValueError,ArithmeticError)):validate_percent(p)
        for n in ['NaN','Infinity','-1','0']:
            with self.assertRaises(ValueError):scale({'a':'100'},n)

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaises(ValueError):parse_answer('{"name_sv":"A","name_sv":"B"}')

    def test_bad_generation_returns_explicit_fallback(self):
        for mode in ['broken','wrong_formula','hallucinated_description']:
            response=Assistant(FakeModel(mode),self.catalog).respond({'text':'Amber','batch_g':10})
            self.assertEqual(response['generation_source'],'catalog_fallback')
            self.assertEqual(response['answer']['components_percent_ww'],{'a':'20','b':'80'})
            self.assertIsNotNone(response['fallback_reason'])

    def test_valid_model_response_identified(self):
        response=Assistant(FakeModel('valid'),self.catalog).respond({'text':'Amber'})
        self.assertEqual(response['generation_source'],'language_model')

    def test_invalid_batch_is_rejected_before_generation(self):
        lm=FakeModel('valid')
        with self.assertRaises(ValueError):Assistant(lm,self.catalog).respond({'text':'Amber','batch_g':'NaN'})
        self.assertEqual(lm.calls,0)

    def test_new_formula_never_inherits_approval(self):
        lm=FakeModel('valid');a=Assistant(lm,self.catalog)
        response=a.respond({'task':'formula_to_text','components_percent_ww':{'a':'21','b':'79'}})
        self.assertEqual(response['status'],'outside_catalog');self.assertEqual(lm.calls,0)
        self.assertEqual(response['sensory_status'],'not_documented_for_this_formula')

    def test_unknown_material_rejected(self):
        with self.assertRaises(ValueError):Assistant(FakeModel('valid'),self.catalog).respond(
            {'task':'formula_to_text','components_percent_ww':{'unknown':'100'}})

    def test_empty_or_unmatched_text_not_silently_approved(self):
        a=Assistant(FakeModel('valid'),self.catalog)
        with self.assertRaises(ValueError):a.respond({'text':''})
        self.assertEqual(a.respond({'text':'zxqvjk'})['status'],'choose_accord')
        self.assertEqual(a.respond({'text':'En doft av hav och salt'})['status'],'choose_accord')

    def test_registered_variant_is_recognized(self):
        path=Path(self.tmp.name)/'registered-formulas.jsonl'
        r={**self.row,'formula_id':'f2','components':[{'ingredient_id':'a','name_sv':'A','weight_percent':'21'},
            {'ingredient_id':'b','name_sv':'B','weight_percent':'79'}]}
        path.write_text(json.dumps(r)+'\n',encoding='utf-8');cat=Catalog(path.with_name('catalog.json'))
        response=Assistant(FakeModel('valid'),cat).respond({'task':'formula_to_text','components_percent_ww':{'a':21,'b':79}})
        self.assertEqual(response['formula_id'],'f2');self.assertEqual(response['generation_source'],'language_model')

if __name__=='__main__':unittest.main()
