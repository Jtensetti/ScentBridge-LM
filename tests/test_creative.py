import json,sys,tempfile,unittest
from pathlib import Path
from decimal import Decimal
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from scentbridge.creative import RecipeDesigner,explicit_facets,stock_plan,rounded_composition,MODULES
from scentbridge.runtime import Catalog
from scentbridge.contracts import validate_percent

class Model:
    def __init__(self,weights=None,avoid=None,unsupported=None,broken=False):
        self.plan={'weights':weights or {},'avoid':avoid or [],'unsupported':unsupported or []};self.broken=broken;self.calls=0
    def generate_structured(self,chat,schema,tokens):
        self.calls+=1
        # No formula, composition or target answer may be given to the encoder.
        assert 'components_percent_ww' not in json.dumps(chat)
        return ('broken' if self.broken else json.dumps(self.plan)),{'seconds':0}

class CreativeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        r={'prior_id':'amber','formula_id':'source1','name_sv':'Amber','descriptions_sv':['Varm amber'],
           'components':[{'ingredient_id':'vanillin','name_sv':'Vanillin','weight_percent':20},
                         {'ingredient_id':'labdanum_absolute','name_sv':'Labdanum','weight_percent':80}]}
        p=Path(self.tmp.name)/'catalog.json';p.write_text(json.dumps([r]),encoding='utf-8');self.cat=Catalog(p)
    def design(self,text,model=None,**kw):return RecipeDesigner(model or Model(),self.cat).respond({'text':text,**kw})
    def test_fire_and_ash_creates_novel_formula_without_reference_answer(self):
        r=self.design('jag vill att det ska lukta aska och eld')
        self.assertEqual(r['status'],'ok');validate_percent(r['answer']['components_percent_ww'])
        self.assertIn('guaiacol',r['answer']['components_percent_ww'])
        self.assertEqual(r['answer']['sensory_status'],'generated_not_smell_tested')
        self.assertNotIn('formula_id',r);self.assertTrue(r['new_materials'])
    def test_opposite_briefs_change_formula(self):
        a=self.design('rök och trä')['answer']['components_percent_ww']
        b=self.design('vanilj och blommor')['answer']['components_percent_ww']
        self.assertNotEqual(a,b);self.assertNotIn('guaiacol',b)
    def test_negation_beats_model_and_removes_indirect_materials(self):
        r=self.design('Trä utan rök',Model({'wood':3,'smoke':5}))
        self.assertNotIn('guaiacol',r['answer']['components_percent_ww'])
        self.assertNotIn('guaiacwood_oil',r['answer']['components_percent_ww'])
    def test_list_negation_and_contrast(self):
        w,n=explicit_facets('Utan rök och vanilj men med ros och jord',self.cat)
        self.assertEqual(n,{'smoke','vanilla'});self.assertEqual(set(w),{'rose','earth'})
    def test_perfume_scaffold_respects_flower_and_musk_exclusion(self):
        r=self.design('Trä utan blommor och mysk',recipe_type='perfume')
        self.assertNotIn('galaxolide',r['answer']['components_percent_ww'])
        self.assertNotIn('hedione',r['answer']['components_percent_ww'])
    def test_parent_approval_does_not_transfer(self):
        r=self.design('amber och citrus')
        self.assertEqual(r['answer']['sensory_status'],'generated_not_smell_tested')
        self.assertTrue(any(x.get('parent_formula_id')=='source1' for x in r['lineage']))
    def test_mass_and_active_mass_conserved_with_stock_override(self):
        p={'a':'99.9','guaiacol':'.1'};r=stock_plan(p,'20',{'a':'50','guaiacol':'.5'})
        m={k:Decimal(v) for k,v in r['stock_masses_g'].items()}
        self.assertEqual(sum(m.values())+Decimal(r['additional_carrier_g']),Decimal(20))
        active=sum(m[k]*Decimal(r['stock_strengths_percent_ww'][k])/100 for k in m)
        self.assertEqual(active,Decimal('.2'))
    def test_invalid_dilutions_and_impossible_carrier(self):
        for n in ['NaN','0','-1','101','Infinity','abc','.001']:
            with self.assertRaises(ValueError):stock_plan({'a':'100'},'20',{'a':n})
    def test_refine_changes_quantities_preserves_zero_slider(self):
        a=self.design('aska och rök');b=self.design('aska och rök',facet_weights={'ash':3,'smoke':0})
        self.assertNotEqual(a['recipe_id'],b['recipe_id']);self.assertEqual(b['intent']['weights']['smoke'],0)
        self.assertNotIn('guaiacol',b['answer']['components_percent_ww'])
    def test_unknown_and_partial_unknown_not_silently_dropped(self):
        self.assertEqual(self.design('rök och bensin',Model({'smoke':3},unsupported=['bensin']))['status'],'needs_detail')
        self.assertEqual(self.design('zxqxz',Model(broken=True))['status'],'needs_detail')
        self.assertEqual(self.design('kaffe',Model({'vanilla':3}))['status'],'needs_detail')
        self.assertEqual(self.design('rök och bensin',Model({'smoke':3}))['status'],'needs_detail')
    def test_unrequested_model_facets_never_enter_formula(self):
        r=self.design('rök och aska',Model({'smoke':3,'ash':3,'vanilla':5}))
        self.assertEqual(set(r['intent']['weights']),{'smoke','ash'})
        self.assertEqual(r['generation_source'],'explicit_terms_corrected')
        self.assertNotIn('vanillin',r['answer']['components_percent_ww'])
    def test_model_unknown_flags_cannot_override_explicit_supported_words(self):
        r=self.design('vanilj och rök',Model({'vanilla':3},unsupported=['smoke']))
        self.assertEqual(r['status'],'ok');self.assertIn('smoke',r['intent']['weights'])
    def test_failure_is_visible_and_bad_batch_never_calls_model(self):
        self.assertEqual(self.design('trä',Model(broken=True))['generation_source'],'explicit_terms_fallback')
        m=Model()
        with self.assertRaises(ValueError):self.design('trä',m,batch_g='NaN')
        self.assertEqual(m.calls,0)
    def test_rounding_and_module_integrity(self):
        for p in MODULES.values():validate_percent(p)
        p=rounded_composition({'a':Decimal(1),'b':Decimal(1),'c':Decimal(1)})
        self.assertEqual(sum(map(Decimal,p.values())),Decimal(100))

if __name__=='__main__':unittest.main()
