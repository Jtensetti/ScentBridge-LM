"""Import the supplied quantitative archive with an explicit user attestation.

Original archive files remain byte-identical. No bundled scripts are executed.
Use --confirmed-all only after explicit confirmation covering all formulas/texts.
"""
import argparse
import collections
import hashlib
import json
from decimal import Decimal as D
from pathlib import Path
from zipfile import ZipFile

def digest(b):
    return hashlib.sha256(b).hexdigest()

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def composition_key(p):
    return tuple(sorted((k, D(v)) for k, v in p.items()))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--archive', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--confirmed-all', action='store_true')
    args=ap.parse_args()
    if not args.confirmed_all:
        ap.error('Explicit confirmation of all formulas AND descriptions is required.')
    if digest(args.archive.read_bytes())!='828c7be3283e63bf9715be2c74c54761f49a6ab161874f2b2b89a7c180dd1a36':
        ap.error('This dated attestation applies only to the audited archive; supply a new importer/attestation for a different source.')
    out=args.out.resolve(); source=out/'originals'; source.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.archive) as z:
        for e in z.infolist():
            target=(source/e.filename).resolve()
            if not target.is_relative_to(source) or e.file_size>500_000_000:
                raise ValueError('Unsafe archive entry: '+e.filename)
            if not e.is_dir():
                target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(z.read(e))
    root=source/'scentbridge_synthetic_v0_2'
    verified=[]
    for line in (root/'SHA256SUMS.txt').read_text().splitlines():
        sha, name=line.split(None, 1); name=name.lstrip('*')
        target=(root/name).resolve()
        if not target.is_relative_to(root) or digest(target.read_bytes())!=sha:
            raise ValueError('Checksum mismatch: '+name)
        verified.append(name)
    attestation={'id':'user-confirmation-2026-10-13-all', 'date':'2026-10-13',
      'statement_sv':'Allt', 'question_scope':'All 4992 formula variants, their proportions and associated odor descriptions',
      'status':'user_reported_smell_tested_and_approved', 'independently_verified':False,
      'source_archive_sha256':digest(args.archive.read_bytes()),
      'protocol':None, 'test_dates':None, 'panel_size':None, 'numeric_sensory_ratings':None,
      'scope_exclusions':['novel compositions produced by arithmetic tasks','human preference rankings','commercial rights clearance'],
      'supersedes':'sensory untested flags for these formulas and descriptions only'}
    dump(out/'attestation.json', attestation)
    forms=[json.loads(l) for l in (root/'data/formulas.jsonl').read_text(encoding='utf-8').splitlines()]
    briefs=json.loads((root/'config/target_briefs_sv.json').read_text(encoding='utf-8'))
    # Remove obsolete trial-status clauses, retaining odor wording and original texts.
    edits={
      '; hur väl mängderna fångar den behöver undersökas.':'.',
      ' Vilka delar av jasminens skiftande uttryck som fångas måste luktprovas.':'',
      '; exakt dofteffekt och dos är inte mätta.':'.'}
    cleaned={k:[replace_all(s,edits) for s in texts] for k,texts in briefs.items()}
    dump(out/'derived/target_descriptions.json',cleaned)
    ids={}; compositions={}; families=collections.defaultdict(set); fps=set()
    form_counts=collections.Counter(); prior_counts=collections.defaultdict(set)
    with (out/'derived/formulas.jsonl').open('w',encoding='utf-8') as dest:
        for f in forms:
            p={x['ingredient_id']:x['weight_percent'] for x in f['components']}
            assert sum(map(D,p.values()))==100 and all(D(v)>=0 for v in p.values())
            assert f['formula_id'] not in ids and f['formula_fingerprint'] not in fps
            for c in f['components']:
                assert D(c['material_equivalent_mass_g'])==D(c['weight_percent'])*D(f['reference_batch_g'])/100
            ids[f['formula_id']]=f; fps.add(f['formula_fingerprint'])
            compositions[composition_key(p)]=f['formula_id']
            families[f['accord_family_id']].add(f['split']);form_counts[f['split']]+=1
            prior_counts[f['split']].add(f['prior_id'])
            f['original_evidence']={k:f[k] for k in ['sensory_validated','proportion_evidence','target_profile_status','observed_sensory_profile']}
            f.update(sensory_validated=True,sensory_evidence='user_attestation',attestation_id=attestation['id'],
                proportion_evidence='rule_based_origin_subsequently_user_reported_tested',
                target_profile_status='user_approved_descriptions',
                approved_descriptions_sv=cleaned[f['prior_id']])
            # Missing quantitative measurements are not fabricated from text approval.
            dest.write(json.dumps(f,ensure_ascii=False)+'\n')
    assert all(len(v)==1 for v in families.values())
    assert len(compositions)==len(forms)
    selected=json.loads((root/'examples/start_train_2000.selection.json').read_text())['record_ids']
    selected_set=set(selected); selected_rows={}; sft_counts=collections.Counter(); task_counts=collections.Counter()
    system=('Du arbetar med doftackord och blandningsmatematik. Svara med JSON. Procent avser massa/massa. '
      'Använd angiven evidens: användaren har bekräftat luktprov och godkänt de registrerade formlerna och beskrivningarna. '
      'Godkännandet gäller inte automatiskt nya proportioner. Numeriska doftbetyg saknas. '
      'Skilj råvaruekvivalent från stamlösning och slutblandning. För naturmaterial avses hela extraktet. '
      'Härled inte säkerhet eller löslighet från luktgodkännande eller matematik.')
    for split in ['train','validation','test']:
        destpath=out/'derived/sft'/f'{split}.jsonl';destpath.parent.mkdir(parents=True,exist_ok=True)
        with destpath.open('w',encoding='utf-8') as dest:
            for line in (root/'data/sft'/f'{split}.jsonl').open(encoding='utf-8'):
                r=json.loads(line); f=ids[r['formula_id']];task=r['task']
                assert r['split']==f['split']==split and r['accord_family_id']==f['accord_family_id']
                if r['record_id'] in selected_set: selected_rows[r['record_id']]=json.loads(json.dumps({'messages':r['messages']}))
                r['original_record_id']=r['record_id'];r['record_id']+='__attested_v1'
                r['attestation_id']=attestation['id'];r['formula_sensory_evidence']='user_attestation'
                r['synthetic']=True # The TASKS remain programmatically authored.
                r['sensory_observation']=False # No new trial record is invented.
                r['messages'][0]['content']=system
                q=r['problem'];a=json.loads(r['messages'][2]['content'])
                q['registered_formula_id']=f['formula_id']
                q['formula_sensory_evidence']='user_reported_smell_tested_and_approved'
                prefix=r['messages'][1]['content'].split('\n\nUnderlag:')[0]
                if task=='propose_with_fixed_share':
                    prefix='Återge det godkända katalogackordets proportioner inom paletten och procentvillkoret.'
                    q['description_status']='user_approved_description'
                    q['target_description_sv']=replace_all(q['target_description_sv'],edits)
                    a.update(proportion_status='user_reported_tested_and_approved',sensory_validated=True,sensory_evidence='user_attestation')
                    r['evidence_layer']='user_attested_formula';r['suggested_sampling_weight']=1.0
                elif task=='validation_status':
                    q['quantity_source']='generated_then_user_reported_tested'
                    q['user_attestation']={'status':attestation['status'],'id':attestation['id']}
                    q['approved_descriptions_sv']=cleaned[f['prior_id']]
                    a.update(sensory_validated=True,sensory_evidence='user_attestation',
                        approved_descriptions_sv=cleaned[f['prior_id']])
                elif task=='source_accord_retrieval':
                    prefix='Koppla ingredienslistan till källackordet. Redovisa användarens godkännande separat från numeriska doftbetyg.'
                    a['registered_formula_sensory_evidence']='user_reported_tested_and_approved'
                elif task=='change_component':
                    matched=compositions.get(composition_key(a['components_percent_ww']))
                    a['sensory_status']='user_reported_tested_and_approved' if matched else 'not_documented_for_modified_formula'
                    a['matched_registered_formula_id']=matched
                r['messages'][1]['content']=prefix+'\n\nUnderlag:\n'+json.dumps(q,ensure_ascii=False,sort_keys=True)
                r['messages'][2]['content']=json.dumps(a,ensure_ascii=False,sort_keys=True)
                dest.write(json.dumps(r,ensure_ascii=False)+'\n');sft_counts[split]+=1;task_counts[task]+=1
    original_starter=[json.loads(l) for l in (root/'examples/start_train_2000.jsonl').read_text(encoding='utf-8').splitlines()]
    assert original_starter==[selected_rows[i] for i in selected]
    with (out/'derived/start_train_2000.jsonl').open('w',encoding='utf-8') as dest:
        for line in (out/'derived/sft/train.jsonl').open(encoding='utf-8'):
            if json.loads(line)['original_record_id'] in selected_set:dest.write(line)
    report={'passed':True,'archive_sha256':attestation['source_archive_sha256'], 'checksums_verified':len(verified),
      'formulas':len(forms),'formula_splits':dict(form_counts),'priors':sum(map(len,prior_counts.values())),
      'prior_splits':{k:len(v) for k,v in prior_counts.items()},'families':len(families),
      'cross_split_family_overlap':0,'duplicate_compositions':0,'sft_records':sum(sft_counts.values()),
      'sft_splits':dict(sft_counts),'tasks':dict(task_counts),'starter_records':len(selected),
      'starter_lineage_verified':True,'user_attested_formulas':len(forms),
      'unique_description_texts':len(set(s for a in cleaned.values() for s in a)),
      'numeric_sensory_ratings_available':False,'preference_pairs_excluded':9984,
      'preference_exclusion_reason':'Generated arithmetic/evidence checks are not human sensory preferences; stale evidence labels retained only in originals.'}
    dump(out/'intake-report.json',report);print(json.dumps(report,indent=2))

def replace_all(s, edits):
    for a,b in edits.items():s=s.replace(a,b)
    return s

if __name__=='__main__':main()
