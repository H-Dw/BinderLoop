#!/usr/bin/env python3
"""SC2RBD contract smoke: deterministic local artifacts, no GPU/LLM/Taiji."""
import copy, csv, hashlib, json, tempfile, sys
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from binderloop.analysis.structure_features import parse_structure
from binderloop.analysis.template_alignment import align_target_patch
from binderloop.analysis.motif_attribution import attribute_candidate_lineages
from binderloop.agents.design_spec_agent import DesignSpecAgent
from binderloop.agents.result_ingestion_agent import ResultIngestionAgent
from binderloop.execution_governance import build_template_application_plan, resolve_round_budget, stable_digest
from binderloop.lineage import make_backbone_id,make_candidate_id,make_inverse_fold_sequence_id,make_run_id,make_stage_record,write_jsonl
from binderloop.models.base import DesignJob
from binderloop.resume import build_template_execution_identity,classify_template_replay,file_sha256
from binderloop.templates.length_mapping import plan_length_transform
from binderloop.templates.outcome_ledger import OutcomeLedger,assert_matched_pair,matched_comparison_signature,matched_group_id
ROOT=Path(__file__).resolve().parents[1]; SC=ROOT/'examples/bg_example/SC2RBD.cif'

def write_contract_fixture(path, target_rows, binder_shift=0.0):
 lines=[]; serial=1
 for i in range(1,13):
  x=float(i); y=3.0+(1.0 if i==6 else 0.0)+binder_shift
  lines.append(f"ATOM  {serial:5d}  CA  ALA A{i:4d}    {x:8.3f}{y:8.3f}{0.:8.3f}  1.00  0.00           C"); serial+=1
 for number,icode,name,(x,y,z) in target_rows:
  lines.append(f"ATOM  {serial:5d}  CA  {name:3s} E{number:4d}{icode:1s}   {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"); serial+=1
 path.write_text('\n'.join(lines+['END','']))

def main():
 assert SC.is_file(); target_digest=file_sha256(SC)
 atoms=[a for a in parse_structure(SC) if a.chain=='E' and a.name=='CA'][:4]
 target_rows=[(a.resseq,a.icode,a.resname,a.coord) for a in atoms]
 with tempfile.TemporaryDirectory() as tmp:
  root=Path(tmp); source=root/'synthetic_source_from_sc2rbd_patch.pdb'; current=root/'synthetic_current_sc2rbd_patch.pdb'
  write_contract_fixture(source,target_rows); write_contract_fixture(current,target_rows)
  patch=[f'E:{a.resseq}{a.icode}' for a in atoms]
  alignment=align_target_patch(str(source),str(current),source_target_chain='E',current_target_chain='E',residue_ids=patch)
  assert alignment.status=='aligned' and alignment.target_patch_rmsd<1e-6
  transform=plan_length_transform([f'A:{i}' for i in range(1,13)],14,motif_residues=['A:5','A:6','A:7'],contact_residues=['A:5'],insertion_preference=('c_terminal',),min_designable_residues=8)
  assert transform.status=='applied' and transform.chain_continuous and len(transform.source_to_effective_residue_map)==12
  template={'template_id':'sc2rbd_contract_fixture','source_structure_file':str(source),'source_digest':file_sha256(source),'binder_chain':'A','binder_residue_ids':['A:5','A:6','A:7'],'fixed_res_index':'5..7','within_proximity':7.5,'mode':'structure_redesign','target_contact_residues':patch,'fixed_res_index':'5..7','within_proximity':7.5,'quality_score':.9,'target_alignment':alignment.to_dict(),'source_to_effective_residue_map':dict(transform.source_to_effective_residue_map),'length_transform':transform.to_dict(),'source_target_identity':{'fixture':'derived_from_SC2RBD','sc2rbd_sha256':target_digest}}
  plan=build_template_application_plan(template,current_target=str(SC),current_target_chain='E',round_fraction=.5,allocated_num_designs=2)
  assert plan.applicability['applicable'] and plan.current_target_identity['digest']==target_digest
  budget=resolve_round_budget(4,[{'id':'template','bucket':'template_conditioned'},{'id':'control','bucket':'template_free'}],requested_conditioned_fraction=.5)
  assert sum(x['num_designs'] for x in budget.allocations)==4 and budget.bucket_allocations=={'template_conditioned':2,'template_free':2,'other':0}
  base={'protocol':'protein-anything','num_designs':2,'max_binders_per_round':2,'budget':2,'devices':1,'binder_lengths':[14],'inverse_fold_num_sequences':1,'run_filtering':True,'template_application_plan':plan.to_dict(),'target_identity_digest':target_digest,'lineage_schema_version':2}
  signature=matched_comparison_signature(base,target_structure=str(SC),chain_id='E',binder_length=14); group=matched_group_id(target_digest,template['template_id'],0,signature)
  tp={**base,'template_conditioned':True,'binder_template':template,'matched_group_id':group,'matched_comparison':signature}; cp={**base,'template_conditioned':False,'matched_group_id':group,'matched_comparison':signature}
  assert_matched_pair(tp,cp)
  job=DesignJob('sc2rbd_contract_template',str(SC),'E',patch,14,params=tp,output_dir=str(root/'dry_run'))
  spec=DesignSpecAgent(ROOT.parent/'boltzgen').create_boltzgen_run_spec(job,params=tp)
  assert Path(spec.design_spec_path).is_file() and Path(spec.expected_outputs['redesign_mask']).is_file() and Path(spec.expected_outputs['effective_execution_plan']).is_file()
  execution=json.loads(Path(spec.expected_outputs['effective_execution_plan']).read_text()); assert execution['template_artifact_digests']['source']==file_sha256(source) and execution['template_artifact_digests']['alignment']==alignment.digest
  out=root/'synthetic_results'; out.mkdir(); structures=[]
  for stage in ('initial','before','final'):
   path=out/f'{stage}.pdb'; write_contract_fixture(path,target_rows); structures.append(path)
  ctx={'run_namespace':'sc2rbd/offline','branch':'template','run_id':make_run_id('sc2rbd/offline','template'),'round_id':'0','job_id':job.job_id,'template_id':template['template_id'],'digests':dict(execution['template_artifact_digests'])}
  bb=make_backbone_id(ctx['run_namespace'],ctx['branch'],0); initial=make_candidate_id(ctx['run_namespace'],ctx['branch'],0,bb); seq=make_inverse_fold_sequence_id(bb,0,'AAAAAAAAAAAAAA'); child=make_candidate_id(ctx['run_namespace'],ctx['branch'],0,bb,seq); childctx={**ctx,'root_candidate_id':initial}
  artifact=lambda p:{'path':p.name,'sha256':file_sha256(p)}
  records=[make_stage_record(context=ctx,stage='initial_design',logical_ordinal=0,backbone_id=bb,candidate_id=initial,parent_candidate_id=None,artifacts={'structure':artifact(structures[0])}),make_stage_record(context=childctx,stage='inverse_folded',logical_ordinal=0,backbone_id=bb,candidate_id=child,parent_candidate_id=initial,inverse_fold_sequence_id=seq,artifacts={'sequence':{'path':'seq.fasta','sha256':hashlib.sha256(b'>x\nAAAAAAAAAAAAAA\n').hexdigest()}},structural=False),make_stage_record(context=childctx,stage='before_refolding',logical_ordinal=0,backbone_id=bb,candidate_id=child,parent_candidate_id=child,inverse_fold_sequence_id=seq,artifacts={'structure':artifact(structures[1])}),make_stage_record(context=childctx,stage='final_refold',logical_ordinal=0,backbone_id=bb,candidate_id=child,parent_candidate_id=child,inverse_fold_sequence_id=seq,artifacts={'structure':artifact(structures[2])})]
  (out/'seq.fasta').write_text('>x\nAAAAAAAAAAAAAA\n'); mp=write_jsonl(out/'candidate_manifest.jsonl',records)
  metrics=out/'final_ranked_designs'/'final_designs_metrics_1.csv'; metrics.parent.mkdir();
  final_record=records[-1]; final_alias=final_record['canonical_alias']; final_digest=final_record['artifacts']['structure']['sha256']
  with metrics.open('w',newline='') as h: w=csv.DictWriter(h,fieldnames=['id','canonical_alias','global_candidate_id','artifact_digest','quality_score','primary_coverage','retention','clash']); w.writeheader(); w.writerow({'id':final_alias,'canonical_alias':final_alias,'global_candidate_id':child,'artifact_digest':final_digest,'quality_score':.8,'primary_coverage':.9,'retention':.9,'clash':.05})
  manifest={'schema_version':3,'run':{k:ctx[k] for k in ('run_id','run_namespace','branch','round_id','job_id','template_id')},'files':[str(mp.relative_to(out)),str(metrics.relative_to(out))]+[p.name for p in structures]+['seq.fasta'],'candidate_manifests':[mp.name],'digests':ctx['digests']}; (out/'result_manifest.json').write_text(json.dumps(manifest))
  ingested=ResultIngestionAgent().ingest_boltzgen_output(out); assert ingested.exact_attribution and len(ingested.lineage_records)==4 and ingested.post_ingest_parity['status']=='validated'
  attr=attribute_candidate_lineages(out,ingested.lineage_records,template); assert attr['status']=='evaluated'; assert any(x['from_stage']=='initial_design' and x['to_stage']=='inverse_folded' and x['status']=='not_available' for x in attr['comparisons']); assert any(x['from_stage']=='source' and x['to_stage']=='final_refold' and x['status']=='evaluated' for x in attr['comparisons'])
  ledger=OutcomeLedger.open(root/'ledger.json'); bad={'quality':.2,'primary_coverage':.2,'retention':.2,'clash':.4}; good={'quality':.8,'primary_coverage':.8,'retention':.8,'clash':.1}
  ledger.record_failure(target_digest,template['template_id'],round_id=0,failure_type='runtime_failure'); assert ledger.eligible(target_digest,template['template_id'],1)
  ledger.record_outcome(target_digest,template['template_id'],round_id=1,template_metrics=bad,control_metrics=good,confidence=1,matched_group_id=group,cooldown_failures=2); ledger.record_outcome(target_digest,template['template_id'],round_id=2,template_metrics=bad,control_metrics=good,confidence=1,matched_group_id=group,cooldown_failures=2); assert not ledger.eligible(target_digest,template['template_id'],2)
  ledger.record_failure(target_digest,'hard',round_id=1,failure_type='digest_mismatch'); assert not ledger.eligible(target_digest,'hard',999)
  identity=build_template_execution_identity(tp,target_structure=SC,target_chain='E',lineage_schema_version=2,lineage_manifest_digest=ingested.lineage_summary['digest']); assert classify_template_replay(identity,identity)['status']=='exact_replay' and classify_template_replay(identity,identity)['exact_attribution']
  for field,value in [('source_digest','changed')]:
   changed=copy.deepcopy(tp); changed['binder_template'][field]=value; changed['binder_template']['source_structure_file']=str(root/'missing-source.cif')
   assert classify_template_replay(identity,build_template_execution_identity(changed,target_structure=SC,target_chain='E',lineage_schema_version=2,lineage_manifest_digest=ingested.lineage_summary['digest']))['status']=='reject_replay'
  changed=copy.deepcopy(tp); changed['binder_template']['target_alignment']={**alignment.to_dict(),'mapping_coverage':.5}
  assert classify_template_replay(identity,build_template_execution_identity(changed,target_structure=SC,target_chain='E',lineage_schema_version=2,lineage_manifest_digest=ingested.lineage_summary['digest']))['status']=='reject_replay'
  changed=copy.deepcopy(tp); changed['binder_template']['source_to_effective_residue_map']['A:5']='A:99'
  assert classify_template_replay(identity,build_template_execution_identity(changed,target_structure=SC,target_chain='E',lineage_schema_version=2,lineage_manifest_digest=ingested.lineage_summary['digest']))['status']=='reject_replay'
  failed=resolve_round_budget(4,[{'id':'bad','bucket':'template_conditioned','valid':False,'rejection_reason':'digest_mismatch'},{'id':'control','bucket':'template_free'}],requested_conditioned_fraction=.5); assert failed.bucket_allocations['template_free']==4 and failed.rematerialization[0]['policy']=='reject_and_rematerialize'
 print('SC2RBD offline template contract smoke passed')
if __name__=='__main__': main()
