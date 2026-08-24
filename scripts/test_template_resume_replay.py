#!/usr/bin/env python3
import copy
import tempfile
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binderloop.resume import build_template_execution_identity, classify_template_replay


def fixture(root: Path):
    source=root/'source.cif'; target=root/'target.cif'
    source.write_text('source-v1\n'); target.write_text('target-v1\n')
    params={
      'harness_template_policy': {'enabled':True,'round_conditioned_fraction':.5},
      'binder_template': {'template_id':'t1','source_structure_file':str(source),
        'source_digest':'ignored-when-file-present',
        'target_alignment':{'status':'aligned','digest':'a','mapping_coverage':1.0,'target_patch_rmsd':0.0},
        'source_to_effective_residue_map':{'A:1':'A:1'},
        'length_transform':{'status':'identity','digest':'l','effective_length':1}},
      'template_application_plan': {'schema_version':1,'template_id':'t1','source_digest':'s',
        'source_structure':str(source),'current_target_identity':{'structure':str(target),'chain':'E'},
        'alignment':{'status':'aligned','digest':'a'},'source_to_effective_residue_map':{'A:1':'A:1'},
        'length_transform':{'status':'identity','digest':'l'},'allocated_num_designs':2},
      'matched_group_id':'g','matched_comparison':{'effective_length':1},
      'host':'host-a','devices':1,'gpu':0,'shard_index':0,
    }
    return source,target,params


def main():
  with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp); source,target,params=fixture(root)
    old=build_template_execution_identity(params,target_structure=target,target_chain='E',output_dir='/old',lineage_schema_version=2,lineage_manifest_digest='manifest')
    same=build_template_execution_identity(params,target_structure=target,target_chain='E',output_dir='/old',lineage_schema_version=2,lineage_manifest_digest='manifest')
    assert classify_template_replay(old,same)['status']=='exact_replay'
    expected_components={'policy','target','source','alignment','residue_map','length_transform','template_application_plan','matched_group'}
    assert set(old['semantic']['component_digests'])==expected_components
    moved=root/'moved'; moved.mkdir(); moved_source=moved/'renamed.cif'; moved_target=moved/'renamed_target.cif'
    moved_source.write_bytes(source.read_bytes()); moved_target.write_bytes(target.read_bytes())
    operational=copy.deepcopy(params); operational['host']='host-b'; operational['devices']=8; operational['gpu']=7; operational['shard_index']=12
    operational['binder_template']['source_structure_file']=str(moved_source); operational['template_application_plan']['source_structure']=str(moved_source); operational['template_application_plan']['current_target_identity']['structure']=str(moved_target)
    changed_path=build_template_execution_identity(operational,target_structure=moved_target,target_chain='E',output_dir='/new',lineage_schema_version=2,lineage_manifest_digest='manifest')
    verdict=classify_template_replay(old,changed_path); assert verdict['status']=='exact_replay' and verdict['operational_changed']
    source.write_text('source-v2\n')
    changed_content=build_template_execution_identity(params,target_structure=target,target_chain='E',lineage_schema_version=2,lineage_manifest_digest='manifest')
    assert classify_template_replay(old,changed_content)['status']=='reject_replay'
    source.write_text('source-v1\n')
    for key,value in [('target_alignment',{'status':'aligned','digest':'different'}),('source_to_effective_residue_map',{'A:1':'A:2'}),('length_transform',{'status':'applied','digest':'different'})]:
      changed=copy.deepcopy(params); changed['binder_template'][key]=value
      now=build_template_execution_identity(changed,target_structure=target,target_chain='E',lineage_schema_version=2,lineage_manifest_digest='manifest')
      verdict=classify_template_replay(old,now)
      assert verdict['status']=='reject_replay', key
      component={'target_alignment':'alignment','source_to_effective_residue_map':'residue_map','length_transform':'length_transform'}[key]
      assert old['semantic']['component_digests'][component] != now['semantic']['component_digests'][component]
    changed=copy.deepcopy(params); changed['harness_template_policy']['round_conditioned_fraction']=.25
    assert classify_template_replay(old,build_template_execution_identity(changed,target_structure=target,target_chain='E',lineage_schema_version=2,lineage_manifest_digest='manifest'))['status']=='reject_replay'
    changed=copy.deepcopy(params); changed['template_application_plan']['allocated_num_designs']=3
    now=build_template_execution_identity(changed,target_structure=target,target_chain='E',lineage_schema_version=2,lineage_manifest_digest='manifest')
    assert classify_template_replay(old,now)['status']=='reject_replay'
    assert old['semantic']['component_digests']['template_application_plan'] != now['semantic']['component_digests']['template_application_plan']
    changed=copy.deepcopy(params); changed['matched_group_id']='other'
    now=build_template_execution_identity(changed,target_structure=target,target_chain='E',lineage_schema_version=2,lineage_manifest_digest='manifest')
    assert classify_template_replay(old,now)['status']=='reject_replay'
    assert old['semantic']['component_digests']['matched_group'] != now['semantic']['component_digests']['matched_group']
    changed_lineage=build_template_execution_identity(params,target_structure=target,target_chain='E',lineage_schema_version=2,lineage_manifest_digest='other-manifest')
    verdict=classify_template_replay(old,changed_lineage)
    assert verdict['status']=='exact_replay' and not verdict['exact_attribution']
    missing=copy.deepcopy(params); missing_source=root/'missing-source.cif'
    missing['binder_template']['source_structure_file']=str(missing_source); missing['binder_template']['source_digest']=old['semantic']['source']['content']['sha256']
    missing['template_application_plan']['source_structure']=str(missing_source)
    remat=build_template_execution_identity(missing,target_structure=target,target_chain='E',lineage_schema_version=2,lineage_manifest_digest='manifest')
    verdict=classify_template_replay(old,remat); assert verdict['status']=='rematerialize_replay' and not verdict['exact_attribution']
    for schema,digest in [(1,'legacy'),('v22','legacy'),(22,'legacy'),(None,'')]:
      historical=build_template_execution_identity(params,target_structure=target,target_chain='E',lineage_schema_version=schema,lineage_manifest_digest=digest)
      verdict=classify_template_replay(historical,old); assert verdict['status']=='exact_replay' and verdict['audit_ingestion_allowed'] and not verdict['exact_attribution']
    needs_materialization=build_template_execution_identity(params,target_structure=target,target_chain='E',lineage_schema_version=2,lineage_manifest_digest='')
    verdict=classify_template_replay(old,needs_materialization); assert verdict['status']=='exact_replay' and not verdict['exact_attribution']
  print('template resume replay tests passed')
if __name__=='__main__': main()


def test_template_digest_pairs_are_not_crossed():
    source = (Path(__file__).resolve().parents[1] / "binderloop" / "agents" / "design_spec_agent.py").read_text(encoding="utf-8")
    assert '("coherent_aligned", template.get("coherent_frame_source_structure_file") or template.get("source_structure_file"), template.get("source_digest"))' in source
    assert '("staged_unaligned", template.get("staged_source_structure_file") or template.get("unaligned_source_structure_file"), template.get("unaligned_source_digest"))' in source
    assert 'template.get("staged_source_structure_file") or template.get("source_structure_file")' not in source
