#!/usr/bin/env python3
import hashlib
import json
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.analysis.motif_attribution import attribute_candidate_lineages, compare_structures
from binderloop.analysis.post_ingest_parity import validate_post_ingest_parity
from binderloop.lineage import make_backbone_id, make_candidate_id, make_run_id, make_stage_record, write_jsonl
from binderloop.templates.residue_identity import parse_residue_identity

AA={"A":"ALA","G":"GLY","S":"SER","V":"VAL"}
def pdb(path, motif_shift=(0,0,0), patch_shift=(0,0,0), deform=0, sequence="AAA", contact=True, insertion=False):
    lines=[]; serial=1
    for i,aa in enumerate(sequence,1):
        x=float(i*3); y=float(deform if i==2 else 0); z=0
        if not contact: y += 20
        icode="A" if insertion and i==2 else " "
        lines.append(f"ATOM  {serial:5d}  CA  {AA[aa]:3s} A{i:4d}{icode}   {x+motif_shift[0]:8.3f}{y+motif_shift[1]:8.3f}{z+motif_shift[2]:8.3f}  1.00  0.00           C")
        serial+=1
    for i in range(1,4):
        x=float(i*3); y=4; z=0
        lines.append(f"ATOM  {serial:5d}  CA  GLY B{i:4d}    {x+patch_shift[0]:8.3f}{y+patch_shift[1]:8.3f}{z+patch_shift[2]:8.3f}  1.00  0.00           C")
        serial+=1
    path.write_text("\n".join(lines+["END",""]))

def value(metrics,key): return metrics[key]["value"]

def main():
  with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp); base=root/'base.pdb'; deform=root/'deform.pdb'; drift=root/'drift.pdb'; mutant=root/'mutant.pdb'; lost=root/'lost.pdb'; ins=root/'ins.pdb'
    pdb(base); pdb(deform,deform=2); pdb(drift,motif_shift=(8,0,0)); pdb(mutant,sequence='AGA'); pdb(lost,contact=False); pdb(ins,insertion=True)
    motif=[parse_residue_identity(f'A:{i}') for i in range(1,4)]; patch=[parse_residue_identity(f'B:{i}') for i in range(1,4)]
    d=compare_structures(base,deform,left_motif=motif,right_motif=motif,left_patch=patch,right_patch=patch)
    assert value(d,'motif_self_aligned_rmsd')>.5
    pose=compare_structures(base,drift,left_motif=motif,right_motif=motif,left_patch=patch,right_patch=patch)
    assert value(pose,'motif_self_aligned_rmsd')<1e-5 and value(pose,'target_patch_aligned_motif_rmsd')>5
    mutation=compare_structures(base,mutant,left_motif=motif,right_motif=motif,left_patch=patch,right_patch=patch)
    assert abs(value(mutation,'mapped_sequence_identity')-2/3)<1e-6
    contacts=compare_structures(base,lost,left_motif=motif,right_motif=motif,left_patch=patch,right_patch=patch)
    assert value(contacts,'contact_retention')==0
    insertion=[parse_residue_identity('A:1'),parse_residue_identity('A:2A'),parse_residue_identity('A:3')]
    assert value(compare_structures(ins,ins,left_motif=insertion,right_motif=insertion,left_patch=patch,right_patch=patch),'motif_mapping_coverage')==1

    ctx={'run_namespace':'n','branch':'b','run_id':make_run_id('n','b'),'template_id':'t'}; bb=make_backbone_id('n','b',0); cid=make_candidate_id('n','b',0,bb)
    initial=root/'initial.pdb'; pdb(initial)
    row=make_stage_record(context=ctx,stage='initial_design',logical_ordinal=0,backbone_id=bb,candidate_id=cid,parent_candidate_id=None,artifacts={'structure':{'path':'initial.pdb','sha256':hashlib.sha256(initial.read_bytes()).hexdigest()}})
    template={'template_id':'t','source_structure_file':str(base),'binder_chain':'A','binder_residue_ids':['A:1','A:2','A:3'],'source_to_effective_residue_map':{'A:1':'A:1','A:2':'A:2','A:3':'A:3'},'target_contact_residues':['B:1','B:2','B:3'],'target_alignment':{'source_target_chain':'B','residue_map':{'B:1':'B:1','B:2':'B:2','B:3':'B:3'}}}
    doc=attribute_candidate_lineages(root,[row],template)
    statuses={(x['from_stage'],x['to_stage']):x['status'] for x in doc['comparisons']}
    assert statuses[('source','initial_design')]=='evaluated' and statuses[('source','final_refold')]=='not_available'
    row_seq=make_stage_record(context={**ctx,'root_candidate_id':cid},stage='inverse_folded',logical_ordinal=0,backbone_id=bb,candidate_id=cid,parent_candidate_id=cid,artifacts={'sequence':'seq.fa'},structural=False)
    # This intentionally bypasses full lineage validation to verify public missing path separately.
    assert attribute_candidate_lineages(root,[],template)['status']=='lineage_unavailable'
    bad=attribute_candidate_lineages(root,[row],template,parity_error='artifact_digest_mismatch')
    assert all(x['status']=='not_evaluable' for x in bad['comparisons'])

    manifest_path=write_jsonl(root/'candidate_manifest.jsonl',[row])
    manifest={'schema_version':2,'run':{'run_id':ctx['run_id'],'run_namespace':'n','branch':'b','round_id':'1','job_id':'j','template_id':''},'files':['candidate_manifest.jsonl','initial.pdb'],'candidate_manifests':['candidate_manifest.jsonl'],'digests':{},'lineage_summary':{'record_count':1}}
    parity=validate_post_ingest_parity(root,manifest,[row]); assert parity.evaluable, parity.failures
    initial.write_text(initial.read_text()+'REMARK mutation\n')
    parity=validate_post_ingest_parity(root,manifest,[row]); assert not parity.evaluable and any('artifact_digest_mismatch' in x for x in parity.failures)
  print('template attribution tests passed')
if __name__=='__main__': main()
