"""Residue-aware target patch alignment for executable fragment templates."""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
from typing import Any, Dict, Mapping, Sequence
import numpy as np
from binderloop.analysis.structure_features import AA3_TO_1, parse_structure
from binderloop.execution_governance import stable_digest
from binderloop.templates.residue_identity import ResidueIdentity, canonical_residue_digest, parse_residue_identity

@dataclass(frozen=True)
class TargetPatchAlignment:
    status: str; source_target_chain: str; current_target_chain: str; residue_map: Dict[str,str]
    matched_residue_count: int; requested_residue_count: int; mapping_coverage: float; target_patch_rmsd: float
    rotation: Sequence[Sequence[float]]; translation: Sequence[float]; digest: str; reason: str=""
    mapping_method: str="none"; mapping_input_digest: str=""; source_target_identity_digest: str=""; current_target_identity_digest: str=""; sequence_fallback_status: str="not_attempted"
    def to_dict(self): return asdict(self)

def _residues(atoms, chain):
    seen={}
    for atom in atoms:
        if atom.chain==chain and atom.name=="CA": seen.setdefault(ResidueIdentity(chain,atom.resseq,atom.icode.strip(),name=atom.resname),atom)
    return list(seen.items())

def _file_identity(path, chain, residues):
    file_digest=hashlib.sha256(Path(path).read_bytes()).hexdigest(); return stable_digest({"file":file_digest,"chain":chain,"residues":[r.token for r,_ in residues]})

def align_target_patch(source_structure: str, current_target: str, *, source_target_chain: str, current_target_chain: str, residue_ids: Sequence[Any], min_coverage: float=.75, max_rmsd: float=2.5, allow_sequence_fallback: bool=True, min_sequence_identity: float=1.0) -> TargetPatchAlignment:
    source_atoms=parse_structure(source_structure); current_atoms=parse_structure(current_target)
    source_res=_residues(source_atoms,source_target_chain); current_res=_residues(current_atoms,current_target_chain)
    source_by={r.token:(r,a) for r,a in source_res}; current_by={r.token:(r,a) for r,a in current_res}
    requested=[]
    for value in residue_ids:
        try: requested.append(parse_residue_identity(value,default_chain=source_target_chain))
        except ValueError: pass
    requested=list(dict((r.token,r) for r in requested).values()); input_digest=stable_digest({"source":source_structure,"current":current_target,"source_chain":source_target_chain,"current_chain":current_target_chain,"requested":[r.token for r in requested],"fallback":allow_sequence_fallback,"minimum_identity":min_sequence_identity})
    source_identity=_file_identity(source_structure,source_target_chain,source_res); current_identity=_file_identity(current_target,current_target_chain,current_res)
    pairs=[]
    for residue in requested:
        current_key=ResidueIdentity(current_target_chain,residue.author_residue_number,residue.insertion_code).token
        if residue.token in source_by and current_key in current_by: pairs.append((source_by[residue.token],current_by[current_key]))
    method="exact_author_identity"; fallback_status="not_needed"
    coverage=len(pairs)/max(1,len(requested))
    if (len(pairs)<3 or coverage<min_coverage) and allow_sequence_fallback:
        fallback_status="attempted"; source_patch=[source_by[r.token] for r in requested if r.token in source_by]
        sequence="".join(AA3_TO_1.get(r.name or "","X") for r,_ in source_patch); current_sequence="".join(AA3_TO_1.get(r.name or "","X") for r,_ in current_res)
        starts=[i for i in range(max(0,len(current_sequence)-len(sequence)+1)) if sequence and sum(a==b for a,b in zip(sequence,current_sequence[i:i+len(sequence)]))/len(sequence)>=min_sequence_identity]
        if len(starts)==1 and len(source_patch)==len(requested):
            pairs=list(zip(source_patch,current_res[starts[0]:starts[0]+len(source_patch)])); method="unique_sequence_fallback"; fallback_status="unique_match"; coverage=1.0
        else: fallback_status="ambiguous" if len(starts)>1 else "no_match"
    if len(pairs)<3 or coverage<min_coverage:
        return _failed(source_target_chain,current_target_chain,requested,pairs,coverage,"insufficient_target_patch_mapping",method,input_digest,source_identity,current_identity,fallback_status)
    source_xyz=np.asarray([a.coord for (_,a),_ in pairs]); current_xyz=np.asarray([a.coord for _,(_,a) in pairs]); rotation,translation,rmsd=_rigid_transform(source_xyz,current_xyz)
    if rmsd>max_rmsd: return _failed(source_target_chain,current_target_chain,requested,pairs,coverage,f"target_patch_rmsd_exceeds:{rmsd:.4f}",method,input_digest,source_identity,current_identity,fallback_status,rmsd)
    residue_map={s.token:c.token for (s,_),(c,_) in pairs}; body=dict(status="aligned",source_target_chain=source_target_chain,current_target_chain=current_target_chain,residue_map=residue_map,matched_residue_count=len(pairs),requested_residue_count=len(requested),mapping_coverage=round(coverage,6),target_patch_rmsd=round(rmsd,6),rotation=rotation.tolist(),translation=translation.tolist(),reason="",mapping_method=method,mapping_input_digest=input_digest,source_target_identity_digest=source_identity,current_target_identity_digest=current_identity,sequence_fallback_status=fallback_status)
    return TargetPatchAlignment(**body,digest=stable_digest(body))

def transform_structure_coordinates(coords,alignment):
    xyz=np.asarray(list(coords),dtype=float); return (xyz@np.asarray(alignment["rotation"],dtype=float)+np.asarray(alignment["translation"],dtype=float)).tolist()
def _rigid_transform(source,target):
    sc=source.mean(0); tc=target.mean(0); u,_,vt=np.linalg.svd((source-sc).T@(target-tc)); rotation=u@vt
    if np.linalg.det(rotation)<0: u[:,-1]*=-1; rotation=u@vt
    translation=tc-sc@rotation; aligned=source@rotation+translation; return rotation,translation,float(np.sqrt(np.mean(np.sum((aligned-target)**2,axis=1))))
def _failed(sc,cc,requested,pairs,coverage,reason,method,input_digest,source_identity,current_identity,fallback_status,rmsd=float("inf")):
    residue_map={s.token:c.token for (s,_),(c,_) in pairs}; body=dict(status="rejected",source_target_chain=sc,current_target_chain=cc,residue_map=residue_map,matched_residue_count=len(pairs),requested_residue_count=len(requested),mapping_coverage=round(coverage,6),target_patch_rmsd=rmsd,rotation=[],translation=[],reason=reason,mapping_method=method,mapping_input_digest=input_digest,source_target_identity_digest=source_identity,current_target_identity_digest=current_identity,sequence_fallback_status=fallback_status); return TargetPatchAlignment(**body,digest=stable_digest(body))
def write_aligned_binder_template(source_structure,output_path,*,binder_chain,alignment):
    if alignment.get("status")!="aligned": raise ValueError("cannot write template from rejected alignment")
    atoms=[a for a in parse_structure(source_structure) if a.chain==binder_chain]; coords=transform_structure_coordinates([a.coord for a in atoms],alignment); lines=[]
    for serial,(atom,(x,y,z)) in enumerate(zip(atoms,coords),1): lines.append(f"ATOM  {serial:5d} {atom.name:>4s} {atom.resname:>3s} {binder_chain:1s}{atom.resseq:4d}{atom.icode:1s}   {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {(atom.element or atom.name[:1]).upper():>2s}")
    output=Path(output_path); output.parent.mkdir(parents=True,exist_ok=True); output.write_text("\n".join(lines+["TER","END",""]),encoding="utf-8"); return str(output)
