"""Fail-closed source-to-effective binder length planner."""
from dataclasses import asdict, dataclass
import hashlib, json
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple
from .residue_identity import ResidueIdentity, parse_residue_identity

def _digest(x): return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
@dataclass(frozen=True)
class LengthMappingEntry:
    source: Optional[ResidueIdentity]
    effective: Optional[ResidueIdentity]
    role: str
    motif: bool = False
    contact: bool = False
    designable: bool = True
    def to_dict(self):
        d=asdict(self); d.update(source_token=self.source.token if self.source else None, effective_token=self.effective.token if self.effective else None); return d
@dataclass(frozen=True)
class LengthTransform:
    status: str; method: str; source_length: int; effective_length: int
    entries: Tuple[LengthMappingEntry, ...]; source_to_effective_residue_map: Mapping[str,str]
    fixed_residue_tokens: Tuple[str,...]; designable_residue_tokens: Tuple[str,...]
    fixed_fraction: float; designable_count: int; chain_continuous: bool; reason: str; digest: str
    def to_dict(self):
        d=asdict(self); d["entries"]=[e.to_dict() for e in self.entries]; d["source_to_effective_residue_map"]=dict(self.source_to_effective_residue_map); return d

def plan_length_transform(source_residues: Sequence[Any], effective_length: int, *, motif_residues: Iterable[Any]=(), contact_residues: Iterable[Any]=(), fixed_residues: Optional[Iterable[Any]]=None, insertion_preference: Sequence[str]=("c_terminal","n_terminal"), safe_linker_after: Iterable[Any]=(), max_fixed_fraction: float=.5, min_designable_residues: int=8) -> LengthTransform:
    src=tuple(parse_residue_identity(x) for x in source_residues); target=int(effective_length)
    if not src or target<=0: return _reject(src,target,"empty_or_invalid_length")
    if len({x.chain for x in src})!=1 or len({x.token for x in src})!=len(src): return _reject(src,target,"non_unique_or_multi_chain_source")
    all_tokens={x.token for x in src}; motif={parse_residue_identity(x).token for x in motif_residues}; contact={parse_residue_identity(x).token for x in contact_residues}; fixed={parse_residue_identity(x).token for x in (fixed_residues if fixed_residues is not None else motif_residues)}
    if (motif|contact|fixed)-all_tokens: return _reject(src,target,"protected_residue_not_in_source")
    n=len(src); kept=list(range(n)); insert_at=None; method="unchanged"
    if target<n:
        crop=n-target; protected=motif|contact; n_ok=all(src[i].token not in protected for i in range(crop)); c_ok=all(src[i].token not in protected for i in range(n-crop,n))
        if "n_terminal" in insertion_preference and n_ok: kept,method=list(range(crop,n)),"n_terminal_crop"
        elif c_ok: kept,method=list(range(n-crop)),"c_terminal_crop"
        elif n_ok: kept,method=list(range(crop,n)),"n_terminal_crop"
        else: return _reject(src,target,"unsafe_internal_deletion_required")
    elif target>n:
        method=""; safe={parse_residue_identity(x).token for x in safe_linker_after}
        for pref in insertion_preference:
            if pref=="n_terminal": insert_at,method=0,"n_terminal_insertion"
            elif pref=="c_terminal": insert_at,method=n,"c_terminal_insertion"
            elif pref=="linker":
                sites=[i+1 for i in range(n-1) if src[i].token in safe and not ({src[i].token,src[i+1].token}&(motif|contact))]
                if len(sites)==1: insert_at,method=sites[0],"linker_insertion"
            if method: break
        if not method: return _reject(src,target,"no_safe_insertion_site")
    delta=max(0,target-len(kept)); entries=[]; mapping={}; ei=0
    for i in range(n+1):
        if delta and i==insert_at:
            role={"n_terminal_insertion":"inserted_n","c_terminal_insertion":"inserted_c","linker_insertion":"inserted_linker"}[method]
            for _ in range(delta): ei+=1; entries.append(LengthMappingEntry(None,ResidueIdentity(src[0].chain,ei,label="inserted"),role))
        if i==n: break
        residue=src[i]
        if i not in kept: entries.append(LengthMappingEntry(residue,None,"cropped_n" if method=="n_terminal_crop" else "cropped_c",residue.token in motif,residue.token in contact,residue.token not in fixed)); continue
        ei+=1; eff=ResidueIdentity(residue.chain,ei,label=residue.token,name=residue.name); mapping[residue.token]=eff.token; entries.append(LengthMappingEntry(residue,eff,"unchanged",residue.token in motif,residue.token in contact,residue.token not in fixed))
    fixed_eff=tuple(mapping[x] for x in sorted(fixed) if x in mapping); designable=tuple(e.effective.token for e in entries if e.effective and e.designable); fraction=len(fixed_eff)/float(target)
    reason="validated"
    if (motif|contact)-set(mapping): reason="protected_residue_cropped"
    elif fraction>max_fixed_fraction: reason="fixed_fraction_exceeds_limit"
    elif len(designable)<min_designable_residues: reason="insufficient_designable_residues"
    if reason!="validated": return _reject(src,target,reason,tuple(entries))
    status="identity" if method=="unchanged" else "applied"; body=dict(status=status,method=method,source_length=n,effective_length=target,entries=[e.to_dict() for e in entries],source_to_effective_residue_map=mapping,fixed_residue_tokens=fixed_eff,designable_residue_tokens=designable,fixed_fraction=round(fraction,8),designable_count=len(designable),chain_continuous=ei==target,reason=reason)
    return LengthTransform(status,method,n,target,tuple(entries),mapping,fixed_eff,designable,body["fixed_fraction"],len(designable),ei==target,reason,_digest(body))

def _reject(src,target,reason,entries=()):
    body=dict(status="rejected",method="rejected",source_length=len(src),effective_length=int(target or 0),entries=[e.to_dict() for e in entries],source_to_effective_residue_map={},fixed_residue_tokens=(),designable_residue_tokens=(),fixed_fraction=0.,designable_count=0,chain_continuous=False,reason=reason)
    return LengthTransform("rejected","rejected",len(src),int(target or 0),tuple(entries),{},(),(),0.,0,False,reason,_digest(body))
