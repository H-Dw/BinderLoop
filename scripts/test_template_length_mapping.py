#!/usr/bin/env python3
import sys, tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from binderloop.templates import ResidueIdentity, parse_residue_identity, plan_length_transform
from binderloop.analysis.template_alignment import align_target_patch, transform_structure_coordinates

def residues(n): return [f"A:{i}" for i in range(1,n+1)]
def pdb(path, chain, rows):
    lines=[]
    for i,(number,icode,name,xyz) in enumerate(rows,1):
        x,y,z=xyz; lines.append(f"ATOM  {i:5d}   CA {name:>3s} {chain}{number:4d}{icode:1s}   {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C")
    path.write_text("\n".join(lines+["END",""]))

def test_identity_and_mapping():
    assert parse_residue_identity("B:12A")==ResidueIdentity("B",12,"A")
    t=plan_length_transform(residues(12),12,motif_residues=["A:5","A:6"],min_designable_residues=1)
    assert t.status=="identity" and t.source_to_effective_residue_map["A:5"]=="A:5"

def test_insert_crop_linker_and_reject():
    n=plan_length_transform(residues(10),12,motif_residues=["A:5"],insertion_preference=("n_terminal",),min_designable_residues=1); assert n.method=="n_terminal_insertion" and n.source_to_effective_residue_map["A:5"]=="A:7"
    c=plan_length_transform(residues(10),12,motif_residues=["A:5"],insertion_preference=("c_terminal",),min_designable_residues=1); assert c.method=="c_terminal_insertion" and c.source_to_effective_residue_map["A:5"]=="A:5"
    l=plan_length_transform(residues(10),12,motif_residues=["A:2"],insertion_preference=("linker",),safe_linker_after=["A:6"],min_designable_residues=1); assert l.method=="linker_insertion"
    crop=plan_length_transform(residues(10),8,motif_residues=["A:4"],contact_residues=["A:5"],insertion_preference=("c_terminal",),min_designable_residues=1); assert crop.method=="c_terminal_crop" and len(crop.source_to_effective_residue_map)==8
    bad=plan_length_transform(residues(10),4,motif_residues=["A:1","A:10"],contact_residues=["A:2","A:9"],min_designable_residues=1); assert bad.status=="rejected"

def test_alignment_rigid_insertion_and_fallback(tmp_path):
    src=tmp_path/"src.pdb"; cur=tmp_path/"cur.pdb"
    rows=[(10,"","ALA",(0,0,0)),(10,"A","CYS",(1,0,0)),(11,"","ASP",(0,1,0))]
    pdb(src,"T",rows); pdb(cur,"Q",[(n,ic,name,(xyz[0]+3,xyz[1]-2,xyz[2]+1)) for n,ic,name,xyz in rows])
    a=align_target_patch(str(src),str(cur),source_target_chain="T",current_target_chain="Q",residue_ids=["T:10","T:10A","T:11"]); assert a.status=="aligned" and a.mapping_method=="exact_author_identity" and a.target_patch_rmsd<1e-6 and a.residue_map["T:10A"]=="Q:10A"
    cur2=tmp_path/"cur2.pdb"; pdb(cur2,"Q",[(101,"",name,xyz) for (_,_,name,xyz) in rows]); b=align_target_patch(str(src),str(cur2),source_target_chain="T",current_target_chain="Q",residue_ids=["T:10","T:10A","T:11"]); assert b.status=="aligned" and b.mapping_method=="unique_sequence_fallback"
    amb=tmp_path/"amb.pdb"; pdb(amb,"Q",[(i+1,"",name,(i,0,0)) for i,name in enumerate(["ALA","CYS","ASP","ALA","CYS","ASP"])]); r=align_target_patch(str(src),str(amb),source_target_chain="T",current_target_chain="Q",residue_ids=["T:10","T:10A","T:11"]); assert r.status=="rejected" and r.sequence_fallback_status=="ambiguous"

def main():
    test_identity_and_mapping(); test_insert_crop_linker_and_reject()
    with tempfile.TemporaryDirectory() as d: test_alignment_rigid_insertion_and_fallback(Path(d))
    print("ALL TEMPLATE LENGTH MAPPING TESTS PASSED")
if __name__=="__main__": main()
