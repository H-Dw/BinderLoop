from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set
import yaml

from binderloop.parameter_decision import ParameterDecisionSpec

class ConfigError(ValueError): pass

def _reject(section, data, known):
    if data is None: data={}
    if not isinstance(data, dict): raise ConfigError(f"{section} must be a mapping")
    bad=sorted(set(data)-set(known))
    if bad: raise ConfigError(f"{section}.{bad[0]} is not a supported config field")

def _merge(cls,data,section):
    _reject(section,data,{f.name for f in fields(cls)})
    return cls(**(data or {}))

@dataclass
class TargetSpec:
    structure_path: str
    chain_id: str='A'; hotspots: List[str]=field(default_factory=list); notes: Optional[str]=None
    profile: Dict[str,Any]=field(default_factory=dict); include: List[Dict[str,Any]]=field(default_factory=list)
    binding_types: List[Dict[str,Any]]=field(default_factory=list); structure_groups: Optional[str]=None
@dataclass
class TaskInputSpec:
    task_name: str='binder_task'; target_structure_path: Optional[str]=None; boltzgen_input_path: Optional[str]=None
    target_chain_id: str='A'; hotspots: List[str]=field(default_factory=list); binder_length_range: Optional[Any]=None
    binder_length_step: int=10; max_binders_per_round: Optional[int]=None; target_include: List[Dict[str,Any]]=field(default_factory=list)
    target_binding_types: List[Dict[str,Any]]=field(default_factory=list); structure_groups: Optional[str]=None; notes: Optional[str]=None
    freeze_target_definition: bool=True; freeze_binder_length_range: bool=True; freeze_round_budget: bool=True
@dataclass
class SearchSpace:
    binder_lengths: List[int]=field(default_factory=lambda:[60,80,100]); binder_length_range: Optional[Any]=None
    binder_length_step: int=10; num_designs_per_round: int=32; max_binders_per_round: Optional[int]=None
    model_order: List[str]=field(default_factory=lambda:['boltzgen','odesign']); boltzgen: Dict[str,Any]=field(default_factory=dict); odesign: Dict[str,Any]=field(default_factory=dict); rfd3: Dict[str,Any]=field(default_factory=dict)
@dataclass
class ScoringWeights:
    interface_confidence: float=.30; hotspot_contact: float=.25; binder_plddt: float=.15; clash_penalty: float=.15; diversity: float=.10; sequence_designability: float=.05
@dataclass
class ActiveLearningSpec:
    max_rounds:int=5; max_retries:int=3; strategy:str='successive_halving'; top_k:int=8; exploration_ratio:float=.30
    branch_width:int=1; promote_top_branches:int=1; branch_budget_policy:str='equal'; min_designs_per_branch:int=1
    def __post_init__(self):
        self.branch_width=int(self.branch_width); self.promote_top_branches=int(self.promote_top_branches)
        if self.branch_width < 1: raise ConfigError('active_learning branch_width must be >= 1')
        if self.branch_budget_policy != 'equal': raise ConfigError('only equal branch_budget_policy is supported')
    enable_backtracking:bool=True; regression_tolerance:float=.25; rollback_patience:int=2; enable_strategy_skills:bool=False
    enable_exploitation_arms:bool=False; min_current_positives_for_exploit:int=2; prior_positive_decay_after_zero_rounds:int=2
    near_miss_top_k:int=4; near_miss_min_confidence:float=.30; near_miss_weight:float=.25
@dataclass
class ModelRuntimeSpec:
    conda_env: str
    weights_path: Optional[str]=None
    checkpoint_dir: Optional[str]=None
    cache_dir: Optional[str]=None
    moldir: Optional[str]=None
    def __post_init__(self):
        self.conda_env=str(self.conda_env or '').strip()
        if not self.conda_env: raise ConfigError('model runtime conda_env cannot be empty')
        for name in ('weights_path','checkpoint_dir','cache_dir','moldir'):
            value=getattr(self,name)
            if value is not None:
                value=str(value).strip()
                if not value: raise ConfigError(f'model runtime {name} cannot be empty')
                setattr(self,name,value)

def _default_model_runtimes():
    return {
        'boltzgen': ModelRuntimeSpec(conda_env='bg'),
        'rfd3': ModelRuntimeSpec(conda_env='foundry'),
        'odesign': ModelRuntimeSpec(conda_env='odesign'),
    }

@dataclass
class RuntimeSpec:
    project_root:str='..'; skill_registry_path:Optional[str]='configs/skills/binder_skills.yaml'; boltzgen_root:str='../boltzgen'; odesign_root:str='../ODesign'; foundry_root:str='models/foundry'
    output_dir:str='outputs/run'; python_bin:str='python'; extend_memory:bool=False
    conda_base:str='/data/miniconda3'; conda_executable:str='conda'; model_runtimes:Dict[str,ModelRuntimeSpec]=field(default_factory=_default_model_runtimes)
    def __post_init__(self):
        self.conda_base=str(self.conda_base or '').strip()
        self.conda_executable=str(self.conda_executable or '').strip()
        if not self.conda_executable: raise ConfigError('runtime.conda_executable cannot be empty')
        supplied=self.model_runtimes or {}
        if not isinstance(supplied,Mapping): raise ConfigError('runtime.model_runtimes must be a mapping')
        normalized=_default_model_runtimes()
        for raw_name,raw_spec in supplied.items():
            name=str(raw_name).strip().lower()
            if not name: raise ConfigError('runtime.model_runtimes model name cannot be empty')
            if isinstance(raw_spec,ModelRuntimeSpec):
                normalized[name]=raw_spec
            elif isinstance(raw_spec,Mapping):
                inherited={'conda_env': normalized[name].conda_env} if name in normalized else {}
                normalized[name]=_merge(ModelRuntimeSpec,{**inherited,**dict(raw_spec)},f'owner.runtime_resources.runtime.model_runtimes.{name}')
            else:
                raise ConfigError(f'owner.runtime_resources.runtime.model_runtimes.{name} must be a mapping')
        self.model_runtimes=normalized
    def model_runtime(self,model_name):
        name=str(model_name or '').strip().lower()
        if name not in self.model_runtimes: raise ConfigError(f'no runtime configured for model {name!r}')
        return self.model_runtimes[name]
@dataclass
class MemorySpec:
    enabled:bool=False; index_items:bool=False; retrieval:bool=False; semantic_rerank:bool=False; compression:bool=False; apply_prompt_budget:bool=False
    retrieval_candidate_limit:int=24; retrieval_top_k:int=8; mmr_lambda:float=.7; max_active_items:int=24; compression_batch_size:int=6; max_summary_chars:int=1200; prompt_max_bytes:int=750000
    def wants_index_items(self): return self.index_items or self.enabled
    def wants_retrieval(self): return self.retrieval or self.enabled
    def wants_semantic_rerank(self): return self.semantic_rerank and self.wants_retrieval()
    def wants_compression(self): return self.compression or self.enabled
    def wants_prompt_budget(self): return (self.apply_prompt_budget or self.enabled) and self.prompt_max_bytes>0
    def any_optimization_enabled(self): return any((self.wants_index_items(),self.wants_retrieval(),self.wants_compression(),self.wants_prompt_budget(),self.semantic_rerank))
def apply_memory_cli_overrides(memory,**kwargs):
    for k,v in kwargs.items():
        if v and hasattr(memory,k): setattr(memory,k,True)
    if kwargs.get('semantic_rerank'): memory.retrieval=True
    return memory
@dataclass
class SelfImprovementSpec:
    enabled:bool=False; skill_path:Optional[str]=None; max_active_rules:int=6; max_rules:int=48; promotion_min_support:int=2; retirement_contradictions:int=2
    reward_improvement_threshold:float=.01; strong_improvement_threshold:float=.05; semantic_candidate_limit:int=8; semantic_confidence_threshold:float=.72
    conflict_resolution_enabled:bool=True; prompt_max_bytes:int=24000; recent_round_window:int=5
    def __post_init__(self):
        if self.max_active_rules < 1 or self.max_rules < 1 or self.max_active_rules > self.max_rules:
            raise ConfigError('self improvement rule limits are invalid')
        if not 0 <= float(self.semantic_confidence_threshold) <= 1:
            raise ConfigError('semantic_confidence_threshold must be within [0, 1]')
        if float(self.strong_improvement_threshold) <= float(self.reward_improvement_threshold):
            raise ConfigError('strong_improvement_threshold must exceed reward_improvement_threshold')
@dataclass
class HotspotSelectionSpec:
    enabled: bool = False
    allow_web_search: bool = False
    min_hotspots: int = 3
    max_hotspots: int = 6
    max_change_per_round: int = 2
    max_residues_in_prompt: int = 200
    require_llm: bool = True
    model: Optional[str] = None

    def __post_init__(self):
        self.min_hotspots = int(self.min_hotspots)
        self.max_hotspots = int(self.max_hotspots)
        self.max_change_per_round = int(self.max_change_per_round)
        self.max_residues_in_prompt = int(self.max_residues_in_prompt)
        if self.min_hotspots < 1:
            raise ConfigError('owner.llm_context_learning.hotspot_selection.min_hotspots must be >= 1')
        if self.max_hotspots < self.min_hotspots:
            raise ConfigError('owner.llm_context_learning.hotspot_selection.max_hotspots cannot be below min_hotspots')
        if self.max_change_per_round < 0:
            raise ConfigError('owner.llm_context_learning.hotspot_selection.max_change_per_round must be >= 0')
        if self.max_residues_in_prompt < 8:
            raise ConfigError('owner.llm_context_learning.hotspot_selection.max_residues_in_prompt must be >= 8')
        if self.model is not None:
            model = str(self.model).strip()
            self.model = model or None

@dataclass
class QualityCollaborationSpec:
    enabled:bool=False; performance_drop_tolerance:float=0.; reward_noise_band:float=.02; recovery_ratio:float=.97
    max_consecutive_multi_rounds:int=2; compute_gate_degradation_ratio:float=.90; pae_degradation_ratio:float=1.10; hotspot_degradation_ratio:float=.90
    low_confidence_threshold:float=.55; high_impact_parameter_count:int=2; recovery_tolerance:float=1e-6; request_timeout_seconds:int=105
    failure_cooldown_seconds:int=20; specialist_max_tokens:int=1400; manager_max_tokens:int=1800
    specialist_reasoning_mode:str='low'; manager_reasoning_mode:str='low'
    max_completion_tokens:int=65536; reasoning_budget_tokens:int=0; visible_json_budget_tokens:int=4096
    max_api_calls:int=8
    # Deprecated compatibility aliases; used only when the new explicit fields are absent.
    specialist_output_tokens:Optional[int]=None; manager_output_tokens:Optional[int]=None
    max_revisions:Optional[int]=None; final_max_tokens:Optional[int]=None
    def __post_init__(self):
        if self.specialist_reasoning_mode not in {'low','medium','high'}: raise ConfigError('specialist_reasoning_mode must be low, medium, or high')
        if self.manager_reasoning_mode not in {'low','medium','high'}: raise ConfigError('manager_reasoning_mode must be low, medium, or high')
        if int(self.visible_json_budget_tokens) < 1024: raise ConfigError('visible_json_budget_tokens must be >= 1024')
        if int(self.max_completion_tokens) < int(self.visible_json_budget_tokens): raise ConfigError('max_completion_tokens must cover visible_json_budget_tokens')
        if int(self.max_completion_tokens) > 65536: raise ConfigError('max_completion_tokens must be <= 65536')
        if int(self.reasoning_budget_tokens) < 0: raise ConfigError('reasoning_budget_tokens must be >= 0')
        if self.specialist_output_tokens is not None and self.specialist_max_tokens == 1400: self.specialist_max_tokens=int(self.specialist_output_tokens)
        if self.manager_output_tokens is not None and self.manager_max_tokens == 1800: self.manager_max_tokens=int(self.manager_output_tokens)
@dataclass
class ResourceSpec:
    backend:str='direct'; host_num:int=1; host_gpu_num:int=1; taiji_multi_host_mode:str='native'; gpu_name:str='V100'; max_parallel_jobs:int=1
    template_json:Optional[str]=None; image_full_name:Optional[str]=None; timeout_seconds:int=3600; taiji_options:Dict[str,Any]=field(default_factory=dict)
    def __post_init__(self):
        self.backend=str(self.backend or 'direct').strip().lower()
        self.host_num=int(self.host_num); self.host_gpu_num=int(self.host_gpu_num)
        if self.host_num<1 or self.host_gpu_num<1: raise ConfigError('resource hosts must be >= 1')
        self.taiji_multi_host_mode={'unified':'native','fanout':'split_jobs','split':'split_jobs'}.get(self.taiji_multi_host_mode,self.taiji_multi_host_mode)
        if self.taiji_multi_host_mode not in {'native','split_jobs'}: raise ConfigError('invalid taiji_multi_host_mode')
        if self.backend not in {'direct','local','taiji','dry_run'}: raise ConfigError('invalid resource backend')
    def to_taiji_options(self):
        o=dict(self.taiji_options); o.setdefault('host_num',self.host_num); o.setdefault('host_gpu_num',self.host_gpu_num); o.setdefault('GPUName',self.gpu_name); o.setdefault('taiji_timeout',self.timeout_seconds); return o
@dataclass
class TaskHardConstraints:
    target_structure_path:Optional[str]; num_designs:int; task_name:str='binder_task'; boltzgen_input_path:Optional[str]=None; target_chain_id:str='A'
    hotspots:List[str]=field(default_factory=list); binder_length_range:Optional[Any]=None; binder_length_step:int=10; target_include:List[Dict[str,Any]]=field(default_factory=list)
    target_binding_types:List[Dict[str,Any]]=field(default_factory=list); structure_groups:Optional[str]=None; notes:Optional[str]=None
    freeze_target_definition:bool=True; freeze_binder_length_range:bool=True; freeze_round_budget:bool=True
    def __post_init__(self):
        if isinstance(self.num_designs, bool) or not isinstance(self.num_designs, int):
            raise ConfigError('owner.task_hard_constraints.num_designs must be an integer')
        if self.num_designs < 1:
            raise ConfigError('owner.task_hard_constraints.num_designs must be >= 1')
@dataclass
class SequenceModuleSpec:
    tool: Optional[str] = None

@dataclass
class RefoldingModuleSpec:
    tool: Optional[str] = None

@dataclass
class SamplerBounds:
    noise_scale:Optional[Dict[str,float]]=None; step_scale:Optional[Dict[str,float]]=None; alpha:Optional[Dict[str,float]]=None; gamma_0:Optional[Dict[str,float]]=None
    def __post_init__(self):
        from binderloop.models.search_profile import absolute_bounds_for_axis
        for n in ('noise_scale','step_scale','alpha','gamma_0'):
            v=getattr(self,n)
            if v is None: continue
            if not isinstance(v,Mapping) or set(v)!={'min','max'}: raise ConfigError(f'owner.sampler_bounds.{n} must contain exactly min and max')
            lo,hi=float(v['min']),float(v['max'])
            if lo>hi: raise ConfigError(f'owner.sampler_bounds.{n}.min cannot exceed max')
            try:
                absolute_bounds_for_axis(n, lo, hi)
            except Exception as exc:
                raise ConfigError(str(exc)) from exc
            setattr(self,n,{'min':lo,'max':hi})
@dataclass
class FilteringBudget:
    budget:int; run_filtering:bool=True; keep_unfiltered_for_failure_analysis:bool=True; additional_filters:List[str]=field(default_factory=list)
    def __post_init__(self):
        self.budget=int(self.budget)
        if self.budget<1: raise ConfigError('owner.filtering_budget.budget must be >= 1')

@dataclass
class HarnessTemplatePolicy:
    enabled: bool = False
    gate: str = 'interchain_pae'
    interchain_pae_max: float = 10.0
    min_quality: float = 0.70
    top_k: int = 1
    round_conditioned_fraction: float = 0.5
    proximity: float = 8.0
    max_templates: int = 20
    library_size: int = 30
    max_fixed_fraction: float = 0.5
    min_designable_residues: int = 8
    min_alignment_coverage: float = 0.75
    max_target_patch_rmsd: float = 2.5
    require_pae: bool = True
    package_failure_policy: str = 'reject_and_rematerialize'
    utility_decay: float = 0.90
    cooldown_failures: int = 2
    blacklist_failures: int = 3
    def __post_init__(self):
        if self.gate not in {'interchain_pae', 'iptm'}: raise ConfigError('owner.harness_template_policy.gate is invalid')
        if not 0.0 <= float(self.round_conditioned_fraction) <= 0.8: raise ConfigError('round_conditioned_fraction must be within [0, 0.8]')
        if not 0.0 < float(self.min_alignment_coverage) <= 1.0: raise ConfigError('min_alignment_coverage must be within (0, 1]')
        if not 0.0 < float(self.max_fixed_fraction) < 1.0: raise ConfigError('max_fixed_fraction must be within (0, 1)')
        if self.top_k < 1 or self.max_templates < 1 or self.library_size < 1 or self.min_designable_residues < 1: raise ConfigError('template count limits must be positive')
        if self.package_failure_policy != 'reject_and_rematerialize': raise ConfigError('only reject_and_rematerialize is supported')

@dataclass
class OwnerConfig:
    task_hard_constraints: TaskHardConstraints
    sampler_bounds: SamplerBounds = field(default_factory=SamplerBounds)
    filtering_budget: Optional[FilteringBudget] = None
    harness_template_policy: HarnessTemplatePolicy = field(default_factory=HarnessTemplatePolicy)
    parameter_decision: ParameterDecisionSpec = field(default_factory=ParameterDecisionSpec)
@dataclass
class HarnessConfig:
    target:TargetSpec; owner:Optional[OwnerConfig]=None; task_name:str='binder_task'; task:TaskInputSpec=field(default_factory=TaskInputSpec)
    search_space:SearchSpace=field(default_factory=SearchSpace); scoring:ScoringWeights=field(default_factory=ScoringWeights); active_learning:ActiveLearningSpec=field(default_factory=ActiveLearningSpec)
    runtime:RuntimeSpec=field(default_factory=RuntimeSpec); memory:MemorySpec=field(default_factory=MemorySpec); self_improvement:SelfImprovementSpec=field(default_factory=SelfImprovementSpec)
    quality_collaboration:QualityCollaborationSpec=field(default_factory=QualityCollaborationSpec); resource:ResourceSpec=field(default_factory=ResourceSpec)
    sequence:SequenceModuleSpec=field(default_factory=SequenceModuleSpec); refolding:RefoldingModuleSpec=field(default_factory=RefoldingModuleSpec)
    hotspot_selection:HotspotSelectionSpec=field(default_factory=HotspotSelectionSpec)

def _section(owner,key):
    v=owner.get(key) or {}
    if not isinstance(v,dict): raise ConfigError(f'owner.{key} must be a mapping')
    return dict(v)

def _normalize_boltzgen_choice_flag(field,value,choices):
    """Canonicalize YAML bool/str choice flags to BoltzGen argparse tokens.

    BoltzGen rejects capitalized ``True``/``False``. YAML ``false`` is a Python
    bool, so this must run at load time before any ``str(value)`` CLI rendering.
    """
    from binderloop.agents.model_input_spec import normalize_choice_flag
    token=normalize_choice_flag(value,frozenset(choices))
    if token is None:
        raise ConfigError(f'{field} must be one of {sorted(choices)}')
    return token

def load_config(path)->HarnessConfig:
    path=Path(path); raw=yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    _reject('config',raw,{'schema_version','owner'})
    if raw.get('schema_version')!=1: raise ConfigError('schema_version: 1 is required')
    owner=raw.get('owner'); allowed={'task_hard_constraints','boltzgen_design_native','boltzgen_inverse_fold_and_validation','boltzgen_filtering_ranking','rfd3_design_native','rfd3_inverse_fold_and_validation','rfd3_filtering_ranking','sampler_bounds','filtering_budget','harness_search_space','harness_selection_and_evidence','harness_template_policy','parameter_decision','active_learning_and_rollback','runtime_resources','llm_context_learning','sequence','refolding'}
    _reject('owner',owner,allowed); owner=dict(owner or {})
    llm=_section(owner,'llm_context_learning'); _reject('owner.llm_context_learning',llm,{'memory','self_improvement','quality_collaboration','hotspot_selection'})
    hotspot_selection=_merge(HotspotSelectionSpec,llm.get('hotspot_selection'),'owner.llm_context_learning.hotspot_selection')
    hardraw=owner.get('task_hard_constraints')
    if not isinstance(hardraw,dict): raise ConfigError('owner.task_hard_constraints is required')
    if 'num_designs' not in hardraw: raise ConfigError('owner.task_hard_constraints.num_designs is required')
    hard=_merge(TaskHardConstraints,hardraw,'owner.task_hard_constraints'); bounds=_merge(SamplerBounds,owner.get('sampler_bounds'),'owner.sampler_bounds'); template_policy=_merge(HarnessTemplatePolicy,owner.get('harness_template_policy'),'owner.harness_template_policy'); parameter_decision=_merge(ParameterDecisionSpec,owner.get('parameter_decision'),'owner.parameter_decision')
    fb=_merge(FilteringBudget,owner.get('filtering_budget'),'owner.filtering_budget') if 'filtering_budget' in owner else None
    native=_section(owner,'boltzgen_design_native'); inverse=_section(owner,'boltzgen_inverse_fold_and_validation'); filtering=_section(owner,'boltzgen_filtering_ranking')
    _reject('owner.boltzgen_design_native',native,{'protocol','diffusion_batch_size','design_checkpoints','steps','checkpoint_dir','cache','moldir','use_kernels','num_workers','reuse','silence','log_heartbeat_seconds','auto_binder_length','epitope_crop_mode','binder_chain'})
    _reject('owner.boltzgen_inverse_fold_and_validation',inverse,{'inverse_fold_num_sequences','inverse_fold_avoid','inverse_fold_checkpoint','folding_checkpoint','affinity_checkpoint','skip_inverse_folding','only_inverse_fold'})
    _reject('owner.boltzgen_filtering_ranking',filtering,{'alpha','filter_biased','refolding_rmsd_threshold','metrics_override','size_buckets','config_overrides'})
    if 'use_kernels' in native:
        native['use_kernels']=_normalize_boltzgen_choice_flag('owner.boltzgen_design_native.use_kernels',native.get('use_kernels'),{'auto','true','false'})
    if 'filter_biased' in filtering:
        filtering['filter_biased']=_normalize_boltzgen_choice_flag('owner.boltzgen_filtering_ranking.filter_biased',filtering.get('filter_biased'),{'true','false'})
    rfd3_native=_section(owner,'rfd3_design_native'); rfd3_inverse=_section(owner,'rfd3_inverse_fold_and_validation'); rfd3_filtering=_section(owner,'rfd3_filtering_ranking')
    _reject('owner.rfd3_design_native',rfd3_native,{'steps','diffusion_batch_size','n_batches','ckpt_path','rfd3_checkpoint','checkpoint_dir','weights_path','infer_ori_strategy','is_non_loopy','redesign_motif_sidechains','dialect','step_scale','gamma_0','num_timesteps','noise_scale','skip_existing','prevalidate_inputs','dump_trajectories','low_memory_mode','global_prefix','auto_binder_length','binder_chain','target_chain','target_res_index','contig','select_hotspots','select_hbond_donor','select_hbond_acceptor','sample_name','residue_id_scheme','rfd3_source_id_scheme','rfd3_residue_scheme','rfd3_adapt_structure','rfd3_convert_residue_ids'})
    _reject('owner.rfd3_inverse_fold_and_validation',rfd3_inverse,{'model_type','is_legacy_weights','inverse_fold_num_sequences','temperature','designed_chains','mpnn_checkpoint','checkpoint_path','batch_size','number_of_batches','write_fasta','write_structures'})
    _reject('owner.rfd3_filtering_ranking',rfd3_filtering,{'refolding_rmsd_threshold','early_stopping_plddt_threshold','n_recycles','num_steps','rf3_checkpoint','folding_checkpoint'})
    bz={**native,**inverse,**filtering}
    rfd3={**rfd3_native,**rfd3_inverse,**rfd3_filtering}
    bz.update({
        'fragment_templates_enabled': template_policy.enabled,
        'fragment_template_gate': template_policy.gate,
        'fragment_interchain_pae_max': template_policy.interchain_pae_max,
        'fragment_template_min_quality': template_policy.min_quality,
        'fragment_template_top_k': template_policy.top_k,
        'template_conditioned_fraction': template_policy.round_conditioned_fraction,
        'binder_template_proximity': template_policy.proximity,
        'fragment_template_max_templates': template_policy.max_templates,
        'fragment_template_library_size': template_policy.library_size,
        'fragment_template_max_fixed_fraction': template_policy.max_fixed_fraction,
        'fragment_template_min_designable_residues': template_policy.min_designable_residues,
        'fragment_template_min_alignment_coverage': template_policy.min_alignment_coverage,
        'fragment_template_max_target_patch_rmsd': template_policy.max_target_patch_rmsd,
        'fragment_template_require_pae': template_policy.require_pae,
        'fragment_template_package_failure_policy': template_policy.package_failure_policy,
        'fragment_template_utility_decay': template_policy.utility_decay,
        'fragment_template_cooldown_failures': template_policy.cooldown_failures,
        'fragment_template_blacklist_failures': template_policy.blacklist_failures,
    })
    if fb:
        if not fb.run_filtering: raise ConfigError('owner.filtering_budget.run_filtering cannot be false')
        bz.update(budget=fb.budget,run_filtering=True,keep_unfiltered_for_failure_analysis=fb.keep_unfiltered_for_failure_analysis,additional_filters=list(fb.additional_filters))
    task=TaskInputSpec(task_name=hard.task_name,target_structure_path=hard.target_structure_path,boltzgen_input_path=hard.boltzgen_input_path,target_chain_id=hard.target_chain_id,hotspots=list(hard.hotspots),binder_length_range=hard.binder_length_range,binder_length_step=hard.binder_length_step,max_binders_per_round=hard.num_designs,target_include=list(hard.target_include),target_binding_types=list(hard.target_binding_types),structure_groups=hard.structure_groups,notes=hard.notes,freeze_target_definition=hard.freeze_target_definition,freeze_binder_length_range=hard.freeze_binder_length_range,freeze_round_budget=hard.freeze_round_budget)
    if task.boltzgen_input_path: _merge_boltzgen_input(task,path.parent,reject_hotspot_priors=hotspot_selection.enabled)
    if not task.target_structure_path: raise ConfigError('owner.task_hard_constraints.target_structure_path is required')
    if task.binder_length_range is None: raise ConfigError('owner.task_hard_constraints.binder_length_range is required')
    selection=_section(owner,'harness_selection_and_evidence'); _reject('owner.harness_selection_and_evidence',selection,{'contact_cutoff','clash_cutoff','clash_density_max','fragment_window','fragment_stride','weighted_hotspot_conditioning'});
    if selection.get('weighted_hotspot_conditioning'): raise ConfigError('weighted_hotspot_conditioning is unsupported by current BoltzGen checkpoints')
    for key,value in selection.items(): bz[key]=value
    hs=_section(owner,'harness_search_space'); _reject('owner.harness_search_space',hs,{'model_order','odesign','rfd3'})
    search=SearchSpace(model_order=list(hs.get('model_order') or ['boltzgen']),odesign=dict(hs.get('odesign') or {}),boltzgen=bz,rfd3=rfd3,binder_length_range=task.binder_length_range,binder_length_step=task.binder_length_step,max_binders_per_round=hard.num_designs)
    search.binder_lengths=_expand_length_range(search.binder_length_range,search.binder_length_step); search.num_designs_per_round=hard.num_designs; search.boltzgen.setdefault('epitope_crop_mode','disabled'); search.boltzgen['num_designs']=hard.num_designs
    search.rfd3['num_designs']=hard.num_designs
    search.rfd3.setdefault('target_chain',task.target_chain_id)
    if fb: search.rfd3.setdefault('budget',fb.budget)
    extra_rfd3=dict(hs.get('rfd3') or {}); search.rfd3.update(extra_rfd3)
    if task.target_include: search.boltzgen.setdefault('target_include',task.target_include); search.rfd3.setdefault('target_include',task.target_include)
    if task.target_binding_types: search.boltzgen.setdefault('target_binding_types',task.target_binding_types)
    if task.structure_groups: search.boltzgen.setdefault('structure_groups',task.structure_groups)
    rr=_section(owner,'runtime_resources'); _reject('owner.runtime_resources',rr,{'runtime','resource'})
    if hotspot_selection.enabled:
        _reject_llm_hotspot_priors(hard=hard, task=task, rfd3=search.rfd3)
    target=TargetSpec(task.target_structure_path,task.target_chain_id,list(task.hotspots),task.notes,include=list(task.target_include),binding_types=list(task.target_binding_types),structure_groups=task.structure_groups)
    active_raw=dict(owner.get('active_learning_and_rollback') or {}); active_raw.setdefault('branch_width', 2)
    active_learning=_merge(ActiveLearningSpec,active_raw,'owner.active_learning_and_rollback')
    if active_learning.branch_width not in {2,4}: raise ConfigError('owner.active_learning_and_rollback.branch_width must be exactly 2 or 4')
    if hard.num_designs < active_learning.branch_width: raise ConfigError('owner.task_hard_constraints.num_designs must be >= active_learning branch_width')
    primary=str((search.model_order or ['boltzgen'])[0]).strip().lower()
    raw_pd=owner.get('parameter_decision') or {}
    if primary=='rfd3':
        from binderloop.models.search_profile import RFD3_DEFAULT_GAMMA_0_CANDIDATES, RFD3_DEFAULT_STEP_SCALE_CANDIDATES, RFD3_SAMPLER_AXES
        if 'sampler_axes' not in raw_pd:
            parameter_decision.sampler_axes=RFD3_SAMPLER_AXES
        if 'step_scale_candidates' not in raw_pd:
            parameter_decision.step_scale_candidates=RFD3_DEFAULT_STEP_SCALE_CANDIDATES
        if 'gamma_0_candidates' not in raw_pd:
            parameter_decision.gamma_0_candidates=RFD3_DEFAULT_GAMMA_0_CANDIDATES
        parameter_decision.__post_init__()
    sequence=_merge(SequenceModuleSpec,owner.get('sequence'),'owner.sequence')
    refolding=_merge(RefoldingModuleSpec,owner.get('refolding'),'owner.refolding')
    return HarnessConfig(target=target,owner=OwnerConfig(hard,bounds,fb,template_policy,parameter_decision),task_name=task.task_name,task=task,search_space=search,scoring=ScoringWeights(),active_learning=active_learning,runtime=_merge(RuntimeSpec,rr.get('runtime'),'owner.runtime_resources.runtime'),resource=_merge(ResourceSpec,rr.get('resource'),'owner.runtime_resources.resource'),memory=_merge(MemorySpec,llm.get('memory'),'owner.llm_context_learning.memory'),self_improvement=_merge(SelfImprovementSpec,llm.get('self_improvement'),'owner.llm_context_learning.self_improvement'),quality_collaboration=_merge(QualityCollaborationSpec,llm.get('quality_collaboration'),'owner.llm_context_learning.quality_collaboration'),sequence=sequence,refolding=refolding,hotspot_selection=hotspot_selection)

def binder_generation_cap(cfg): return max(1,int(cfg.search_space.max_binders_per_round or cfg.search_space.num_designs_per_round or 1))
def primary_design_model(cfg):
    order=[str(item).strip().lower() for item in (cfg.search_space.model_order or []) if str(item).strip()]
    return order[0] if order else 'boltzgen'
def _expand_length_range(value,step=10):
    if isinstance(value,dict): lo=int(value.get('min',value.get('start'))); hi=int(value.get('max',value.get('end'))); step=int(value.get('step',step))
    elif isinstance(value,str): lo,hi=map(int,value.replace('..','-').replace(':','-').split('-',1))
    else:
        seq=list(value)
        if not seq: raise ConfigError('binder_length_range cannot be empty')
        lo=int(seq[0]); hi=int(seq[1] if len(seq)>1 else seq[0]); step=int(seq[2] if len(seq)>2 else step)
    lo,hi=min(lo,hi),max(lo,hi); step=max(1,step); out=list(range(lo,hi+1,step))
    if out[-1]!=hi: out.append(hi)
    return sorted(set(out))
def _nonempty_hotspot_tokens(values):
    return [str(item).strip() for item in (values or []) if str(item).strip()]

def _reject_llm_hotspot_priors(*, hard, task, rfd3):
    if _nonempty_hotspot_tokens(getattr(hard, 'hotspots', None)):
        raise ConfigError('owner.llm_context_learning.hotspot_selection.enabled forbids owner.task_hard_constraints.hotspots')
    if list(getattr(hard, 'target_binding_types', None) or []):
        raise ConfigError('owner.llm_context_learning.hotspot_selection.enabled forbids owner.task_hard_constraints.target_binding_types')
    if _nonempty_hotspot_tokens(getattr(task, 'hotspots', None)):
        raise ConfigError('owner.llm_context_learning.hotspot_selection.enabled forbids hotspot priors on the resolved task')
    if list(getattr(task, 'target_binding_types', None) or []):
        raise ConfigError('owner.llm_context_learning.hotspot_selection.enabled forbids resolved target_binding_types')
    select_hotspots = str((rfd3 or {}).get('select_hotspots') or '').strip()
    if select_hotspots:
        raise ConfigError('owner.llm_context_learning.hotspot_selection.enabled forbids owner.rfd3_design_native.select_hotspots')

def _merge_boltzgen_input(task,config_dir,reject_hotspot_priors=False):
    p=Path(task.boltzgen_input_path or ''); p=p if p.is_absolute() else config_dir/p
    if not p.exists(): return
    d=yaml.safe_load(p.read_text(encoding='utf-8')) or {}
    for e in d.get('entities',[]) or []:
        if 'file' in e:
            f=e['file'] or {}
            binding_types=list(f.get('binding_types') or [])
            if reject_hotspot_priors and binding_types:
                raise ConfigError('owner.llm_context_learning.hotspot_selection.enabled forbids boltzgen_input binding_types')
            if not task.target_structure_path and f.get('path'): task.target_structure_path=str((p.parent/f['path']).resolve())
            if not task.target_include: task.target_include=list(f.get('include') or [])
            if not task.target_binding_types: task.target_binding_types=binding_types
            if not task.hotspots: task.hotspots=_hotspots_from_binding_types(task.target_binding_types,task.target_chain_id)
            task.structure_groups=task.structure_groups or f.get('structure_groups')
def _hotspots_from_binding_types(items,default):
    out=[]
    for x in items:
        c=(x.get('chain') or {}) if isinstance(x,dict) else {}; cid=str(c.get('id') or default)
        out += [f'{cid}:{r.strip()}' for r in str(c.get('binding') or '').split(',') if r.strip()]
    return out
