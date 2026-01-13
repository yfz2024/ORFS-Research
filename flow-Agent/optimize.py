#!/usr/bin/env python3
import json
import os
import sys
import re
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import anthropic
import csv
import random
from rag.util import answerWithRAG
from rag.index import load_embeddings_and_docs, build_and_save_embeddings
from sentence_transformers import SentenceTransformer
import torch
from pathlib import Path
from openai import OpenAI
from inspectfuncs import *
from modelfuncs import *
from agglomfuncs import *
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

print("Python executable:", sys.executable)
print("sys.path:", sys.path[:3])
print("Current working dir:", os.getcwd())

for pkg in ["anthropic", "sentence_transformers", "torch", "openai"]:
    try:
        __import__(pkg)
        print(f" Imported {pkg}")
    except ImportError as e:
        print(f" Failed to import {pkg}: {e}")

def process_log_file(log_path: str) -> Dict[str, Any]:
    """Process a single log file to extract relevant metrics"""
    run_data = {
        'file': os.path.basename(log_path),
        'success': False,
        'metrics': {},
        'errors': []
    }
    
    with open(log_path, 'r') as f:
        content = f.read()
        # print(f"File size: {len(content)} characters")
        
        # Extract timing information
        timing_metrics = extract_timing_metrics(content)
        run_data['metrics'].update(timing_metrics)
        
        # Extract wirelength information
        wl_metrics = extract_wirelength_metrics(content, log_path)
        run_data['metrics'].update(wl_metrics)
        
        # Extract any errors
        errors = extract_errors(content)
        run_data['errors'].extend(errors)
        # === Inject a simulated error for testing RAG debugging ===
        # if "run3" in log_path:  #  you can change to any run，eg. run1 or run2
        #     fake_error = "[ERROR] TEST_SIM: Simulated routing congestion failure"
        #     print(f"[DEBUG] Injected simulated error in {log_path}: {fake_error}")
        #     run_data['errors'].append(fake_error)
        #     run_data['success'] = False
        
        # Determine success based on completion markers and errors
        run_data['success'] = is_run_successful(content, errors)
        
    return run_data

def extract_timing_metrics(log_content: str) -> Dict[str, float]:
    """Extract timing-related metrics from log content"""
    metrics = {}
    
    # Extract worst slack (final)
    slack_match = re.search(r'(?:worst slack|WNS):\s*([-\d.]+)', log_content, re.IGNORECASE)
    if slack_match:
        metrics['worst_slack'] = float(slack_match.group(1))
        
    # Extract CTS worst slack
    # cts_slack_match = re.search(r'(?:CTS|Clock Tree) (?:worst slack|WNS):\s*([-\d.]+)', log_content, re.IGNORECASE)
    cts_slack_match = re.search(r'Timing-driven:\s*(?:worst slack|WNS)\s+([-\d.eE]+)', log_content, re.IGNORECASE)
    if cts_slack_match:
        metrics['cts_ws'] = float(cts_slack_match.group(1))
        
    # Extract clock period
    # period_match = re.search(r'[Cc]lock period:\s*([\d.]+)', log_content)
    period_match = re.search(r'clock period to\s*([\d.]+)', log_content)
    if period_match:
        metrics['clock_period'] = float(period_match.group(1))
        
    # Extract TNS
    # tns_match = re.search(r'(?:Total negative slack|TNS):\s*([-\d.]+)', log_content, re.IGNORECASE)
    tns_match = re.search(r'"finish__timing__setup__tns"\s*:\s*([-\d.eE]+)', log_content)
    if tns_match:
        metrics['tns'] = float(tns_match.group(1))
        
    return metrics

def extract_wirelength_metrics(log_content: str, log_path: Optional[str] = None) -> Dict[str, float]:
    metrics = {}
    # Prefer detailed route JSON: ./logs/<platform>/<design>/base_i/5_2_route.json
    if log_path:
        base_dir = Path(log_path).parent
        route_json = None

        # log_path 可能在 logs/<p>/<d>/ 或 logs/<p>/<d>/base_i/，统一定位 base_i
        if base_dir.name.startswith("base_"):
            route_json = base_dir / "5_2_route.json"
        else:
            # 找到包含 base_* 的子目录
            for p in base_dir.iterdir() if base_dir.is_dir() else []:
                if p.is_dir() and p.name.startswith("base_"):
                    candidate = p / "5_2_route.json"
                    if candidate.exists():
                        route_json = candidate
                        break
            # 再向上查找父目录中的 base_*
            if route_json is None:
                parents_base = [p for p in base_dir.parents if p.name.startswith("base_")]
                if parents_base:
                    route_json = parents_base[0] / "5_2_route.json"
                else:
                    route_json = base_dir / "5_2_route.json"

        if route_json.exists():
            try:
                with open(route_json, "r") as jf:
                    route_data = json.load(jf)
                wl = route_data.get("detailedroute__route__wirelength")
                if wl is not None:
                    metrics["total_wirelength"] = float(wl)
                    print(f"[WIRE] Using detailed route wirelength from {route_json}")
                else:
                    print(f"[WIRE] {route_json} found but no detailedroute__route__wirelength key")
            except Exception as e:
                print(f"[WARN] Failed to read wirelength from {route_json}: {e}")

    # Fall back to log parsing if JSON not available
    if "total_wirelength" not in metrics:
        wl_match = re.search(r'Total wirelength: ([\d.]+)', log_content)
        if wl_match:
            metrics['total_wirelength'] = float(wl_match.group(1))
            print(f"[WIRE] Fallback to CTS wirelength from log")

    return metrics

def extract_errors(log_content: str) -> List[str]:
    """Extract error messages from log content"""
    errors = []
    
    # Look for common error patterns
    error_patterns = [
        r'Error: .*',
        r'ERROR: .*',
        r'FATAL: .*',
        r'Failed: .*'
    ]
    
    for pattern in error_patterns:
        matches = re.finditer(pattern, log_content, re.MULTILINE)
        errors.extend(match.group(0) for match in matches)
        
    return errors

def is_run_successful(log_content: str, errors: List[str]) -> bool:
    """Determine if a run was successful based on log content and errors"""
    if errors:
        print("Errors encountered, Return False.")
        return False
        
    # Look for completion markers
    completion_markers = [
        'finish report_design_area',
        '6_report',
        'Report metrics stage 6, finish...'
    ]
    flag = any(marker in log_content for marker in completion_markers)
    if not flag:
        print(log_content)
    return flag

class OptimizationWorkflow:
    def __init__(self, platform: str, design: str, objective: str):
        self.platform = platform
        self.design = design
        self.objective = objective.upper()  # Convert to uppercase
        self.history_root = Path(f"backup_dir/{self.platform}/{self.design}")
        self.history_root = Path(f"backup_dir/{self.platform}/{self.design}")
        
        # Load configuration
        self.config = self._load_config()

        # [TextGrad Integration] 1. Convert static Prompts into Optimizable State Variables
        self.tool_instructions = {
            'inspect': (
                f"**Stage: Inspect**\n"
                "In this stage, we analyze the data from previous optimization runs to identify patterns, trends, and insights.\n\n"
                "Please analyze the data and provide insights on:\n"
                "1. Key patterns in successful vs unsuccessful runs.\n"
                "2. Parameter ranges that appear promising.\n"
                "3. Any timing or wirelength trends.\n"
                "4. Recommendations for subsequent runs.\n"
                "5. We hope to further reduce the total wirelength.\n"
            ),
            
            'model': (
                f"**Stage: Model**\n"
                "In this stage, we decide how to model the optimization problem based on the data analysis.\n\n"
                "Please suggest:\n"
                "1. Appropriate modeling techniques.\n"
                "2. Key parameters to focus on.\n"
                "3. Surrogate model recommendations.\n"
                "4. Acquisition function choices.\n"
            ),
            
            'agglomerate': (
                f"**Stage: Agglomerate**\n"
                "In this stage, we generate new parameter combinations to explore in the next optimization runs.\n\n"
                "Please provide a list of new parameter sets to try, ensuring they respect the domain constraints below.\n\n"
                "Each parameter set should be a dictionary with parameter names and their suggested values.\n"
                "**Important:** Make sure all parameter sets satisfy the domain constraints before suggesting them.\n"
            )
        }
        
        # Find the design configuration
        configurations = self.config.get('configurations', [])
        for config in configurations:
            if (
                config['platform'].lower() == self.platform.lower()
                and config['design'].lower() == self.design.lower()
                and config['goal'].upper() == self.objective
            ):
                self.design_config = config
                break
        else:
            raise ValueError(
                f"No configuration found for platform={self.platform}, design={self.design}, objective={self.objective}"
            )
        
        # Define initial clock periods for each design and platform
        initial_clock_periods = {
            ('asap7', 'aes'): 400,     # in picoseconds
            ('asap7', 'ibex'): 1260,   # in picoseconds
            ('asap7', 'jpeg'): 1100,   # in picoseconds
            ('sky130hd', 'aes'): 4.5,  # in nanoseconds
            ('sky130hd', 'ibex'): 10.0,# in nanoseconds
            ('sky130hd', 'jpeg'): 8.0, # in nanoseconds
        }

        # Get the initial clock period for the current platform and design
        key = (self.platform.lower(), self.design.lower())
        if key in initial_clock_periods:
            self.initial_clk_period = initial_clock_periods[key]
        else:
            raise ValueError(f"Initial clock period not defined for platform={self.platform}, design={self.design}")

        # Calculate clock period range
        min_clk = self.initial_clk_period * 0.7
        max_clk = self.initial_clk_period * 1.3

        # Define parameter names in the expected order
        self.parameter_names = [
            'core_util',
            'cell_pad_global',
            'cell_pad_detail', 
            'enable_dpo',
            'clk_period'
        ]
        
        # Set all parameter constraints including clock period range
        self.param_constraints = {
            'core_util': {'type': 'int', 'range': [20, 99]},
            'cell_pad_global': {'type': 'int', 'range': [0, 4]},
            'cell_pad_detail': {'type': 'int', 'range': [0, 4]},
            'enable_dpo': {'type': 'int', 'range': [0, 1]},
            'clk_period': {'type': 'float', 'range': [min_clk, max_clk]}
        }
        
        # Load initial parameters and SDC context
        self.initial_params = self._load_initial_params()
        self.sdc_context = self._load_sdc_context()

        emb_np, docs, docsDict = load_embeddings_and_docs()
        self.rag_embeddings = torch.tensor(emb_np).cpu()
        self.rag_docs = docs
        self.rag_docsDict = docsDict
        print("[INFO] Loading embedding model...")
        model_path = Path(__file__).parent / "models" / "mxbai-embed-large-v1"
        model_path = model_path.resolve()
        # self.rag_model = SentenceTransformer(str(model_path))
        self.rag_model = SentenceTransformer(str(model_path), device='cpu')
        
        # LLM clients – require keys from environment for portability/security
        self.default_base_url = os.environ.get("OPENAI_BASE_URL", "https://ai.gitee.com/v1")
        self.default_api_key = os.environ.get("OPENAI_API_KEY")
        if not self.default_api_key:
            raise ValueError("OPENAI_API_KEY must be set for optimize_textgrad to run.")
        
        # Secondary supervisor model for dual-LLM flow. All values can be overridden via env variables.
        self.supervisor_client = OpenAI(
            base_url=os.environ.get("SUPERVISOR_BASE_URL", self.default_base_url),
            api_key=os.environ.get("SUPERVISOR_API_KEY", self.default_api_key),
        )
        self.supervisor_model_name = os.environ.get("SUPERVISOR_MODEL", "DeepSeek-R1")
        # self.supervision_feedback: str = ""
        self.feedback: str = ""

    def _optimize_prompt_with_textgrad(self, stage: str, feedback: str):
        """
        [TextGrad Integration] Core function: Optimize Prompt(Variable) based on feedback (Gradient).
        """
        print(f"\n[TextGrad] Optimizing '{stage}' instructions based on feedback...")
        
        current_instruction = self.tool_instructions.get(stage, "")
        
        # This is a meta-prompt used to make LLM act as an "optimizer".
        optimizer_prompt = f"""
        You are a Prompt Optimizer for an EDA Logic Synthesis Agent.
        
        **Current System Instruction**:
        {current_instruction}
        
        **Critique/Gradient (Feedback from Supervisor)**:
        {feedback}
        
        **Task**:
        Rewrite the "Current System Instruction" to explicitly address the Critique. 
        - If the critic says the agent ignored errors, strengthen the error-handling instructions.
        - If the critic says the agent explored too little, emphasize exploration in the instructions.
        - Keep the format consistent, but change the content to be more effective.
        
        Return ONLY the new instruction text.
        """

        try:
            response = self.supervisor_client.chat.completions.create(
                model=self.supervisor_model_name,
                messages=[{"role": "user", "content": optimizer_prompt}],
                temperature=0.2
            )
            new_instruction = response.choices[0].message.content.strip()
            
            # update (Variable Update)
            self.tool_instructions[stage] = new_instruction
            print(f"[TextGrad] Successfully updated instructions for stage '{stage}'.")
            print(f"[TextGrad] Successfully updated instructions :'{self.tool_instructions[stage]}'.")
            
        except Exception as e:
            print(f"[TextGrad] Failed to optimize prompt: {e}")

    def _load_config(self) -> Dict[str, Any]:
        """Load the configuration from 'opt_config.json'."""
        with open('opt_config.json', 'r') as f:
            config = json.load(f)
        return config
          
    def _load_initial_params(self) -> Dict[str, Any]:
        """Load initial parameters from the design's config.mk file"""
        config_path = f"designs/{self.platform}/{self.design}/config.mk"
        params = {}
        with open(config_path, 'r') as f:
            for line in f:
                if line.startswith('export'):
                    parts = line.strip().split('=', 1)
                    if len(parts) == 2:
                        key = parts[0].replace('export', '').strip()
                        value = parts[1].strip()
                        params[key] = value
        return params
        
    def _load_sdc_context(self) -> Dict[str, Any]:
        """Load SDC context including special case for JPEG"""
        sdc_context = {}
        
        # Determine SDC filename based on platform/design
        if self.platform.lower() == 'asap7' and self.design.lower() == 'jpeg':
            sdc_file = 'jpeg_encoder15_7nm.sdc'
        else:
            sdc_file = 'constraint.sdc'
            
        sdc_path = f"designs/{self.platform}/{self.design}/{sdc_file}"
        
        # Read SDC file
        try:
            with open(sdc_path, 'r') as f:
                sdc_content = f.read()
                
            # Extract clock period
            clock_period_match = re.search(r'set clk_period\s+([\d.]+)', sdc_content)
            if clock_period_match:
                sdc_context['clock_period'] = float(clock_period_match.group(1))
                
            # Store full content for context
            sdc_context['content'] = sdc_content
            sdc_context['filename'] = sdc_file
            
        except Exception as e:
            print(f"Warning: Could not load SDC file {sdc_path}: {str(e)}")
            sdc_context['error'] = str(e)
            
        return sdc_context
        
    def _load_run_parameters(self, run_id: str, log_path: str) -> Dict[str, Any]:
        """
        Attempt to recover parameters used in a run so downstream analysis does not
        fall back to initial_params. Tries (in order):
        1) Sidecar JSON next to the log (e.g., <log>.params.json or params.json in the same folder).
        2) A CSV row that aligns with the run_id index (base_<n>) in the design CSV.
        """
        sidecars = [
            f"{log_path}.params.json",
            os.path.join(os.path.dirname(log_path), "params.json"),
        ]
        for sidecar in sidecars:
            if os.path.exists(sidecar):
                try:
                    with open(sidecar, "r") as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        return data.get("parameters", data)
                    if isinstance(data, list):
                        return data[0] if data else {}
                except Exception as e:
                    print(f"[WARN] Failed to read sidecar params {sidecar}: {e}")

        # Try mapping base_<n> to nth row of the design CSV (1-based)
        csv_path = f"designs/{self.platform}/{self.design}/{self.platform}_{self.design}.csv"
        m = re.search(r'(\d+)$', run_id)
        if os.path.exists(csv_path) and m:
            row_idx = int(m.group(1)) - 1  # zero-based index
            try:
                with open(csv_path, newline='') as f:
                    reader = csv.reader(f)
                    header = next(reader, [])
                    rows = list(reader)
                if 0 <= row_idx < len(rows) and header:
                    row = rows[row_idx]
                    params = {}
                    for name, value in zip(header, row):
                        try:
                            params[name] = float(value)
                        except ValueError:
                            params[name] = value
                    return params
            except Exception as e:
                print(f"[WARN] Failed to map run_id {run_id} to CSV row: {e}")

        return {}

    def _generate_llm_prompt(self, stage: str, data: Dict[str, Any]) -> str:

        constraints_text = "Parameter Constraints:\n"
        for param, info in self.param_constraints.items():
            param_type = info['type']
            param_range = info['range']
            constraints_text += f"- {param} ({param_type}, range: {param_range})\n"

        # stage_descriptions = {
        #     'inspect': (
        #         "You are an expert EDA optimization analyst. Analyze the optimization run data to identify patterns "
        #         "and insights. Use the available tools to examine data distributions, correlations, and successful "
        #         "parameter ranges. Your goal is to understand what makes runs successful and provide recommendations "
        #         "for the modeling stage."
        #     ),
        #     'model': (
        #         "You are an expert machine learning engineer for EDA optimization. Based on the inspection results, "
        #         "configure the modeling approach for Bayesian optimization. Consider kernel selection, acquisition "
        #         "functions, and surrogate modeling strategies. Balance exploration and exploitation based on the data."
        #     ),
        #     'agglomerate': (
        #         "You are an expert parameter optimization specialist. Generate new parameter combinations for the "
        #         "next optimization iteration. Use insights from previous stages to focus on promising regions while "
        #         "maintaining diversity. Ensure all parameters satisfy the domain constraints."
        #     )
        # }
    
        # data_context = ""
        # if stage == 'inspect':
        #      data_context = f"\n[data_context] Data to analyze:\n{json.dumps(data, indent=2, default=str)}\n"
        # elif stage == 'model':
        #      data_context = f"\n[data_context] Data for modeling:\n{json.dumps(data, indent=2, default=str)}\n"
        # elif stage == 'agglomerate':
        #      data_context = f"\n[data_context] Domain Constraints to respect:\n{constraints_text}\n"

        # stage_contexts = {
        #     'inspect': (
        #         f"metrics analyze: Total runs={data.get('log_data', {}).get('summary', {}).get('total_runs', 0)}, "
        #         f"successful runs ={data.get('log_data', {}).get('summary', {}).get('successful_runs', 0)}"
        #     ),
        #     'model': "Configure modeling methods based on inspection results",
        #     'agglomerate': "Configure parameter selection strategy based on modeling results"
        # }

        # constraints_text = "Parameter Constraints:\n"
        # for param, info in self.param_constraints.items():
        #     constraints_text += f"- {param} ({info['type']}, range: {info['range']})\n"

        constraints_text += (
            "\nParameter Notes:\n"
            "- cell_pad_global: The cell spacing during global routing/placement: smaller spacing saves area but increases congestion; if frequent congestion/routing DRC errors occur, the spacing can be increased accordingly; however, if area/timing constraints are tight, the spacing should be kept as small as possible.\n"
            "- cell_pad_detail: The cell spacing during the detailed placement stage is similar to that of the global placement; it can be increased if there is layout congestion or detailed routing failure, and decreased when aiming for shorter connections or smaller area.\n"
            "- constraints: the value of cell_pad_detail cannot be greater than the value of cell_pad_global.\n"
        )

        prompt = f"""**stages:{stage.upper()}**

        {self.tool_instructions.get(stage, '')}

        **Important: Use **tool calls only**, and generate text responses!**
        """

        if getattr(self, "supervision_feedback", ""):
            prompt += f"""

        ### Supervisor Feedback
        {self.supervision_feedback}

        Please prioritize these supervisor notes when making decisions in this stage.
        """
            print("[SUPERVISOR][DEBUG] Added supervisor feedback into prompt.")

        # ========= Splicing RAG  =========
        # rag_context = ""
        # if self.rag_model is not None and self.rag_embeddings is not None:
        #     try:
        #         print("[DEBUG] RAG: Starting retrieval...")

        #         # --------------------------------------------------
        #         # Step 1: Extract recent error information
        #         # --------------------------------------------------
        #         error_summary = ""
        #         if "log_data" in data:
        #             all_errors = []
        #             for run in data["log_data"].get("runs", []):
        #                 if run.get("errors"):
        #                     all_errors.extend(run["errors"])

        #             if all_errors:
        #                 top_errors = all_errors[-3:]
        #                 error_summary = "\n".join(top_errors)
        #                 print(f"[DEBUG] Detected {len(all_errors)} error messages. Sample:\n{error_summary}")

        #         # --------------------------------------------------
        #         # Step 2: Construct RAG query
        #         # --------------------------------------------------
        #         if error_summary:
        #             query = (
        #                 f"Stage: {stage}. Objective: {self.objective}. "
        #                 f"The recent optimization runs failed with errors:\n{error_summary}\n\n"
        #                 "Retrieve relevant OpenROAD documentation, known failure modes, and "
        #                 "potential parameter tuning suggestions that may fix these issues."
        #             )
        #         else:
        #             query = (
        #                 f"Stage: {stage}. Objective: {self.objective}. "
        #                 f"Focus on analyzing {stage}-related optimization results. "
        #                 f"Important metrics include wirelength, timing, area, and success rate. "
        #                 f"Find relevant OpenROAD documentation, parameter tuning guides, and failure pattern examples."
        #             )

        #         # --------------------------------------------------
        #         # Step 3: Execute RAG search
        #         # --------------------------------------------------
        #         rag_result = answerWithRAG(
        #             query,
        #             self.rag_embeddings,
        #             self.rag_model,
        #             self.rag_docs,
        #             self.rag_docsDict
        #         )

        #         retrieved_sections = []
        #         print("[DEBUG] rag_result type:", type(rag_result))
        #         print("[DEBUG] rag_result content:", rag_result)

        #         # --------------------------------------------------
        #         # Step 4: Select top-K most relevant docs
        #         # --------------------------------------------------
        #         docs = []
        #         if isinstance(rag_result, dict):
        #             docs = rag_result.get("docs", [])
        #         elif isinstance(rag_result, list):
        #             docs = rag_result
        #         elif isinstance(rag_result, str):
        #             print("[WARN] RAG returned string output, not structured docs.")

        #         if docs:
        #             docs = sorted(docs, key=lambda d: d.get("score", 0.0), reverse=True)

        #             TOP_K = 3
        #             MAX_CHARS_PER_DOC = 300  

        #             for idx, doc in enumerate(docs[:TOP_K], 1):
        #                 content = doc.get("content", "").strip()
        #                 excerpt = content[:MAX_CHARS_PER_DOC]

        #                 section = (
        #                     f"[Doc {idx} | score={doc.get('score', 0.0):.3f} | "
        #                     f"source={doc.get('source', 'unknown')}]\n"
        #                     f"{excerpt}"
        #                 )
        #                 retrieved_sections.append(section)

        #             print(f"[DEBUG] RAG: Selected top-{len(retrieved_sections)} documents.")

        #         # --------------------------------------------------
        #         # Step 5: Inject controlled RAG context into prompt
        #         # --------------------------------------------------
        #         if retrieved_sections:
        #             prompt += (
        #                 "\n\n=== Retrieved OpenROAD Documentation (Top-K via RAG) ===\n"
        #                 + "\n\n".join(retrieved_sections)
        #                 + "\n=============================================="
        #             )
        #             print("[DEBUG] RAG: Context successfully added to prompt.")
        #         else:
        #             print("[DEBUG] RAG: No relevant documents retrieved.")

        #     except Exception as e:
        #         prompt += f"\n\n[WARN] RAG retrieval failed: {e}"

        rag_context = ""
        if self.rag_model is not None and self.rag_embeddings is not None:
            try:
                print("[DEBUG] RAG: Starting retrieval...")

                # Step 1: Extract error information
                error_summary = ""
                if "log_data" in data:
                    all_errors = []
                    for run in data["log_data"].get("runs", []):
                        if run.get("errors"):
                            all_errors.extend(run["errors"])
                    if all_errors:
                        # Retrieve the most recent or frequent errors
                        top_errors = all_errors[-3:]  # Last 3 posts
                        error_summary = "\n".join(top_errors)
                        print(f"[DEBUG] Detected {len(all_errors)} error messages. Sample:\n{error_summary}")

                # Step 2: Construct RAG query
                if error_summary:
                    query = (
                        f"Stage: {stage}. Objective: {self.objective}. "
                        f"The recent optimization runs failed with errors:\n{error_summary}\n\n"
                        "Retrieve relevant OpenROAD documentation, known failure modes, and "
                        "potential parameter tuning suggestions that may fix these issues."
                    )
                else:
                    query = (
                        f"Stage: {stage}. Objective: {self.objective}. "
                        f"Focus on analyzing {stage}-related optimization results. "
                        f"Important metrics include wirelength, timing, area, and success rate. "
                        f"Find relevant OpenROAD documentation, parameter tuning guides, and failure pattern examples."
                    )

                # Step 3: Execute search
                rag_context = answerWithRAG(
                    query,
                    self.rag_embeddings,
                    self.rag_model,
                    self.rag_docs,
                    self.rag_docsDict
                )

                if isinstance(rag_context, dict):  # If returning dict
                    print(f"[DEBUG] RAG Retrieved {len(rag_context.get('docs', []))} related entries.")
                    for doc in rag_context.get("docs", [])[:3]:
                        print(f"  ↳ {doc['title']} (score={doc['score']:.3f}) from {doc['source']}")
                    retrieved_text = rag_context.get("content", "")
                else:
                    retrieved_text = rag_context

                # Step 4: 加入 prompt
                print(f"[DEBUG] RAG: Retrieved context length = {len(rag_context)}")
                if retrieved_text.strip():
                    prompt += (
                        "\n\n=== Retrieved OpenROAD Documentation (via RAG) ===\n"
                        f"{retrieved_text[:10]}\n"
                        "=============================================="
                    )
                    print("[DEBUG] RAG: Context successfully added to prompt.")
                else:
                    print("[DEBUG] RAG: Empty context retrieved.")
            except Exception as e:
                prompt += f"\n\n[WARN] RAG retrieval failed: {e}"

        if "log_data" in data:
            all_errors = []
            for run in data["log_data"].get("runs", []):
                if run.get("errors"):
                    all_errors.extend(run["errors"])

            if all_errors:
                error_summary = "\n".join(all_errors[-3:])  # The last 5 errors
                prompt += f"""

        ### Detected Run Errors
        {error_summary}

        The model must now act as an **EDA Debugging Assistant**.
        Using the retrieved documentation and knowledge, analyze the errors above and provide a short explanation for each suggestion.

        """
                print("[DEBUG] Added error-fix section to prompt.")      
        return prompt
        
        # **Current Stage**: {stage.upper()}
        # **Objective**: {self.objective}
        # **Platform**: {self.platform}
        # **Design**: {self.design}

        # {stage_descriptions[stage]}
        # {constraints_text}
        # {data_context}
        # {stage_contexts.get(stage, '')}

        # **Available Data**:
        # - Total runs: {data.get('log_data', {}).get('summary', {}).get('total_runs', 0)}
        # - Successful runs: {data.get('log_data', {}).get('summary', {}).get('successful_runs', 0)}
        # - Failed runs: {data.get('log_data', {}).get('summary', {}).get('failed_runs', 0)}

        # Using the retrieved documentation and knowledge, analyze the errors above and:
        # 1. Identify the root causes of each error (e.g., tool misconfiguration, parameter overflow, timing issues, etc.)
        # 2. Suggest **specific parameter changes or flow adjustments** to prevent these errors.
        # 3. Indicate whether the error is likely due to constraints, routing congestion, or timing margin.
        # 4. Provide a short explanation for each suggestion.

        # Output format:
        # Error Diagnosis:
        # - Root Cause: ...
        # - Recommended Fix: ...
        # - Suggested Parameter Change: ...
        # - Reason: ...

    def _collect_recent_errors(self, data: Dict[str, Any], max_errors=5) -> str:
        errors = []
        for run in data.get("log_data", {}).get("runs", []):
            if run.get("errors"):
                errors.extend(run["errors"])
        print(f"[DEBUG] Found total {len(errors)} errors in all runs.")
        return "\n".join(errors[-max_errors:]) if errors else "No errors detected."

    def _call_llm(self, stage: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Call llm to get recommendations for optimization parameters"""
        print(f"\n=== Calling LLM for {stage} stage ===")
        
        # Hardcode API key
        # api_key = ""#PUT YOUR KEY HERE
        
        # client = anthropic.Anthropic(api_key=api_key)
        client = OpenAI(
            base_url="https://ai.gitee.com/v1",
            api_key=self.default_api_key,
            )
        
        tools = [
            {
                "type": "function",  
                "function": {
                    "name": "configure_inspection",
                    "description": "Configure data inspection parameters",
                    "parameters": { 
                        "type": "object",
                        "properties": {
                            "n_clusters": {
                                "type": "integer",
                                "description": "Number of clusters for structure analysis",
                                "minimum": 2,
                                "maximum": 10
                            },
                            "correlation_threshold": {
                                "type": "number",
                                "description": "Threshold for considering correlations significant",
                                "minimum": 0,
                                "maximum": 1
                            }
                        },
                        "required": ["n_clusters", "correlation_threshold"]
                    }
                }
            },
            {
                "type": "function", 
                "function": {
                    "name": "configure_model",
                    "description": "Configure modeling approach",
                    "parameters": {  
                        "type": "object",
                        "properties": {
                            "kernel_type": {
                                "type": "string",
                                "enum": ["rbf", "matern", "rational"],
                                "description": "Type of kernel to use"
                            },
                            "preprocessing": {
                                "type": "string",
                                "enum": ["standard", "robust", "none"],
                                "description": "Type of preprocessing to apply"
                            },
                            "acquisition": {
                                "type": "string",
                                "enum": ["ei", "ucb", "pi"],
                                "description": "Acquisition function to use"
                            },
                            "surrogate_weight": {
                                "type": "number",
                                "description": "Weight to give surrogate values",
                                "minimum": 0,
                                "maximum": 1
                            }
                        },
                        "required": ["kernel_type", "preprocessing", "acquisition", "surrogate_weight"]
                    }
                }
            },
            {
                "type": "function",  
                "function": {
                    "name": "configure_selection",
                    "description": "Configure point selection strategy",
                    "parameters": { 
                        "type": "object",
                        "properties": {
                            "method": {
                                "type": "string",
                                "enum": ["entropy", "kmeans", "hybrid", "graph"],
                                "description": "Selection method to use"
                            },
                            "quality_weight": {
                                "type": "number",
                                "description": "Weight between quality and diversity",
                                "minimum": 0,
                                "maximum": 1
                            },
                            "uncertainty_bonus": {
                                "type": "number",
                                "description": "Weight for uncertainty in quality scores",
                                "minimum": 0,
                                "maximum": 1
                            }
                        },
                        "required": ["method", "quality_weight", "uncertainty_bonus"]
                    }
                }
            }
        ]

        # Get data summaries
        if stage == 'inspect':
            print("Generating data distribution and structure summaries...")
            # Extract X and Y from successful runs if available
            print("=== DEBUG: Checking data structure ===")
            print(f"data keys: {list(data.keys()) if data else 'Empty data'}")
            X = []
            Y = []
            if 'log_data' in data and 'runs' in data['log_data']:
                print("✓ 'log_data' and 'runs' found in data")
                print(f"Number of runs: {len(data['log_data']['runs'])}")
                for run in data['log_data']['runs']:
                    if run['success'] and 'metrics' in run:
                        obj = self._calculate_objective(run)
                        if obj['value'] is not None or obj['surrogate'] is not None:
                            # Use parameters as X
                            params = []
                            for param in self.param_constraints.keys():
                                # params.append(float(run.get('parameters', {}).get(param, 0)))
                                params.append(float(run.get('parameters', {}).get(param, self.initial_params.get(param, 0))))
                            X.append(params)
                            # Use objective as Y
                            Y.append(obj['value'] if obj['value'] is not None else obj['surrogate'])
                            print(f"  ✓ Added to X and Y: X={params}, Y={obj['value'] if obj['value'] is not None else obj['surrogate']}")
            
            if X and Y:
                dist_summary = inspect_data_distribution(np.array(X), np.array(Y))
                # struct_summary = inspect_data_structure(np.array(X))
                struct_summary = inspect_data_structure(np.array(X), np.array(Y))
                print("dist_summary =", dist_summary)
                print("struct_summary =", struct_summary)
                print("DEBUG: len(X)=", len(X), " len(Y)=", len(Y))
            else:
                print("No successful runs found, using empty summaries")
                dist_summary = {}
                struct_summary = {}
        else:
            dist_summary = {}
            struct_summary = {}

        # Generate context message
        context_message = self._generate_llm_prompt(stage, data)
        print(f"Generated prompt with context for {stage}")
        print("=== Context Message ===")
        print(context_message)
        print("========================")
        
        print("Making LLM API call...")
        # response = client.messages.create(
        response = client.chat.completions.create(
            # model="claude-3-sonnet-20240229",
            model="DeepSeek-R1",
            messages=[{
                "role": "user",
                "content": context_message
            }],
            # max_tokens=2048,
            tools=tools,
            tool_choice="auto",
            # stream=True,
            max_tokens=32000,
            temperature=0.1,
            # top_p=0.8,
            # extra_body={
            # "top_k": 20,
            # },
            # frequency_penalty=1.1,

            # messages=[{
            #     "role": "user",
            #     "content": context_message
            # }]
        )

        # Extract configurations
        configs = {}
        print("=== 完整API响应 ===")
        print(f"Response type: {type(response)}")
        print(f"Choices count: {len(response.choices)}")
        print("\nReceived tool calls from LLM:")
        message = response.choices[0].message
        print(f"Message content: {getattr(message, 'content', 'No content')}")
        print(f"Tool calls: {getattr(message, 'tool_calls', 'No tool calls')}")
        
        # 检查是否有工具调用
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                print(f"- {function_name}: {function_args}")
                
                if function_name == 'configure_inspection':
                    configs['inspection'] = function_args
                elif function_name == 'configure_model':
                    configs['model'] = function_args
                elif function_name == 'configure_selection':
                    configs['selection'] = function_args

        # for block in response.content:
        #     if block.type == 'tool_use':
        #         print(f"- {block.name}: {block.input}")
        #         if block.name == 'configure_inspection':
        #             configs['inspection'] = block.input
        #         elif block.name == 'configure_model':
        #             configs['model'] = block.input
        #         elif block.name == 'configure_selection':
        #             configs['selection'] = block.input

        # Add default configs if missing
        for key, default in [
            ('inspection', {"n_clusters": 5, "correlation_threshold": 0.5}),
            ('model', {"kernel_type": "matern", "preprocessing": "robust", 
                      "acquisition": "ei", "surrogate_weight": 0.8}),
            ('selection', {"method": "hybrid", "quality_weight": 0.7, 
                          "uncertainty_bonus": 0.2})
        ]:
            if key not in configs:
                print(f"Warning: No {key} config from LLM, using defaults: {default}")
                configs[key] = default
        
        return configs

    def run_iteration(self, num_runs: int) -> None:
        """Run a complete iteration of the optimization workflow"""
        print(f"\n=== Starting optimization iteration for {self.platform}/{self.design} ===")
        print(f"Objective: {self.objective}")
        print(f"Number of runs requested: {num_runs}")
        
        # Step 1: Inspect logs
        print("\nStep 1: Inspecting logs...")
        log_data = self.inspect_logs()
        print(f"Found {log_data['summary']['total_runs']} total runs, "
              f"{log_data['summary']['successful_runs']} successful")
        
        # Get LLM recommendations for inspection and analysis
        print("\nGetting LLM recommendations for inspection...")
        inspect_configs = self._call_llm('inspect', {
            'log_data': log_data,
            'initial_params': self.initial_params,
            'sdc_context': self.sdc_context
        })
        inspection_config = inspect_configs.get('inspect', {})
        print(f"LLM inspection config: {inspection_config}")
        
        # Step 2: Analyze metrics with LLM config
        print("\nStep 2: Analyzing metrics...")
        metrics = self.analyze_metrics(
            log_data, 
            n_clusters=inspect_configs['inspection']['n_clusters'],
            correlation_threshold=inspect_configs['inspection']['correlation_threshold']
        )
        print(f"Processed metrics for {len(metrics.get('objectives', []))} successful runs")
        
        # Get LLM recommendations for modeling based on inspection results
        print("\nGetting LLM recommendations for modeling...")
        model_configs = self._call_llm('model', {
            'log_data': log_data,
            'metrics': metrics,
            'initial_params': self.initial_params,
            'sdc_context': self.sdc_context,
            'inspection_results': inspect_configs
        })
        model_config = model_configs.get('model', {})
        print(f"LLM model config:  {model_config}")
        
        # Step 3: Evaluate models with LLM config
        print("\nStep 3: Evaluating models...")
        model_results = self.evaluate_models(
            log_data, metrics,
            kernel_type=model_configs['model']['kernel_type'],
            preprocessing=model_configs['model']['preprocessing'],
            acquisition=model_configs['model']['acquisition'],
            surrogate_weight=model_configs['model']['surrogate_weight']
        )

        # ReAct-style feedback loop: use model performance to refine model_config
        print("\n[ReAct] Adjusting model config based on GPR performance feedback...")
        model_config = self._adjust_model_config_with_feedback(model_results, model_configs.get('model', {}))
        
        # Get LLM recommendations for parameter selection based on all previous results
        print("\nGetting LLM recommendations for parameter selection...")
        selection_configs = self._call_llm('agglomerate', {
            'log_data': log_data,
            'metrics': metrics,
            'model_results': model_results,
            'initial_params': self.initial_params,
            'sdc_context': self.sdc_context,
            'inspection_results': inspect_configs,
            'model_configs': model_configs
        })
        selection_config = selection_configs.get('agglomerate', {})
        print(f"LLM selection config: {selection_config}")
        
        # Step 4: Generate parameters with LLM config
        print("\nStep 4: Generating parameters...")
        self.generate_parameters(
            log_data, metrics, model_results, num_runs,
            selection_method=selection_configs['selection']['method'],
            quality_weight=selection_configs['selection']['quality_weight'],
            uncertainty_bonus=selection_configs['selection']['uncertainty_bonus']
        )

        flow_summary = {
            # "flow_index": flow_index,
            "log_data": log_data,
            "metrics": metrics,
            "model_results": model_results,
            "inspection_config": inspection_config,
            "model_config": model_config,
            "selection_config": selection_config,
            "num_runs": num_runs,
            "recent_errors": self._collect_recent_errors({"log_data": log_data})
        }

        # Get gradient (Supervisor Feedback)
        supervisor_feedback = self._run_supervisor_review(flow_summary)
        self.supervision_feedback = supervisor_feedback or ""
            
        if supervisor_feedback:
            # [TextGrad Integration] 3. Perform Backpropagation Steps
            # Here we assume we need to optimize the Prompt in the 'agglomerate' stage, as this is crucial for parameter generation.
            # You can also intelligently decide which stage to optimize based on the feedback content.
            print("[TextGrad] Backpropagating feedback to Agent Prompts...")
            self._optimize_prompt_with_textgrad('agglomerate', supervisor_feedback)
            self._optimize_prompt_with_textgrad('model', supervisor_feedback)
        else:
            print("[SUPERVISOR] No feedback or final iteration; skipping optimization.")        

    def _run_supervisor_review(self, flow_summary: Dict[str, Any]) -> str:
        """
        Use a secondary model to critique the previous flow and return feedback
        for the next prompt.
        """
        flow_snapshot = {
            "log_summary": flow_summary.get("log_data", {}).get("summary", {}),
            "inspection_config": flow_summary.get("inspection_config", {}),
            "model_config": flow_summary.get("model_config", {}),
            "selection_config": flow_summary.get("selection_config", {}),
            "recent_errors": flow_summary.get("recent_errors", ""),
            "objective": self.objective,
            "platform": self.platform,
            "design": self.design,
        }

        supervisor_prompt = (
            "You are the supervisor model for an EDA optimization agent.\n"
            "The primary agent just finished an inspect-optimize-agglomerate flow.\n"
            "Provide concise feedback (<=300 words) to guide the next flow. "
            "Focus on:\n"
            "1) What to adjust in inspection/model/selection configs,\n"
            "2) Specific parameter or constraint reminders,\n"
            "3) Warnings about errors or data issues.\n"
            "Return plain text; start the message with 'Supervisor Feedback:'.\n"
            f"Flow summary (JSON):\n{json.dumps(flow_snapshot, ensure_ascii=False, indent=2)}"
        )
        print(f"[SUPERVISOR] Supervisor prompt: {supervisor_prompt}")

        try:
            response = self.supervisor_client.chat.completions.create(
                model=self.supervisor_model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior EDA optimization reviewer. Be concise and actionable.",
                    },
                    {"role": "user", "content": supervisor_prompt},
                ],
                temperature=0.2,
                max_tokens=4096,
            )
            feedback = response.choices[0].message.content.strip()
            print(f"[SUPERVISOR] Feedback generated (len={len(feedback)}).")
            return feedback
        except Exception as e:
            print(f"[SUPERVISOR] Failed to generate feedback: {e}")
            return ""

    def inspect_logs(self) -> Dict[str, Any]:
        """Step 1: Inspect all .log and .json logs recursively"""
        log_dir = "logs"
        pattern = f"{self.platform}_{self.design}_run"
        
        log_data = {
            'runs': [],
            'summary': {
                'total_runs': 0,
                'successful_runs': 0,
                'failed_runs': 0
            }
        }

        if not os.path.exists(log_dir):
            return log_data

        run_groups = {}

        def _load_run_parameters(run_id: str, log_path: str) -> Dict[str, Any]:
            """Recover parameters for a run from sidecar JSON or design CSV."""
            # Sidecar json next to log
            sidecar = Path(log_path).with_suffix(".params.json")
            if sidecar.exists():
                try:
                    data = json.load(open(sidecar))
                    if isinstance(data, dict):
                        return data.get("parameters", data)
                except Exception:
                    pass
            # Map base_n to nth row in design CSV
            m = re.search(r'(\d+)', run_id)
            if m:
                idx = int(m.group(1)) - 1
                csv_path = Path(f"designs/{self.platform}/{self.design}/{self.platform}_{self.design}.csv")
                if csv_path.exists():
                    try:
                        with csv_path.open() as f:
                            reader = list(csv.reader(f))
                        if not reader:
                            return {}
                        header, rows = reader[0], reader[1:]
                        if 0 <= idx < len(rows):
                            row = rows[idx]
                            params = {}
                            for name, val in zip(header, row):
                                try:
                                    params[name] = float(val)
                                except Exception:
                                    params[name] = val
                            return params
                    except Exception:
                        return {}
            return {}

        def _ingest_log_path(log_path: str, run_id: str):
            run_data = process_log_file(log_path)
            if run_id not in run_groups:
                run_groups[run_id] = {
                    'run_id': run_id,
                    'success': True,
                    'metrics': {},
                    'errors': [],
                    'files': []
                }
            run_groups[run_id]['success'] &= run_data.get('success', True)
            run_groups[run_id]['files'].append(log_path)
            if 'metrics' in run_data and run_data['metrics']:
                run_groups[run_id]['metrics'].update(run_data['metrics'])
            if 'errors' in run_data and run_data['errors']:
                run_groups[run_id]['errors'].extend(run_data['errors'])
            if 'parameters' not in run_groups[run_id]:
                params = _load_run_parameters(run_id, log_path)
                if params:
                    run_groups[run_id]['parameters'] = params

        # 当前迭代 logs/
        for root, _, files in os.walk(log_dir):
            for log_file in files:
                if not log_file.endswith(('.log', '.json')):
                    continue
                log_path = os.path.join(root, log_file)
                if log_file.startswith(pattern) or (self.platform in root and self.design in root):
                    run_id = "main"
                    m = re.search(r'_run(\d+)\.log$', log_file)
                    if m:
                        run_id = f"base_{m.group(1)}"
                    elif 'base_' in root:
                        run_id = os.path.basename(root)
                    _ingest_log_path(log_path, run_id)

        # # 历史迭代 backup_dir/<platform>/<design>/result_dump_*/
        # if self.history_root.exists():
        #     for dump_dir in sorted(self.history_root.glob("result_dump_*")):
        #         hist_logs = dump_dir.glob("logs_dump/**/*")
        #         for log_path in hist_logs:
        #             if log_path.suffix not in ('.log', '.json'):
        #                 continue
        #             run_id = f"{dump_dir.name}_{log_path.stem}"
        #             _ingest_log_path(str(log_path), run_id)

        for run_id, run_data in run_groups.items():
            print(f"DEBUG: Run {run_id} success: {run_data['success']}")
            print(f"DEBUG: Run {run_id} metrics: {run_data['metrics']}")
            print(f"DEBUG: Run {run_id} errors: {run_data['errors']}")
            # print(f"DEBUG: Run {run_id} files: {run_data['files']}")

            log_data['runs'].append(run_data)
            log_data['summary']['total_runs'] += 1
            if run_data['success']:
                log_data['summary']['successful_runs'] += 1
            else:
                log_data['summary']['failed_runs'] += 1

        return log_data
        
    def analyze_metrics(self, log_data: Dict[str, Any], n_clusters: int, correlation_threshold: float) -> Dict[str, Any]:
        """Step 2: Analyze metrics from log data with improved analysis"""
        metrics = {
            'objectives': [],
            'surrogates': [],
            'correlations': {},
            'structure_analysis': {}
        }
        
        # Extract feature vectors and objectives
        feature_vectors = []
        objective_values = []
        surrogate_values = []
        
        for run in log_data['runs']:
            if run['success'] and 'metrics' in run:
                run_metrics = run['metrics']
                obj_values = self._calculate_objective(run)
                
                if obj_values['value'] is not None or obj_values['surrogate'] is not None:
                    # Extract parameter values as features
                    params = []
                    for param in self.parameter_names:
                        params.append(float(run.get('parameters', {}).get(param, self.initial_params.get(param, 0))))
                    feature_vectors.append(params)
                    
                    objective_values.append(obj_values['value'] if obj_values['value'] is not None else np.nan)
                    surrogate_values.append(obj_values['surrogate'] if obj_values['surrogate'] is not None else np.nan)
                    
                    # Track correlations and other metrics
                    if obj_values['value'] is not None:
                        metrics['objectives'].append(obj_values['value'])
                    if obj_values['surrogate'] is not None:
                        metrics['surrogates'].append(obj_values['surrogate'])
                    
                    if obj_values['value'] is not None and obj_values['surrogate'] is not None:
                        if 'real_vs_surrogate' not in metrics['correlations']:
                            metrics['correlations']['real_vs_surrogate'] = []
                        metrics['correlations']['real_vs_surrogate'].append({
                            'real': obj_values['value'],
                            'surrogate': obj_values['surrogate']
                        })
        
        if feature_vectors:
            X = np.array(feature_vectors)
            Y = np.array(objective_values)
            Y_surrogate = np.array(surrogate_values)
            # 缓存给后续阶段使用
            metrics['X'] = X
            metrics['Y'] = Y
            metrics['Y_surrogate'] = Y_surrogate
            metrics['X_list'] = X.tolist()
            metrics['Y_list'] = Y.tolist()
            metrics['Y_surrogate_list'] = Y_surrogate.tolist()
            
            # Use improved inspection functions
            metrics['distribution_analysis'] = inspect_data_distribution(X, Y, Y_surrogate)
            metrics['structure_analysis'] = inspect_data_structure(X, Y, {
                'n_clusters': n_clusters,
                'correlation_threshold': correlation_threshold
            })
            
            # Extract model recommendations if available (from new functions)
            if 'model_recommendations' in metrics['structure_analysis']:
                metrics['model_recommendations'] = metrics['structure_analysis']['model_recommendations']
            
            # Add objective-specific metrics
            if self.objective == 'ECP':
                metrics['clock_period_impact'] = {}
                for run in log_data['runs']:
                    if run['success'] and 'metrics' in run:
                        period = float(run['metrics'].get('clock_period', 0))
                        if period > 0:
                            if period not in metrics['clock_period_impact']:
                                metrics['clock_period_impact'][period] = {
                                    'final_slack': [],
                                    'cts_slack': []
                                }
                            if 'worst_slack' in run['metrics']:
                                metrics['clock_period_impact'][period]['final_slack'].append(
                                    float(run['metrics']['worst_slack']))
                            if 'cts_ws' in run['metrics']:
                                metrics['clock_period_impact'][period]['cts_slack'].append(
                                    float(run['metrics']['cts_ws']))
            
            elif self.objective == 'DWL':
                metrics['wirelength_progression'] = []
                for run in log_data['runs']:
                    if run['success'] and 'metrics' in run:
                        if 'cts_wirelength' in run['metrics'] and 'total_wirelength' in run['metrics']:
                            metrics['wirelength_progression'].append({
                                'cts': float(run['metrics']['cts_wirelength']),
                                'final': float(run['metrics']['total_wirelength'])
                            })

            # 训练集（优先用真实值，否则用 surrogate）
            y_train = np.where(np.isnan(Y), Y_surrogate, Y)
            metrics['X_train'] = X
            metrics['Y_train'] = y_train
            metrics['X_train_list'] = X.tolist()
            metrics['Y_train_list'] = y_train.tolist()
        
        return metrics

    def evaluate_models(self, log_data: Dict[str, Any], metrics: Dict[str, Any], 
                       kernel_type: str, preprocessing: str, acquisition: str, surrogate_weight: float) -> Dict[str, Any]:
        """Step 3: Use improved model functions to evaluate parameter quality"""
        model_results = {}
        
        # Build context from log data and metrics
        context = {
            'log_data': log_data,
            'metrics': metrics,
            'design_config': self.design_config,
            'initial_params': self.initial_params,
            'model_recommendations': metrics.get('model_recommendations', {})
        }
        
        # Use model recommendations if available (from new functions)
        if 'model_recommendations' in metrics:
            kernel_type = metrics['model_recommendations'].get('kernel_type', kernel_type)
            preprocessing = 'robust' if metrics['model_recommendations'].get('needs_feature_scaling', True) else 'none'
        
        # Call appropriate model functions with improved modeling
        if self.objective == 'ECP':
            model_results = evaluate_timing_model(context)
        elif self.objective == 'DWL':
            model_results = evaluate_wirelength_model(context)
        
        # Add model configuration used
            model_results['configuration'] = {
                'kernel_type': kernel_type,
                'preprocessing': preprocessing,
                'acquisition': acquisition,
                'surrogate_weight': surrogate_weight
            }
        
        return model_results

    def _adjust_model_config_with_feedback(self, model_results: Dict[str, Any], base_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        ReAct-style adjustment: summarize model performance and ask LLM to suggest an updated
        model config (kernel_type, preprocessing, acquisition, surrogate_weight).
        """
        prompt = (
            "You are a Bayesian optimization expert. Given the model performance summary below,\n"
            "suggest an updated model configuration to improve predictive quality and balance\n"
            "exploration/exploitation. Respond in JSON with keys: kernel_type, preprocessing,\n"
            "acquisition, surrogate_weight (0-1 float). Keep values concise.\n\n"
            f"Current config: {json.dumps(base_config)}\n"
            f"Model results summary: {json.dumps(model_results, default=str)[:3000]}\n"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=2048,
            )
            content = response.choices[0].message.content
            print(f"[ReAct][ModelFeedback] Raw suggestion: {content}")
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                suggested = json.loads(match.group())
                new_config = {
                    'kernel_type': suggested.get('kernel_type', base_config.get('kernel_type', 'matern')),
                    'preprocessing': suggested.get('preprocessing', base_config.get('preprocessing', 'robust')),
                    'acquisition': suggested.get('acquisition', base_config.get('acquisition', 'ei')),
                    'surrogate_weight': float(suggested.get('surrogate_weight', base_config.get('surrogate_weight', 0.8))),
                }
                print(f"[ReAct][ModelFeedback] Applied updated model config: {new_config}")
                return {'model': new_config}
        except Exception as e:
            print(f"[ReAct][ModelFeedback] Failed to adjust model config: {e}")

        # fallback
        return {'model': base_config if base_config else {
            'kernel_type': 'matern',
            'preprocessing': 'robust',
            'acquisition': 'ei',
            'surrogate_weight': 0.8,
        }}
        
    def generate_parameters(self, log_data: Dict[str, Any], metrics: Dict[str, Any],
                          model_results: Dict[str, Any], num_runs: int,
                          selection_method: str = 'hybrid', quality_weight: float = 0.7,
                          uncertainty_bonus: float = 0.2, model_config: Dict[str, Any] = None) -> None:
        """Step 4: Generate parameter combinations and write to CSV"""
        # Get list of parameters to optimize from constraints
        param_names = list(self.param_constraints.keys())
        print(f"\nGenerating parameters for {len(param_names)} variables:")
        for name in param_names:
            print(f"  - {name}: {self.param_constraints[name]}")
        
        # Extract successful runs for training data
        if metrics.get('X_train') is not None and metrics.get('Y_train') is not None:
            X = np.array(metrics['X_train'])
            y = np.array(metrics['Y_train'])
            print(f"Using cached training data: X_train={X.shape}, Y_train={y.shape}")
        else:
            successful_params = []
            successful_objectives = []
            for run in log_data.get('runs', []):
                if run.get('success', False) and 'metrics' in run:
                    params = {}
                    for param in param_names:
                        params[param] = float(run.get('parameters', {}).get(param, self.initial_params.get(param, 0)))
                    successful_params.append(params)
                    
                    obj = self._calculate_objective(run)
                    successful_objectives.append(obj['value'] if obj['value'] is not None else obj['surrogate'])
            
            print(f"Using {len(successful_params)} successful runs as training data")
            
            # Convert to numpy arrays
            if successful_params:
                X = np.array([[params[name] for name in param_names] for params in successful_params])
                y = np.array(successful_objectives)
            else:
                print("Warning: No successful runs, using initial parameters as base")
                X = np.array([[float(self.initial_params.get(name, 0)) for name in param_names]])
                y = np.array([0.0])
            metrics['X_train'] = X
            metrics['Y_train'] = y
        
        # Get kernel type from model config
        kernel_type = model_config.get('kernel_type', 'matern') if model_config else 'matern'
        print(f"\nCreating surrogate model with {kernel_type} kernel")
        model = create_model(X, y, kernel_type=kernel_type)
        
        # Generate candidates
        n_candidates = num_runs * 10
        print(f"Generating {n_candidates} candidate points")
        candidates = latin_hypercube(n_candidates, len(param_names))
        
        # Scale candidates to parameter ranges
        for i, param in enumerate(param_names):
            constraints = self.param_constraints[param]
            min_val = float(constraints['range'][0])
            max_val = float(constraints['range'][1])
            candidates[:, i] = candidates[:, i] * (max_val - min_val) + min_val
        
        # Get predictions
        predictions, uncertainties = model.predict(candidates, return_std=True)
        
        # Select points
        print(f"\nSelecting points using {selection_method} method")
        print(f"Quality weight: {quality_weight}, Uncertainty bonus: {uncertainty_bonus}")
        quality_scores = create_quality_scores(candidates, y, predictions, uncertainties)
        selected_indices = select_points(
            candidates, quality_scores,
            method=selection_method,
            n_points=num_runs,
            config={"quality_weight": quality_weight, 
                   "uncertainty_bonus": uncertainty_bonus}
        )
        
        selected_params = candidates[selected_indices]
        
        # Validate domain constraints
        print("\nValidating domain constraints...")
        valid_params = []
        for params in selected_params:
            param_dict = {name: value for name, value in zip(param_names, params)}
            if self._validate_domain_constraints(param_dict):
                valid_params.append(param_dict)
            else:
                print("Warning: Parameter set failed domain constraints")
        
        print(f"Found {len(valid_params)} valid parameter sets out of {len(selected_params)}")
        
        # Generate more if needed
        while len(valid_params) < num_runs:
            needed = num_runs - len(valid_params)
            print(f"Generating {needed} additional parameter sets...")
            new_candidates = latin_hypercube(needed, len(param_names))
            for i, param in enumerate(param_names):
                constraints = self.param_constraints[param]
                min_val = float(constraints['range'][0])
                max_val = float(constraints['range'][1])
                new_candidates[:, i] = new_candidates[:, i] * (max_val - min_val) + min_val
            
            for params in new_candidates:
                param_dict = {name: value for name, value in zip(param_names, params)}
                if self._validate_domain_constraints(param_dict):
                    valid_params.append(param_dict)
                if len(valid_params) >= num_runs:
                    break
        
        # Write to CSV
        print(f"\nWriting {num_runs} parameter sets to CSV...")
        self._write_params_to_csv(valid_params[:num_runs])
    
    def _calculate_objective(self, run: Dict[str, Any]) -> Dict[str, float]:
        """Calculate objective value and surrogate from run metrics"""
        metrics = run.get('metrics', {})
        result = {'value': None, 'surrogate': None}
        
        if self.objective == 'ECP':
            if 'clock_period' in metrics:
                period = float(metrics['clock_period'])
                # Real ECP from final worst slack
                if 'worst_slack' in metrics:
                    result['value'] = period - float(metrics['worst_slack'])
                # Surrogate ECP from CTS worst slack
                if 'cts_ws' in metrics:
                    result['surrogate'] = period - float(metrics['cts_ws'])
                    
        elif self.objective == 'DWL':
            # Real wirelength from detailed route
            if 'total_wirelength' in metrics:
                result['value'] = float(metrics['total_wirelength'])
            # Surrogate wirelength from CTS
            if 'cts_wirelength' in metrics:
                result['surrogate'] = float(metrics['cts_wirelength'])
                
        elif self.objective == 'COMBO':
            # Get weights from environment variables
            try:
                ecp_weight = float(os.environ.get('ECP_WEIGHT'))
                wl_weight = float(os.environ.get('WL_WEIGHT'))
                ecp_weight_surrogate = float(os.environ.get('ECP_WEIGHT_SURROGATE'))
                wl_weight_surrogate = float(os.environ.get('WL_WEIGHT_SURROGATE'))
            except (ValueError, TypeError):
                raise ValueError("Weights not properly set in environment variables.")
            
            # Calculate real ECP and WL values
            ecp_value = None
            wl_value = None
            if 'clock_period' in metrics:
                clock_period = float(metrics['clock_period'])
                if 'worst_slack' in metrics:
                    ecp_value = clock_period - float(metrics['worst_slack'])
                if 'total_wirelength' in metrics:
                    wl_value = float(metrics['total_wirelength'])

            # Calculate surrogate ECP and WL values
            ecp_surrogate = None
            wl_surrogate = None
            if 'clock_period' in metrics:
                clock_period = float(metrics['clock_period'])
                if 'cts_ws' in metrics:
                    ecp_surrogate = clock_period - float(metrics['cts_ws'])
                if 'cts_wirelength' in metrics:
                    wl_surrogate = float(metrics['cts_wirelength'])

            # Calculate weighted objective for real values
            if ecp_value is not None and wl_value is not None:
                result['value'] = ecp_weight * ecp_value + wl_weight * wl_value
            elif ecp_value is not None:
                result['value'] = ecp_weight * ecp_value
            elif wl_value is not None:
                result['value'] = wl_weight * wl_value

            # Calculate weighted surrogate objective
            if ecp_surrogate is not None and wl_surrogate is not None:
                result['surrogate'] = ecp_weight_surrogate * ecp_surrogate + wl_weight_surrogate * wl_surrogate
            elif ecp_surrogate is not None:
                result['surrogate'] = ecp_weight_surrogate * ecp_surrogate
            elif wl_surrogate is not None:
                result['surrogate'] = wl_weight_surrogate * wl_surrogate

        return result
    
    def _validate_domain_constraints(self, params: Dict[str, Any]) -> bool:
        """Validate parameter combinations against domain constraints"""
        # 1. Core utilization vs cell padding
        core_util = float(params.get('core_util', 0))
        gp_pad = float(params.get('cell_pad_global', 0))
        dp_pad = float(params.get('cell_pad_detail', 0))
        
        if core_util > 80 and (gp_pad > 2 or dp_pad > 2):
            print(f"Domain constraint failed: core_util > 80 and gp_pad > 2 or dp_pad > 2")
            return False
            
        # 2. TNS end percent vs place density
        tns_end = float(params.get('tns', 0))
        place_density = float(params.get('lb_addon', 0))
        
        if tns_end < 70 and place_density > 0.7:
            print(f"Domain constraint failed: tns_end < 70 and place_density > 0.7")
            return False
            
        # 3. CTS cluster constraints
        cts_size = float(params.get('cts_size', 0))
        cts_diameter = float(params.get('cts_diameter', 0))
        
        if cts_size > 30 and cts_diameter < 100:
            print(f"Domain constraint failed: cts_size > 30 and cts_diameter < 100")
            return False
            
        return True
    
    def _write_params_to_csv(self, params_list: List[Dict[str, Any]]) -> None:
        """Write the new parameter sets to the expected CSV file for the next iteration"""
        csv_file = f"designs/{self.platform}/{self.design}/{self.platform}_{self.design}.csv"
        
        # Ensure parameters are in correct order and properly typed
        with open(csv_file, 'w', newline='\n') as f:  # Explicitly use Unix line endings
            writer = csv.writer(f)
            # Write header in correct order
            writer.writerow(self.parameter_names)
            
            # Write parameter rows
            for params in params_list:
                row = []
                for param_name in self.parameter_names:
                    value = params.get(param_name)
                    if value is None:
                        print(f"Warning: Missing value for parameter {param_name}")
                        continue
                        
                    # Apply type constraints
                    constraint = self.param_constraints[param_name]
                    param_type = constraint['type']
                    param_range = constraint['range']
                    
                    try:
                        # First convert to float for uniform handling
                        value = float(value)
                            
                        # Apply range constraints if they exist
                        if param_range is not None:
                            min_val, max_val = param_range
                            range_size = max_val - min_val
                            
                            # If value is too far out of range, resample uniformly
                            if value > max_val + range_size or value < min_val - range_size:
                                value = random.uniform(min_val, max_val)
                                print(f"Resampled {param_name} to {value} (was too far out of range)")
                            else:
                                # Otherwise just clamp to range
                                value = max(min_val, min(max_val, value))

                        # Convert to final type after range enforcement
                        if param_type == 'int':
                            value = int(round(value))
                            
                    except (ValueError, TypeError) as e:
                        print(f"Error converting {param_name} value '{value}' to {param_type}: {e}")
                        continue
                        
                    row.append(value)
                    
                if len(row) == len(self.parameter_names):
                    writer.writerow(row)
                else:
                    print(f"Warning: Skipping incomplete parameter set")
                    
        print(f"New parameter sets written to {csv_file}")

    def _parse_llm_output(self, llm_response: str) -> List[Dict[str, Any]]:
        """Parse the LLM's response and enforce parameter constraints"""
        try:
            param_sets = json.loads(llm_response)
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM response: {e}")
            return []

        constrained_params = []
        for param_set in param_sets:
            ordered_params = {}
            valid = True
            for param in self.parameter_names:
                value = param_set.get(param)
                if value is None:
                    print(f"Parameter '{param}' is missing in the LLM output.")
                    valid = False
                    break
                    
                constraint = self.param_constraints.get(param)
                if constraint:
                    param_type = constraint['type']
                    param_range = constraint['range']
                    
                    try:
                        # First convert to float for uniform handling
                        value = float(value)

                        # Apply range constraints if they exist
                        if param_range is not None:
                            min_val, max_val = param_range
                            range_size = max_val - min_val
                            
                            # If value is too far out of range, resample uniformly
                            if value > max_val + range_size or value < min_val - range_size:
                                value = random.uniform(min_val, max_val)
                                print(f"Resampled {param} to {value} (was too far out of range)")
                            else:
                                # Otherwise just clamp to range
                                value = max(min_val, min(max_val, value))

                        # Convert to final type after range enforcement
                        if param_type == 'int':
                            value = int(round(value))
                        elif param_type != 'float':
                            raise ValueError(f"Unsupported parameter type: {param_type}")

                        ordered_params[param] = value
                    except (ValueError, TypeError) as e:
                        print(f"Parameter '{param}' has invalid value '{value}': {e}")
                        valid = False
                        break
                else:
                    print(f"Unknown parameter '{param}' in parameter set.")
                    valid = False
                    break
                    
            if valid and self._validate_domain_constraints(ordered_params):
                constrained_params.append(ordered_params)
            else:
                print(f"Parameter set {ordered_params} failed validation and will be discarded.")
                
        return constrained_params

    def generate_initial_parameters(self, num_runs: int) -> None:
        """Generate initial random parameters and write them to CSV using the same method as subsequent iterations"""
        params_list = []
        for _ in range(num_runs):
            params = {}
            for param in self.parameter_names:
                info = self.param_constraints[param]
                param_type = info['type']
                min_value, max_value = info['range']
                if param_type == 'int':
                    value = random.randint(int(min_value), int(max_value))
                elif param_type == 'float':
                    value = random.uniform(min_value, max_value)
                else:
                    continue  # Skip unsupported types
                params[param] = value
            params_list.append(params)

        # Use the same method to write parameters to CSV
        self._write_params_to_csv(params_list)

def main():
    if len(sys.argv) != 5:
        print("Usage: optimize.py <platform> <design> <objective> <num_runs> ")
        sys.exit(1)
        
    platform = sys.argv[1]
    design = sys.argv[2]
    objective = sys.argv[3]
    num_runs = int(sys.argv[4])
    
    workflow = OptimizationWorkflow(platform, design, objective)
    workflow.run_iteration(num_runs)  # Use the run_iteration method instead of individual steps

if __name__ == "__main__":
    main()  
