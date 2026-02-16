Output

# ML Modeling Pipeline — Final Architecture Plan
## AWS Strands · GraphBuilder · DuckDB · Human-in-the-Loop

---

## 1. Architecture Summary

```
USER
  │
  ▼
CHAT AGENT (Strands Agent — standalone, pre-pipeline)
  │ Collects: model types, performance targets, quality thresholds,
  │           doc format/reference, data source path
  │ Output: PipelineConfig (Pydantic validated)
  │
  ▼
PIPELINE INITIALIZER
  │ Creates: DuckDB file, loads raw_data, initializes logger
  │ Builds: invocation_state dict from PipelineConfig
  │ Sets up: S3SessionManager for graph-level persistence
  │
  ▼
GRAPH PIPELINE (Strands GraphBuilder)
  │
  │ ┌─────────────────────┐     ┌─────────────────────────┐
  │ │  DuckDB Version Mgr │     │   Pipeline Logger       │
  │ │  (read/write tables,│     │   (structured action    │
  │ │   auto-profile,     │     │    logs → __pipeline_log │
  │ │   lineage tracking) │     │    with doc_section tags)│
  │ └────────┬────────────┘     └────────────┬────────────┘
  │          │      invocation_state          │
  │          ▼          (shared bus)           ▼
  │
  │  Node 1: DATA SUPERVISOR ──── quality gate ────→
  │  Node 2: FEATURE SUPERVISOR ── feature gate ───→
  │  Node 3: MODEL SUPERVISOR ──── model gate ─────→
  │  Node 4: DOCUMENTATION SUPERVISOR ─────────────→ DONE
  │
  │  Human-in-the-loop: Strands Interrupt at any gate failure
  │
  ▼
OUTPUT
  ├── S3: final datasets (parquet), model artifact, documentation (.docx)
  ├── DuckDB: full version history, quality snapshots, pipeline logs
  └── S3: pipeline.duckdb backup (checkpoint after each supervisor)
```

---

## 2. DuckDB Design

### 2.1 Table Layout

```
Pipeline Tables (agents read/write by name):
─────────────────────────────────────────────
  raw_data                 Immutable. Loaded once at pipeline init.
  cleaned_data             Written by data treatment agent. Overwritten on retry.
  train_data               Written by feature creation agent (post train/test split).
  test_data                Written by feature creation agent.
  feature_matrix_train     Written by feature selection agent (final selected features).
  feature_matrix_test      Written by feature selection agent.
  model_predictions        Written by evaluation agent (best model predictions on test).
  evaluation_results       Written by evaluation agent (metrics table).

System Tables (managed by version manager and logger):
──────────────────────────────────────────────────────
  __version_registry       Every table write is recorded here with lineage.
  __pipeline_log           Every agent action logged here with doc_section tags.
  __quality_snapshots      Auto-generated column-level profiles on every table write.
```

### 2.2 How Agents Interact with DuckDB

Agents never deal with versions or file paths. They call two tools:

```python
read_from_duckdb(table_name: str) → str
    # Loads table into invocation_state["_working_df"]
    # Returns shape + column summary for the LLM

write_to_duckdb(table_name: str, description: str) → str
    # Saves invocation_state["_working_df"] as the named table
    # Auto-creates version entry in __version_registry
    # Auto-creates quality snapshot in __quality_snapshots
    # Updates invocation_state["versions"][table_name]
    # Returns confirmation with version_id
```

### 2.3 Version Manager Module

```python
class DuckDBVersionManager:
    """
    Local DuckDB file with S3 checkpoint sync.
    
    Location: /tmp/pipeline_{run_id}.duckdb (local during execution)
    Backup:   s3://{bucket}/runs/{run_id}/pipeline.duckdb (synced at checkpoints)
    
    Key behaviors:
    - write_table() always does CREATE OR REPLACE (latest version wins)
    - Every write auto-records in __version_registry (full lineage)
    - Every write auto-profiles all columns into __quality_snapshots
    - read_table() always returns the current table (no version lookup needed)
    - get_lineage() reconstructs full parent chain (for documentation)
    - sync_to_s3() called after each supervisor completes (checkpoint)
    """
    
    def __init__(self, db_path, s3_bucket, run_id):
        self.db_path = db_path
        self.s3_bucket = s3_bucket
        self.run_id = run_id
        self.conn = duckdb.connect(db_path)
        self._init_system_tables()
        self._version_counter = 0
    
    def write_table(self, table_name, df, created_by, supervisor,
                    parent_table=None, description=""):
        """Write DataFrame as named table. Returns version_id."""
        self._version_counter += 1
        version_id = f"v_{self._version_counter:03d}_{table_name}"
        
        # Find parent version
        parent_version = None
        if parent_table:
            parent_version = self._get_latest_version_id(parent_table)
        
        # Overwrite table
        self.conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")
        
        # Register version
        self.conn.execute("""
            INSERT INTO __version_registry 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, [version_id, table_name, created_by, supervisor,
              parent_version, self.run_id, len(df), len(df.columns),
              json.dumps(list(df.columns)), description])
        
        # Auto-profile
        self._snapshot_quality(version_id, table_name, df)
        
        return version_id
    
    def read_table(self, table_name):
        """Read current version of a table as DataFrame."""
        return self.conn.execute(f"SELECT * FROM {table_name}").fetchdf()
    
    def list_active_tables(self):
        """Returns {table_name: version_id} for current run."""
        rows = self.conn.execute("""
            SELECT table_name, version_id FROM __version_registry
            WHERE run_id = ?
            ORDER BY table_name, created_at DESC
        """, [self.run_id]).fetchall()
        # Deduplicate — keep latest per table_name
        result = {}
        for table_name, version_id in rows:
            if table_name not in result:
                result[table_name] = version_id
        return result
    
    def get_lineage(self, table_name):
        """Recursive lineage for documentation agent."""
        return self.conn.execute("""
            WITH RECURSIVE lineage AS (
                SELECT * FROM __version_registry 
                WHERE version_id = (
                    SELECT version_id FROM __version_registry 
                    WHERE table_name = ? AND run_id = ?
                    ORDER BY created_at DESC LIMIT 1
                )
                UNION ALL
                SELECT vr.* FROM __version_registry vr
                JOIN lineage l ON vr.version_id = l.parent_version
            )
            SELECT * FROM lineage ORDER BY created_at ASC
        """, [table_name, self.run_id]).fetchdf().to_dict('records')
    
    def get_quality_comparison(self, version_id_before, version_id_after):
        """Compare quality snapshots between two versions (for review agents)."""
        before = self.conn.execute(
            "SELECT * FROM __quality_snapshots WHERE version_id = ?", 
            [version_id_before]
        ).fetchdf()
        after = self.conn.execute(
            "SELECT * FROM __quality_snapshots WHERE version_id = ?", 
            [version_id_after]
        ).fetchdf()
        return {"before": before.to_dict('records'), "after": after.to_dict('records')}
    
    def sync_to_s3(self, s3_client):
        """Checkpoint: flush WAL and upload .duckdb to S3."""
        self.conn.execute("CHECKPOINT")
        s3_client.upload_file(
            self.db_path, self.s3_bucket,
            f"runs/{self.run_id}/pipeline.duckdb"
        )
    
    def export_final_to_s3(self, table_name, s3_client):
        """Export a table to S3 as parquet (for final deliverables)."""
        s3_key = f"runs/{self.run_id}/final/{table_name}.parquet"
        s3_path = f"s3://{self.s3_bucket}/{s3_key}"
        self.conn.execute(f"COPY {table_name} TO '{s3_path}' (FORMAT PARQUET)")
        return s3_path
    
    def _snapshot_quality(self, version_id, table_name, df):
        """Auto-profile every column."""
        snapshots = []
        for col in df.columns:
            snap = {
                'snapshot_id': f"snap_{version_id}_{col}",
                'version_id': version_id,
                'table_name': table_name,
                'column_name': col,
                'dtype': str(df[col].dtype),
                'null_count': int(df[col].isnull().sum()),
                'null_rate': round(float(df[col].isnull().mean()), 4),
                'distinct_count': int(df[col].nunique()),
            }
            if pd.api.types.is_numeric_dtype(df[col]):
                desc = df[col].describe()
                snap.update({
                    'mean': round(float(desc.get('mean', 0)), 4),
                    'std': round(float(desc.get('std', 0)), 4),
                    'min': round(float(desc.get('min', 0)), 4),
                    'p25': round(float(desc.get('25%', 0)), 4),
                    'median': round(float(desc.get('50%', 0)), 4),
                    'p75': round(float(desc.get('75%', 0)), 4),
                    'max': round(float(desc.get('max', 0)), 4),
                })
            snapshots.append(snap)
        snap_df = pd.DataFrame(snapshots)
        self.conn.execute("INSERT INTO __quality_snapshots SELECT * FROM snap_df")
```

### 2.4 Checkpoint Strategy

```
After Data Supervisor completes:
  → pipeline_logger.flush()           # buffer → __pipeline_log
  → db_manager.sync_to_s3(s3_client)  # .duckdb → S3
  
After Feature Supervisor completes:
  → pipeline_logger.flush()
  → db_manager.sync_to_s3(s3_client)
  
After Model Supervisor completes:
  → pipeline_logger.flush()
  → db_manager.export_final_to_s3("feature_matrix_train", s3_client)
  → db_manager.export_final_to_s3("feature_matrix_test", s3_client)
  → db_manager.export_final_to_s3("model_predictions", s3_client)
  → db_manager.sync_to_s3(s3_client)
  
After Documentation Supervisor completes:
  → pipeline_logger.flush()
  → db_manager.export_final_to_s3("evaluation_results", s3_client)
  → db_manager.sync_to_s3(s3_client)  # final checkpoint
```

---

## 3. Pipeline Logger Design

### 3.1 Log Schema

Every agent action produces one log entry:

```python
{
    "log_id":          "uuid",
    "run_id":          "run_20260213_143022",
    "timestamp":       "2026-02-13T14:31:05",
    "supervisor":      "data_supervisor",
    "agent":           "data_quality_agent",
    "action_type":     "data_quality_check",
    "summary":         "Raw data profiled. Found 8% nulls in annual_income, ...",
    "metrics":         {"null_rate_annual_income": 0.08, "duplicate_rate": 0.016},
    "input_version":   "v_001_raw_data",
    "output_version":  null,            # quality check doesn't produce new data
    "tool_calls":      [
        {"tool": "read_from_duckdb", "params": {"table": "raw_data"}, "duration_ms": 45},
        {"tool": "profile_dataset", "params": {}, "duration_ms": 230}
    ],
    "status":          "success",
    "iteration":       1,
    "doc_section":     "3.1 Data Sources and Quality"
}
```

### 3.2 Doc Section Mapping

```python
DOC_SECTION_MAP = {
    # Data stage
    "data_quality_check":       "3.1 Data Sources and Quality",
    "data_treatment":           "3.2 Data Preparation and Cleaning",
    "data_review":              "3.3 Data Quality Assurance",
    
    # Feature stage
    "feature_creation":         "4.1 Variable Construction",
    "feature_selection":        "4.2 Variable Selection and Rationale",
    "feature_validation":       "4.3 Variable Stability Analysis",
    
    # Model stage
    "model_training":           "5.1 Model Development Methodology",
    "hyperparameter_tuning":    "5.2 Model Calibration",
    "model_evaluation":         "5.3 Model Performance Assessment",
    
    # Documentation stage
    "doc_generation":           "7.1 Documentation Completeness",
    "doc_review":               "7.2 Documentation Review",
    
    # Human-in-the-loop
    "human_feedback":           "6.1 Expert Review and Feedback",
}
```

### 3.3 Logger Module

```python
class PipelineLogger:
    """
    Structured action logging for pipeline documentation.
    
    Logs are buffered in memory during supervisor execution and flushed
    to DuckDB's __pipeline_log after each supervisor completes.
    
    Two access patterns:
    1. get_summary_for_agent() — compressed text for LLM context (short)
    2. get_logs_for_doc() — full structured logs for documentation agent
    """
    
    def __init__(self, db_manager, run_id):
        self.db_manager = db_manager
        self.run_id = run_id
        self.buffer = []
    
    def log(self, supervisor, agent, action_type, summary,
            metrics=None, input_version=None, output_version=None,
            tool_calls=None, status="success", iteration=1):
        """Buffer a log entry."""
        self.buffer.append({
            "log_id": str(uuid.uuid4()),
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "supervisor": supervisor,
            "agent": agent,
            "action_type": action_type,
            "summary": summary,
            "metrics": json.dumps(metrics) if metrics else None,
            "input_version": input_version,
            "output_version": output_version,
            "tool_calls": json.dumps(tool_calls) if tool_calls else None,
            "status": status,
            "iteration": iteration,
            "doc_section": DOC_SECTION_MAP.get(action_type, "Other"),
        })
    
    def flush(self):
        """Write buffer to DuckDB."""
        if self.buffer:
            df = pd.DataFrame(self.buffer)
            self.db_manager.conn.execute(
                "INSERT INTO __pipeline_log SELECT * FROM df"
            )
            self.buffer.clear()
    
    def get_summary_for_agent(self, supervisor=None, max_lines=20):
        """Compressed summary for LLM context. Avoids context overflow."""
        query = """
            SELECT agent, action_type, summary, metrics, status 
            FROM __pipeline_log WHERE run_id = ?
        """
        params = [self.run_id]
        if supervisor:
            query += " AND supervisor = ?"
            params.append(supervisor)
        query += " ORDER BY timestamp ASC"
        
        rows = self.db_manager.conn.execute(query, params).fetchall()
        lines = []
        for agent, action, summary, metrics, status in rows[-max_lines:]:
            line = f"[{agent}] {action}: {summary[:150]}"
            if status != "success":
                line += f" ⚠️ {status}"
            lines.append(line)
        return "\n".join(lines)
    
    def get_logs_for_doc(self, section=None):
        """Full structured logs for documentation generation."""
        query = "SELECT * FROM __pipeline_log WHERE run_id = ?"
        params = [self.run_id]
        if section:
            query += " AND doc_section = ?"
            params.append(section)
        query += " ORDER BY timestamp ASC"
        return self.db_manager.conn.execute(query, params).fetchdf().to_dict('records')
```

### 3.4 The log_result Tool (Called by Agents)

```python
@tool(context=True)
def log_result(
    action_type: str,
    summary: str,
    metrics: str = "{}",
    tool_context: ToolContext = None
) -> str:
    """Log the result of a completed action for pipeline documentation.
    
    Call this AFTER completing any significant action (quality check,
    treatment, feature creation, model training, etc.).
    
    Args:
        action_type: Type of action. Must be one of:
            data_quality_check, data_treatment, data_review,
            feature_creation, feature_selection, feature_validation,
            model_training, hyperparameter_tuning, model_evaluation
        summary: Clear, factual summary of what was done and key findings.
            Include specific numbers. Example: "Imputed 387 nulls in 
            annual_income using median (54231). Capped 11 outliers in 
            loan_amount to [2103, 198432] using IQR method."
        metrics: JSON string of key metrics. Example:
            '{"null_rate": 0.0, "rows_affected": 387, "outliers_capped": 11}'
    """
    logger = tool_context.invocation_state["pipeline_logger"]
    db_manager = tool_context.invocation_state["db_manager"]
    versions = tool_context.invocation_state.get("versions", {})
    
    # Parse metrics
    try:
        metrics_dict = json.loads(metrics) if metrics else {}
    except json.JSONDecodeError:
        metrics_dict = {"raw": metrics}
    
    logger.log(
        supervisor=tool_context.invocation_state.get("_current_supervisor", "unknown"),
        agent=tool_context.invocation_state.get("_current_agent", "unknown"),
        action_type=action_type,
        summary=summary,
        metrics=metrics_dict,
    )
    
    return f"✓ Logged [{action_type}]: {summary[:80]}..."
```

---

## 4. Complete invocation_state Schema

```python
invocation_state = {
    # ── Run Identity ──
    "run_id": "run_20260213_143022",
    
    # ── AWS Clients ──
    "s3_client": boto3.client("s3"),
    "artifact_bucket": "ml-pipeline-artifacts",
    
    # ── From Chat Agent (pipeline config) ──
    "model_preferences": ["xgb", "lgb"],
    "performance_targets": {
        "min_auc": 0.75,
        "max_auc": 0.85,
        "max_train_test_gap": 0.10,
        "min_gini": 0.40,
    },
    "quality_thresholds": {
        "max_null_rate": 0.05,
        "max_duplicate_rate": 0.01,
        "max_outlier_rate": 0.05,
        "min_rows": 1000,
    },
    "doc_config": {
        "format": "SR_11_7",
        "reference_doc_path": "s3://templates/sr_11_7_template.docx",
        "include_variable_rationale": True,
    },
    
    # ── DuckDB (core data layer) ──
    "db_manager": "<DuckDBVersionManager instance>",
    "versions": {
        "raw_data": "v_001_raw_data",
        # grows as pipeline progresses:
        # "cleaned_data": "v_002_cleaned_data",
        # "train_data": "v_003_train_data",
        # "test_data": "v_004_test_data",
        # "feature_matrix_train": "v_005_feature_matrix_train",
        # ...
    },
    
    # ── Logger ──
    "pipeline_logger": "<PipelineLogger instance>",
    
    # ── Working State (transient, within-supervisor) ──
    "_working_df": None,                  # current DataFrame in memory
    "_current_supervisor": None,          # set by each graph node
    "_current_agent": None,               # set by each agent-as-tool
    "latest_quality_report": None,        # passed between data sub-agents
    "latest_treatment_log": None,         # passed between data sub-agents
}
```

---

## 5. Graph Definition

```python
from strands import Agent
from strands.multiagent import GraphBuilder
from strands.multiagent.graph import GraphState
from strands.session.s3_session_manager import S3SessionManager

def build_pipeline(invocation_state):
    
    # ── Supervisor Agents ──
    
    data_supervisor = Agent(
        name="data_supervisor",
        system_prompt=DATA_SUPERVISOR_PROMPT,
        tools=[data_quality_agent, data_treatment_agent, data_review_agent,
               read_from_duckdb, log_result],
    )
    
    feature_supervisor = Agent(
        name="feature_supervisor",
        system_prompt=FEATURE_SUPERVISOR_PROMPT,
        tools=[feature_creation_agent, feature_selection_agent, 
               feature_validation_agent, read_from_duckdb, log_result],
    )
    
    model_supervisor = Agent(
        name="model_supervisor",
        system_prompt=MODEL_SUPERVISOR_PROMPT,
        tools=[model_training_agent, hpo_agent, evaluation_agent,
               read_from_duckdb, log_result],
    )
    
    doc_supervisor = Agent(
        name="doc_supervisor",
        system_prompt=DOC_SUPERVISOR_PROMPT,
        tools=[doc_generation_agent, doc_review_agent,
               read_pipeline_logs, read_version_lineage, log_result],
    )
    
    # ── Quality Gate Conditions ──
    
    def data_gate_passed(state: GraphState) -> bool:
        result = state.results.get("data_preparation")
        if not result:
            return False
        return "approved" in str(result.result).lower()
    
    def data_gate_failed(state: GraphState) -> bool:
        return not data_gate_passed(state)
    
    def feature_gate_passed(state: GraphState) -> bool:
        result = state.results.get("feature_engineering")
        if not result:
            return False
        return "approved" in str(result.result).lower()
    
    def feature_gate_failed(state: GraphState) -> bool:
        return not feature_gate_passed(state)
    
    def model_gate_passed(state: GraphState) -> bool:
        result = state.results.get("model_training")
        if not result:
            return False
        text = str(result.result).lower()
        return "approved" in text and "escalat" not in text
    
    def model_gate_failed(state: GraphState) -> bool:
        return not model_gate_passed(state)
    
    # ── Build Graph ──
    
    builder = GraphBuilder()
    
    # Nodes
    builder.add_node(data_supervisor, "data_preparation")
    builder.add_node(feature_supervisor, "feature_engineering")
    builder.add_node(model_supervisor, "model_training")
    builder.add_node(doc_supervisor, "documentation")
    
    # Edges with quality gates
    builder.add_edge("data_preparation", "feature_engineering", 
                     condition=data_gate_passed)
    builder.add_edge("data_preparation", "data_preparation", 
                     condition=data_gate_failed)      # retry loop
    
    builder.add_edge("feature_engineering", "model_training", 
                     condition=feature_gate_passed)
    builder.add_edge("feature_engineering", "feature_engineering", 
                     condition=feature_gate_failed)    # retry loop
    
    builder.add_edge("model_training", "documentation", 
                     condition=model_gate_passed)
    # model_gate_failed → human-in-the-loop (handled inside supervisor)
    
    # Safety limits
    builder.set_entry_point("data_preparation")
    builder.set_max_node_executions(15)
    builder.set_execution_timeout(3600)
    builder.reset_on_revisit(True)
    
    return builder.build()
```

---

## 6. Detailed Agent Flow — All Four Supervisors

### 6.1 Data Supervisor

```
ENTRY: "Prepare raw_data for modeling"

SUPERVISOR DECIDES → data_quality_agent()
  │ 
  │  Internally:
  │  ├─ read_from_duckdb("raw_data")         → loads into _working_df
  │  ├─ profile_dataset()                     → JSON quality report
  │  └─ log_result(
  │        action_type="data_quality_check",
  │        summary="5080 rows, 8 cols. Issues: annual_income 8% nulls,
  │                 employment_length 12% nulls, 1.6% duplicates,
  │                 11 extreme outliers in loan_amount",
  │        metrics='{"null_rate_annual_income": 0.08, 
  │                  "null_rate_employment_length": 0.12,
  │                  "duplicate_rate": 0.016, 
  │                  "outlier_cols": ["loan_amount","interest_rate"]}'
  │     )
  │
  │  Returns: quality report text
  │  Stored in: invocation_state["latest_quality_report"]
  │
SUPERVISOR DECIDES → data_treatment_agent()
  │
  │  Reads: latest_quality_report from invocation_state
  │  Internally:
  │  ├─ read_from_duckdb("raw_data")
  │  ├─ remove_duplicates()                   → 80 rows removed
  │  ├─ impute_nulls("median", "annual_income,employment_length")
  │  │                                        → 387 + 589 nulls filled
  │  ├─ cap_outliers("loan_amount,interest_rate,debt_to_income", "iqr")
  │  │                                        → 31 values capped
  │  ├─ write_to_duckdb("cleaned_data", 
  │  │      description="Deduped, imputed, outliers capped")
  │  │  → DuckDB: cleaned_data table created
  │  │  → __version_registry: v_002_cleaned_data recorded
  │  │  → __quality_snapshots: all columns profiled
  │  │  → invocation_state["versions"]["cleaned_data"] = "v_002_..."
  │  │
  │  └─ log_result(
  │        action_type="data_treatment",
  │        summary="Applied 3 treatments: dedup (-80 rows), median imputation
  │                 (annual_income, employment_length), IQR outlier capping
  │                 (loan_amount, interest_rate, debt_to_income). 
  │                 Final shape: 4920 x 8",
  │        metrics='{"rows_removed": 80, "nulls_imputed": 976, 
  │                  "outliers_capped": 31, "final_shape": [4920, 8]}'
  │     )
  │
  │  Returns: treatment summary
  │  Stored in: invocation_state["latest_treatment_log"]
  │
SUPERVISOR DECIDES → data_review_agent()
  │
  │  Internally:
  │  ├─ read_from_duckdb("cleaned_data")      → loads treated version
  │  ├─ profile_dataset()                     → all thresholds met
  │  └─ log_result(
  │        action_type="data_review",
  │        summary="APPROVED. All thresholds met. Null rates: 0% across all
  │                 columns. Duplicate rate: 0%. Max outlier rate: 3.1%
  │                 (within 5% threshold). Row count: 4920 (above min 1000).",
  │        metrics='{"max_null_rate": 0.0, "duplicate_rate": 0.0,
  │                  "max_outlier_rate": 0.031, "row_count": 4920}'
  │     )
  │
  │  Returns: "APPROVED"

SUPERVISOR OUTPUT: "APPROVED — cleaned_data ready at v_002. 
                    1 iteration. 3 treatments applied."

──── pipeline_logger.flush() ────
──── db_manager.sync_to_s3()  ────
──── quality gate: "approved" in output → PASSES ────
```

### 6.2 Feature Supervisor

```
ENTRY: Previous node output + "Engineer features from cleaned_data"

SUPERVISOR DECIDES → feature_creation_agent()
  │
  │  ├─ read_from_duckdb("cleaned_data")
  │  ├─ create_derived_features()
  │  │  → income_to_debt = annual_income / debt_to_income
  │  │  → loan_income_ratio = loan_amount / annual_income
  │  │  → credit_score_bucket = binned credit_score
  │  │  → high_risk_flag = (credit_score < 600) & (debt_to_income > 20)
  │  │  → ... (12 new features total)
  │  │
  │  ├─ split_train_test(target="default_flag", test_size=0.3, stratify=True)
  │  ├─ write_to_duckdb("train_data", parent="cleaned_data")
  │  ├─ write_to_duckdb("test_data", parent="cleaned_data")
  │  │
  │  └─ log_result(action_type="feature_creation",
  │        summary="Created 12 derived features. Split 70/30 stratified.
  │                 Train: 3444 rows x 20 cols. Test: 1476 rows x 20 cols.",
  │        metrics='{"features_created": 12, "total_features": 20,
  │                  "train_rows": 3444, "test_rows": 1476}')
  │
SUPERVISOR DECIDES → feature_selection_agent()
  │
  │  ├─ read_from_duckdb("train_data")
  │  ├─ calculate_vif()           → flag VIF > 5
  │  ├─ correlation_matrix()      → flag |corr| > 0.85
  │  ├─ feature_importance()      → preliminary XGB importance
  │  ├─ drop_flagged_features()
  │  ├─ write_to_duckdb("feature_matrix_train", parent="train_data")
  │  ├─ write_to_duckdb("feature_matrix_test", parent="test_data")
  │  │   (same feature selection applied to test)
  │  │
  │  └─ log_result(action_type="feature_selection",
  │        summary="Selected 14 of 20 features. Removed 6:
  │                 income_to_debt (VIF=12.3), credit_utilization (VIF=8.7),
  │                 loan_income_ratio (corr=0.91 with loan_amount),
  │                 high_risk_v2 (importance=0.001), ...",
  │        metrics='{"features_selected": 14, "features_removed": 6,
  │                  "max_vif_remaining": 4.2, "max_corr_remaining": 0.71}')
  │
SUPERVISOR DECIDES → feature_validation_agent()
  │
  │  ├─ read_from_duckdb("feature_matrix_train")
  │  ├─ read_from_duckdb("feature_matrix_test")
  │  ├─ calculate_psi()           → PSI per feature (train vs test)
  │  ├─ distribution_comparison() → visual/statistical comparison
  │  │
  │  └─ log_result(action_type="feature_validation",
  │        summary="APPROVED. All features stable between train/test.
  │                 Max PSI: 0.08 (loan_amount). No features above 0.25.",
  │        metrics='{"max_psi": 0.08, "features_above_0.1": 0,
  │                  "features_above_0.25": 0}')

SUPERVISOR OUTPUT: "APPROVED — feature matrices ready. 14 features selected."

──── flush + checkpoint ────
──── feature gate → PASSES ────
```

### 6.3 Model Supervisor

```
ENTRY: Feature gate passed + model_preferences=["xgb", "lgb"]

SUPERVISOR DECIDES → model_training_agent()
  │
  │  Reads invocation_state["model_preferences"] → ["xgb", "lgb"]
  │  ├─ read_from_duckdb("feature_matrix_train")
  │  ├─ train XGBoost (5-fold CV, default params) → mean AUC: 0.71
  │  ├─ train LightGBM (5-fold CV, default params) → mean AUC: 0.73
  │  │
  │  └─ log_result(action_type="model_training",
  │        summary="Baseline models trained. XGB: AUC=0.71. LGB: AUC=0.73.
  │                 Both below min target 0.75. LGB stronger baseline.",
  │        metrics='{"xgb_auc_cv": 0.71, "lgb_auc_cv": 0.73}')
  │
  │  Returns: "Both below target, recommend HPO on LGB"
  │
SUPERVISOR DECIDES → hpo_agent()
  │
  │  Reads invocation_state["performance_targets"]
  │  ├─ Focus on LGB (better baseline)
  │  ├─ Bayesian HPO round 1 → AUC: 0.76 ✓ (crosses 0.75 threshold)
  │  ├─ Bayesian HPO round 2 → AUC: 0.78
  │  ├─ Bayesian HPO round 3 → AUC: 0.79 (diminishing returns, stop)
  │  │
  │  ├─ Also tune XGB → best AUC: 0.74 (still below)
  │  │
  │  └─ log_result(action_type="hyperparameter_tuning",
  │        summary="HPO complete. LGB best AUC=0.79 (within target 0.75-0.85).
  │                 XGB best AUC=0.74 (below target). Best LGB params:
  │                 n_estimators=500, max_depth=6, learning_rate=0.05,
  │                 min_child_samples=20, subsample=0.8",
  │        metrics='{"lgb_best_auc": 0.79, "xgb_best_auc": 0.74,
  │                  "hpo_rounds_lgb": 3, "hpo_rounds_xgb": 2,
  │                  "best_params": {"n_estimators": 500, ...}}')
  │
SUPERVISOR DECIDES → evaluation_agent()
  │
  │  ├─ read_from_duckdb("feature_matrix_test")
  │  ├─ Predict with best LGB model
  │  ├─ Calculate metrics:
  │  │   AUC: 0.77 (train: 0.79, gap: 0.02 ✓ within 0.10)
  │  │   Gini: 0.54 ✓ (above 0.40)
  │  │   KS: 0.38 ✓
  │  │   Calibration: Hosmer-Lemeshow p=0.34 ✓
  │  │
  │  ├─ write_to_duckdb("model_predictions", ...)
  │  ├─ write_to_duckdb("evaluation_results", ...)
  │  ├─ Save model artifact → S3
  │  │
  │  └─ log_result(action_type="model_evaluation",
  │        summary="APPROVED. LGB final: test AUC=0.77, Gini=0.54, KS=0.38.
  │                 Train-test gap=0.02 (threshold 0.10). Calibration good
  │                 (HL p=0.34). Model saved to S3.",
  │        metrics='{"test_auc": 0.77, "train_auc": 0.79, "gini": 0.54,
  │                  "ks": 0.38, "gap": 0.02, "hl_p_value": 0.34}')

SUPERVISOR OUTPUT: "APPROVED — LGB model meets all targets."

  ┌──────────────────────────────────────────────────────────┐
  │  ⚠️  FAILURE PATH: If best AUC = 0.72 after HPO         │
  │                                                          │
  │  Supervisor detects: below performance_targets.min_auc   │
  │                                                          │
  │  Raises Strands Interrupt:                               │
  │    Interrupt(                                            │
  │      reason="performance_target_not_met",                │
  │      data={                                              │
  │        "best_model": "lgb", "best_auc": 0.72,           │
  │        "target": 0.75,                                   │
  │        "options": [                                      │
  │          "1. Relax min_auc to 0.70",                     │
  │          "2. Return to feature engineering",             │
  │          "3. Add model types (rf, neural_net)",          │
  │          "4. Abort pipeline"                             │
  │        ]                                                 │
  │      }                                                   │
  │    )                                                     │
  │                                                          │
  │  → User responds via chat: "Try option 2"               │
  │  → Pipeline routes back to feature_supervisor            │
  │  → New features created, model retrained                 │
  └──────────────────────────────────────────────────────────┘

──── flush + checkpoint + export finals to S3 ────
──── model gate → PASSES ────
```

### 6.4 Documentation Supervisor

```
ENTRY: "Generate model documentation from pipeline logs"

SUPERVISOR DECIDES → doc_generation_agent()
  │
  │  Reads: invocation_state["doc_config"] → {format: "SR_11_7", ...}
  │  
  │  For each section, queries pipeline_logger:
  │
  │  ┌─ Section 3.1: Data Sources and Quality ─────────────────────┐
  │  │  logs = logger.get_logs_for_doc(section="3.1 Data Sources")  │
  │  │  → "The source dataset contained 5,080 observations with     │
  │  │     8 variables. Quality assessment identified:               │
  │  │     - Null rates exceeding 5% threshold in annual_income     │
  │  │       (8.0%) and employment_length (12.0%)                   │
  │  │     - Duplicate rate of 1.6% (threshold: 1.0%)              │
  │  │     - Outlier violations in loan_amount (11 extreme values)  │
  │  │       and interest_rate (10 negative values)"                │
  │  └──────────────────────────────────────────────────────────────┘
  │  
  │  ┌─ Section 3.2: Data Preparation and Cleaning ────────────────┐
  │  │  logs = logger.get_logs_for_doc(section="3.2 Data Prep")     │
  │  │  → "Three treatment categories were applied:                  │
  │  │     1. Deduplication: 80 duplicate rows removed              │
  │  │     2. Null imputation: Median imputation applied to         │
  │  │        annual_income (387 values, median=54,231) and         │
  │  │        employment_length (589 values, median=8.0)            │
  │  │     3. Outlier capping: IQR method applied to loan_amount    │
  │  │        ([2,103 - 198,432]), interest_rate ([3.2 - 20.8]),   │
  │  │        and debt_to_income. Total 31 values capped.           │
  │  │     Final dataset: 4,920 observations, 8 variables."         │
  │  └──────────────────────────────────────────────────────────────┘
  │  
  │  ┌─ Section 4.2: Variable Selection and Rationale ─────────────┐
  │  │  logs = logger.get_logs_for_doc(section="4.2 Variable Sel")  │
  │  │  lineage = db_manager.get_lineage("feature_matrix_train")    │
  │  │  → "14 of 20 candidate variables were selected. Removal      │
  │  │     rationale:                                                │
  │  │     - income_to_debt: VIF = 12.3 (threshold: 5.0)           │
  │  │     - credit_utilization: VIF = 8.7                          │
  │  │     - loan_income_ratio: correlation = 0.91 with loan_amount │
  │  │     - high_risk_v2: importance = 0.001 (near zero)           │
  │  │     All retained variables showed PSI < 0.10 between         │
  │  │     development and validation samples."                      │
  │  └──────────────────────────────────────────────────────────────┘
  │  
  │  ┌─ Section 5.3: Model Performance Assessment ─────────────────┐
  │  │  logs = logger.get_logs_for_doc(section="5.3 Model Perf")    │
  │  │  → "LightGBM was selected as the final model. Performance:   │
  │  │     - Development AUC: 0.79 | Validation AUC: 0.77           │
  │  │     - Gini coefficient: 0.54                                  │
  │  │     - KS statistic: 0.38                                      │
  │  │     - Train-validation gap: 0.02 (threshold: 0.10)           │
  │  │     - Hosmer-Lemeshow calibration p-value: 0.34               │
  │  │     XGBoost was also evaluated (best AUC: 0.74) but did      │
  │  │     not meet minimum performance thresholds."                 │
  │  └──────────────────────────────────────────────────────────────┘
  │  
  │  ... (remaining sections)
  │  
  │  Generates .docx → saves to S3
  │
SUPERVISOR DECIDES → doc_review_agent()
  │
  │  Checks against SR 11-7 template:
  │  ├─ All required sections populated? ✓
  │  ├─ Every metric claim traceable to __pipeline_log? ✓
  │  ├─ Variable rationale backed by selection logs? ✓
  │  ├─ No unsupported claims or fabricated numbers? ✓
  │  │
  │  └─ Returns: "APPROVED — documentation complete and verified"

SUPERVISOR OUTPUT: "Documentation generated and reviewed. 
                    Saved to s3://bucket/runs/run_id/documentation/model_doc.docx"

──── final flush + final S3 checkpoint ────
──── PIPELINE COMPLETE ────
```

---

## 7. Failure Mitigation Matrix

### Data Failures

| # | What Fails | How Detected | What Happens |
|---|-----------|-------------|-------------|
| 1 | Raw data empty/corrupted | Version manager checks row_count=0 on load | Pipeline fails fast before graph starts. Clear error message. |
| 2 | Treatment introduces new issues | Review agent re-profiles cleaned_data | Retry loop: supervisor calls quality → treatment → review again (max 3x) |
| 3 | Too many rows lost after cleanup | Review agent checks against min_rows threshold | If below min_rows: Interrupt → ask user to provide more data or relax threshold |
| 4 | Treatment strategy is non-reproducible | Pipeline logger records exact strategy per column | System prompt prescribes rules: median for numeric, mode for categorical. Logged for audit. |
| 5 | DuckDB table doesn't exist when read | read_from_duckdb returns available tables on error | Agent sees error, adjusts (e.g., reads raw_data instead of cleaned_data) |

### Feature Failures

| # | What Fails | How Detected | What Happens |
|---|-----------|-------------|-------------|
| 6 | Feature explosion (>50 features) | Feature creation agent has hard cap in tool | Tool enforces max_features=50. Excess dropped by importance. |
| 7 | All features have high VIF | Selection agent finds <3 features remaining | Interrupt → ask user: relax VIF threshold or suggest interaction terms |
| 8 | PSI violation (train/test drift) | Validation agent checks PSI per feature | Remove drifting features, re-validate. If too many drift: Interrupt. |
| 9 | Train/test split has target imbalance | Split tool checks target distribution | Enforce stratified split. Log warning if minority < 5%. |

### Model Failures

| # | What Fails | How Detected | What Happens |
|---|-----------|-------------|-------------|
| 10 | Can't meet performance target | HPO agent compares best_auc vs min_auc | Interrupt with options: relax target, more features, new model types, abort |
| 11 | Severe overfitting (gap > threshold) | Evaluation agent checks train-test gap | Route to HPO with stronger regularization. If persistent: Interrupt. |
| 12 | Model training OOM/timeout | try-except in training tool | Reduce data sample or simplify model. Log failure. Retry once. |
| 13 | All models perform similarly poorly | Evaluation agent compares across models | Likely data/feature issue. Route back to feature supervisor via Interrupt. |

### Infrastructure Failures

| # | What Fails | How Detected | What Happens |
|---|-----------|-------------|-------------|
| 14 | DuckDB file corrupted | Exception on read/write | Restore from last S3 checkpoint (.duckdb backup) |
| 15 | LLM API timeout | Strands retry strategy | Exponential backoff (built into Strands). Session state preserved. |
| 16 | Pipeline crash mid-supervisor | S3SessionManager detects incomplete graph | Resume from last completed node. DuckDB has all intermediate data. |
| 17 | S3 write failure | Exception in sync_to_s3/export | Retry with backoff. Local .duckdb is intact as fallback. |

### Agent Reasoning Failures

| # | What Fails | How Detected | What Happens |
|---|-----------|-------------|-------------|
| 18 | Agent picks wrong table to read | DuckDB returns error (table not found) | read_from_duckdb lists available tables. Agent retries with correct name. |
| 19 | Agent hallucinates tool results | Tools return structured JSON, not prose | Review agents independently re-verify by calling tools themselves. |
| 20 | Doc agent invents rationale | Doc review agent cross-checks vs logs | System prompt: "ONLY use facts from __pipeline_log. Never infer." Review catches gaps. |
| 21 | Supervisor infinite retry loop | GraphBuilder max_node_executions=15 | Hard limit. After 15 node executions total, pipeline stops with error. |

### Human-in-the-Loop Edge Cases

| # | What Fails | How Detected | What Happens |
|---|-----------|-------------|-------------|
| 22 | User doesn't respond to Interrupt | Timeout on interrupt | Default to conservative: pause pipeline, save state, notify user |
| 23 | User gives contradictory config change | Chat agent re-validates config | Ask for clarification before resuming |
| 24 | User wants full restart | Explicit command | New run_id, fresh DuckDB, chat agent config preserved |
| 25 | User modifies config mid-pipeline | Config change via interrupt response | Allow: relax thresholds, add model types. Disallow: change data source (requires restart) |

---

## 8. Implementation Phases

### Phase 1: Core Infrastructure + Data Supervisor
**Deliverables:**
- DuckDBVersionManager module (write/read/lineage/snapshot/sync)
- PipelineLogger module (log/flush/get_summary/get_logs_for_doc)
- DuckDB tools: read_from_duckdb, write_to_duckdb, profile_dataset
- Treatment tools: impute_nulls, cap_outliers, remove_duplicates
- log_result tool
- Chat agent with structured output (PipelineConfig)
- Data Supervisor with 3 sub-agents (quality, treatment, review)
- Single-node GraphBuilder (data supervisor only)
- S3 checkpoint sync
- Basic quality gate condition

**Test:** Run end-to-end on synthetic dirty dataset. 
Verify: DuckDB has raw_data + cleaned_data, __version_registry has 2 entries,
__pipeline_log has ~3 entries, __quality_snapshots populated.

### Phase 2: Feature + Model Supervisors
**Deliverables:**
- Feature tools: create_features, split_train_test, calculate_vif, 
  correlation_filter, feature_importance, calculate_psi
- Feature Supervisor with 3 sub-agents
- Model tools: train_model, run_hpo, evaluate_model, save_model
- Model Supervisor with 3 sub-agents
- Full graph with 3 nodes + quality gate conditions
- Human-in-the-loop Interrupts for model failure

**Test:** Full pipeline on synthetic data through model training.
Verify: All DuckDB tables populated, S3 has model artifact,
pipeline log has complete trace.

### Phase 3: Documentation + Polish
**Deliverables:**
- Doc tools: read_pipeline_logs, read_version_lineage, 
  read_quality_snapshots, generate_docx
- Documentation Supervisor with 2 sub-agents (generation + review)
- SR 11-7 section mapping and template integration
- Full 4-node graph
- Export final datasets to S3 parquet

**Test:** Full pipeline produces .docx with all SR 11-7 sections populated.
Verify: Every claim in doc traceable to __pipeline_log entry.

### Phase 4: Production Hardening
**Deliverables:**
- S3SessionManager integration for full resume
- Recovery function: rebuild invocation_state from DuckDB on resume
- Strands OTEL tracing → CloudWatch
- Config validation in chat agent
- Error handling in all tools (graceful failures, clear messages)
- Add evidently for drift reports (feature validation)
- End-to-end integration tests on real data

**Test:** Kill pipeline mid-model-training, resume from checkpoint.
Verify: Feature data preserved, only model training re-runs.
```

---

## 9. Package Decisions

### Use from Day 1 (Phase 1):
- **strands-agents** + **strands-agents-tools**: Core framework
- **duckdb**: Fast intermediate data layer
- **pandas** + **numpy**: Data manipulation
- **boto3**: S3 integration
- **pyarrow**: Parquet read/write

### Add in Phase 2:
- **xgboost** + **lightgbm**: Model training (per user preferences)
- **scikit-learn**: Train/test split, metrics, cross-validation
- **optuna** or **hyperopt**: Bayesian HPO

### Add in Phase 3:
- **python-docx**: Generate SR 11-7 documentation as .docx

### Add in Phase 4 (if needed):
- **evidently**: Drift detection in feature validation
  → Use if: PSI calculation needs statistical depth beyond custom code
  → Skip if: Custom PSI in profile_dataset is sufficient
  
- **great_expectations**: Deterministic quality assertions
  → Use if: LLM review agents prove unreliable at quality gates
  → Skip if: Custom threshold checks in profile_dataset work well
  
- **mlflow**: Model experiment tracking
  → Use if: You need a UI to browse experiments or many model types
  → Skip if: DuckDB __pipeline_log covers your needs

### Explicitly NOT Using:
- **whylogs**: Overlaps with version manager's auto-snapshot. No need for both.
- **LangChain/LangGraph**: Using Strands natively. No need for another framework.
- **Airflow/Step Functions**: Pipeline orchestration is handled by GraphBuilder.
  These would add unnecessary complexity at this stage.
Done